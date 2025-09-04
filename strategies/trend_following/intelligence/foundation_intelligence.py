# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V2.5 最终架构版】基础情报分析总指挥
        - 核心职责: 统一调度所有基础情报的生成，并最终通过协同中心进行提纯。
        - 核心升级 (本次修改):
          - [重构] 将 diagnose_synergy_states, diagnose_multi_timeframe_synergy, 和
                    diagnose_static_multi_dynamic_synergy 三个方法合并为统一的
                    diagnose_synergy_intelligence (协同情报中心)。
          - 优化了整个分析流程，使其更符合“先原子，后协同”的逻辑。
        """
        print("        -> [基础情报分析总指挥 V2.5] 启动...") # 更新版本号和打印信息
        states = {}
        df = self.strategy.df_indicators

        # === 第一阶段: 生成所有基础原子情报 ===
        # 步骤1: 诊断波动率统一情报
        states.update(self.diagnose_volatility_intelligence(df))
        # 步骤2: 诊断震荡与动能统一情报
        states.update(self.diagnose_oscillator_intelligence(df))
        # 步骤3: 诊断资金流与绝对波幅状态
        states.update(self.diagnose_capital_and_range_states(df))
        # 步骤4: 诊断经典技术指标
        states.update(self.diagnose_classic_indicators(df))
        # 步骤5: 诊断市场特征数值化评分
        states.update(self.diagnose_market_character_scores(df))
        
        # 将所有生成的基础原子情报更新到策略实例，为协同诊断做准备
        self.strategy.atomic_states.update(states)

        # === 第二阶段: 运行协同情报中心，生成高级复合信号 ===
        # 步骤6: [重构] 运行统一的协同情报中心
        synergy_intelligence = self.diagnose_synergy_intelligence(df) # 调用全新的统一协同方法
        states.update(synergy_intelligence)

        # 最终更新，确保协同信号也被添加
        self.strategy.atomic_states.update(synergy_intelligence)
        
        print("          -> [基础情报分析总指挥 V2.5] 所有情报已生成。") # 更新打印信息
        return states

    def diagnose_synergy_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.3 全天候作战版】协同情报中心 (Synergy Intelligence Center)
        - 核心职责: 构建一个完备、对称的信号系统，能够“多空双向”识别机会与风险。
        - 核心升级 (本次重构):
          - [新增] Part 3: 引入对称的“下跌共振” (Breakdown) 信号，与“上升共振” (Breakout) 对应。
          - 明确区分四类核心战法:
            1. 上升共振 (机会): 低位压缩后的向上突破。
            2. 下跌共振 (风险): 高位放量后的向下崩溃。
            3. 底部反转 (机会): 趋势衰竭后的向上反转。
            4. 顶部反转 (风险): 趋势衰竭后的向下反转。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 统一军备检查 (Unified Arms Check) ---
        required_cols = [
            'RSI_13_D', 'ACCEL_5_RSI_13_D', 'RSI_13_W', 'SLOPE_5_RSI_13_W',
            'ACCEL_5_close_D', 'SLOPE_5_BBW_21_2.0_D', 'BIAS_55_D',
            'SLOPE_5_EMA_5_D', 'SLOPE_13_EMA_13_D', 'SLOPE_21_EMA_21_D', 'SLOPE_55_EMA_55_D',
            'ACCEL_5_EMA_5_D', 'ACCEL_13_EMA_13_D', 'ACCEL_21_EMA_21_D', 'ACCEL_55_EMA_55_D'
        ]
        required_states = [
            'VOL_STATE_HIGH_VOLATILITY', 'OSC_STATE_RSI_OVERBOUGHT',
            'VOL_STATE_EXTREME_SQUEEZE', 'OSC_STATE_RSI_OVERSOLD',
            'SCORE_VOL_COMPRESSION_POTENTIAL', 'OPP_STATE_NEGATIVE_DEVIATION'
        ]
        
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 协同情报中心缺少必需列: {missing}，模块已跳过。")
            return states
        if not all(s in atomic for s in required_states):
            missing = [s for s in required_states if s not in atomic]
            print(f"          -> [警告] 协同情报中心缺少原子状态: {missing}，模块已跳过。")
            return states

        # === Part 1 & 2: 基础协同与多时间维度协同 - 保持不变 ===
        is_high_vol = atomic.get('VOL_STATE_HIGH_VOLATILITY', default_series)
        is_overbought = atomic.get('OSC_STATE_RSI_OVERBOUGHT', default_series)
        states['RISK_VOL_BLOWOFF_TOP_A'] = is_high_vol & is_overbought
        is_squeezing = atomic.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        is_oversold = atomic.get('OSC_STATE_RSI_OVERSOLD', default_series)
        states['OPP_VOL_SQUEEZE_BOTTOM_B'] = is_squeezing & is_oversold
        is_weekly_bullish = df['SLOPE_5_RSI_13_W'] > 0
        is_daily_igniting = df['ACCEL_5_RSI_13_D'] > 0
        states['OPP_MTF_RSI_BULLISH_ALIGNMENT_A'] = is_weekly_bullish & is_daily_igniting
        is_weekly_diverging = df['SLOPE_5_RSI_13_W'] < 0
        is_daily_overbought = df['RSI_13_D'] > 75
        states['RISK_MTF_RSI_BEARISH_DIVERGENCE_S'] = is_weekly_diverging & is_daily_overbought

        # === Part 3: 置信度分级的共振突破 (Breakout) 与崩溃 (Breakdown) ===
        # --- 3.1 通用计算: 趋势与加速度的共振分数 ---
        slope_cols = ['SLOPE_5_EMA_5_D', 'SLOPE_13_EMA_13_D', 'SLOPE_21_EMA_21_D', 'SLOPE_55_EMA_55_D']
        score_trend_confluence_bullish = sum([(df[col] > 0).astype(int) for col in slope_cols]) / len(slope_cols)
        score_trend_confluence_bearish = sum([(df[col] < 0).astype(int) for col in slope_cols]) / len(slope_cols) # 下跌趋势共振分
        states['SCORE_TREND_CONFLUENCE_BULLISH'] = score_trend_confluence_bullish
        states['SCORE_TREND_CONFLUENCE_BEARISH'] = score_trend_confluence_bearish 
        
        accel_cols = ['ACCEL_5_EMA_5_D', 'ACCEL_13_EMA_13_D', 'ACCEL_21_EMA_21_D', 'ACCEL_55_EMA_55_D']
        score_accel_confluence_bullish = sum([(df[col] > 0).astype(int) for col in accel_cols]) / len(accel_cols)
        score_accel_confluence_bearish = sum([(df[col] < 0).astype(int) for col in accel_cols]) / len(accel_cols) # 下跌加速共振分
        states['SCORE_ACCEL_CONFLUENCE_BULLISH'] = score_accel_confluence_bullish
        states['SCORE_ACCEL_CONFLUENCE_BEARISH'] = score_accel_confluence_bearish 

        # --- 3.2 上升共振机会 (Squeeze Breakout Opportunity) ---
        is_breakout_setup = is_squeezing
        states['OPP_SQUEEZE_BREAKOUT_TACTICAL_B'] = is_breakout_setup & (score_trend_confluence_bullish >= 0.75)
        states['OPP_SQUEEZE_BREAKOUT_STRATEGIC_A'] = is_breakout_setup & (score_trend_confluence_bullish == 1.0)
        states['OPP_SQUEEZE_BREAKOUT_TOTAL_S'] = is_breakout_setup & (score_trend_confluence_bullish == 1.0) & (score_accel_confluence_bullish >= 0.75)
        setup_score = atomic.get('SCORE_VOL_COMPRESSION_POTENTIAL', default_series)
        trigger_score_bullish = (score_trend_confluence_bullish + score_accel_confluence_bullish) / 2
        states['SCORE_SYNERGY_OPPORTUNITY_S'] = (setup_score * trigger_score_bullish).fillna(0).astype(np.float32)

        # --- 3.3 [新增] 下跌共振风险 (High-Vol Breakdown Risk) ---
        is_breakdown_setup = is_high_vol
        states['RISK_HIGH_VOL_BREAKDOWN_TACTICAL_B'] = is_breakdown_setup & (score_trend_confluence_bearish >= 0.75)
        states['RISK_HIGH_VOL_BREAKDOWN_STRATEGIC_A'] = is_breakdown_setup & (score_trend_confluence_bearish == 1.0)
        states['RISK_HIGH_VOL_BREAKDOWN_TOTAL_S'] = is_breakdown_setup & (score_trend_confluence_bearish == 1.0) & (score_accel_confluence_bearish >= 0.75)
        risk_env_score = is_breakdown_setup.astype(int)
        trigger_score_bearish = (score_trend_confluence_bearish + score_accel_confluence_bearish) / 2
        states['SCORE_SYNERGY_RISK_S'] = (risk_env_score * trigger_score_bearish).fillna(0).astype(np.float32)

        # === Part 4: 置信度分级的趋势反转 (Reversal) - 保持不变，逻辑已完备 ===
        # --- 4.1 底部反转机会 (Bullish Reversal Opportunity) ---
        is_long_term_down = df['SLOPE_55_EMA_55_D'] < 0
        is_exhaustion_setup_bullish = is_long_term_down & is_oversold & atomic.get('OPP_STATE_NEGATIVE_DEVIATION', default_series)
        trigger_b_bullish = (df['SLOPE_5_EMA_5_D'] > 0) & (df['SLOPE_5_EMA_5_D'].shift(1) <= 0)
        trigger_a_bullish = trigger_b_bullish & (df['ACCEL_5_EMA_5_D'] > 0)
        is_mid_term_stabilizing = df['SLOPE_13_EMA_13_D'] >= df['SLOPE_13_EMA_13_D'].shift(1)
        trigger_s_bullish = trigger_a_bullish & is_mid_term_stabilizing
        states['OPP_REVERSAL_BOTTOM_TENTATIVE_B'] = is_exhaustion_setup_bullish & trigger_b_bullish
        states['OPP_REVERSAL_BOTTOM_CONFIRMED_A'] = is_exhaustion_setup_bullish & trigger_a_bullish
        states['OPP_REVERSAL_BOTTOM_STRONG_S'] = is_exhaustion_setup_bullish & trigger_s_bullish
        exhaustion_score_bullish = self._normalize_score(df['BIAS_55_D'], ascending=False)
        reversal_strength_bullish = self._normalize_score(df['ACCEL_5_EMA_5_D'])
        states['SCORE_REVERSAL_OPPORTUNITY_S'] = (exhaustion_score_bullish * reversal_strength_bullish * is_exhaustion_setup_bullish).fillna(0).astype(np.float32)

        # --- 4.2 顶部反转风险 (Bearish Reversal Risk) ---
        is_long_term_up = df['SLOPE_55_EMA_55_D'] > 0
        is_exhaustion_setup_bearish = is_long_term_up & is_overbought & is_high_vol
        trigger_b_bearish = (df['SLOPE_5_EMA_5_D'] < 0) & (df['SLOPE_5_EMA_5_D'].shift(1) >= 0)
        trigger_a_bearish = trigger_b_bearish & (df['ACCEL_5_EMA_5_D'] < 0)
        is_mid_term_topping = df['SLOPE_13_EMA_13_D'] <= df['SLOPE_13_EMA_13_D'].shift(1)
        trigger_s_bearish = trigger_a_bearish & is_mid_term_topping
        states['RISK_REVERSAL_TOP_TENTATIVE_B'] = is_exhaustion_setup_bearish & trigger_b_bearish
        states['RISK_REVERSAL_TOP_CONFIRMED_A'] = is_exhaustion_setup_bearish & trigger_a_bearish
        states['RISK_REVERSAL_TOP_STRONG_S'] = is_exhaustion_setup_bearish & trigger_s_bearish
        exhaustion_score_bearish = self._normalize_score(df['BIAS_55_D'], ascending=True)
        reversal_strength_bearish = self._normalize_score(df['ACCEL_5_EMA_5_D'].abs(), ascending=True)
        states['SCORE_REVERSAL_RISK_S'] = (exhaustion_score_bearish * reversal_strength_bearish * is_exhaustion_setup_bearish).fillna(0).astype(np.float32)

        return states

    def diagnose_oscillator_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 全天候作战版】震荡与动能统一情报中心
        - 核心职责: 构建一个完备、对称的信号系统，能够“多空双向”识别震荡指标的机会与风险。
        - 核心升级 (本次重构):
          - [结构] 完全对称化重构，所有多头信号均有对应的空头风险信号。
          - [新增] 补全了看涨背离、正乖离风险、空头钉住、空头动能共振等缺失的对立信号。
          - [新增] 新增S级风险信号“趋势政权下的协同崩溃”，与原有的S级机会信号完美对应。
          - [命名] 全面遵循 OPP/RISK 前缀和 B/A/S 后缀的命名规范。
        """
        states = {}
        p = get_params_block(self.strategy, 'oscillator_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 军备检查 (统一) ---
        required_cols = [
            'RSI_13_D', 'SLOPE_5_RSI_13_D', 'ACCEL_5_RSI_13_D', 'MACDh_13_34_8_D',
            'MACD_HIST_ZSCORE_D', 'BIAS_55_D', 'close_D', 'open_D', 'high_D', 'low_D', # 增加了 high_D, low_D 用于背离计算
            'SLOPE_5_EMA_13_D', 'SLOPE_21_EMA_21_D', 'SLOPE_55_EMA_55_D'
        ]
        required_states = ['VOL_REGIME_TRENDING']
        
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 震荡与动能情报中心缺少必需列: {missing}，模块已跳过。")
            return states
        if not all(s in atomic for s in required_states):
            missing = [s for s in required_states if s not in atomic]
            print(f"          -> [警告] 震荡与动能情报中心缺少原子状态: {missing}，模块已跳过。")
            return states

        # === Part 1: 基础静态诊断 (Foundational Static States) ===
        rsi_col = 'RSI_13_D'
        bias_col = 'BIAS_55_D'
        
        # 1.1 RSI 极值区状态
        states['OSC_STATE_RSI_OVERBOUGHT'] = df[rsi_col] > get_param_value(p.get('rsi_overbought'), 80)
        states['OSC_STATE_RSI_OVERSOLD'] = df[rsi_col] < get_param_value(p.get('rsi_oversold'), 25)

        # 1.2 MACD 方向状态
        states['OSC_STATE_MACD_BULLISH'] = df['MACDh_13_34_8_D'] > 0
        states['OSC_STATE_MACD_BEARISH'] = df['MACDh_13_34_8_D'] < 0 # 对称的空头状态

        # 1.3 BIAS 乖离机会与风险
        p_bias = p.get('bias_dynamic_threshold', {})
        window = get_param_value(p_bias.get('window'), 120)
        quantile = get_param_value(p_bias.get('quantile'), 0.1)
        # 机会: 动态负乖离 (超卖)
        dynamic_oversold_threshold = df[bias_col].rolling(window=window).quantile(quantile)
        is_oversold_bias = df[bias_col] < dynamic_oversold_threshold
        is_rebound_attempt = df['close_D'] > df['open_D']
        states['OPP_STATE_NEGATIVE_DEVIATION'] = is_oversold_bias & is_rebound_attempt
        # 风险: 动态正乖离 (超买)
        dynamic_overbought_threshold = df[bias_col].rolling(window=window).quantile(1 - quantile)
        is_overbought_bias = df[bias_col] > dynamic_overbought_threshold
        is_pullback_attempt = df['close_D'] < df['open_D']
        states['RISK_STATE_POSITIVE_DEVIATION_A'] = is_overbought_bias & is_pullback_attempt

        # === Part 2: 动态与持续性诊断 (Dynamic & Persistent States) ===
        # 2.1 RSI 动态 (加速/减速)
        states['OSC_DYN_RSI_ACCELERATING_BULLISH'] = (df[rsi_col] > 50) & (df['ACCEL_5_RSI_13_D'] > 0)
        states['OSC_DYN_RSI_ACCELERATING_BEARISH'] = (df[rsi_col] < 50) & (df['ACCEL_5_RSI_13_D'] < 0) # 对称的空头加速
        states['RISK_RSI_TOP_DECELERATION_B'] = (df[rsi_col] > 70) & (df['ACCEL_5_RSI_13_D'] < 0) # 命名规范化
        states['OPP_RSI_BOTTOM_ACCELERATION_B'] = (df[rsi_col] < 30) & (df['ACCEL_5_RSI_13_D'] > 0) # 对称的底部加速机会

        # 2.2 MACD 背离
        # 风险: 看跌顶背离 (价格新高, MACD未新高)
        is_price_higher = df['high_D'] > df['high_D'].rolling(10).max().shift(1)
        is_macd_z_lower = df['MACD_HIST_ZSCORE_D'] < df['MACD_HIST_ZSCORE_D'].rolling(10).max().shift(1)
        states['RISK_MACD_BEARISH_DIVERGENCE_A'] = is_price_higher & is_macd_z_lower # 命名规范化
        # 机会: 看涨底背离 (价格新低, MACD未新低)
        is_price_lower = df['low_D'] < df['low_D'].rolling(10).min().shift(1)
        is_macd_z_higher = df['MACD_HIST_ZSCORE_D'] > df['MACD_HIST_ZSCORE_D'].rolling(10).min().shift(1)
        states['OPP_MACD_BULLISH_DIVERGENCE_A'] = is_price_lower & is_macd_z_higher

        # 2.3 RSI 钉住状态 (强势/弱势持续)
        window_peg = 5
        threshold_days = 3
        # 多头钉住: 持续处于超买区
        overbought_level = get_param_value(p.get('rsi_overbought'), 80)
        bullish_persistence_count = (df[rsi_col] > overbought_level).rolling(window=window_peg, min_periods=threshold_days).sum()
        states['OSC_STATE_RSI_PEGGED_BULLISH'] = bullish_persistence_count >= threshold_days
        states['SCORE_OSC_BULLISH_PERSISTENCE'] = (bullish_persistence_count / window_peg).fillna(0).astype(np.float32)
        # 空头钉住: 持续处于超卖区
        oversold_level = get_param_value(p.get('rsi_oversold'), 25)
        bearish_persistence_count = (df[rsi_col] < oversold_level).rolling(window=window_peg, min_periods=threshold_days).sum()
        states['OSC_STATE_RSI_PEGGED_BEARISH'] = bearish_persistence_count >= threshold_days
        states['SCORE_OSC_BEARISH_PERSISTENCE'] = (bearish_persistence_count / window_peg).fillna(0).astype(np.float32)
        
        # === Part 3: 多维度协同与S级信号 (Multi-Dimensional Synergy & S-Class Signals) ===
        # 3.1 多时间维度动能协同
        # 多头共振
        is_short_bullish = df['SLOPE_5_EMA_13_D'] > 0
        is_mid_bullish = df['SLOPE_21_EMA_21_D'] > 0
        is_long_bullish = df['SLOPE_55_EMA_55_D'] > 0
        is_bullish_confluence = is_short_bullish & is_mid_bullish & is_long_bullish
        states['OSC_STATE_MTF_MOMENTUM_BULLISH_CONFLUENCE'] = is_bullish_confluence
        states['SCORE_OSC_BULLISH_CONFLUENCE'] = ((is_short_bullish.astype(int) + is_mid_bullish.astype(int) + is_long_bullish.astype(int)) / 3).astype(np.float32)
        # 空头共振
        is_short_bearish = df['SLOPE_5_EMA_13_D'] < 0
        is_mid_bearish = df['SLOPE_21_EMA_21_D'] < 0
        is_long_bearish = df['SLOPE_55_EMA_55_D'] < 0
        is_bearish_confluence = is_short_bearish & is_mid_bearish & is_long_bearish
        states['OSC_STATE_MTF_MOMENTUM_BEARISH_CONFLUENCE'] = is_bearish_confluence
        states['SCORE_OSC_BEARISH_CONFLUENCE'] = ((is_short_bearish.astype(int) + is_mid_bearish.astype(int) + is_long_bearish.astype(int)) / 3).astype(np.float32)

        # 3.2 S级信号: 趋势政权下的协同触发
        is_trending_regime = atomic.get('VOL_REGIME_TRENDING', default_series)
        # S级机会: 趋势政权 + 多头共振 + RSI加速点火
        is_rsi_accelerating_bullish = df['ACCEL_5_RSI_13_D'] > 0
        states['OPP_REGIME_CONFLUENCE_IGNITION_S'] = is_trending_regime & is_bullish_confluence & is_rsi_accelerating_bullish
        # S级风险: 趋势政权 + 空头共振 + RSI加速崩溃
        is_rsi_accelerating_bearish = df['ACCEL_5_RSI_13_D'] < 0
        states['RISK_REGIME_CONFLUENCE_BREAKDOWN_S'] = is_trending_regime & is_bearish_confluence & is_rsi_accelerating_bearish

        return states

    def diagnose_volatility_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 全天候作战版】波动率统一情报中心
        - 核心职责: 基于多维交叉验证，构建一个对称的、置信度分级的波动率共振与反转信号体系。
        - 核心升级 (本次重构):
          - [数据驱动] 严格基于军械库清单，对日线BBW(静态/斜率/加速度)和周线BBW(静态)进行交叉验证。
          - [逻辑升级] 引入“环境(Setup) + 触发(Trigger)”的分析框架。
          - [信号体系] 生成对称的、B/A/S三级的“上升共振(Breakout)”和“下跌共振(Breakdown)”信号。
          - [新增] 引入基于波动率动态的“临界点(Tipping Point)”信号，作为反转的先行指标。
        """
        states = {}
        p = get_params_block(self.strategy, 'volatility_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 军备检查 (Arsenal Check) ---
        required_cols = [
            'BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_BBW_21_2.0_D', # 日线三维
            'BBW_20_2.0_W',                                                # 周线静态
            'SLOPE_5_close_D', 'ACCEL_5_close_D',                           # 价格动态
            'volume_D', 'VOL_MA_21_D',                                     # 成交量
            'hurst_120d_D'                                                 # 市场政权
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 波动率情报中心缺少必需列: {missing}，模块已跳过。")
            return states

        # === Part 1: 战略环境定义 (Strategic Setup Definition) ===
        # 基于多时间周期的静态交叉验证，定义战场环境
        # 1.1 压缩环境 (Compression Environment)
        is_squeeze_daily = df['BBW_21_2.0_D'] < df['BBW_21_2.0_D'].rolling(120).quantile(0.15)
        is_squeeze_weekly = df['BBW_20_2.0_W'] < df['BBW_20_2.0_W'].rolling(52).quantile(0.20)
        states['VOL_SETUP_COMPRESSION_D'] = is_squeeze_daily
        states['VOL_SETUP_COMPRESSION_W'] = is_squeeze_weekly
        # S级压缩环境：周线和日线同时处于极度压缩状态，能量积蓄最充分
        is_compression_setup_S = is_squeeze_daily & is_squeeze_weekly
        states['VOL_SETUP_COMPRESSION_MTF_S'] = is_compression_setup_S

        # 1.2 高波/扩张环境 (High-Volatility / Expansion Environment)
        is_high_vol_daily = df['BBW_21_2.0_D'] > df['BBW_21_2.0_D'].rolling(120).quantile(0.85)
        is_high_vol_weekly = df['BBW_20_2.0_W'] > df['BBW_20_2.0_W'].rolling(52).quantile(0.80)
        states['VOL_SETUP_HIGH_VOL_D'] = is_high_vol_daily
        states['VOL_SETUP_HIGH_VOL_W'] = is_high_vol_weekly
        # S级高波环境：周线和日线同时处于高波动，风险或机会被放大
        is_expansion_setup_S = is_high_vol_daily & is_high_vol_weekly
        states['VOL_SETUP_EXPANSION_MTF_S'] = is_expansion_setup_S

        # === Part 2: 动态触发条件定义 (Dynamic Trigger Definition) ===
        # 基于斜率和加速度的动态变化
        is_vol_expanding = df['SLOPE_5_BBW_21_2.0_D'] > 0
        is_vol_accelerating = df['ACCEL_5_BBW_21_2.0_D'] > 0
        is_price_trending_up = df['SLOPE_5_close_D'] > 0
        is_price_accelerating_up = df['ACCEL_5_close_D'] > 0
        is_price_trending_down = df['SLOPE_5_close_D'] < 0
        is_price_accelerating_down = df['ACCEL_5_close_D'] < 0
        volume_break_ratio = get_param_value(p.get('volume_break_ratio'), 1.5)
        is_volume_confirming = df['volume_D'] > df['VOL_MA_21_D'] * volume_break_ratio

        # === Part 3: 上升共振 (Breakout) 信号合成 ===
        # 在“压缩环境”下，寻找“向上爆发”的触发
        # B级 (战术级): 环境成立 + 波动率与价格方向初步一致
        trigger_b_bullish = is_vol_expanding & is_price_trending_up
        states['OPP_VOL_BREAKOUT_TACTICAL_B'] = is_compression_setup_S & trigger_b_bullish
        # A级 (战略级): B级条件 + 成交量确认
        trigger_a_bullish = trigger_b_bullish & is_volume_confirming
        states['OPP_VOL_BREAKOUT_STRATEGIC_A'] = is_compression_setup_S & trigger_a_bullish
        # S级 (全局级): A级条件 + 波动率与价格双重加速，形成最强共振
        trigger_s_bullish = trigger_a_bullish & is_vol_accelerating & is_price_accelerating_up
        states['OPP_VOL_BREAKOUT_TOTAL_S'] = is_compression_setup_S & trigger_s_bullish

        # === Part 4: 下跌共振 (Breakdown) 信号合成 ===
        # 在“高波环境”下，寻找“向下崩溃”的触发
        # B级 (战术级): 环境成立 + 波动率扩张 + 价格趋势向下
        trigger_b_bearish = is_vol_expanding & is_price_trending_down
        states['RISK_VOL_BREAKDOWN_TACTICAL_B'] = is_expansion_setup_S & trigger_b_bearish
        # A级 (战略级): B级条件 + 成交量确认 (放量下跌)
        trigger_a_bearish = trigger_b_bearish & is_volume_confirming
        states['RISK_VOL_BREAKDOWN_STRATEGIC_A'] = is_expansion_setup_S & trigger_a_bearish
        # S级 (全局级): A级条件 + 波动率与价格双重加速，形成恐慌式下跌
        trigger_s_bearish = trigger_a_bearish & is_vol_accelerating & is_price_accelerating_down
        states['RISK_VOL_BREAKDOWN_TOTAL_S'] = is_expansion_setup_S & trigger_s_bearish

        # === Part 5: 波动率反转临界点 (Tipping Point) 信号 ===
        # 这些是更领先的信号，预示着状态可能即将改变
        # 机会: 压缩至极，波动率斜率首次转正 (底部反转/突破前夜)
        is_tipping_point_bottom = (df['SLOPE_5_BBW_21_2.0_D'] > 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) <= 0)
        states['OPP_VOL_TIPPING_POINT_BOTTOM_B'] = is_compression_setup_S & is_tipping_point_bottom
        # 风险: 扩张至极，波动率斜率首次转负 (顶部反转/滞涨迹象)
        is_tipping_point_top = (df['SLOPE_5_BBW_21_2.0_D'] < 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) >= 0)
        states['RISK_VOL_TIPPING_POINT_TOP_B'] = is_expansion_setup_S & is_tipping_point_top
        
        # === Part 6: 市场政权与数值化评分 ===
        # 沿用并整合原有评分逻辑
        is_trending_regime = df['hurst_120d_D'] > 0.55
        states['VOL_REGIME_TRENDING'] = is_trending_regime
        
        # 突破潜力分 = 压缩程度 * 趋势政权
        compression_score = self._normalize_score(df['BBW_21_2.0_D'], ascending=False)
        hurst_score = self._normalize_score(df['hurst_120d_D'])
        states['SCORE_VOL_BREAKOUT_POTENTIAL_S'] = compression_score * hurst_score
        
        # 扩张风险分 = 扩张程度 * (1 - 趋势政权得分) -> 在非趋势市中高波更危险
        expansion_score = self._normalize_score(df['BBW_21_2.0_D'], ascending=True)
        states['SCORE_VOL_BREAKDOWN_RISK_S'] = expansion_score * (1 - hurst_score)

        return states

    def _normalize_score(self, series: pd.Series, window: int = 120, ascending: bool = True) -> pd.Series:
        """
        辅助函数：将一个Series进行滚动窗口排名归一化，生成0-1分。
        :param series: 原始数据Series。
        :param window: 归一化滚动窗口。
        :param ascending: 归一化方向，True表示值越大分数越高。
        :return: 归一化后的0-1分数Series。
        """
        min_periods = max(1, window // 5)
        rank = series.rolling(window=window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score = rank if ascending else 1 - rank
        return score.astype(np.float32)

    def diagnose_market_character_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 全天候作战版】市场特征与情绪统一情报中心
        - 核心职责: 基于核心情绪指标(获利盘比例)的“静态/斜率/加速度”三维数据，构建一个对称的、
                    能够识别市场情绪共振与反转的信号体系。
        - 核心升级 (本次重构):
          - [数据驱动] 锁定`total_winner_rate_D`及其多周期斜率/加速度作为分析核心。
          - [逻辑升级] 引入“多时间周期”与“同周期多维度”的交叉验证框架。
          - [信号体系] 生成对称的、B/A/S三级的“上升/下跌共振”与“顶部/底部反转”信号。
          - [评分体系] 建立基于共振强度的数值化评分，量化市场情绪的动能。
        """
        states = {}
        p = get_params_block(self.strategy, 'market_character_params')
        if not get_param_value(p.get('enabled'), False): return states

        # --- 军备检查 (Arsenal Check) ---
        required_cols = [
            'total_winner_rate_D',
            'SLOPE_5_total_winner_rate_D', 'ACCEL_5_total_winner_rate_D', # 短期动态
            'SLOPE_21_total_winner_rate_D', 'ACCEL_21_total_winner_rate_D' # 中期动态
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 市场特征情报中心缺少必需列: {missing}，模块已跳过。")
            return states

        # === Part 1: 基础条件定义 (Fundamental Conditions) ===
        # 将核心指标的各个维度状态布尔化，作为信号合成的积木
        winner_rate = df['total_winner_rate_D']
        
        # 1.1 静态位置 (Static Position)
        is_high_sentiment = winner_rate > winner_rate.rolling(120).quantile(0.85)
        is_low_sentiment = winner_rate < winner_rate.rolling(120).quantile(0.15)

        # 1.2 斜率/速度 (Slope / Velocity)
        is_slope_up_short = df['SLOPE_5_total_winner_rate_D'] > 0
        is_slope_up_mid = df['SLOPE_21_total_winner_rate_D'] > 0
        is_slope_down_short = df['SLOPE_5_total_winner_rate_D'] < 0
        is_slope_down_mid = df['SLOPE_21_total_winner_rate_D'] < 0

        # 1.3 加速度 (Acceleration)
        is_accel_up_short = df['ACCEL_5_total_winner_rate_D'] > 0
        is_accel_up_mid = df['ACCEL_21_total_winner_rate_D'] > 0
        is_accel_down_short = df['ACCEL_5_total_winner_rate_D'] < 0
        is_accel_down_mid = df['ACCEL_21_total_winner_rate_D'] < 0

        # === Part 2: 共振信号合成 (Confluence Signal Synthesis) ===
        # 2.1 上升共振 (赚钱效应扩散)
        # B级: 短期和中期情绪趋势一致向上
        is_bullish_confluence_B = is_slope_up_short & is_slope_up_mid
        states['OPP_MKT_BULLISH_CONFLUENCE_B'] = is_bullish_confluence_B
        # A级: B级基础上，短期情绪仍在加速
        is_bullish_confluence_A = is_bullish_confluence_B & is_accel_up_short
        states['OPP_MKT_BULLISH_CONfluence_A'] = is_bullish_confluence_A
        # S级: A级基础上，中期情绪也同时加速，形成最强共振
        is_bullish_confluence_S = is_bullish_confluence_A & is_accel_up_mid
        states['OPP_MKT_BULLISH_CONFLUENCE_S'] = is_bullish_confluence_S

        # 2.2 下跌共振 (亏钱效应扩散)
        # B级: 短期和中期情绪趋势一致向下
        is_bearish_confluence_B = is_slope_down_short & is_slope_down_mid
        states['RISK_MKT_BEARISH_CONFLUENCE_B'] = is_bearish_confluence_B
        # A级: B级基础上，短期情绪仍在加速下跌
        is_bearish_confluence_A = is_bearish_confluence_B & is_accel_down_short
        states['RISK_MKT_BEARISH_CONFLUENCE_A'] = is_bearish_confluence_A
        # S级: A级基础上，中期情绪也同时加速下跌，形成恐慌共振
        is_bearish_confluence_S = is_bearish_confluence_A & is_accel_down_mid
        states['RISK_MKT_BEARISH_CONFLUENCE_S'] = is_bearish_confluence_S

        # === Part 3: 反转信号合成 (Reversal Signal Synthesis) ===
        # 3.1 底部反转 (冰点回暖)
        # B级: 处于情绪低位区，且下跌趋势开始减速 (短期加速度转正)
        is_bottom_reversal_B = is_low_sentiment & is_accel_up_short
        states['OPP_MKT_BOTTOM_REVERSAL_B'] = is_bottom_reversal_B
        # A级: B级基础上，短期趋势已确认反转 (短期斜率转正)
        is_bottom_reversal_A = is_bottom_reversal_B & is_slope_up_short
        states['OPP_MKT_BOTTOM_REVERSAL_A'] = is_bottom_reversal_A

        # 3.2 顶部反转 (高烧降温)
        # B级: 处于情绪高位区，且上涨趋势开始减速 (短期加速度转负)
        is_top_reversal_B = is_high_sentiment & is_accel_down_short
        states['RISK_MKT_TOP_REVERSAL_B'] = is_top_reversal_B
        # A级: B级基础上，短期趋势已确认反转 (短期斜率转负)
        is_top_reversal_A = is_top_reversal_B & is_slope_down_short
        states['RISK_MKT_TOP_REVERSAL_A'] = is_top_reversal_A

        # === Part 4: 数值化评分 (Quantitative Scoring) ===
        # 将布尔信号转化为0-1之间的连续得分，以量化情绪动能
        # 市场看涨动能分: 综合短期和中期的上升趋势与加速度
        bullish_score = (
            is_slope_up_short.astype(int) * 0.3 +
            is_slope_up_mid.astype(int) * 0.3 +
            is_accel_up_short.astype(int) * 0.2 +
            is_accel_up_mid.astype(int) * 0.2
        ).astype(np.float32)
        states['SCORE_MKT_BULLISH_MOMENTUM'] = bullish_score

        # 市场看跌动能分: 综合短期和中期的下降趋势与加速度
        bearish_score = (
            is_slope_down_short.astype(int) * 0.3 +
            is_slope_down_mid.astype(int) * 0.3 +
            is_accel_down_short.astype(int) * 0.2 +
            is_accel_down_mid.astype(int) * 0.2
        ).astype(np.float32)
        states['SCORE_MKT_BEARISH_MOMENTUM'] = bearish_score

        # 综合市场健康分: 结合静态位置和动态动能
        static_score = self._normalize_score(winner_rate)
        dynamic_score = bullish_score - bearish_score # 范围[-1, 1]
        # 将dynamic_score映射到[0, 1]
        normalized_dynamic_score = (dynamic_score + 1) / 2
        
        # 最终得分: 静态位置权重40%，动态动能权重60%
        states['SCORE_MKT_HEALTH_S'] = (
            static_score * 0.4 + normalized_dynamic_score * 0.6
        ).astype(np.float32)

        return states

    def diagnose_capital_and_range_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 全天候作战版】资金流与绝对波幅统一情报中心
        - 核心职责: 基于CMF和ATR的“静态/斜率/加速度”三维数据，构建一个对称的、
                    能够识别资金共振与反转、以及波幅状态临界点的信号体系。
        - 核心升级 (本次重构):
          - [数据驱动] 严格依赖军备库提供的CMF和ATR的多周期、多维度数据。
          - [逻辑升级] 引入多维交叉验证，生成对称的、B/A/S三级的资金“流入/流出共振”信号。
          - [新增] 新增资金“顶部/底部反转”信号，捕捉资金流向的早期拐点。
          - [新增] 新增ATR“扩张临界点”和“衰竭临界点”信号，作为波动率分析的补充。
        """
        states = {}
        p_capital = get_params_block(self.strategy, 'capital_state_params')
        if not get_param_value(p_capital.get('enabled'), False): return states

        # --- 军备检查 (Arsenal Check) ---
        # 检查基于未来理想数据扩充后的清单
        required_cols = [
            'CMF_21_D',
            'SLOPE_5_CMF_21_D', 'ACCEL_5_CMF_21_D',   # CMF短期动态
            'SLOPE_21_CMF_21_D', 'ACCEL_21_CMF_21_D', # CMF中期动态
            'ATR_14_D',
            'SLOPE_5_ATR_14_D'                       # ATR短期动态
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 资金流情报中心缺少必需列: {missing}，模块已跳过。")
            print(f"          -> [提示] 请确保数据工程层已按要求扩充CMF和ATR的斜率/加速度指标。")
            return states

        # === Part 1: 基础条件定义 (Fundamental Conditions) ===
        cmf = df['CMF_21_D']
        atr = df['ATR_14_D']

        # 1.1 CMF 静态位置与动态
        is_cmf_high = cmf > cmf.rolling(120).quantile(0.85)
        is_cmf_low = cmf < cmf.rolling(120).quantile(0.15)
        is_cmf_inflow = cmf > 0 # 资金净流入
        is_cmf_outflow = cmf < 0 # 资金净流出
        
        is_cmf_slope_up_short = df['SLOPE_5_CMF_21_D'] > 0
        is_cmf_slope_up_mid = df['SLOPE_21_CMF_21_D'] > 0
        is_cmf_slope_down_short = df['SLOPE_5_CMF_21_D'] < 0
        is_cmf_slope_down_mid = df['SLOPE_21_CMF_21_D'] < 0

        is_cmf_accel_up_short = df['ACCEL_5_CMF_21_D'] > 0
        is_cmf_accel_up_mid = df['ACCEL_21_CMF_21_D'] > 0
        is_cmf_accel_down_short = df['ACCEL_5_CMF_21_D'] < 0
        is_cmf_accel_down_mid = df['ACCEL_21_CMF_21_D'] < 0

        # === Part 2: 资金流入共振 (Accumulation Confluence) 信号合成 ===
        # B级: 短期和中期资金流趋势一致向上
        is_inflow_confluence_B = is_cmf_inflow & is_cmf_slope_up_short & is_cmf_slope_up_mid
        states['OPP_CAPITAL_INFLOW_CONFLUENCE_B'] = is_inflow_confluence_B
        # A级: B级基础上，短期流入仍在加速
        is_inflow_confluence_A = is_inflow_confluence_B & is_cmf_accel_up_short
        states['OPP_CAPITAL_INFLOW_CONFLUENCE_A'] = is_inflow_confluence_A
        # S级: A级基础上，中期流入也同时加速，形成最强吸筹共振
        is_inflow_confluence_S = is_inflow_confluence_A & is_cmf_accel_up_mid
        states['OPP_CAPITAL_INFLOW_CONFLUENCE_S'] = is_inflow_confluence_S

        # === Part 3: 资金流出共振 (Distribution Confluence) 信号合成 ===
        # B级: 短期和中期资金流趋势一致向下
        is_outflow_confluence_B = is_cmf_outflow & is_cmf_slope_down_short & is_cmf_slope_down_mid
        states['RISK_CAPITAL_OUTFLOW_CONFLUENCE_B'] = is_outflow_confluence_B
        # A级: B级基础上，短期流出仍在加速
        is_outflow_confluence_A = is_outflow_confluence_B & is_cmf_accel_down_short
        states['RISK_CAPITAL_OUTFLOW_CONFLUENCE_A'] = is_outflow_confluence_A
        # S级: A级基础上，中期流出也同时加速，形成最强派发共振
        is_outflow_confluence_S = is_outflow_confluence_A & is_cmf_accel_down_mid
        states['RISK_CAPITAL_OUTFLOW_CONFLUENCE_S'] = is_outflow_confluence_S

        # === Part 4: 资金流反转 (Reversal) 信号合成 ===
        # 4.1 底部反转 (资金开始回流)
        # B级: 处于资金低位区，且流出趋势开始减速 (短期加速度转正)
        is_bottom_reversal_B = is_cmf_low & is_cmf_accel_up_short
        states['OPP_CAPITAL_BOTTOM_REVERSAL_B'] = is_bottom_reversal_B
        # A级: B级基础上，短期资金流趋势已确认反转 (短期斜率转正)
        is_bottom_reversal_A = is_bottom_reversal_B & is_cmf_slope_up_short
        states['OPP_CAPITAL_BOTTOM_REVERSAL_A'] = is_bottom_reversal_A

        # 4.2 顶部反转 (资金开始离场)
        # B级: 处于资金高位区，且流入趋势开始减速 (短期加速度转负)
        is_top_reversal_B = is_cmf_high & is_cmf_accel_down_short
        states['RISK_CAPITAL_TOP_REVERSAL_B'] = is_top_reversal_B
        # A级: B级基础上，短期资金流趋势已确认反转 (短期斜率转负)
        is_top_reversal_A = is_top_reversal_B & is_cmf_slope_down_short
        states['RISK_CAPITAL_TOP_REVERSAL_A'] = is_top_reversal_A

        # === Part 5: ATR 绝对波幅状态与临界点信号 ===
        # 5.1 静态状态
        is_atr_compression = atr < atr.rolling(120).quantile(0.10)
        is_atr_expansion = atr > atr.rolling(120).quantile(0.90)
        states['VOL_STATE_ATR_COMPRESSION'] = is_atr_compression
        states['VOL_STATE_ATR_EXPANSION'] = is_atr_expansion

        # 5.2 临界点 (Tipping Point)
        # 机会: 绝对波幅压缩至极后，首次开始扩张，预示行情可能启动
        is_tipping_point_expansion = (df['SLOPE_5_ATR_14_D'] > 0) & (df['SLOPE_5_ATR_14_D'].shift(1) <= 0)
        states['OPP_ATR_EXPANSION_IGNITION_B'] = is_atr_compression & is_tipping_point_expansion
        # 风险/观察: 绝对波幅扩张至极后，首次开始收缩，预示趋势可能衰竭或进入盘整
        is_tipping_point_exhaustion = (df['SLOPE_5_ATR_14_D'] < 0) & (df['SLOPE_5_ATR_14_D'].shift(1) >= 0)
        states['RISK_ATR_EXPANSION_EXHAUSTION_B'] = is_atr_expansion & is_tipping_point_exhaustion

        # === Part 6: 数值化评分 ===
        # 资金流入动能分
        inflow_score = (
            is_cmf_slope_up_short.astype(int) * 0.35 +
            is_cmf_slope_up_mid.astype(int) * 0.35 +
            is_cmf_accel_up_short.astype(int) * 0.15 +
            is_cmf_accel_up_mid.astype(int) * 0.15
        ).astype(np.float32)
        states['SCORE_CAPITAL_INFLOW_MOMENTUM'] = inflow_score * is_cmf_inflow.astype(int)

        # 资金流出风险分
        outflow_score = (
            is_cmf_slope_down_short.astype(int) * 0.35 +
            is_cmf_slope_down_mid.astype(int) * 0.35 +
            is_cmf_accel_down_short.astype(int) * 0.15 +
            is_cmf_accel_down_mid.astype(int) * 0.15
        ).astype(np.float32)
        states['SCORE_CAPITAL_OUTFLOW_RISK'] = outflow_score * is_cmf_outflow.astype(int)

        return states

    def diagnose_classic_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 全天候作战版】经典指标统一情报中心
        - 核心职责: 将经典的MACD和成交量分析从“离散事件”升级为基于“静态/斜率/加速度”的
                    “动态过程”分析，构建对称的、置信度分级的共振与反转信号。
        - 核心升级 (本次重构):
          - [数据驱动] 严格依赖军备库提供的MACD柱和成交量的多周期、多维度数据。
          - [逻辑升级] 废弃简单的“金叉/死叉”，引入MACD动能的“上升/下跌共振”信号体系。
          - [新增] 新增MACD动能“顶部/底部反转”信号，更早捕捉趋势拐点。
          - [逻辑升级] 将“静态放量”升级为“动态加速放量”，提升量价信号的可靠性。
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states

        # --- 军备检查 (Arsenal Check) ---
        required_cols = [
            'MACDh_13_34_8_D', 'MACD_HIST_ZSCORE_D',
            'SLOPE_5_MACDh_13_34_8_D', 'ACCEL_5_MACDh_13_34_8_D',   # MACD短期动态
            'SLOPE_21_MACDh_13_34_8_D', 'ACCEL_21_MACDh_13_34_8_D', # MACD中期动态
            'volume_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D',    # 成交量动态
            'close_D', 'open_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 经典指标情报中心缺少必需列: {missing}，模块已跳过。")
            print(f"          -> [提示] 请确保数据工程层已按要求扩充MACDh的斜率/加速度指标。")
            return states

        # === Part 1: MACD 动能分析 (Momentum Analysis) ===
        # 1.1 基础条件定义
        macdh_zscore = df['MACD_HIST_ZSCORE_D']
        is_macdh_positive = df['MACDh_13_34_8_D'] > 0
        is_macdh_negative = df['MACDh_13_34_8_D'] < 0
        
        is_macdh_high = macdh_zscore > 1.5 # Z-Score > 1.5, 动能处于统计高位
        is_macdh_low = macdh_zscore < -1.5 # Z-Score < -1.5, 动能处于统计低位

        is_slope_up_short = df['SLOPE_5_MACDh_13_34_8_D'] > 0
        is_slope_up_mid = df['SLOPE_21_MACDh_13_34_8_D'] > 0
        is_slope_down_short = df['SLOPE_5_MACDh_13_34_8_D'] < 0
        is_slope_down_mid = df['SLOPE_21_MACDh_13_34_8_D'] < 0

        is_accel_up_short = df['ACCEL_5_MACDh_13_34_8_D'] > 0
        is_accel_up_mid = df['ACCEL_21_MACDh_13_34_8_D'] > 0
        is_accel_down_short = df['ACCEL_5_MACDh_13_34_8_D'] < 0
        is_accel_down_mid = df['ACCEL_21_MACDh_13_34_8_D'] < 0

        # 1.2 动能上升共振 (Bullish Momentum Confluence)
        # B级: 短中期动能趋势一致向上，且动能为正
        is_bullish_confluence_B = is_macdh_positive & is_slope_up_short & is_slope_up_mid
        states['OPP_MACD_BULLISH_CONFLUENCE_B'] = is_bullish_confluence_B
        # A级: B级基础上，短期动能仍在加速
        is_bullish_confluence_A = is_bullish_confluence_B & is_accel_up_short
        states['OPP_MACD_BULLISH_CONFLUENCE_A'] = is_bullish_confluence_A
        # S级: A级基础上，中长期动能也同时加速，形成最强共振
        is_bullish_confluence_S = is_bullish_confluence_A & is_accel_up_mid
        states['OPP_MACD_BULLISH_CONFLUENCE_S'] = is_bullish_confluence_S

        # 1.3 动能下跌共振 (Bearish Momentum Confluence)
        # B级: 短中期动能趋势一致向下，且动能为负
        is_bearish_confluence_B = is_macdh_negative & is_slope_down_short & is_slope_down_mid
        states['RISK_MACD_BEARISH_CONFLUENCE_B'] = is_bearish_confluence_B
        # A级: B级基础上，短期动能仍在加速下跌
        is_bearish_confluence_A = is_bearish_confluence_B & is_accel_down_short
        states['RISK_MACD_BEARISH_CONFLUENCE_A'] = is_bearish_confluence_A
        # S级: A级基础上，中长期动能也加速下跌，形成恐慌共振
        is_bearish_confluence_S = is_bearish_confluence_A & is_accel_down_mid
        states['RISK_MACD_BEARISH_CONFLUENCE_S'] = is_bearish_confluence_S

        # 1.4 动能反转 (Momentum Reversal)
        # 机会: 底部反转 (B级) - 动能处于低位，且下跌趋势开始减速(加速度转正)
        is_bottom_reversal_B = is_macdh_low & is_accel_up_short
        states['OPP_MACD_BOTTOM_REVERSAL_B'] = is_bottom_reversal_B
        # 机会: 底部反转 (A级) - B级基础上，短期动能趋势已确认反转(斜率转正)
        is_bottom_reversal_A = is_bottom_reversal_B & is_slope_up_short
        states['OPP_MACD_BOTTOM_REVERSAL_A'] = is_bottom_reversal_A

        # 风险: 顶部反转 (B级) - 动能处于高位，且上涨趋势开始减速(加速度转负)
        is_top_reversal_B = is_macdh_high & is_accel_down_short
        states['RISK_MACD_TOP_REVERSAL_B'] = is_top_reversal_B
        # 风险: 顶部反转 (A级) - B级基础上，短期动能趋势已确认反转(斜率转负)
        is_top_reversal_A = is_top_reversal_B & is_slope_down_short
        states['RISK_MACD_TOP_REVERSAL_A'] = is_top_reversal_A

        # === Part 2: 成交量动态分析 (Volume Dynamics Analysis) ===
        is_price_up = df['close_D'] > df['open_D']
        is_price_down = df['close_D'] < df['open_D']
        
        # 定义成交量动态点火/恐慌条件：趋势向上且仍在加速
        is_volume_igniting = (df['SLOPE_5_volume_D'] > 0) & (df['ACCEL_5_volume_D'] > 0)

        # 机会信号 (A级): [升级] 价格上涨 + 成交量动态点火，是强烈的买盘确认信号
        states['VOL_PRICE_IGNITION_UP_A'] = is_price_up & is_volume_igniting
        
        # 风险信号 (A级): [升级] 价格下跌 + 成交量动态点火(恐慌)，是强烈的风险预警
        states['RISK_VOL_PRICE_CAPITULATION_DOWN_A'] = is_price_down & is_volume_igniting

        # === Part 3: 数值化评分 ===
        bullish_momentum_score = (
            is_slope_up_short.astype(int) * 0.4 +
            is_slope_up_mid.astype(int) * 0.3 +
            is_accel_up_short.astype(int) * 0.2 +
            is_accel_up_mid.astype(int) * 0.1
        ).astype(np.float32)
        states['SCORE_MACD_BULLISH_MOMENTUM'] = bullish_momentum_score * is_macdh_positive.astype(int)

        bearish_momentum_score = (
            is_slope_down_short.astype(int) * 0.4 +
            is_slope_down_mid.astype(int) * 0.3 +
            is_accel_down_short.astype(int) * 0.2 +
            is_accel_down_mid.astype(int) * 0.1
        ).astype(np.float32)
        states['SCORE_MACD_BEARISH_MOMENTUM'] = bearish_momentum_score * is_macdh_negative.astype(int)

        return states










