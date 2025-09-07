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

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V2.5 最终架构版】基础情报分析总指挥
        - 核心职责: 统一调度所有基础情报的生成，并最终通过协同中心进行提纯。
        - 核心升级 (本次修改):
          - 将 diagnose_synergy_states, diagnose_multi_timeframe_synergy, 和
                    diagnose_static_multi_dynamic_synergy 三个方法合并为统一的
                    diagnose_synergy_intelligence (协同情报中心)。
          - 优化了整个分析流程，使其更符合“先原子，后协同”的逻辑。
        """
        print("        -> [基础情报分析总指挥 V2.5] 启动...")
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
        # 步骤6: 运行统一的协同情报中心
        synergy_intelligence = self.diagnose_synergy_intelligence(df) # 调用全新的统一协同方法
        states.update(synergy_intelligence)

        # 最终更新，确保协同信号也被添加
        self.strategy.atomic_states.update(synergy_intelligence)
        
        print("          -> [基础情报分析总指挥 V2.5] 所有情报已生成。") # 更新打印信息
        return states

    def diagnose_synergy_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.6 最终数值化版】协同情报中心 (Synergy Intelligence Center)
        - 核心职责: (原有注释)
        - 核心升级 (本次重构):
          - [最终数值化] 将内部所有关于趋势和反转的布尔判断，全部替换为基于
                          _normalize_score 的连续评分，实现100%数值化协同。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 军备检查 ---
        required_cols = [
            'RSI_13_D', 'ACCEL_5_RSI_13_D', 'RSI_13_W', 'SLOPE_5_RSI_13_W',
            'SLOPE_5_EMA_5_D', 'SLOPE_13_EMA_13_D', 'SLOPE_21_EMA_21_D', 'SLOPE_55_EMA_55_D',
            'ACCEL_5_EMA_5_D', 'ACCEL_13_EMA_13_D', 'ACCEL_21_EMA_21_D', 'ACCEL_55_EMA_55_D'
        ]
        required_states = [
            'SCORE_VOL_EXPANSION_LEVEL', 'SCORE_RSI_OVERBOUGHT_EXTENT',
            'SCORE_VOL_COMPRESSION_LEVEL', 'SCORE_RSI_OVERSOLD_EXTENT',
            'SCORE_BIAS_OVERSOLD_EXTENT'
        ]
        if not all(c in df.columns for c in required_cols) or not all(s in atomic for s in required_states):
            print(f"          -> [警告] 协同情报中心缺少依赖，模块已跳过。")
            return states
        # === Part 1 & 2: 基础与MTF协同 (已数值化) ===
        states['SCORE_RISK_BLOWOFF_TOP_A'] = (atomic.get('SCORE_VOL_EXPANSION_LEVEL', default_score) * atomic.get('SCORE_RSI_OVERBOUGHT_EXTENT', default_score)).astype(np.float32)
        states['SCORE_OPP_SQUEEZE_BOTTOM_B'] = (atomic.get('SCORE_VOL_COMPRESSION_LEVEL', default_score) * atomic.get('SCORE_RSI_OVERSOLD_EXTENT', default_score)).astype(np.float32)
        score_weekly_bullish = self._normalize_score(df['SLOPE_5_RSI_13_W'])
        score_daily_igniting = self._normalize_score(df['ACCEL_5_RSI_13_D'])
        states['SCORE_OPP_MTF_RSI_ALIGNMENT_A'] = (score_weekly_bullish * score_daily_igniting).astype(np.float32)
        score_weekly_diverging = self._normalize_score(df['SLOPE_5_RSI_13_W'], ascending=False)
        score_daily_overbought = self._normalize_score(df['RSI_13_D'].clip(70, 100))
        states['SCORE_RISK_MTF_RSI_DIVERGENCE_S'] = (score_weekly_diverging * score_daily_overbought).astype(np.float32)
        # === Part 3: 共振突破与崩溃 (内部逻辑数值化) ===
        # 将基于布尔计数的共振分，升级为基于归一化强度的平均分
        slope_cols = ['SLOPE_5_EMA_5_D', 'SLOPE_13_EMA_13_D', 'SLOPE_21_EMA_21_D', 'SLOPE_55_EMA_55_D']
        score_trend_confluence_bullish = sum([self._normalize_score(df[col].clip(lower=0)) for col in slope_cols]) / len(slope_cols)
        score_trend_confluence_bearish = sum([self._normalize_score(df[col].clip(upper=0), ascending=False) for col in slope_cols]) / len(slope_cols)
        states['SCORE_TREND_CONFLUENCE_BULLISH'] = score_trend_confluence_bullish.astype(np.float32)
        states['SCORE_TREND_CONFLUENCE_BEARISH'] = score_trend_confluence_bearish.astype(np.float32)
        accel_cols = ['ACCEL_5_EMA_5_D', 'ACCEL_13_EMA_13_D', 'ACCEL_21_EMA_21_D', 'ACCEL_55_EMA_55_D']
        score_accel_confluence_bullish = sum([self._normalize_score(df[col].clip(lower=0)) for col in accel_cols]) / len(accel_cols)
        score_accel_confluence_bearish = sum([self._normalize_score(df[col].clip(upper=0), ascending=False) for col in accel_cols]) / len(accel_cols)
        states['SCORE_ACCEL_CONFLUENCE_BULLISH'] = score_accel_confluence_bullish.astype(np.float32)
        states['SCORE_ACCEL_CONFLUENCE_BEARISH'] = score_accel_confluence_bearish.astype(np.float32)
        trigger_score_bullish = (score_trend_confluence_bullish * 0.6 + score_accel_confluence_bullish * 0.4)
        states['SCORE_SQUEEZE_BREAKOUT_OPP_S'] = (atomic.get('SCORE_VOL_COMPRESSION_LEVEL', default_score) * trigger_score_bullish).astype(np.float32)
        trigger_score_bearish = (score_trend_confluence_bearish * 0.6 + score_accel_confluence_bearish * 0.4)
        states['SCORE_HIGH_VOL_BREAKDOWN_RISK_S'] = (atomic.get('SCORE_VOL_EXPANSION_LEVEL', default_score) * trigger_score_bearish).astype(np.float32)
        # === Part 4: 趋势反转 (内部逻辑数值化) ===
        # 将反转逻辑中的布尔判断全部数值化
        score_long_term_down = self._normalize_score(df['SLOPE_55_EMA_55_D'].clip(upper=0), ascending=False)
        score_exhaustion_bullish = (score_long_term_down * atomic.get('SCORE_RSI_OVERSOLD_EXTENT', default_score) * atomic.get('SCORE_BIAS_OVERSOLD_EXTENT', default_score))
        trigger_bullish_turn = ((df['SLOPE_5_EMA_5_D'] > 0) & (df['SLOPE_5_EMA_5_D'].shift(1) <= 0)).astype(int) # 拐点事件保持布尔
        score_reversal_strength_bullish = self._normalize_score(df['ACCEL_5_EMA_5_D'])
        score_mid_term_stabilizing = self._normalize_score((df['SLOPE_13_EMA_13_D'] - df['SLOPE_13_EMA_13_D'].shift(1)).clip(lower=0))
        score_trigger_bullish = (trigger_bullish_turn * 0.5 + score_reversal_strength_bullish * 0.3 + score_mid_term_stabilizing * 0.2)
        states['SCORE_REVERSAL_BOTTOM_OPP_S'] = (score_exhaustion_bullish * score_trigger_bullish).astype(np.float32)
        score_long_term_up = self._normalize_score(df['SLOPE_55_EMA_55_D'].clip(lower=0))
        score_exhaustion_bearish = (score_long_term_up * atomic.get('SCORE_RSI_OVERBOUGHT_EXTENT', default_score) * atomic.get('SCORE_VOL_EXPANSION_LEVEL', default_score))
        trigger_bearish_turn = ((df['SLOPE_5_EMA_5_D'] < 0) & (df['SLOPE_5_EMA_5_D'].shift(1) >= 0)).astype(int) # 拐点事件保持布尔
        score_reversal_strength_bearish = self._normalize_score(df['ACCEL_5_EMA_5_D'].abs(), ascending=False)
        score_mid_term_topping = self._normalize_score((df['SLOPE_13_EMA_13_D'].shift(1) - df['SLOPE_13_EMA_13_D']).clip(lower=0))
        score_trigger_bearish = (trigger_bearish_turn * 0.5 + score_reversal_strength_bearish * 0.3 + score_mid_term_topping * 0.2)
        states['SCORE_REVERSAL_TOP_RISK_S'] = (score_exhaustion_bearish * score_trigger_bearish).astype(np.float32)
        return states

    def diagnose_oscillator_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.4 最终数值化版】震荡与动能统一情报中心
        - 核心职责: (原有注释)
        - 核心升级 (本次重构):
          - [最终数值化] 将RSI钉住状态的计算逻辑，从“计数”升级为对“程度分”的
                          滚动求和，使其能精确反映持续状态的强度。
        """
        states = {}
        p = get_params_block(self.strategy, 'oscillator_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        # --- 军备检查 (统一) ---
        required_cols = [
            'RSI_13_D', 'ACCEL_5_RSI_13_D', 'MACD_HIST_ZSCORE_D', 'BIAS_55_D',
            'high_D', 'low_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 震荡与动能情报中心缺少必需列: {missing}，模块已跳过。")
            return states
        # === Part 1: 基础静态诊断 (已数值化) ===
        rsi_col = 'RSI_13_D'
        bias_col = 'BIAS_55_D'
        overbought_threshold = get_param_value(p.get('rsi_overbought_start'), 70)
        score_overbought = (df[rsi_col] - overbought_threshold) / (100 - overbought_threshold)
        states['SCORE_RSI_OVERBOUGHT_EXTENT'] = score_overbought.clip(0, 1).astype(np.float32)
        oversold_threshold = get_param_value(p.get('rsi_oversold_start'), 30)
        score_oversold = (oversold_threshold - df[rsi_col]) / (oversold_threshold - 0)
        states['SCORE_RSI_OVERSOLD_EXTENT'] = score_oversold.clip(0, 1).astype(np.float32)
        p_bias = p.get('bias_dynamic_threshold', {})
        window = get_param_value(p_bias.get('window'), 120)
        quantile = get_param_value(p_bias.get('quantile'), 0.1)
        dynamic_oversold_threshold = df[bias_col].rolling(window=window).quantile(quantile)
        negative_deviation = (dynamic_oversold_threshold - df[bias_col]).clip(lower=0)
        states['SCORE_BIAS_OVERSOLD_EXTENT'] = self._normalize_score(negative_deviation, window=window)
        dynamic_overbought_threshold = df[bias_col].rolling(window=window).quantile(1 - quantile)
        positive_deviation = (df[bias_col] - dynamic_overbought_threshold).clip(lower=0)
        states['SCORE_BIAS_OVERBOUGHT_EXTENT'] = self._normalize_score(positive_deviation, window=window)
        # === Part 2: 动态与持续性诊断 (已数值化) ===
        rsi_pos_factor = ((df[rsi_col] - 50) / 50).clip(0, 1)
        rsi_neg_factor = ((50 - df[rsi_col]) / 50).clip(0, 1)
        rsi_accel_score = self._normalize_score(df['ACCEL_5_RSI_13_D'])
        rsi_decel_score = self._normalize_score(df['ACCEL_5_RSI_13_D'], ascending=False)
        states['SCORE_RSI_BULLISH_ACCEL'] = rsi_pos_factor * rsi_accel_score
        states['SCORE_RSI_BEARISH_ACCEL_RISK'] = rsi_neg_factor * rsi_decel_score
        states['SCORE_RSI_TOP_DECELERATION_RISK'] = states['SCORE_RSI_OVERBOUGHT_EXTENT'] * rsi_decel_score
        states['SCORE_RSI_BOTTOM_ACCELERATION_OPP'] = states['SCORE_RSI_OVERSOLD_EXTENT'] * rsi_accel_score
        price_overshoot = (df['high_D'] - df['high_D'].rolling(10).max().shift(1)).clip(lower=0)
        macd_undershoot = (df['MACD_HIST_ZSCORE_D'].rolling(10).max().shift(1) - df['MACD_HIST_ZSCORE_D']).clip(lower=0)
        states['SCORE_MACD_BEARISH_DIVERGENCE_RISK'] = self._normalize_score(price_overshoot) * self._normalize_score(macd_undershoot)
        price_undershoot = (df['low_D'].rolling(10).min().shift(1) - df['low_D']).clip(lower=0)
        macd_overshoot = (df['MACD_HIST_ZSCORE_D'] - df['MACD_HIST_ZSCORE_D'].rolling(10).min().shift(1)).clip(lower=0)
        states['SCORE_MACD_BULLISH_DIVERGENCE_OPP'] = self._normalize_score(price_undershoot) * self._normalize_score(macd_overshoot)
        # 将RSI钉住状态的计算从“计数”升级为对“程度分”的滚动求和
        window_peg = 5
        # 多头钉住：对超买“程度分”进行滚动求和
        bullish_persistence_sum = states['SCORE_RSI_OVERBOUGHT_EXTENT'].rolling(window=window_peg).sum()
        states['SCORE_OSC_BULLISH_PERSISTENCE'] = (bullish_persistence_sum / window_peg).fillna(0).astype(np.float32)
        # 空头钉住：对超卖“程度分”进行滚动求和
        bearish_persistence_sum = states['SCORE_RSI_OVERSOLD_EXTENT'].rolling(window=window_peg).sum()
        states['SCORE_OSC_BEARISH_PERSISTENCE'] = (bearish_persistence_sum / window_peg).fillna(0).astype(np.float32)
        # print("        -> [震荡情报中心] 已将RSI钉住状态升级为基于程度分的滚动求和。")
        return states

    def diagnose_volatility_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.5 最终数值化版】波动率统一情报中心
        - 核心职责: (原有注释)
        - 核心升级 (本次修改):
          - [最终数值化] 将基于分级(0, 0.4, 0.7, 1.0)的压缩/扩张评分体系，升级为
                          一个完全平滑的、基于日线和周线BBW归一化得分的加权平均值，
                          实现压缩/扩张程度的连续、平滑度量。
        """
        states = {}
        p = get_params_block(self.strategy, 'volatility_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        # --- 军备检查 (Arsenal Check) ---
        required_cols = [
            'BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'BBW_21_2.0_W', 'hurst_120d_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 波动率情报中心缺少必需列: {missing}，模块已跳过。")
            return states
        # === Part 1: 战略环境定义 (最终数值化) ===
        # 将分级的压缩/扩张评分，升级为完全平滑的连续评分
        # 1.1 压缩环境 (Compression Environment)
        score_squeeze_daily = self._normalize_score(df['BBW_21_2.0_D'], ascending=False)
        score_squeeze_weekly = self._normalize_score(df['BBW_21_2.0_W'], ascending=False)
        # 使用加权平均（周线权重更高）生成平滑的最终分
        states['SCORE_VOL_COMPRESSION_LEVEL'] = (score_squeeze_daily * 0.4 + score_squeeze_weekly * 0.6).astype(np.float32)
        # 1.2 高波/扩张环境 (High-Volatility / Expansion Environment)
        score_expansion_daily = self._normalize_score(df['BBW_21_2.0_D'], ascending=True)
        score_expansion_weekly = self._normalize_score(df['BBW_21_2.0_W'], ascending=True)
        # 对称的扩张评分逻辑
        states['SCORE_VOL_EXPANSION_LEVEL'] = (score_expansion_daily * 0.4 + score_expansion_weekly * 0.6).astype(np.float32)
        # print("        -> [波动率情报中心] 已将压缩/扩张等级分升级为平滑连续评分。")
        # === Part 5: 波动率反转临界点 (逻辑不变，但消费了新的平滑分) ===
        is_tipping_point_bottom = (df['SLOPE_5_BBW_21_2.0_D'] > 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) <= 0)
        states['SCORE_VOL_TIPPING_POINT_BOTTOM_OPP'] = states['SCORE_VOL_COMPRESSION_LEVEL'] * is_tipping_point_bottom.astype(np.float32)
        is_tipping_point_top = (df['SLOPE_5_BBW_21_2.0_D'] < 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) >= 0)
        states['SCORE_VOL_TIPPING_POINT_TOP_RISK'] = states['SCORE_VOL_EXPANSION_LEVEL'] * is_tipping_point_top.astype(np.float32)
        # === Part 6: 市场政权与数值化评分 (逻辑不变) ===
        hurst_score = self._normalize_score(df['hurst_120d_D'])
        states['SCORE_TRENDING_REGIME'] = hurst_score
        # 注意: 这里的 breakout_potential 和 breakdown_risk 使用的是原始日线分，逻辑更纯粹
        states['SCORE_VOL_BREAKOUT_POTENTIAL_S'] = score_squeeze_daily * hurst_score
        states['SCORE_VOL_BREAKDOWN_RISK_S'] = score_expansion_daily * (1 - hurst_score)
        return states

    def diagnose_market_character_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.2 最终数值化版】市场特征与情绪统一情报中心
        - 核心职责: (原有注释)
        - 核心升级 (本次重构):
          - [最终数值化] 将内部所有布尔判断 (如 is_high_sentiment, is_slope_up)
                          替换为基于 _normalize_score 的连续评分，实现100%数值化。
        """
        states = {}
        p = get_params_block(self.strategy, 'market_character_params')
        if not get_param_value(p.get('enabled'), False): return states
        # --- 军备检查 (Arsenal Check) ---
        required_cols = [
            'total_winner_rate_D', 'SLOPE_5_total_winner_rate_D', 'ACCEL_5_total_winner_rate_D',
            'SLOPE_21_total_winner_rate_D', 'ACCEL_21_total_winner_rate_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 市场特征情报中心缺少必需列: {missing}，模块已跳过。")
            return states
        # === Part 1: 基础条件数值化 (Fundamental Conditions Scoring) ===
        # 将所有布尔条件替换为0-1的连续评分
        winner_rate = df['total_winner_rate_D']
        score_high_sentiment = self._normalize_score(winner_rate.clip(lower=winner_rate.rolling(120).quantile(0.85)))
        score_low_sentiment = self._normalize_score(winner_rate.clip(upper=winner_rate.rolling(120).quantile(0.15)), ascending=False)
        score_slope_up_short = self._normalize_score(df['SLOPE_5_total_winner_rate_D'].clip(lower=0))
        score_slope_up_mid = self._normalize_score(df['SLOPE_21_total_winner_rate_D'].clip(lower=0))
        score_slope_down_short = self._normalize_score(df['SLOPE_5_total_winner_rate_D'].clip(upper=0), ascending=False)
        score_slope_down_mid = self._normalize_score(df['SLOPE_21_total_winner_rate_D'].clip(upper=0), ascending=False)
        score_accel_up_short = self._normalize_score(df['ACCEL_5_total_winner_rate_D'].clip(lower=0))
        score_accel_up_mid = self._normalize_score(df['ACCEL_21_total_winner_rate_D'].clip(lower=0))
        score_accel_down_short = self._normalize_score(df['ACCEL_5_total_winner_rate_D'].clip(upper=0), ascending=False)
        score_accel_down_mid = self._normalize_score(df['ACCEL_21_total_winner_rate_D'].clip(upper=0), ascending=False)
        # === Part 2 & 3: 共振与反转信号数值化 (使用新评分) ===
        # 上升共振评分
        score_bullish_confluence = (
            score_slope_up_short * 0.35 + score_slope_up_mid * 0.35 +
            score_accel_up_short * 0.20 + score_accel_up_mid * 0.10
        ).astype(np.float32)
        states['SCORE_MKT_BULLISH_CONFLUENCE'] = score_bullish_confluence
        # 下跌共振风险评分
        score_bearish_confluence = (
            score_slope_down_short * 0.35 + score_slope_down_mid * 0.35 +
            score_accel_down_short * 0.20 + score_accel_down_mid * 0.10
        ).astype(np.float32)
        states['SCORE_MKT_BEARISH_CONFLUENCE_RISK'] = score_bearish_confluence
        # 底部反转机会评分
        states['SCORE_MKT_BOTTOM_REVERSAL_OPP'] = (
            score_low_sentiment * 0.4 + score_accel_up_short * 0.4 + score_slope_up_short * 0.2
        ).astype(np.float32)
        # 顶部反转风险评分
        states['SCORE_MKT_TOP_REVERSAL_RISK'] = (
            score_high_sentiment * 0.4 + score_accel_down_short * 0.4 + score_slope_down_short * 0.2
        ).astype(np.float32)
        # === Part 4: 综合市场健康分 (逻辑不变) ===
        static_score = self._normalize_score(winner_rate)
        dynamic_score = score_bullish_confluence - score_bearish_confluence
        normalized_dynamic_score = (dynamic_score + 1) / 2
        states['SCORE_MKT_HEALTH_S'] = (
            static_score * 0.4 + normalized_dynamic_score * 0.6
        ).astype(np.float32)
        return states

    def diagnose_capital_and_range_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.2 最终数值化版】资金流与绝对波幅统一情报中心
        - 核心职责: (原有注释)
        - 核心升级 (本次重构):
          - [最终数值化] 将内部所有布尔判断 (如 is_cmf_inflow, is_cmf_slope_up)
                          替换为基于 _normalize_score 的连续评分，实现100%数值化。
        """
        states = {}
        p_capital = get_params_block(self.strategy, 'capital_state_params')
        if not get_param_value(p_capital.get('enabled'), False): return states
        # --- 军备检查 (Arsenal Check) ---
        required_cols = [
            'CMF_21_D', 'SLOPE_5_CMF_21_D', 'ACCEL_5_CMF_21_D',
            'SLOPE_21_CMF_21_D', 'ACCEL_21_CMF_21_D', 'ATR_14_D', 'SLOPE_5_ATR_14_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 资金流情报中心缺少必需列: {missing}，模块已跳过。")
            return states
        # === Part 1: 基础条件数值化 (Fundamental Conditions Scoring) ===
        # 将所有布尔条件替换为0-1的连续评分
        cmf = df['CMF_21_D']
        score_cmf_high = self._normalize_score(cmf.clip(lower=cmf.rolling(120).quantile(0.85)))
        score_cmf_low = self._normalize_score(cmf.clip(upper=cmf.rolling(120).quantile(0.15)), ascending=False)
        score_cmf_inflow = self._normalize_score(cmf.clip(lower=0))
        score_cmf_outflow = self._normalize_score(cmf.clip(upper=0), ascending=False)
        score_cmf_slope_up_short = self._normalize_score(df['SLOPE_5_CMF_21_D'].clip(lower=0))
        score_cmf_slope_up_mid = self._normalize_score(df['SLOPE_21_CMF_21_D'].clip(lower=0))
        score_cmf_slope_down_short = self._normalize_score(df['SLOPE_5_CMF_21_D'].clip(upper=0), ascending=False)
        score_cmf_slope_down_mid = self._normalize_score(df['SLOPE_21_CMF_21_D'].clip(upper=0), ascending=False)
        score_cmf_accel_up_short = self._normalize_score(df['ACCEL_5_CMF_21_D'].clip(lower=0))
        score_cmf_accel_up_mid = self._normalize_score(df['ACCEL_21_CMF_21_D'].clip(lower=0))
        score_cmf_accel_down_short = self._normalize_score(df['ACCEL_5_CMF_21_D'].clip(upper=0), ascending=False)
        # === Part 2, 3, 4: 资金流信号数值化 (使用新评分) ===
        # 资金流入共振评分
        states['SCORE_CAPITAL_INFLOW_CONFLUENCE'] = (
            score_cmf_inflow * 0.3 +
            score_cmf_slope_up_short * 0.25 + score_cmf_slope_up_mid * 0.25 +
            score_cmf_accel_up_short * 0.1 + score_cmf_accel_up_mid * 0.1
        ).astype(np.float32)
        # 资金流出共振风险评分
        states['SCORE_CAPITAL_OUTFLOW_CONFLUENCE_RISK'] = (
            score_cmf_outflow * 0.4 +
            score_cmf_slope_down_short * 0.3 + score_cmf_slope_down_mid * 0.2 +
            score_cmf_accel_down_short * 0.1
        ).astype(np.float32)
        # 资金底部反转机会评分
        states['SCORE_CAPITAL_BOTTOM_REVERSAL_OPP'] = (
            score_cmf_low * 0.5 +
            score_cmf_accel_up_short * 0.3 +
            score_cmf_slope_up_short * 0.2
        ).astype(np.float32)
        # 资金顶部反转风险评分
        states['SCORE_CAPITAL_TOP_REVERSAL_RISK'] = (
            score_cmf_high * 0.5 +
            score_cmf_accel_down_short * 0.3 +
            score_cmf_slope_down_short * 0.2
        ).astype(np.float32)
        # === Part 5: ATR 绝对波幅状态与临界点信号 (逻辑不变) ===
        atr = df['ATR_14_D']
        score_atr_compression = self._normalize_score(atr, ascending=False)
        score_atr_expansion = self._normalize_score(atr, ascending=True)
        states['SCORE_ATR_COMPRESSION_LEVEL'] = score_atr_compression
        states['SCORE_ATR_EXPANSION_LEVEL'] = score_atr_expansion
        is_tipping_point_expansion = (df['SLOPE_5_ATR_14_D'] > 0) & (df['SLOPE_5_ATR_14_D'].shift(1) <= 0)
        states['SCORE_ATR_EXPANSION_IGNITION_OPP'] = score_atr_compression * is_tipping_point_expansion.astype(np.float32)
        is_tipping_point_exhaustion = (df['SLOPE_5_ATR_14_D'] < 0) & (df['SLOPE_5_ATR_14_D'].shift(1) >= 0)
        states['SCORE_ATR_EXPANSION_EXHAUSTION_RISK'] = score_atr_expansion * is_tipping_point_exhaustion.astype(np.float32)
        return states

    def diagnose_classic_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.3 最终数值化版】经典指标统一情报中心
        - 核心职责: (原有注释)
        - 核心升级 (本次重构):
          - [最终数值化] 将量价分析中的 `is_price_up/down` 布尔判断，升级为衡量
                          K线实体相对强度的连续评分，使信号强度与价格运动幅度挂钩。
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states
        # --- 军备检查 (Arsenal Check) ---
        required_cols = [
            'MACDh_13_34_8_D', 'MACD_HIST_ZSCORE_D', 'SLOPE_5_MACDh_13_34_8_D',
            'ACCEL_5_MACDh_13_34_8_D', 'SLOPE_21_MACDh_13_34_8_D',
            'volume_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D', 'close_D', 'open_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 经典指标情报中心缺少必需列: {missing}，模块已跳过。")
            return states
        # === Part 1: MACD 动能分析 (已数值化) ===
        score_macdh_positive = self._normalize_score(df['MACDh_13_34_8_D'].clip(lower=0))
        score_macdh_negative = self._normalize_score(df['MACDh_13_34_8_D'].clip(upper=0), ascending=False)
        score_macdh_high = self._normalize_score(df['MACD_HIST_ZSCORE_D'].clip(lower=1.5))
        score_macdh_low = self._normalize_score(df['MACD_HIST_ZSCORE_D'].clip(upper=-1.5), ascending=False)
        score_slope_up_short = self._normalize_score(df['SLOPE_5_MACDh_13_34_8_D'].clip(lower=0))
        score_slope_up_mid = self._normalize_score(df['SLOPE_21_MACDh_13_34_8_D'].clip(lower=0))
        score_slope_down_short = self._normalize_score(df['SLOPE_5_MACDh_13_34_8_D'].clip(upper=0), ascending=False)
        score_slope_down_mid = self._normalize_score(df['SLOPE_21_MACDh_13_34_8_D'].clip(upper=0), ascending=False)
        score_accel_up_short = self._normalize_score(df['ACCEL_5_MACDh_13_34_8_D'].clip(lower=0))
        score_accel_down_short = self._normalize_score(df['ACCEL_5_MACDh_13_34_8_D'].clip(upper=0), ascending=False)
        states['SCORE_MACD_BULLISH_CONFLUENCE'] = (score_macdh_positive * 0.4 + score_slope_up_short * 0.3 + score_slope_up_mid * 0.2 + score_accel_up_short * 0.1).astype(np.float32)
        states['SCORE_MACD_BEARISH_CONFLUENCE_RISK'] = (score_macdh_negative * 0.4 + score_slope_down_short * 0.3 + score_slope_down_mid * 0.2 + score_accel_down_short * 0.1).astype(np.float32)
        states['SCORE_MACD_BOTTOM_REVERSAL_OPP'] = (score_macdh_low * 0.5 + score_accel_up_short * 0.3 + score_slope_up_short * 0.2).astype(np.float32)
        states['SCORE_MACD_TOP_REVERSAL_RISK'] = (score_macdh_high * 0.5 + score_accel_down_short * 0.3 + score_slope_down_short * 0.2).astype(np.float32)
        # === Part 2: 成交量动态分析 (最终数值化) ===
        # 将 is_price_up/down 的布尔判断，升级为衡量K线实体强度的连续评分
        candle_body_up = (df['close_D'] - df['open_D']).clip(lower=0)
        candle_body_down = (df['open_D'] - df['close_D']).clip(lower=0)
        score_price_up_strength = self._normalize_score(candle_body_up)
        score_price_down_strength = self._normalize_score(candle_body_down)
        score_vol_slope_up = self._normalize_score(df['SLOPE_5_volume_D'].clip(lower=0))
        score_vol_accel_up = self._normalize_score(df['ACCEL_5_volume_D'].clip(lower=0))
        score_volume_igniting = score_vol_slope_up * score_vol_accel_up
        # 使用新的价格强度分计算最终得分
        states['SCORE_VOL_PRICE_IGNITION_UP'] = score_price_up_strength * score_volume_igniting
        states['SCORE_VOL_PRICE_PANIC_DOWN_RISK'] = score_price_down_strength * score_volume_igniting
        # print("        -> [经典指标情报中心] 已将量价点火信号中的价格部分数值化。")
        return states










