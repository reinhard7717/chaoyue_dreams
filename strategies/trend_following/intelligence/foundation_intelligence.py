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
        # print("        -> [基础情报分析总指挥 V2.5] 启动...")
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
        states.update(self.diagnose_ema_synergy(df))
        
        # 将所有生成的基础原子情报更新到策略实例，为协同诊断做准备
        self.strategy.atomic_states.update(states)

        # === 第二阶段: 运行协同情报中心，生成高级复合信号 ===
        # 步骤6: 运行统一的协同情报中心
        synergy_intelligence = self.diagnose_foundation_synergy(df)
        states.update(synergy_intelligence)

        # 最终更新，确保协同信号也被添加
        self.strategy.atomic_states.update(synergy_intelligence)
        
        print("          -> [基础情报分析总指挥 V2.5] 所有情报已生成。") # 更新打印信息
        return states

    def diagnose_ema_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 终极交叉验证版】EMA均线协同诊断模块
        - 核心重构 (本次修改):
          - [交叉验证] 引入“静态(排列) x 动态(斜率) x 加速”三维交叉验证，生成B/A/S三级置信度信号。
          - [1日维度] 将1日周期的斜率和加速度纳入计算，捕捉最即时的边际变化。
        - 核心逻辑:
          - 共振: B级(短期斜率共振) -> A级(B级+静态排列确认) -> S级(A级+短期加速确认)。
          - 反转: B级(短期加速拐点) -> A级(B级+长期趋势环境确认) -> S级(A级+短期斜率确认)。
        - 收益: 信号逻辑更严谨，能更精确地识别趋势的形成、持续与转折。
        """
        states = {}
        p = get_params_block(self.strategy, 'multi_dim_ma_params')
        if not get_param_value(p.get('enabled'), True): return {}
        
        # --- 1. 军备检查 ---
        periods = get_param_value(p.get('ma_periods'), [1, 5, 13, 21, 55])
        required_cols = [f'EMA_{p}_D' if p > 1 else 'close_D' for p in periods] # 1日EMA即为收盘价
        required_cols.extend([f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D' for p in periods])
        required_cols.extend([f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D' for p in periods])
        
        if not all(c in df.columns for c in required_cols):
            print(f"          -> [警告] EMA协同模块缺少依赖，模块已跳过。")
            return states
        # --- 2. 核心要素数值化 ---
        # 2.1 静态排列分 (多头排列程度)
        alignment_scores = []
        for i in range(len(periods) - 1):
            # 适配1日周期使用close_D
            short_col = f'EMA_{periods[i]}_D' if periods[i] > 1 else 'close_D'
            long_col = f'EMA_{periods[i+1]}_D'
            alignment_scores.append((df[short_col] > df[long_col]).astype(float))
        score_static_bullish = pd.Series(np.mean(alignment_scores, axis=0), index=df.index)
        # 2.2 动态斜率分
        slope_scores = {p: self._normalize_score(df[f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D']) for p in periods}
        # 2.3 动态加速分
        accel_scores = {p: self._normalize_score(df[f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D']) for p in periods}
        # --- 3. 上升/下跌共振信号 ---
        # 严格遵循 B/A/S 逻辑
        # B级: 短中期斜率共振
        bullish_momentum_b = (slope_scores[1] * slope_scores[5] * slope_scores[13]).astype(np.float32)
        states['SCORE_EMA_BULLISH_RESONANCE_B'] = bullish_momentum_b
        # A级: B级信号 + 静态排列确认
        bullish_momentum_a = (bullish_momentum_b * score_static_bullish).astype(np.float32)
        states['SCORE_EMA_BULLISH_RESONANCE_A'] = bullish_momentum_a
        # S级: A级信号 + 加速确认
        bullish_accel_s = (accel_scores[1] * accel_scores[5]).astype(np.float32)
        states['SCORE_EMA_BULLISH_RESONANCE_S'] = (bullish_momentum_a * bullish_accel_s).astype(np.float32)
        # 对称的下跌共振逻辑
        bearish_momentum_b = ((1 - slope_scores[1]) * (1 - slope_scores[5]) * (1 - slope_scores[13])).astype(np.float32)
        states['SCORE_EMA_BEARISH_RESONANCE_B'] = bearish_momentum_b
        bearish_momentum_a = (bearish_momentum_b * (1 - score_static_bullish)).astype(np.float32)
        states['SCORE_EMA_BEARISH_RESONANCE_A'] = bearish_momentum_a
        bearish_accel_s = ((1 - accel_scores[1]) * (1 - accel_scores[5])).astype(np.float32)
        states['SCORE_EMA_BEARISH_RESONANCE_S'] = (bearish_momentum_a * bearish_accel_s).astype(np.float32)
        # --- 4. 顶部/底部反转信号 ---
        # 严格遵循 B/A/S 逻辑
        # B级: 短期加速拐点
        bottom_trigger_b = accel_scores[1].astype(np.float32)
        states['SCORE_EMA_BOTTOM_REVERSAL_B'] = bottom_trigger_b
        # A级: B级信号 + 长期趋势环境确认
        long_term_down_env = (1 - slope_scores[55])
        bottom_trigger_a = (bottom_trigger_b * long_term_down_env).astype(np.float32)
        states['SCORE_EMA_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        # S级: A级信号 + 短期斜率确认
        short_term_mom_up = slope_scores[5]
        states['SCORE_EMA_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * short_term_mom_up).astype(np.float32)

        # 对称的顶部反转逻辑
        top_trigger_b = (1 - accel_scores[1]).astype(np.float32)
        states['SCORE_EMA_TOP_REVERSAL_B'] = top_trigger_b
        long_term_up_env = slope_scores[55]
        top_trigger_a = (top_trigger_b * long_term_up_env).astype(np.float32)
        states['SCORE_EMA_TOP_REVERSAL_A'] = top_trigger_a
        short_term_mom_down = (1 - slope_scores[5])
        states['SCORE_EMA_TOP_REVERSAL_S'] = (top_trigger_a * short_term_mom_down).astype(np.float32)
        
        return states

    def diagnose_foundation_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 共识引擎版】基础层最终协同诊断模块
        - 核心重构 (本次修改):
          - [逻辑升级] 废除V1.0的“或”门逻辑(np.maximum)，升级为基于加权平均的“共识”逻辑。
          - [信号升级] 生成全新的、代表多领域共识的S级信号，如`SCORE_FOUNDATION_BULLISH_RESONANCE_S`。
        - 核心逻辑:
          - 对来自各子模块（EMA, RSI, MACD等）的同类型S级信号进行加权平均。
          - 一个信号的最终分数，取决于有多少个不同维度的指标在同时为它“投票”，以及每个“投票”的权重。
        - 收益: 极大提升了最终输出信号的置信度，能有效识别出得到市场多方面验证的交易机会。
        """
        # 此方法为全新的顶层融合模块
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        def get_weighted_synergy_score(signal_weights: Dict[str, float]) -> pd.Series:
            """辅助函数，用于计算加权协同分数"""
            total_score = pd.Series(0.0, index=df.index)
            total_weight = 0.0
            for signal, weight in signal_weights.items():
                score_series = atomic.get(signal, default_score)
                total_score += score_series * weight
                total_weight += weight
            return total_score / total_weight if total_weight > 0 else default_score

        # --- 1. 上升共振协同 (Bullish Resonance Synergy) ---
        bullish_resonance_sources = {
            'SCORE_EMA_BULLISH_RESONANCE_S': 0.4,
            'SCORE_RSI_BULLISH_RESONANCE_S': 0.3,
            'SCORE_MACD_BULLISH_RESONANCE_S': 0.2,
            'SCORE_CMF_BULLISH_RESONANCE_S': 0.1,
        }
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_S'] = get_weighted_synergy_score(bullish_resonance_sources)

        # --- 2. 下跌共振协同 (Bearish Resonance Synergy) ---
        bearish_resonance_sources = {
            'SCORE_EMA_BEARISH_RESONANCE_S': 0.4,
            'SCORE_RSI_BEARISH_RESONANCE_S': 0.3,
            'SCORE_MACD_BEARISH_RESONANCE_S': 0.2,
            'SCORE_CMF_BEARISH_RESONANCE_S': 0.1,
        }
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_S'] = get_weighted_synergy_score(bearish_resonance_sources)

        # --- 3. 底部反转协同 (Bottom Reversal Synergy) ---
        bottom_reversal_sources = {
            'SCORE_EMA_BOTTOM_REVERSAL_S': 0.35,
            'SCORE_RSI_BOTTOM_REVERSAL_S': 0.35,
            'SCORE_MACD_BOTTOM_REVERSAL_S': 0.2,
            'SCORE_VOL_TIPPING_POINT_BOTTOM_OPP': 0.1,
        }
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S'] = get_weighted_synergy_score(bottom_reversal_sources)

        # --- 4. 顶部反转协同 (Top Reversal Synergy) ---
        top_reversal_sources = {
            'SCORE_EMA_TOP_REVERSAL_S': 0.35,
            'SCORE_RSI_TOP_REVERSAL_S': 0.35,
            'SCORE_MACD_TOP_REVERSAL_S': 0.2,
            'SCORE_VOL_TIPPING_POINT_TOP_RISK': 0.1,
        }
        states['SCORE_FOUNDATION_TOP_REVERSAL_S'] = get_weighted_synergy_score(top_reversal_sources)

        return states

    def diagnose_oscillator_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 终极交叉验证版】震荡与动能统一情报中心
        - 核心重构 (本次修改):
          - [交叉验证] 引入“静态(RSI区间) x 动态(斜率) x 加速”三维交叉验证。
          - [1日维度] 将1日周期的斜率和加速度纳入计算，捕捉最即时的边际变化。
        - 核心逻辑:
          - 共振: B级(短期斜率) -> A级(B级+静态区间确认) -> S级(A级+短期加速确认)。
          - 反转: B级(短期加速拐点) -> A级(B级+超卖/超买环境确认) -> S级(A级+短期斜率确认)。
        - 收益: 极大提升了RSI信号的可靠性，能有效过滤在“钝化区”的假信号。
        """
        states = {}
        p = get_params_block(self.strategy, 'oscillator_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 1. 军备检查 ---
        required_cols = [
            'RSI_13_D', 'SLOPE_1_RSI_13_D', 'SLOPE_5_RSI_13_D', 'ACCEL_1_RSI_13_D', 'ACCEL_5_RSI_13_D',
            'RSI_13_W', 'SLOPE_5_RSI_13_W',
            'MACD_HIST_ZSCORE_D', 'BIAS_55_D', 'high_D', 'low_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 震荡与动能情报中心缺少必需列: {missing}，模块已跳过。")
            # 增加对数据层缺失的提示
            print(f"          -> [数据需求] 请确保数据工程层已为 RSI_13_D 计算了 1日和5日的斜率与加速度。")
            return states

        # --- 2. 基础静态诊断 (逻辑不变) ---
        rsi_col = 'RSI_13_D'
        overbought_threshold = get_param_value(p.get('rsi_overbought_start'), 70)
        score_overbought = (df[rsi_col] - overbought_threshold) / (100 - overbought_threshold)
        states['SCORE_RSI_OVERBOUGHT_EXTENT'] = score_overbought.clip(0, 1).astype(np.float32)
        oversold_threshold = get_param_value(p.get('rsi_oversold_start'), 30)
        score_oversold = (oversold_threshold - df[rsi_col]) / (oversold_threshold - 0)
        states['SCORE_RSI_OVERSOLD_EXTENT'] = score_oversold.clip(0, 1).astype(np.float32)
        # ... (BIAS 和 MACD Divergence 逻辑保持不变) ...
        p_bias = p.get('bias_dynamic_threshold', {})
        window = get_param_value(p_bias.get('window'), 120)
        quantile = get_param_value(p_bias.get('quantile'), 0.1)
        dynamic_oversold_threshold = df['BIAS_55_D'].rolling(window=window).quantile(quantile)
        negative_deviation = (dynamic_oversold_threshold - df['BIAS_55_D']).clip(lower=0)
        states['SCORE_BIAS_OVERSOLD_EXTENT'] = self._normalize_score(negative_deviation, window=window)
        dynamic_overbought_threshold = df['BIAS_55_D'].rolling(window=window).quantile(1 - quantile)
        positive_deviation = (df['BIAS_55_D'] - dynamic_overbought_threshold).clip(lower=0)
        states['SCORE_BIAS_OVERBOUGHT_EXTENT'] = self._normalize_score(positive_deviation, window=window)
        price_overshoot = (df['high_D'] - df['high_D'].rolling(10).max().shift(1)).clip(lower=0)
        macd_undershoot = (df['MACD_HIST_ZSCORE_D'].rolling(10).max().shift(1) - df['MACD_HIST_ZSCORE_D']).clip(lower=0)
        states['SCORE_MACD_BEARISH_DIVERGENCE_RISK'] = self._normalize_score(price_overshoot) * self._normalize_score(macd_undershoot)
        price_undershoot = (df['low_D'].rolling(10).min().shift(1) - df['low_D']).clip(lower=0)
        macd_overshoot = (df['MACD_HIST_ZSCORE_D'] - df['MACD_HIST_ZSCORE_D'].rolling(10).min().shift(1)).clip(lower=0)
        states['SCORE_MACD_BULLISH_DIVERGENCE_OPP'] = self._normalize_score(price_undershoot) * self._normalize_score(macd_overshoot)

        # --- 3. RSI协同诊断单元 ---
        # 全面重构RSI信号生成逻辑
        # 3.1 核心要素数值化
        score_rsi_d_static_bull = self._normalize_score(df['RSI_13_D'].clip(50, 100))
        score_rsi_d_static_bear = self._normalize_score(df['RSI_13_D'].clip(0, 50), ascending=False)
        score_rsi_d_mom_short = self._normalize_score(df['SLOPE_1_RSI_13_D'])
        score_rsi_d_mom_mid = self._normalize_score(df['SLOPE_5_RSI_13_D'])
        score_rsi_d_accel_short = self._normalize_score(df['ACCEL_1_RSI_13_D'])
        score_rsi_w_mom = self._normalize_score(df['SLOPE_5_RSI_13_W'])

        # 3.2 上升共振 (Bullish Resonance)
        bullish_momentum_b = score_rsi_d_mom_mid
        states['SCORE_RSI_BULLISH_RESONANCE_B'] = bullish_momentum_b.astype(np.float32)
        bullish_momentum_a = (bullish_momentum_b * score_rsi_d_static_bull * score_rsi_w_mom).astype(np.float32)
        states['SCORE_RSI_BULLISH_RESONANCE_A'] = bullish_momentum_a
        states['SCORE_RSI_BULLISH_RESONANCE_S'] = (bullish_momentum_a * score_rsi_d_accel_short).astype(np.float32)

        # 3.3 下跌共振 (Bearish Resonance)
        bearish_momentum_b = (1 - score_rsi_d_mom_mid)
        states['SCORE_RSI_BEARISH_RESONANCE_B'] = bearish_momentum_b.astype(np.float32)
        bearish_momentum_a = (bearish_momentum_b * score_rsi_d_static_bear * (1 - score_rsi_w_mom)).astype(np.float32)
        states['SCORE_RSI_BEARISH_RESONANCE_A'] = bearish_momentum_a
        states['SCORE_RSI_BEARISH_RESONANCE_S'] = (bearish_momentum_a * (1 - score_rsi_d_accel_short)).astype(np.float32)

        # 3.4 底部反转 (Bottom Reversal)
        bottom_trigger_b = score_rsi_d_accel_short
        states['SCORE_RSI_BOTTOM_REVERSAL_B'] = bottom_trigger_b.astype(np.float32)
        bottom_trigger_a = (bottom_trigger_b * states['SCORE_RSI_OVERSOLD_EXTENT']).astype(np.float32)
        states['SCORE_RSI_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        states['SCORE_RSI_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * score_rsi_d_mom_short).astype(np.float32)

        # 3.5 顶部反转 (Top Reversal)
        top_trigger_b = (1 - score_rsi_d_accel_short)
        states['SCORE_RSI_TOP_REVERSAL_B'] = top_trigger_b.astype(np.float32)
        top_trigger_a = (top_trigger_b * states['SCORE_RSI_OVERBOUGHT_EXTENT']).astype(np.float32)
        states['SCORE_RSI_TOP_REVERSAL_A'] = top_trigger_a
        states['SCORE_RSI_TOP_REVERSAL_S'] = (top_trigger_a * (1 - score_rsi_d_mom_short)).astype(np.float32)
        
        return states

    def diagnose_volatility_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 交叉验证版】波动率统一情报中心
        - 核心逻辑:
          - B级 (日线压缩): 基于日线BBW的归一化评分。
          - A级 (跨周期压缩): B级信号与周线BBW评分交叉验证。
          - S级 (静态压缩): A级信号与BBW斜率为负（仍在压缩）交叉验证。
        - 收益: 能够更精确地量化波动率状态，区分“初步压缩”和“极致压缩”。
        """
        states = {}
        p = get_params_block(self.strategy, 'volatility_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_BBW_21_2.0_D', 
            'BBW_21_2.0_W', 'hurst_120d_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 波动率情报中心缺少必需列: {missing}，模块已跳过。")
            return states

        # --- 2. 核心要素数值化 ---
        score_squeeze_daily = self._normalize_score(df['BBW_21_2.0_D'], ascending=False)
        score_squeeze_weekly = self._normalize_score(df['BBW_21_2.0_W'], ascending=False)
        score_squeeze_momentum = self._normalize_score(df['SLOPE_5_BBW_21_2.0_D'], ascending=False)
        
        score_expansion_daily = 1 - score_squeeze_daily
        score_expansion_weekly = 1 - score_squeeze_weekly
        score_expansion_momentum = 1 - score_squeeze_momentum

        # --- 3. 生成B/A/S三级压缩信号 ---
        states['SCORE_VOL_COMPRESSION_B'] = score_squeeze_daily
        states['SCORE_VOL_COMPRESSION_A'] = (score_squeeze_daily * score_squeeze_weekly).astype(np.float32)
        states['SCORE_VOL_COMPRESSION_S'] = (states['SCORE_VOL_COMPRESSION_A'] * score_squeeze_momentum).astype(np.float32)

        # --- 4. 生成B/A/S三级扩张信号 (对称逻辑) ---
        states['SCORE_VOL_EXPANSION_B'] = score_expansion_daily
        states['SCORE_VOL_EXPANSION_A'] = (score_expansion_daily * score_expansion_weekly).astype(np.float32)
        states['SCORE_VOL_EXPANSION_S'] = (states['SCORE_VOL_EXPANSION_A'] * score_expansion_momentum).astype(np.float32)

        # --- 5. 波动率反转临界点 (逻辑优化) ---
        is_tipping_point_bottom = (df['SLOPE_5_BBW_21_2.0_D'] > 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) <= 0)
        # 使用最高置信度的S级压缩分作为环境判断
        states['SCORE_VOL_TIPPING_POINT_BOTTOM_OPP'] = (states['SCORE_VOL_COMPRESSION_S'] * is_tipping_point_bottom).astype(np.float32)
        
        is_tipping_point_top = (df['SLOPE_5_BBW_21_2.0_D'] < 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) >= 0)
        # 使用最高置信度的S级扩张分作为环境判断
        states['SCORE_VOL_TIPPING_POINT_TOP_RISK'] = (states['SCORE_VOL_EXPANSION_S'] * is_tipping_point_top).astype(np.float32)

        # --- 6. 市场政权与数值化评分 (逻辑不变) ---
        hurst_score = self._normalize_score(df['hurst_120d_D'])
        states['SCORE_TRENDING_REGIME'] = hurst_score
        states['SCORE_VOL_BREAKOUT_POTENTIAL_S'] = states['SCORE_VOL_COMPRESSION_S'] * hurst_score
        states['SCORE_VOL_BREAKDOWN_RISK_S'] = states['SCORE_VOL_EXPANSION_S'] * (1 - hurst_score)
        
        return states

    def diagnose_market_character_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 交叉验证版】市场特征与情绪统一情报中心
        - 核心重构 (本次修改):
          - [交叉验证] 引入“静态(情绪高/低) x 动态(斜率) x 加速”三维交叉验证。
          - [信号升级] 生成全新的B/A/S三级市场情绪共振与反转信号。
        - 收益: 对市场情绪的判断更具前瞻性和可靠性。
        """
        states = {}
        p = get_params_block(self.strategy, 'market_character_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'total_winner_rate_D', 'SLOPE_1_total_winner_rate_D', 'SLOPE_5_total_winner_rate_D', 
            'SLOPE_21_total_winner_rate_D', 'ACCEL_1_total_winner_rate_D', 'ACCEL_5_total_winner_rate_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 市场特征情报中心缺少必需列: {missing}，模块已跳过。")
            # 增加对数据层缺失的提示
            print(f"          -> [数据需求] 请确保数据工程层已为 total_winner_rate_D 计算了 1, 5, 21日的斜率与加速度。")
            return states

        # --- 2. 市场情绪协同诊断单元 ---
        # 2.1 核心要素数值化
        winner_rate = df['total_winner_rate_D']
        score_static_high = self._normalize_score(winner_rate.clip(lower=winner_rate.rolling(120).quantile(0.8)))
        score_static_low = self._normalize_score(winner_rate.clip(upper=winner_rate.rolling(120).quantile(0.2)), ascending=False)
        score_mom_short = self._normalize_score(df['SLOPE_1_total_winner_rate_D'])
        score_mom_mid = self._normalize_score(df['SLOPE_5_total_winner_rate_D'])
        score_accel_short = self._normalize_score(df['ACCEL_1_total_winner_rate_D'])

        # 2.2 上升共振 (情绪升温)
        bullish_momentum_b = (score_mom_short * score_mom_mid).astype(np.float32)
        states['SCORE_MKT_BULLISH_RESONANCE_B'] = bullish_momentum_b
        bullish_momentum_a = (bullish_momentum_b * self._normalize_score(winner_rate)).astype(np.float32) # A级用原始静态分确认
        states['SCORE_MKT_BULLISH_RESONANCE_A'] = bullish_momentum_a
        states['SCORE_MKT_BULLISH_RESONANCE_S'] = (bullish_momentum_a * score_accel_short).astype(np.float32)

        # 2.3 下跌共振 (情绪降温)
        bearish_momentum_b = ((1 - score_mom_short) * (1 - score_mom_mid)).astype(np.float32)
        states['SCORE_MKT_BEARISH_RESONANCE_B'] = bearish_momentum_b
        bearish_momentum_a = (bearish_momentum_b * (1 - self._normalize_score(winner_rate))).astype(np.float32)
        states['SCORE_MKT_BEARISH_RESONANCE_A'] = bearish_momentum_a
        states['SCORE_MKT_BEARISH_RESONANCE_S'] = (bearish_momentum_a * (1 - score_accel_short)).astype(np.float32)

        # 2.4 底部反转 (情绪冰点反转)
        bottom_trigger_b = score_accel_short
        states['SCORE_MKT_BOTTOM_REVERSAL_B'] = bottom_trigger_b.astype(np.float32)
        bottom_trigger_a = (bottom_trigger_b * score_static_low).astype(np.float32)
        states['SCORE_MKT_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        states['SCORE_MKT_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * score_mom_short).astype(np.float32)

        # 2.5 顶部反转 (情绪高点反转)
        top_trigger_b = (1 - score_accel_short)
        states['SCORE_MKT_TOP_REVERSAL_B'] = top_trigger_b.astype(np.float32)
        top_trigger_a = (top_trigger_b * score_static_high).astype(np.float32)
        states['SCORE_MKT_TOP_REVERSAL_A'] = top_trigger_a
        states['SCORE_MKT_TOP_REVERSAL_S'] = (top_trigger_a * (1 - score_mom_short)).astype(np.float32)

        # --- 3. 综合市场健康分 (逻辑优化) ---
        # 使用新的共振信号来计算健康分
        dynamic_score = states['SCORE_MKT_BULLISH_RESONANCE_A'] - states['SCORE_MKT_BEARISH_RESONANCE_A']
        normalized_dynamic_score = (dynamic_score + 1) / 2
        states['SCORE_MKT_HEALTH_S'] = (
            self._normalize_score(winner_rate) * 0.4 + normalized_dynamic_score * 0.6
        ).astype(np.float32)
        return states

    def diagnose_capital_and_range_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 交叉验证版】资金流与绝对波幅统一情报中心
        - 核心重构 (本次修改):
          - [交叉验证] 引入“静态(净流入) x 动态(斜率) x 加速”三维交叉验证。
          - [信号升级] 生成全新的B/A/S三级CMF共振与反转信号。
        - 收益: 极大提升了资金流信号的可靠性，能有效过滤噪音。
        """
        states = {}
        p_capital = get_params_block(self.strategy, 'capital_state_params')
        if not get_param_value(p_capital.get('enabled'), False): return states
        
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'CMF_21_D', 'SLOPE_1_CMF_21_D', 'SLOPE_5_CMF_21_D', 'SLOPE_21_CMF_21_D', 
            'ACCEL_1_CMF_21_D', 'ACCEL_5_CMF_21_D', 'ATR_14_D', 'SLOPE_5_ATR_14_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 资金流情报中心缺少必需列: {missing}，模块已跳过。")
            # 增加对数据层缺失的提示
            print(f"          -> [数据需求] 请确保数据工程层已为 CMF_21_D 计算了 1, 5, 21日的斜率与加速度。")
            return states

        # --- 2. CMF协同诊断单元 ---
        # 2.1 核心要素数值化
        score_cmf_static_bull = self._normalize_score(df['CMF_21_D'].clip(lower=0))
        score_cmf_static_bear = self._normalize_score(df['CMF_21_D'].clip(upper=0), ascending=False)
        score_cmf_mom_short = self._normalize_score(df['SLOPE_1_CMF_21_D'])
        score_cmf_mom_mid = self._normalize_score(df['SLOPE_5_CMF_21_D'])
        score_cmf_accel_short = self._normalize_score(df['ACCEL_1_CMF_21_D'])

        # 2.2 上升共振 (Bullish Resonance)
        bullish_momentum_b = (score_cmf_mom_short * score_cmf_mom_mid).astype(np.float32)
        states['SCORE_CMF_BULLISH_RESONANCE_B'] = bullish_momentum_b
        bullish_momentum_a = (bullish_momentum_b * score_cmf_static_bull).astype(np.float32)
        states['SCORE_CMF_BULLISH_RESONANCE_A'] = bullish_momentum_a
        states['SCORE_CMF_BULLISH_RESONANCE_S'] = (bullish_momentum_a * score_cmf_accel_short).astype(np.float32)

        # 2.3 下跌共振 (Bearish Resonance)
        bearish_momentum_b = ((1 - score_cmf_mom_short) * (1 - score_cmf_mom_mid)).astype(np.float32)
        states['SCORE_CMF_BEARISH_RESONANCE_B'] = bearish_momentum_b
        bearish_momentum_a = (bearish_momentum_b * score_cmf_static_bear).astype(np.float32)
        states['SCORE_CMF_BEARISH_RESONANCE_A'] = bearish_momentum_a
        states['SCORE_CMF_BEARISH_RESONANCE_S'] = (bearish_momentum_a * (1 - score_cmf_accel_short)).astype(np.float32)

        # 2.4 底部反转 (Bottom Reversal)
        bottom_trigger_b = score_cmf_accel_short
        states['SCORE_CMF_BOTTOM_REVERSAL_B'] = bottom_trigger_b.astype(np.float32)
        bottom_trigger_a = (bottom_trigger_b * score_cmf_static_bear).astype(np.float32)
        states['SCORE_CMF_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        states['SCORE_CMF_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * score_cmf_mom_short).astype(np.float32)

        # 2.5 顶部反转 (Top Reversal)
        top_trigger_b = (1 - score_cmf_accel_short)
        states['SCORE_CMF_TOP_REVERSAL_B'] = top_trigger_b.astype(np.float32)
        top_trigger_a = (top_trigger_b * score_cmf_static_bull).astype(np.float32)
        states['SCORE_CMF_TOP_REVERSAL_A'] = top_trigger_a
        states['SCORE_CMF_TOP_REVERSAL_S'] = (top_trigger_a * (1 - score_cmf_mom_short)).astype(np.float32)

        # --- 3. ATR 绝对波幅状态与临界点信号 (逻辑不变) ---
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
        【V6.0 交叉验证版】经典指标统一情报中心
        - 核心重构 (本次修改):
          - [交叉验证] 引入MACD“静态(柱值) x 动态(斜率) x 加速”三维交叉验证。
          - [信号升级] 生成全新的B/A/S三级MACD共振与反转信号。
        - 收益: 极大提升了MACD信号的可靠性，能有效过滤假金叉/死叉。
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states
        
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'MACDh_13_34_8_D', 'MACD_HIST_ZSCORE_D', 'SLOPE_1_MACDh_13_34_8_D', 
            'SLOPE_5_MACDh_13_34_8_D', 'ACCEL_1_MACDh_13_34_8_D',
            'volume_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D', 'close_D', 'open_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 经典指标情报中心缺少必需列: {missing}，模块已跳过。")
            # 增加对数据层缺失的提示
            print(f"          -> [数据需求] 请确保数据工程层已为 MACDh_13_34_8_D 计算了 1日和5日的斜率与加速度。")
            return states

        # --- 2. MACD协同诊断单元 ---
        # 2.1 核心要素数值化
        score_macdh_static_bull = self._normalize_score(df['MACDh_13_34_8_D'].clip(lower=0))
        score_macdh_static_bear = self._normalize_score(df['MACDh_13_34_8_D'].clip(upper=0), ascending=False)
        score_macdh_zscore_high = self._normalize_score(df['MACD_HIST_ZSCORE_D'].clip(lower=1.5))
        score_macdh_zscore_low = self._normalize_score(df['MACD_HIST_ZSCORE_D'].clip(upper=-1.5), ascending=False)
        score_mom_short = self._normalize_score(df['SLOPE_1_MACDh_13_34_8_D'])
        score_mom_mid = self._normalize_score(df['SLOPE_5_MACDh_13_34_8_D'])
        score_accel_short = self._normalize_score(df['ACCEL_1_MACDh_13_34_8_D'])

        # 2.2 上升共振 (多头动能)
        bullish_momentum_b = (score_mom_short * score_mom_mid).astype(np.float32)
        states['SCORE_MACD_BULLISH_RESONANCE_B'] = bullish_momentum_b
        bullish_momentum_a = (bullish_momentum_b * score_macdh_static_bull).astype(np.float32)
        states['SCORE_MACD_BULLISH_RESONANCE_A'] = bullish_momentum_a
        states['SCORE_MACD_BULLISH_RESONANCE_S'] = (bullish_momentum_a * score_accel_short).astype(np.float32)

        # 2.3 下跌共振 (空头动能)
        bearish_momentum_b = ((1 - score_mom_short) * (1 - score_mom_mid)).astype(np.float32)
        states['SCORE_MACD_BEARISH_RESONANCE_B'] = bearish_momentum_b
        bearish_momentum_a = (bearish_momentum_b * score_macdh_static_bear).astype(np.float32)
        states['SCORE_MACD_BEARISH_RESONANCE_A'] = bearish_momentum_a
        states['SCORE_MACD_BEARISH_RESONANCE_S'] = (bearish_momentum_a * (1 - score_accel_short)).astype(np.float32)

        # 2.4 底部反转 (金叉)
        bottom_trigger_b = score_accel_short
        states['SCORE_MACD_BOTTOM_REVERSAL_B'] = bottom_trigger_b.astype(np.float32)
        bottom_trigger_a = (bottom_trigger_b * score_macdh_zscore_low).astype(np.float32)
        states['SCORE_MACD_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        states['SCORE_MACD_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * score_mom_short).astype(np.float32)

        # 2.5 顶部反转 (死叉)
        top_trigger_b = (1 - score_accel_short)
        states['SCORE_MACD_TOP_REVERSAL_B'] = top_trigger_b.astype(np.float32)
        top_trigger_a = (top_trigger_b * score_macdh_zscore_high).astype(np.float32)
        states['SCORE_MACD_TOP_REVERSAL_A'] = top_trigger_a
        states['SCORE_MACD_TOP_REVERSAL_S'] = (top_trigger_a * (1 - score_mom_short)).astype(np.float32)

        # --- 3. 成交量动态分析 (逻辑不变) ---
        candle_body_up = (df['close_D'] - df['open_D']).clip(lower=0)
        candle_body_down = (df['open_D'] - df['close_D']).clip(lower=0)
        score_price_up_strength = self._normalize_score(candle_body_up)
        score_price_down_strength = self._normalize_score(candle_body_down)
        score_vol_slope_up = self._normalize_score(df['SLOPE_5_volume_D'].clip(lower=0))
        score_vol_accel_up = self._normalize_score(df['ACCEL_5_volume_D'].clip(lower=0))
        score_volume_igniting = score_vol_slope_up * score_vol_accel_up
        states['SCORE_VOL_PRICE_IGNITION_UP'] = score_price_up_strength * score_volume_igniting
        states['SCORE_VOL_PRICE_PANIC_DOWN_RISK'] = score_price_down_strength * score_volume_igniting
        return states










