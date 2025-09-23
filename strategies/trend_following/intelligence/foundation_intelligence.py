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
        【V3.1 全面诊断版】基础情报分析总指挥
        - 核心重构 (本次修改):
          - [架构修复] 彻底修复了旧版只调用单一引擎的重大缺陷。现在会按顺序调用模块内
                        所有的诊断引擎（终极信号、EMA、震荡指标、波动率、经典指标等），
                        并汇总所有产出的信号。
        - 收益: 确保了基础情报层的所有诊断能力都被完全激活，解决了关键信号（如MACD、波动率）
                无法生成和监控的根本性问题。
        """
        
        # print("      -> [基础情报分析总指挥 V3.1 全面诊断版] 启动...")
        df = self.strategy.df_indicators
        all_states = {}

        # 按顺序调用所有诊断引擎，并汇总结果
        all_states.update(self.diagnose_ultimate_foundation_signals(df))
        all_states.update(self.diagnose_ema_synergy(df))
        all_states.update(self.diagnose_oscillator_intelligence(df))
        all_states.update(self.diagnose_volatility_intelligence(df))
        all_states.update(self.diagnose_classic_indicators(df))
        all_states.update(self.diagnose_market_character_scores(df))
        all_states.update(self.diagnose_capital_and_range_states(df))
        
        print(f"      -> [基础情报分析总指挥 V3.1] 分析完毕，共生成 {len(all_states)} 个基础层信号。")
        return all_states

    def diagnose_ultimate_foundation_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.3 基因改造版】终极基础层信号诊断模块
        - 核心升级 (本次修改):
          - [逻辑重构] 彻底重构了“反转”信号的生成逻辑，引入了基于价格位置的“上下文”判断。
          - [新范式] “底部反转”分数 = “底部上下文分数” * “短期看涨力量” * “长期看跌惯性”。
          - “顶部反转”分数 = “顶部上下文分数” * “短期看跌力量” * “长期看涨惯性”。
        - 收益:
          - 从根本上解决了“底部反转”信号在顶部得分最高的反向指标问题。
        """
        # print("        -> [终极基础层信号诊断模块 V1.3 基因改造版] 启动...") # 修改: 更新版本号
        states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
            
        # --- 定义“位置上下文”分数 ---
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        price_position_in_range = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)
        bottom_context_score = 1 - price_position_in_range
        top_context_score = price_position_in_range
        

        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        required_cols = set()
        for p in periods:
            required_cols.update([
                f'EMA_{p}_D' if p > 1 else 'close_D', f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D', f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D',
                'RSI_13_D', f'SLOPE_{p}_RSI_13_D', f'ACCEL_{p}_RSI_13_D',
                'MACDh_13_34_8_D', f'SLOPE_{p}_MACDh_13_34_8_D', f'ACCEL_{p}_MACDh_13_34_8_D',
                'CMF_21_D', f'SLOPE_{p}_CMF_21_D', f'ACCEL_{p}_CMF_21_D'
            ])
        missing_cols = list(required_cols - set(df.columns))
        if missing_cols:
            print(f"          -> [严重警告] 终极基础层引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化 (批量预处理) ---
        ema_static_scores = {p: self._normalize_score(df[f'EMA_{p}_D' if p > 1 else 'close_D'] - df[f'EMA_{max(periods)}_D']) for p in periods}
        ema_slope_scores = {p: self._normalize_score(df[f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D']) for p in periods}
        ema_accel_scores = {p: self._normalize_score(df[f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D']) for p in periods}
        rsi_static_score_arr = self._normalize_score(df['RSI_13_D']).values
        rsi_slope_scores = {p: self._normalize_score(df[f'SLOPE_{p}_RSI_13_D']) for p in periods}
        rsi_accel_scores = {p: self._normalize_score(df[f'ACCEL_{p}_RSI_13_D']) for p in periods}
        macd_static_score_arr = self._normalize_score(df['MACDh_13_34_8_D']).values
        macd_slope_scores = {p: self._normalize_score(df[f'SLOPE_{p}_MACDh_13_34_8_D']) for p in periods}
        macd_accel_scores = {p: self._normalize_score(df[f'ACCEL_{p}_MACDh_13_34_8_D']) for p in periods}
        cmf_static_score_arr = self._normalize_score(df['CMF_21_D']).values
        cmf_slope_scores = {p: self._normalize_score(df[f'SLOPE_{p}_CMF_21_D']) for p in periods}
        cmf_accel_scores = {p: self._normalize_score(df[f'ACCEL_{p}_CMF_21_D']) for p in periods}
        # --- 3. 融合生成“全面共识健康度” ---
        overall_bullish_health = {}
        for p in periods:
            ema_health_arr = ema_static_scores[p].values * ema_slope_scores[p].values * ema_accel_scores[p].values
            rsi_health_arr = rsi_static_score_arr * rsi_slope_scores[p].values * rsi_accel_scores[p].values
            macd_health_arr = macd_static_score_arr * macd_slope_scores[p].values * macd_accel_scores[p].values
            cmf_health_arr = cmf_static_score_arr * cmf_slope_scores[p].values * cmf_accel_scores[p].values
            overall_health_arr = (ema_health_arr * rsi_health_arr * macd_health_arr * cmf_health_arr)**0.25
            overall_bullish_health[p] = pd.Series(overall_health_arr, index=df.index, dtype=np.float32)
        overall_bearish_health = {p: 1.0 - overall_bullish_health[p] for p in periods}
        # --- 4. 定义信号组件 ---
        bullish_short_force = (overall_bullish_health[1] * overall_bullish_health[5])**0.5
        bullish_medium_trend = (overall_bullish_health[13] * overall_bullish_health[21])**0.5
        bullish_long_inertia = overall_bullish_health[55]
        bearish_short_force = (overall_bearish_health[1] * overall_bearish_health[5])**0.5
        bearish_medium_trend = (overall_bearish_health[13] * overall_bearish_health[21])**0.5
        bearish_long_inertia = overall_bearish_health[55]
        # --- 5. 共振信号合成 ---
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_B'] = overall_bullish_health[5].astype(np.float32)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_A'] = (overall_bullish_health[5] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_S'] = (bullish_short_force * bullish_medium_trend).astype(np.float32)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_S_PLUS'] = (states['SCORE_FOUNDATION_BULLISH_RESONANCE_S'] * bullish_long_inertia).astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_B'] = overall_bearish_health[5].astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_A'] = (overall_bearish_health[5] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_S'] = (bearish_short_force * bearish_medium_trend).astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_S_PLUS'] = (states['SCORE_FOUNDATION_BEARISH_RESONANCE_S'] * bearish_long_inertia).astype(np.float32)
        
        # --- 6. 反转信号合成 ---
        # --- 重构反转信号逻辑 ---
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_B'] = (bottom_context_score * overall_bullish_health[1] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_A'] = (bottom_context_score * overall_bullish_health[5] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S'] = (bottom_context_score * bullish_short_force * bearish_long_inertia).astype(np.float32)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S_PLUS'] = (bottom_context_score * bullish_short_force * bullish_medium_trend * bearish_long_inertia).astype(np.float32)
        
        states['SCORE_FOUNDATION_TOP_REVERSAL_B'] = (top_context_score * overall_bearish_health[1] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_TOP_REVERSAL_A'] = (top_context_score * overall_bearish_health[5] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_TOP_REVERSAL_S'] = (top_context_score * bearish_short_force * bullish_long_inertia).astype(np.float32)
        states['SCORE_FOUNDATION_TOP_REVERSAL_S_PLUS'] = (top_context_score * bearish_short_force * bearish_medium_trend * bullish_long_inertia).astype(np.float32)
        
        
        return states

    def diagnose_ema_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.1 性能优化版】EMA均线协同诊断模块
        - 核心升级 (本次修改):
          - [性能优化] 重构了所有信号的计算过程，全面采用NumPy数组进行计算，最后才转换为Pandas Series。
          - [内存优化] 避免了在计算过程中生成大量的中间Series对象，显著提升了计算效率和内存使用效率。
        - 核心重构 (V3.0逻辑保留):
          - [交叉验证] 引入“静态(排列) x 动态(斜率) x 加速”三维交叉验证。
          - [1日维度] 将1日周期的斜率和加速度纳入计算，捕捉最即时的边际变化。
        - 收益: 信号逻辑更严谨，计算性能更高，能更精确、更快速地识别趋势的形成、持续与转折。
        """
        states = {}
        p = get_params_block(self.strategy, 'multi_dim_ma_params')
        if not get_param_value(p.get('enabled'), True): return {}
        # --- 1. 军备检查 ---
        periods = get_param_value(p.get('ma_periods'), [1, 5, 13, 21, 55])
        required_cols = [f'EMA_{p}_D' if p > 1 else 'close_D' for p in periods]
        required_cols.extend([f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D' for p in periods])
        required_cols.extend([f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D' for p in periods])
        if not all(c in df.columns for c in required_cols):
            print(f"          -> [警告] EMA协同模块缺少依赖，模块已跳过。")
            return states
        # 全面采用NumPy数组进行计算
        # --- 2. 核心要素数值化 (在NumPy层面操作) ---
        # 2.1 静态排列分 (多头排列程度)
        alignment_score_arrays = []
        for i in range(len(periods) - 1):
            short_col = f'EMA_{periods[i]}_D' if periods[i] > 1 else 'close_D'
            long_col = f'EMA_{periods[i+1]}_D'
            # 直接在NumPy数组上比较，得到布尔数组，然后转为float32
            alignment_score_arrays.append((df[short_col].values > df[long_col].values).astype(np.float32))
        # 堆叠成2D数组并计算均值，得到静态分数组
        score_static_bullish_arr = np.mean(np.stack(alignment_score_arrays, axis=0), axis=0)
        # 2.2 动态斜率分
        slope_scores = {p: self._normalize_score(df[f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D']) for p in periods}
        # 2.3 动态加速分
        accel_scores = {p: self._normalize_score(df[f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D']) for p in periods}
        # --- 3. 上升/下跌共振信号 ---
        # B级: 短中期斜率共振
        bullish_momentum_b_arr = slope_scores[1].values * slope_scores[5].values * slope_scores[13].values
        states['SCORE_EMA_BULLISH_RESONANCE_B'] = pd.Series(bullish_momentum_b_arr, index=df.index, dtype=np.float32)
        # A级: B级信号 + 静态排列确认
        bullish_momentum_a_arr = bullish_momentum_b_arr * score_static_bullish_arr
        states['SCORE_EMA_BULLISH_RESONANCE_A'] = pd.Series(bullish_momentum_a_arr, index=df.index, dtype=np.float32)
        # S级: A级信号 + 加速确认
        bullish_accel_s_arr = accel_scores[1].values * accel_scores[5].values
        states['SCORE_EMA_BULLISH_RESONANCE_S'] = pd.Series(bullish_momentum_a_arr * bullish_accel_s_arr, index=df.index, dtype=np.float32)
        # 对称的下跌共振逻辑
        bearish_momentum_b_arr = (1 - slope_scores[1].values) * (1 - slope_scores[5].values) * (1 - slope_scores[13].values)
        states['SCORE_EMA_BEARISH_RESONANCE_B'] = pd.Series(bearish_momentum_b_arr, index=df.index, dtype=np.float32)
        bearish_momentum_a_arr = bearish_momentum_b_arr * (1 - score_static_bullish_arr)
        states['SCORE_EMA_BEARISH_RESONANCE_A'] = pd.Series(bearish_momentum_a_arr, index=df.index, dtype=np.float32)
        bearish_accel_s_arr = (1 - accel_scores[1].values) * (1 - accel_scores[5].values)
        states['SCORE_EMA_BEARISH_RESONANCE_S'] = pd.Series(bearish_momentum_a_arr * bearish_accel_s_arr, index=df.index, dtype=np.float32)
        # --- 4. 顶部/底部反转信号 ---
        # B级: 短期加速拐点
        bottom_trigger_b_arr = accel_scores[1].values
        states['SCORE_EMA_BOTTOM_REVERSAL_B'] = pd.Series(bottom_trigger_b_arr, index=df.index, dtype=np.float32)
        # A级: B级信号 + 长期趋势环境确认
        long_term_down_env_arr = 1 - slope_scores[55].values
        bottom_trigger_a_arr = bottom_trigger_b_arr * long_term_down_env_arr
        states['SCORE_EMA_BOTTOM_REVERSAL_A'] = pd.Series(bottom_trigger_a_arr, index=df.index, dtype=np.float32)
        # S级: A级信号 + 短期斜率确认
        short_term_mom_up_arr = slope_scores[5].values
        states['SCORE_EMA_BOTTOM_REVERSAL_S'] = pd.Series(bottom_trigger_a_arr * short_term_mom_up_arr, index=df.index, dtype=np.float32)
        # 对称的顶部反转逻辑
        top_trigger_b_arr = 1 - accel_scores[1].values
        states['SCORE_EMA_TOP_REVERSAL_B'] = pd.Series(top_trigger_b_arr, index=df.index, dtype=np.float32)
        long_term_up_env_arr = slope_scores[55].values
        top_trigger_a_arr = top_trigger_b_arr * long_term_up_env_arr
        states['SCORE_EMA_TOP_REVERSAL_A'] = pd.Series(top_trigger_a_arr, index=df.index, dtype=np.float32)
        short_term_mom_down_arr = 1 - slope_scores[5].values
        states['SCORE_EMA_TOP_REVERSAL_S'] = pd.Series(top_trigger_a_arr * short_term_mom_down_arr, index=df.index, dtype=np.float32)
        
        return states

    def diagnose_foundation_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 性能优化版】基础层最终协同诊断模块
        - 核心升级 (本次修改):
          - [性能优化] 重构了内部辅助函数 `get_weighted_synergy_score`，使用 `np.stack` 和 `np.average` 代替原有的循环和Series加法。
        - 核心重构 (V2.0逻辑保留):
          - [逻辑升级] 采用基于加权平均的“共识”逻辑，替代旧的“或”门逻辑。
          - [信号升级] 生成代表多领域共识的S级和S+级信号。
        - 收益:
          - 通过避免创建临时DataFrame和循环中的Series操作，显著降低了内存占用并提升了计算速度。
          - 极大提升了最终输出信号的置信度，能有效识别出得到市场多方面验证的交易机会。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # 重构辅助函数以提升性能
        def get_weighted_synergy_score(signal_weights: Dict[str, float]) -> pd.Series:
            """【性能优化版】辅助函数，用于高效计算加权协同分数"""
            # 1. 准备信号Series列表和权重列表
            signals_to_average = []
            weights_for_average = []
            for signal_name, weight in signal_weights.items():
                # 使用 atomic.get 获取信号，如果不存在则使用 default_score
                signals_to_average.append(atomic.get(signal_name, default_score))
                weights_for_average.append(weight)
            # 如果列表为空或权重和为0，直接返回默认分
            if not signals_to_average or not any(weights_for_average):
                return default_score.copy()
            # 2. 将Series列表的底层值提取并堆叠成2D NumPy数组
            # 形状为 (信号数量, 时间序列长度)
            stacked_scores = np.stack([s.values for s in signals_to_average], axis=0)
            # 3. 使用np.average进行高效的加权平均计算
            # axis=0表示沿着信号维度（即对每个时间点上的所有信号）进行计算
            weighted_avg_values = np.average(stacked_scores, axis=0, weights=weights_for_average)
            # 4. 将结果包装回Pandas Series
            return pd.Series(weighted_avg_values, index=df.index, dtype=np.float32)
        
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
        # --- 5. S+ 信号生成 (Synergy-Plus)，代表最高置信度的机会 ---
        bullish_resonance_s = states['SCORE_FOUNDATION_BULLISH_RESONANCE_S']
        breakout_potential_s = atomic.get('SCORE_VOL_BREAKOUT_POTENTIAL_S', default_score)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_S_PLUS'] = (bullish_resonance_s * breakout_potential_s).astype(np.float32)
        bearish_resonance_s = states['SCORE_FOUNDATION_BEARISH_RESONANCE_S']
        breakdown_risk_s = atomic.get('SCORE_VOL_BREAKDOWN_RISK_S', default_score)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_S_PLUS'] = (bearish_resonance_s * breakdown_risk_s).astype(np.float32)
        bottom_reversal_s = states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S']
        vol_ignition_opp = atomic.get('SCORE_VOL_TIPPING_POINT_BOTTOM_OPP', default_score)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S_PLUS'] = (bottom_reversal_s * vol_ignition_opp).astype(np.float32)
        top_reversal_s = states['SCORE_FOUNDATION_TOP_REVERSAL_S']
        vol_exhaustion_risk = atomic.get('SCORE_VOL_TIPPING_POINT_TOP_RISK', default_score)
        states['SCORE_FOUNDATION_TOP_REVERSAL_S_PLUS'] = (top_reversal_s * vol_exhaustion_risk).astype(np.float32)
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

        # --- 2. 基础静态诊断 ---
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
        【V6.4 状态诊断升级版】波动率统一情报中心
        - 核心升级 (本次修改):
          - [逻辑重构] 将“波动率拐点”的判断逻辑从一个严苛的、事件驱动的“斜率过零”模型，升级为一个更鲁棒的、状态驱动的“状态x加速度”模型。
          - [新范式] 新的“波动率拐点分” = “S级波动率压缩分” * “波动率加速分”。
        - 收益: 解决了因“斜率未过零”导致信号得分为零的顽固问题，使信号能更早、更稳定地捕捉到波动率从压缩到扩张的转折状态。
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
        
        # 计算波动率的加速分，用于新的拐点模型
        score_vol_accel_up = self._normalize_score(df['ACCEL_5_BBW_21_2.0_D'], ascending=True)
        score_vol_accel_down = self._normalize_score(df['ACCEL_5_BBW_21_2.0_D'], ascending=False)

        # --- 3. 生成B/A/S三级压缩信号 ---
        states['SCORE_VOL_COMPRESSION_B'] = score_squeeze_daily
        states['SCORE_VOL_COMPRESSION_A'] = (score_squeeze_daily * score_squeeze_weekly).astype(np.float32)
        states['SCORE_VOL_COMPRESSION_S'] = (states['SCORE_VOL_COMPRESSION_A'] * score_squeeze_momentum).astype(np.float32)

        # --- 4. 生成B/A/S三级扩张信号 (对称逻辑) ---
        states['SCORE_VOL_EXPANSION_B'] = score_expansion_daily
        states['SCORE_VOL_EXPANSION_A'] = (score_expansion_daily * score_expansion_weekly).astype(np.float32)
        states['SCORE_VOL_EXPANSION_S'] = (states['SCORE_VOL_EXPANSION_A'] * score_expansion_momentum).astype(np.float32)

        # --- 5. 波动率反转临界点 (逻辑重构：事件 -> 状态) ---
        # 废弃旧的事件驱动逻辑
        # is_tipping_point_bottom_event = (df['SLOPE_5_BBW_21_2.0_D'] > 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) <= 0)
        # persistent_bottom_state = create_persistent_state(df, is_tipping_point_bottom_event, 3, pd.Series(False, index=df.index), 'VOL_TIPPING_POINT_BOTTOM')
        # states['SCORE_VOL_TIPPING_POINT_BOTTOM_OPP'] = (states['SCORE_VOL_COMPRESSION_S'] * persistent_bottom_state).astype(np.float32)
        
        # 采用新的状态驱动逻辑: 拐点分 = 压缩状态分 * 加速分
        states['SCORE_VOL_TIPPING_POINT_BOTTOM_OPP'] = (states['SCORE_VOL_COMPRESSION_S'] * score_vol_accel_up).astype(np.float32)
        
        # 对称地更新顶部拐点逻辑
        # is_tipping_point_top_event = (df['SLOPE_5_BBW_21_2.0_D'] < 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) >= 0)
        # persistent_top_state = create_persistent_state(df, is_tipping_point_top_event, 3, pd.Series(False, index=df.index), 'VOL_TIPPING_POINT_TOP')
        # states['SCORE_VOL_TIPPING_POINT_TOP_RISK'] = (states['SCORE_VOL_EXPANSION_S'] * persistent_top_state).astype(np.float32)
        states['SCORE_VOL_TIPPING_POINT_TOP_RISK'] = (states['SCORE_VOL_EXPANSION_S'] * score_vol_accel_down).astype(np.float32)
        
        # 更新探针逻辑以反映新的计算方式
        debug_params = get_params_block(self.strategy, 'debug_params')
        probe_date_str = get_param_value(debug_params.get('probe_date'))
        if probe_date_str:
            probe_ts = pd.to_datetime(probe_date_str)
            if df.index.tz is not None:
                probe_ts = probe_ts.tz_localize(df.index.tz)
            if probe_ts in df.index:
                print(f"\n          --- [一线探针: 波动率拐点活检 @ {probe_date_str}] ---")
                print(f"          - 因子1 (S级压缩分): {states['SCORE_VOL_COMPRESSION_S'].get(probe_ts, -1):.4f}")
                print(f"          - 因子2 (波动率加速分): {score_vol_accel_up.get(probe_ts, -1):.4f}")
                print(f"          - 最终信号分 (因子1 * 因子2): {states['SCORE_VOL_TIPPING_POINT_BOTTOM_OPP'].get(probe_ts, -1):.4f}")
                print(f"          ----------------------------------------------------------\n")

        # --- 6. 市场政权与数值化评分 ---
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
            # 增加对数据层缺失的精确提示，使其更具可操作性
            print(f"          -> [数据需求] 请数据工程层为 'total_winner_rate_D' 指标补充以下缺失的衍生列: {missing}。")
            print(f"          -> [提示] 衍生列计算方法：'SLOPE_n_...' 为n日斜率, 'ACCEL_n_...' 为n日斜率的加速度。")
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

        # --- 3. ATR 绝对波幅状态与临界点信号 ---
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
        【V6.1 深度活检版】经典指标统一情报中心
        - 核心升级 (本次修改):
          - [深度活检] 植入了一个由 `debug_params.probe_date` 控制的“一线法医探针”。
                        当指定日期时，它会打印出计算“MACD底部反转分”的所有三维共振因子：
                        环境分(Z-score)、动能分(斜率)、加速度分，实现对信号生成的完全透视。
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

        # 植入“MACD活检探针”
        debug_params = get_params_block(self.strategy, 'debug_params')
        probe_date_str = get_param_value(debug_params.get('probe_date'))
        if probe_date_str:
            probe_ts = pd.to_datetime(probe_date_str)
            if df.index.tz is not None:
                probe_ts = probe_ts.tz_localize(df.index.tz)
            if probe_ts in df.index:
                print(f"\n          --- [一线探针: MACD底部反转活检 @ {probe_date_str}] ---")
                print(f"          - B级信号 (加速度分): {states['SCORE_MACD_BOTTOM_REVERSAL_B'].get(probe_ts, -1):.4f}  <-- [活检] 原始加速度分 = {score_accel_short.get(probe_ts, -1):.4f}")
                print(f"          - A级信号 (B级*环境分): {states['SCORE_MACD_BOTTOM_REVERSAL_A'].get(probe_ts, -1):.4f}  <-- [活检] 环境分(Z-score) = {score_macdh_zscore_low.get(probe_ts, -1):.4f}")
                print(f"          - S级信号 (A级*动能分): {states['SCORE_MACD_BOTTOM_REVERSAL_S'].get(probe_ts, -1):.4f}  <-- [活检] 动能分(斜率) = {score_mom_short.get(probe_ts, -1):.4f}")
                print(f"          ----------------------------------------------------------\n")

        # --- 3. 成交量动态分析 ---
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









