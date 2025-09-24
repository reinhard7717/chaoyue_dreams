# 文件: strategies/trend_following/intelligence/micro_behavior_engine.py
# 微观行为诊断引擎
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, fuse_multi_level_scores, create_persistent_state

class MicroBehaviorEngine:
    """
    微观行为诊断引擎
    - 核心职责: 诊断微观层面的、复杂的、但又非常具体的市场行为模式。
                这些模式通常是多个基础信号的精巧组合，用于识别主力的特定意图。
    - 来源: 从臃肿的 CognitiveIntelligence 模块中拆分而来。
    """
    def __init__(self, strategy_instance):
        """
        初始化微观行为诊断引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def _normalize_score(self, series: pd.Series, window: int = 120, ascending: bool = True, default=0.5) -> pd.Series:
        """辅助函数：将一个Series进行滚动窗口排名归一化，生成0-1分。"""
        if series is None or series.empty:
            return pd.Series(default, index=self.strategy.df_indicators.index)
        min_periods = max(1, window // 5)
        rank = series.rolling(window=window, min_periods=min_periods).rank(pct=True)
        score = rank if ascending else 1 - rank
        return score.fillna(default).astype(np.float32)

    def _fuse_multi_level_scores(self, df: pd.DataFrame, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
        """融合S+/S/A/B等多层置信度分数的辅助函数。"""
        if weights is None:
            weights = {'S_PLUS': 1.5, 'S': 1.0, 'A': 0.6, 'B': 0.3}
        total_score = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for level in ['S_PLUS', 'S', 'A', 'B']:
            if level not in weights: continue
            weight = weights[level]
            score_name = f"SCORE_{base_name}_{level}"
            if score_name in self.strategy.atomic_states:
                score_series = self.strategy.atomic_states[score_name]
                if len(score_series) > 0:
                    total_score += score_series.reindex(df.index).fillna(0.0) * weight
                    total_weight += weight
        if total_weight == 0:
            single_score_name = f"SCORE_{base_name}"
            if single_score_name in self.strategy.atomic_states:
                return self.strategy.atomic_states[single_score_name].reindex(df.index).fillna(0.5)
            return pd.Series(0.5, index=df.index)
        return (total_score / total_weight).clip(0, 1)

    def run_micro_behavior_synthesis(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【新增方法】微观行为诊断引擎总指挥
        - 核心职责: 按顺序调用本模块内的所有诊断方法，并汇总其产出的所有信号。
        """
        print("      -> [微观行为诊断引擎] 启动...")
        all_states = {}
        early_momentum_states = self.synthesize_early_momentum_ignition(df)
        all_states.update(early_momentum_states)
        # 从刚刚计算的结果中提取出下游方法需要的信号
        early_ignition_score = early_momentum_states.get(
            'COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', 
            pd.Series(0.0, index=df.index, dtype=np.float32)
        )
        all_states.update(self.synthesize_early_momentum_ignition(df))
        all_states.update(self.diagnose_deceptive_retail_flow(df))
        all_states.update(self.synthesize_microstructure_dynamics(df))
        all_states.update(self.synthesize_euphoric_acceleration_risk(df))
        reversal_states = self.synthesize_reversal_reliability_score(df, early_ignition_score=early_ignition_score)
        all_states.update(reversal_states)
        print(f"      -> [微观行为诊断引擎] 分析完毕，共生成 {len(all_states)} 个微观行为信号。")
        return all_states

    def synthesize_early_momentum_ignition(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 · 纯粹形态版】早期动能点火诊断模块 (回归V6.0)
        - 核心重构 (本次修改):
          - [回归初心] 彻底移除V7.0引入的“下跌趋势”和“底部区域”双重上下文过滤器。
          - [职责净化] 让此信号回归其核心职责：作为一个纯粹的、强大的“大阳线质量分”识别器。
          - [新范式] 最终分数 = K线实体强度分 * K线位置强度分 * K线动能强度分。
        - 收益: 解决了因上下文重复过滤导致“K线质量分”被过度压制的问题。将上下文判断的职责完全交还给上游的 `TRIGGER_DOMINANT_REVERSAL` 模块，使架构更清晰、逻辑更正确。
        """
        print("        -> [早期动能点火诊断模块 V8.0 · 纯粹形态版] 启动...")
        states = {}
        # --- 1. K线实体强度分 ---
        # 阳线实体占当天振幅的比例
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        body_size = (df['close_D'] - df['open_D']).clip(lower=0) # 只考虑阳线
        body_strength_score = (body_size / candle_range).fillna(0.0)
        # --- 2. K线位置强度分 ---
        # 收盘价在当天振幅中的位置
        position_in_range_score = ((df['close_D'] - df['low_D']) / candle_range).fillna(0.0)
        # --- 3. K线动能强度分 ---
        # 对当日涨幅进行归一化，涨幅越大分数越高
        # 使用一个合理的范围（如0%到10%）进行线性归一化，而不是相对排名
        momentum_strength_score = (df['pct_change_D'] / 0.10).clip(0, 1).fillna(0.0)
        # --- 4. 融合生成最终信号 ---
        # 三者相乘，得到一个综合的“大阳线质量分”
        final_score = (body_strength_score * position_in_range_score * momentum_strength_score).astype(np.float32)
        states['COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A'] = final_score
        return states

    def diagnose_deceptive_retail_flow(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 VPA增强版】伪装散户吸筹诊断引擎 (主力分单行为识别)
        - 架构归属: 从 ChipIntelligence 迁移至 CognitiveIntelligence，因为它融合了筹码、资金、价格、量价四大维度。
        - 核心增强: 新增对 VPA 效率的判断，形成四维交叉验证，极大提升信号置信度。
        - 核心逻辑:
          1. 资金流表象: 散户资金持续净流入。
          2. 筹码结构结果: 筹码持续集中。
          3. 价格环境: 股价波动被压制。
          4. 量价效率佐证 (VPA): 成交量很大，但价格波动很小，证明交易未用于推升价格。
        - 产出: SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION_S - 一个高置信度的、识别主力隐蔽吸筹的S级认知信号。
        """
        states = {}
        p = get_params_block(self.strategy, 'deceptive_flow_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states
        required_cols = [
            'retail_net_flow_consensus_D', 'SLOPE_5_concentration_90pct_D',
            'SLOPE_5_close_D', 'VPA_EFFICIENCY_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 伪装散户吸筹诊断引擎缺少关键数据: {missing_cols}，模块已跳过！")
            return states
        norm_window = get_param_value(p.get('norm_window'), 120)
        retail_inflow_score = self._normalize_score(df['retail_net_flow_consensus_D'].clip(lower=0), norm_window, ascending=True)
        chip_concentration_score = self._normalize_score(df['SLOPE_5_concentration_90pct_D'], norm_window, ascending=False)
        price_suppression_score = self._normalize_score(df['SLOPE_5_close_D'].abs(), norm_window, ascending=False)
        vpa_inefficiency_score = self._normalize_score(df['VPA_EFFICIENCY_D'], norm_window, ascending=False)
        final_score = (
            retail_inflow_score * chip_concentration_score *
            price_suppression_score * vpa_inefficiency_score
        ).astype(np.float32)
        states['SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION_S'] = final_score
        if (final_score > 0.85).any():
             print(f"          -> [S级认知信号] 侦测到 {(final_score > 0.85).sum()} 次高度疑似“伪装散户吸筹”的博弈行为！")
        return states

    def synthesize_microstructure_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 完全对称版】市场微观结构动态诊断引擎
        - 核心升级 (本次修改):
          - [对称实现] 补全了所有机会和风险的镜像信号，现在能同时诊断四种状态：
            1. 机会：主导权向主力转移
            2. 风险：主导权向散户转移 (新增)
            3. 机会：主力信念在加强 (新增)
            4. 风险：主力信念在瓦解
        - 收益: 实现了对市场微观结构变化的完全对称、无死角的监控。
        """
        states = {}
        norm_window = 120
        granularity_momentum_up = self._normalize_score(df.get('SLOPE_5_avg_order_value_D'), norm_window, ascending=True)
        granularity_accel_up = self._normalize_score(df.get('ACCEL_5_avg_order_value_D'), norm_window, ascending=True)
        dominance_momentum_up = self._normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), norm_window, ascending=True)
        dominance_accel_up = self._normalize_score(df.get('ACCEL_5_trade_concentration_index_D'), norm_window, ascending=True)
        power_shift_to_main_force_score = (
            granularity_momentum_up * granularity_accel_up *
            dominance_momentum_up * dominance_accel_up
        ).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE'] = power_shift_to_main_force_score
        granularity_momentum_down = self._normalize_score(df.get('SLOPE_5_avg_order_value_D'), norm_window, ascending=False)
        granularity_accel_down = self._normalize_score(df.get('ACCEL_5_avg_order_value_D'), norm_window, ascending=False)
        dominance_momentum_down = self._normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), norm_window, ascending=False)
        dominance_accel_down = self._normalize_score(df.get('ACCEL_5_trade_concentration_index_D'), norm_window, ascending=False)
        power_shift_to_retail_risk = (
            granularity_momentum_down * granularity_accel_down *
            dominance_momentum_down * dominance_accel_down
        ).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = power_shift_to_retail_risk
        conviction_momentum_weakening = self._normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), norm_window, ascending=False)
        conviction_accel_weakening = self._normalize_score(df.get('ACCEL_5_main_force_conviction_ratio_D'), norm_window, ascending=False)
        conviction_weakening_risk = (conviction_momentum_weakening * conviction_accel_weakening).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = conviction_weakening_risk
        conviction_momentum_strengthening = self._normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), norm_window, ascending=True)
        conviction_accel_strengthening = self._normalize_score(df.get('ACCEL_5_main_force_conviction_ratio_D'), norm_window, ascending=True)
        conviction_strengthening_opp = (conviction_momentum_strengthening * conviction_accel_strengthening).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING'] = conviction_strengthening_opp
        return states

    def synthesize_euphoric_acceleration_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 上下文净化版】亢奋加速风险诊断引擎 (Euphoric Acceleration Risk)
        - 核心升级 (本次修改):
          - [上下文净化] 引入了“顶部位置上下文”过滤器。现在，最终风险分 = 原始风险分 * 顶部位置分。
        - 收益: 从根本上解决了该信号在底部区域被错误激活的问题。确保了这个为“顶部”量身定制的风险信号，只在价格处于高位时才生效，避免了对底部反转机会的“乌龙”惩罚。
        """
       
        states = {}
        p_risk = get_params_block(self.strategy, 'euphoric_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True):
            return states
        norm_window = get_param_value(p_risk.get('norm_window'), 120)

        # 新增“顶部位置上下文分”作为过滤器
        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        # 价格在55日区间内的位置分 (0=最低点, 1=最高点)
        top_context_score = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)

        # --- 计算原始风险的四个维度 (逻辑不变) ---
        bias_score = self._normalize_score(df['BIAS_21_D'].abs(), norm_window, ascending=True)
        volume_ratio = (df['volume_D'] / df.get('VOL_MA_55_D', df['volume_D'])).fillna(1.0)
        volume_spike_score = self._normalize_score(volume_ratio, norm_window, ascending=True)
        atr_ratio = (df['ATR_14_D'] / df['close_D']).fillna(0.0)
        volatility_score = self._normalize_score(atr_ratio, norm_window, ascending=True)
        total_range = (df['high_D'] - df['low_D']).replace(0, 1e-9)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        upper_shadow_ratio = (upper_shadow / total_range).clip(0, 1).fillna(0.0)
        upthrust_score = upper_shadow_ratio
        
        # 计算原始风险分
        raw_risk_score = (
            bias_score * volume_spike_score * volatility_score * upthrust_score
        )**(1/4)

        # 将原始风险分与位置上下文过滤器相乘
        final_risk_score = (raw_risk_score * top_context_score).astype(np.float32)
        
        states['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = final_risk_score
        return states

    def synthesize_reversal_reliability_score(self, df: pd.DataFrame, early_ignition_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V4.5 终极优化版】高质量战备可靠性诊断引擎
        - 核心升级 (本次修改):
          - [逻辑重构] 将“深度价值区”的融合逻辑从“乘法”（与逻辑）升级为“取最大值”（或逻辑）。
          - [新范式] 新的“深度价值区分” = MAX(价格位置分, 周线RSI超卖分)，使其能捕捉任一价值区形态。
        - 收益: 解决了因价值区判断条件过于严苛导致奖励项失效的问题，显著提升了王牌信号在真实价值底部的得分能力和区分度。
        """
        
        print("        -> [高质量战备可靠性诊断引擎 V4.5 终极优化版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'reversal_reliability_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        norm_window = get_param_value(p.get('norm_window'), 120)
        # --- 第一幕：背景设定 (The Setup) - 深度价值区 ---
        price_pos_yearly = self._normalize_score(df['close_D'], window=250, ascending=True, default=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        states['INTERNAL_SCORE_DEEP_BOTTOM_CONTEXT'] = deep_bottom_context_score.astype(np.float32)
        rsi_w_oversold_score = self._normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), window=52, ascending=False, default=0.5)
        states['INTERNAL_SCORE_RSI_W_OVERSOLD'] = rsi_w_oversold_score.astype(np.float32)
        # 将乘法改为取最大值，以体现“或”逻辑
        background_score = np.maximum(deep_bottom_context_score, rsi_w_oversold_score).astype(np.float32)
        states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] = background_score
        # --- 第二幕：矛盾冲突 (The Conflict) - 强弱手换庄 ---
        shareholder_turnover_score = np.maximum.reduce([
            atomic.get('SCORE_CHIP_TRUE_ACCUMULATION', default_score).values,
            atomic.get('SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL', default_score).values,
            atomic.get('COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING', default_score).values
        ])
        shareholder_quality_score = pd.Series(shareholder_turnover_score, index=df.index, dtype=np.float32)
        states['SCORE_SHAREHOLDER_QUALITY_IMPROVEMENT'] = shareholder_quality_score
        # --- 第三幕：转折点火 (The Ignition) - 共振初起 ---
        fft_trend_score = atomic.get('SCORE_TRENDING_REGIME_FFT', default_score)
        fft_trend_slope = fft_trend_score.diff(5).fillna(0)
        trend_potential_score = self._normalize_score(fft_trend_slope.clip(lower=0), window=norm_window, ascending=True, default=0.0)
        states['INTERNAL_SCORE_TREND_POTENTIAL'] = trend_potential_score.astype(np.float32)
        vol_compression_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        ignition_weights = get_param_value(p.get('ignition_weights'), {'early': 0.5, 'vol': 0.2, 'potential': 0.3})
        ignition_confirmation_score = (
            early_ignition_score * ignition_weights['early'] +
            vol_compression_score * ignition_weights['vol'] +
            trend_potential_score * ignition_weights['potential']
        ).astype(np.float32)
        states['SCORE_IGNITION_CONFIRMATION'] = ignition_confirmation_score
        # --- 最终剧本触发逻辑 (核心分 + 奖励分) ---
        main_reliability_weights = get_param_value(p.get('main_reliability_weights'), {'shareholder': 0.5, 'ignition': 0.5})
        main_score = (
            shareholder_quality_score * main_reliability_weights['shareholder'] +
            ignition_confirmation_score * main_reliability_weights['ignition']
        )
        bonus_factor = get_param_value(p.get('reversal_reliability_bonus_factor'), 0.5)
        final_reliability_score = (main_score * (1 + background_score * bonus_factor)).astype(np.float32)
        states['COGNITIVE_SCORE_REVERSAL_RELIABILITY'] = final_reliability_score
        # 植入“一线法医探针”
        debug_params = get_params_block(self.strategy, 'debug_params')
        probe_date_str = get_param_value(debug_params.get('probe_date'))
        if probe_date_str:
            probe_ts = pd.to_datetime(probe_date_str)
            if probe_ts in df.index:
                # 更新探针逻辑以反映新的计算方式
                probe_ignition_score = (
                    early_ignition_score.get(probe_ts, -1) * ignition_weights['early'] +
                    vol_compression_score.get(probe_ts, -1) * ignition_weights['vol'] +
                    trend_potential_score.get(probe_ts, -1) * ignition_weights['potential']
                )
                probe_main_score = (
                    shareholder_quality_score.get(probe_ts, -1) * main_reliability_weights['shareholder'] +
                    ignition_confirmation_score.get(probe_ts, -1) * main_reliability_weights['ignition']
                )
                probe_final_score = probe_main_score * (1 + background_score.get(probe_ts, -1) * bonus_factor)
                print(f"\n          --- [一线探针: 高质量战备诊断 @ {probe_date_str}] ---")
                print(f"          --- 企稳点火分 (内部计算) ---")
                print(f"            - 早期动能分: {early_ignition_score.get(probe_ts, -1):.4f} (权重: {ignition_weights['early']})")
                print(f"            - 波动压缩分: {vol_compression_score.get(probe_ts, -1):.4f} (权重: {ignition_weights['vol']})")
                print(f"            - 趋势潜力分: {trend_potential_score.get(probe_ts, -1):.4f} (权重: {ignition_weights['potential']})")
                print(f"            - [探针验算] 企稳点火分: {probe_ignition_score:.4f} vs 实际值: {ignition_confirmation_score.get(probe_ts, -1):.4f}")
                print(f"          --- 王牌信号分 (最终计算) ---")
                print(f"          - 核心分: {main_score.get(probe_ts, -1):.4f}")
                print(f"          - 价值区奖励分: {background_score.get(probe_ts, -1):.4f} (奖励系数: {bonus_factor})")
                print(f"          - 决策公式: 核心分 * (1 + 价值区奖励分 * 奖励系数)")
                print(f"          - [探针验算] 最终可靠性分: {probe_final_score:.4f} vs 实际值: {final_reliability_score.get(probe_ts, -1):.4f}")
                print(f"          ----------------------------------------------------------\n")
        return states









