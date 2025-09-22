# 文件: strategies/trend_following/intelligence/micro_behavior_engine.py
# 微观行为诊断引擎
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value

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
        all_states.update(self.synthesize_early_momentum_ignition(df))
        all_states.update(self.diagnose_deceptive_retail_flow(df))
        all_states.update(self.synthesize_microstructure_dynamics(df))
        all_states.update(self.synthesize_euphoric_acceleration_risk(df))
        all_states.update(self.synthesize_reversal_reliability_score(df))
        print(f"      -> [微观行为诊断引擎] 分析完毕，共生成 {len(all_states)} 个微观行为信号。")
        return all_states

    def synthesize_early_momentum_ignition(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】早期动能点火诊断模块 (东风初起)
        - 核心目标: 识别“万事俱备”之后，动能“刚刚启动”的精确时点，避免追高。
        - 核心逻辑: 融合多个“早期”和“温和”的动能信号，如波动率拐点、MACD低位金叉、
                      价格温和放量等，形成一个综合的“早期点火分”。
        - 产出信号:
          - `COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A`: A级早期动能点火信号，可用于替代或补充
                                                         现有激进的动能信号。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        vol_tipping_point_score = atomic.get('SCORE_VOL_TIPPING_POINT_BOTTOM_OPP', default_score)
        macd_reversal_score = np.maximum(
            atomic.get('SCORE_MACD_BOTTOM_REVERSAL_B', default_score).values,
            atomic.get('SCORE_MACD_BOTTOM_REVERSAL_A', default_score).values
        )
        macd_reversal_series = pd.Series(macd_reversal_score, index=df.index)
        pct_change = df['pct_change_D']
        gentle_rally_score = np.maximum(0, 1 - np.abs(pct_change - 0.025) / 0.025).fillna(0)
        volume_ratio = df['volume_D'] / df.get('VOL_MA_21_D', df['volume_D']).replace(0, np.nan)
        vol_score1 = (volume_ratio - 1.2) / (1.8 - 1.2)
        vol_score2 = (3.0 - volume_ratio) / (3.0 - 1.8)
        gentle_volume_score = np.minimum(vol_score1, vol_score2).clip(0, 1).fillna(0)
        price_accel_score = self._normalize_score(df['ACCEL_1_close_D'].clip(lower=0), default=0.0)
        score_components = [
            vol_tipping_point_score.values,
            macd_reversal_series.values,
            gentle_rally_score.values,
            gentle_volume_score.values,
            price_accel_score.values
        ]
        epsilon = 1e-9
        prod_scores = np.prod([arr + epsilon for arr in score_components], axis=0)
        final_score_arr = prod_scores**(1.0 / len(score_components))
        final_score = pd.Series(final_score_arr, index=df.index, dtype=np.float32)
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
        【V1.0 新增】亢奋加速风险诊断引擎 (Euphoric Acceleration Risk)
        - 核心目标: 识别趋势末端的“高潮式”加速上涨，这是最危险的诱多陷阱。
        - 核心逻辑: 风险 = 高乖离度 * 天量成交 * 巨幅振动 * 冲高回落
        - 产出信号: COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION - 一个S级的顶层风险信号。
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'euphoric_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True):
            return states
        norm_window = get_param_value(p_risk.get('norm_window'), 120)
        bias_score = self._normalize_score(df['BIAS_21_D'].abs(), norm_window, ascending=True)
        volume_ratio = (df['volume_D'] / df.get('VOL_MA_55_D', df['volume_D'])).fillna(1.0)
        volume_spike_score = self._normalize_score(volume_ratio, norm_window, ascending=True)
        atr_ratio = (df['ATR_14_D'] / df['close_D']).fillna(0.0)
        volatility_score = self._normalize_score(atr_ratio, norm_window, ascending=True)
        total_range = (df['high_D'] - df['low_D']).replace(0, 1e-9)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        upper_shadow_ratio = (upper_shadow / total_range).clip(0, 1).fillna(0.0)
        upthrust_score = upper_shadow_ratio
        final_risk_score = (
            bias_score * volume_spike_score * volatility_score * upthrust_score
        )**(1/4)
        states['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = final_risk_score.astype(np.float32)
        return states

    def synthesize_reversal_reliability_score(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 逻辑升维版】高质量战备可靠性诊断引擎 (原: 反转可靠性)
        - 核心重构 (本次修改): 彻底重构了信号的融合逻辑。不再使用“三幕剧”的严格乘积，因为这会导致非底部机会被错误地过滤掉。
                          新的逻辑将“股东换血”和“企稳点火”作为核心基础分，而将“深度价值区”作为一个强大的场景加成项。
        - 核心逻辑: 可靠性分 = (核心逻辑: 股东换血 × 企稳点火) × (场景加成: 1 + 深度价值区得分)
        - 收益: 极大扩展了此王牌信号的适用范围，使其能同时评估“底部反转”和“趋势中继蓄势”两种最重要的战机，
                从根本上解决了策略对非底部拉升机会的“认知盲区”。
        """
        # 代码修改：更新版本号和说明
        print("        -> [高质量战备可靠性诊断引擎 V2.1 逻辑升维版] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'reversal_reliability_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        norm_window = get_param_value(p.get('norm_window'), 120)

        # --- 第一幕：背景设定 (The Setup) - 深度价值区 ---
        # 逻辑不变，但它现在作为“加成项”
        price_pos_yearly = self._normalize_score(df['close_D'], window=250, ascending=True, default=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        rsi_w_oversold_score = self._normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), window=52, ascending=False, default=0.5)
        background_score = (deep_bottom_context_score * rsi_w_oversold_score).astype(np.float32)
        states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] = background_score
        print(f"          - [第一幕-深度价值区(加成项)] 完成, 平均分: {background_score.mean():.2f}")

        # --- 第二幕：矛盾冲突 (The Conflict) - 强弱手换庄 ---
        # 逻辑不变，这是核心基础分的一部分
        shareholder_turnover_score = np.maximum.reduce([
            atomic.get('SCORE_CHIP_TRUE_ACCUMULATION', default_score).values,
            atomic.get('SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL', default_score).values,
            atomic.get('COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING', default_score).values
        ])
        shareholder_quality_score = pd.Series(shareholder_turnover_score, index=df.index, dtype=np.float32)
        states['SCORE_SHAREHOLDER_QUALITY_IMPROVEMENT'] = shareholder_quality_score
        print(f"          - [第二幕-股东换血(核心)] 完成, 平均分: {shareholder_quality_score.mean():.2f}")

        # --- 第三幕：转折点火 (The Ignition) - 共振初起 ---
        # 逻辑不变，这是核心基础分的另一部分
        downtrend_stabilizing_score = self._normalize_score(df['SLOPE_55_EMA_55_D'].abs(), norm_window, ascending=False, default=0.0)
        vol_compression_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        early_ignition_score = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score)
        ignition_confirmation_score = (
            downtrend_stabilizing_score * vol_compression_score * early_ignition_score
        ).astype(np.float32)
        states['SCORE_IGNITION_CONFIRMATION'] = ignition_confirmation_score
        print(f"          - [第三幕-企稳点火(核心)] 完成, 平均分: {ignition_confirmation_score.mean():.2f}")

        # --- 最终剧本触发逻辑 (全新融合范式) ---
        # 核心逻辑分 = 股东换血分 * 企稳点火分
        core_logic_score = (
            states['SCORE_SHAREHOLDER_QUALITY_IMPROVEMENT'] *
            states['SCORE_IGNITION_CONFIRMATION']
        )
        
        # 最终可靠性分 = 核心逻辑分 * (1 + 场景加成)
        # 这里的 “* 1.0” 是加成系数，可以配置，代表底部场景能让分数翻倍
        final_reliability_score = (core_logic_score * (1.0 + states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] * 1.0)).astype(np.float32)
        
        # 同时输出两个信号名，以兼容进攻层和剧本引擎
        states['COGNITIVE_SCORE_REVERSAL_RELIABILITY'] = final_reliability_score
        states['COGNITIVE_SCORE_OPP_POST_REVERSAL_RESONANCE_A_PLUS'] = final_reliability_score
        
        if (final_reliability_score > 0.1).any():
            print(f"  [探针-高质量战备] 侦测到 {(final_reliability_score > 0.1).sum()} 次高可靠性战备机会！最高分: {final_reliability_score.max():.3f}")
        
        return states












