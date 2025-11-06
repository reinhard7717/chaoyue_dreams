import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar

class CognitiveIntelligence:
    """
    【V20.0 · 贝叶斯战术推演引擎】
    - 核心重构: 废弃旧的、分散的信号合成方法，引入统一的“贝叶斯战术推演”框架。
    - 核心思想: 将A股的复杂博弈场景抽象为一系列“战术剧本”。引擎不再是简单地叠加信号，
                  而是基于融合层提供的“战场态势”（先验信念），结合原子层的“微观证据”（似然度），
                  通过贝叶斯推演，计算出每个战术剧本上演的“后验概率”（最终信号分）。
    - 收益: 使认知层的每一个判断都有清晰的数学逻辑和博弈论基础，直指A股本质。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        # 认知层不再需要依赖其他引擎的内部方法，实现完全解耦

    def _get_fused_score(self, name: str, default: float = 0.0) -> pd.Series:
        """安全地从原子状态库中获取由融合层提供的态势分数。"""
        # 认知层只消费融合层和原子层的信号
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        else:
            # print(f"    -> [认知层警告] 融合态势信号 '{name}' 不存在，使用默认值 {default}。")
            return pd.Series(default, index=self.strategy.df_indicators.index)

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V22.0 · 战略推演版】总指挥
        - 核心升级: 扩充战术剧本库，新增“龙头苏醒”、“板块轮动”、“趋势衰竭”、“能量压缩”等多个A股核心博弈场景的推演。
        """
        print("启动【V22.0 · 战略推演版】认知推演引擎...")
        all_cognitive_states = {}
        # --- 步骤一: 建立先验信念 (逻辑不变) ---
        print("  -- [认知层] 步骤一: 正在建立先验信念...")
        priors = self._establish_prior_beliefs()
        all_cognitive_states.update(priors)
        # --- 步骤二: 并行推演所有战术剧本，获得“静态后验概率” ---
        print("  -- [认知层] 步骤二: 正在进行贝叶斯推演，获取静态后验概率...")
        raw_playbook_scores = {}
        # 原有剧本
        raw_playbook_scores.update(self._deduce_suppressive_accumulation(priors))
        raw_playbook_scores.update(self._deduce_chasing_accumulation(priors))
        raw_playbook_scores.update(self._deduce_capitulation_reversal(priors))
        raw_playbook_scores.update(self._deduce_distribution_at_high(priors))
        # 新增剧本
        raw_playbook_scores.update(self._deduce_leading_dragon_awakening(priors))
        raw_playbook_scores.update(self._deduce_sector_rotation_vanguard(priors))
        raw_playbook_scores.update(self._deduce_exhausted_crossbow_at_peak(priors))
        raw_playbook_scores.update(self._deduce_energy_compression_breakout(priors))
        # 将原始推演结果存入，用于调试和追溯
        all_cognitive_states.update(raw_playbook_scores)
        self.strategy.atomic_states.update(raw_playbook_scores)
        # --- 步骤三 (新增): 对静态概率进行“动态锻造” ---
        print("  -- [认知层] 步骤三: 正在对后验概率进行“动态锻造”...")
        forged_playbook_scores = self._perform_dynamic_forging(df, raw_playbook_scores)
        all_cognitive_states.update(forged_playbook_scores)
        self.strategy.atomic_states.update(forged_playbook_scores)
        # --- 步骤四: 基于“动态锻造”后的信号进行最终融合与裁决 ---
        print("  -- [认知层] 步骤四: 正在融合动态信号，输出最终认知...")
        # 注意：现在传入的是经过动态锻造后的分数
        final_scores = self._fuse_and_adjudicate_playbooks(forged_playbook_scores)
        all_cognitive_states.update(final_scores)
        # 更新原子状态库
        self.strategy.atomic_states.update(final_scores)
        print(f"【V22.0 · 战略推演版】分析完成，生成 {len(all_cognitive_states)} 个认知信号。")
        return all_cognitive_states

    def _establish_prior_beliefs(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】建立先验信念
        - 核心职责: 将融合层的四大态势，转化为用于贝叶斯推演的“先验概率”。
        """
        states = {}
        # 先验1: 市场处于趋势状态的概率 P(Trend)
        # 证据: 市场政权为正(趋势) * 趋势质量为正(健康)
        market_regime = self._get_fused_score('FUSION_BIPOLAR_MARKET_REGIME', 0.0)
        trend_quality = self._get_fused_score('FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        prior_trend = (market_regime.clip(lower=0) * trend_quality.clip(lower=0)).pow(0.5)
        states['COGNITIVE_PRIOR_TREND_PROB'] = prior_trend.astype(np.float32)

        # 先验2: 市场处于反转临界点的概率 P(Reversal)
        # 证据: 市场压力巨大 * 市场政权趋向反转
        market_pressure = self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0)
        prior_reversal = (market_pressure.abs() * (1 - market_regime.abs())).pow(0.5)
        states['COGNITIVE_PRIOR_REVERSAL_PROB'] = prior_reversal.astype(np.float32)

        return states

    def _get_atomic_score(self, name: str, default: float = 0.0) -> pd.Series:
        """
        【V2.0 · 统一信号获取版】
        安全地从原子状态库或主数据帧中获取信号。
        """
        # 优先从原子状态库获取，因为那里可能有经过处理的信号
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        # 如果没有，再从主数据帧获取原始指标
        elif name in self.strategy.df_indicators.columns:
            return self.strategy.df_indicators[name]
        else:
            # print(f"    -> [认知层警告] 信号 '{name}' 不存在，使用默认值 {default}。")
            return pd.Series(default, index=self.strategy.df_indicators.index)

    def _fuse_and_adjudicate_playbooks(self, playbook_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.1 · 信号净化版】融合与裁决模块
        - 核心修改: 更新了看跌剧本的名称，从 'EXHAUSTED_CROSSBOW' 更新为 'TREND_EXHAUSTION'。
        """
        states = {}
        df_index = self.strategy.df_indicators.index
        # 融合所有看涨剧本的“锻造后”分数
        bullish_playbooks = [
            'COGNITIVE_FORGED_PLAYBOOK_SUPPRESSIVE_ACCUMULATION',
            'COGNITIVE_FORGED_PLAYBOOK_CHASING_ACCUMULATION',
            'COGNITIVE_FORGED_PLAYBOOK_CAPITULATION_REVERSAL',
            'COGNITIVE_FORGED_PLAYBOOK_LEADING_DRAGON_AWAKENING',
            'COGNITIVE_FORGED_PLAYBOOK_SECTOR_ROTATION_VANGUARD',
            'COGNITIVE_FORGED_PLAYBOOK_ENERGY_COMPRESSION',
        ]
        bullish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bullish_playbooks]
        cognitive_bullish_score = np.maximum.reduce([s.values for s in bullish_scores])
        states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df_index, dtype=np.float32)
        # [代码修改开始]
        # 融合所有看跌剧本的“锻造后”分数
        bearish_playbooks = [
            'COGNITIVE_FORGED_RISK_DISTRIBUTION_AT_HIGH',
            'COGNITIVE_FORGED_RISK_TREND_EXHAUSTION',
        ]
        # [代码修改结束]
        bearish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bearish_playbooks]
        cognitive_bearish_score = np.maximum.reduce([s.values for s in bearish_scores])
        states['COGNITIVE_BEARISH_SCORE'] = pd.Series(cognitive_bearish_score, index=df_index, dtype=np.float32)
        return states

    def _deduce_suppressive_accumulation(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.0 · 全息证据链版】贝叶斯推演：“主力打压吸筹”剧本
        - 核心升级: 引入“筹码”和“过程”维度的证据，构建全息证据链。
        - 证据链:
          1. 核心博弈: 资本对抗中主力占优 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION)。
          2. 瞬时量价: 价格下跌，但有承接力量 (price_falling, dip_absorption_power_D)。
          3. 过程演化: 存在隐蔽吸筹的过程 (PROCESS_META_STEALTH_ACCUMULATION)。
          4. 筹码结构: 筹码正在逆势集中 (SCORE_CHIP_AXIOM_CONCENTRATION)。
        """
        print("    -- [剧本推演] 主力打压吸筹 (全息证据链)...")
        # --- 1. 收集所有相关证据 ---
        # 核心博弈证据
        capital_confrontation = self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0)
        # 瞬时量价证据
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_falling_evidence = price_change_bipolar.clip(upper=0).abs()
        efficiency_evidence = normalize_score(self._get_atomic_score('dip_absorption_power_D'), self.strategy.df_indicators.index, 55)
        # 过程演化证据 (新增)
        process_evidence = self._get_atomic_score('PROCESS_META_STEALTH_ACCUMULATION', 0.0).clip(lower=0)
        # 筹码结构证据 (新增)
        chip_evidence = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0)
        # --- 2. 计算似然度 P(证据 | 打压吸筹) ---
        evidence_scores = np.stack([
            capital_confrontation.values,
            price_falling_evidence.values,
            efficiency_evidence.values,
            process_evidence.values,
            chip_evidence.values
        ], axis=0)
        # 为不同证据赋予权重，筹码和过程证据权重更高
        evidence_weights = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 P(反转) ---
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 (最终信号分) ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_chasing_accumulation(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.0 · 全息证据链版】贝叶斯推演：“主力拉升抢筹”剧本
        - 核心升级: 引入“筹码”和“过程”维度的证据，确认拉升的真实意图。
        - 证据链:
          1. 核心博弈: 资本对抗中主力占优 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION)。
          2. 瞬时量价: 价格强劲上涨，且效率高 (price_rising, VPA_EFFICIENCY_D)。
          3. 过程演化: 主力行为紧迫，赢家信念坚定 (PROCESS_META_MAIN_FORCE_URGENCY, PROCESS_META_WINNER_CONVICTION)。
          4. 筹码结构: 筹码持续集中 (SCORE_CHIP_AXIOM_CONCENTRATION)。
        """
        print("    -- [剧本推演] 主力拉升抢筹 (全息证据链)...")
        # --- 1. 收集所有相关证据 ---
        # 核心博弈证据
        capital_confrontation = self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0)
        # 瞬时量价证据
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score('pct_change_D'), self.strategy.df_indicators.index, 21)
        price_rising_evidence = price_change_bipolar.clip(lower=0)
        efficiency_evidence = normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55)
        # 过程演化证据 (新增)
        urgency_evidence = self._get_atomic_score('PROCESS_META_MAIN_FORCE_URGENCY', 0.0).clip(lower=0)
        conviction_evidence = self._get_atomic_score('PROCESS_META_WINNER_CONVICTION', 0.0).clip(lower=0)
        process_evidence = (urgency_evidence * conviction_evidence).pow(0.5)
        # 筹码结构证据 (新增)
        chip_evidence = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(lower=0)
        # --- 2. 计算似然度 P(证据 | 拉升抢筹) ---
        evidence_scores = np.stack([
            capital_confrontation.values,
            price_rising_evidence.values,
            efficiency_evidence.values,
            process_evidence.values,
            chip_evidence.values
        ], axis=0)
        # 为不同证据赋予权重
        evidence_weights = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 P(趋势) ---
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 (最终信号分) ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_capitulation_reversal(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.0 · 全息证据链版】贝叶斯推演：“恐慌投降反转”剧本
        - 核心升级: 引入“过程”和“微观”证据，确认恐慌盘被真实吸收。
        - 证据链:
          1. 宏观压力: 市场存在向上反转的巨大压力 (FUSION_BIPOLAR_MARKET_PRESSURE)。
          2. 瞬时量价: 深V反弹，成交量激增 (dip_absorption_power_D, volume_spike)。
          3. 过程演化: 套牢盘正在投降 (PROCESS_META_LOSER_CAPITULATION)。
          4. 微观行为: 存在恐慌盘被吸收的微观迹象 (SCORE_MICRO_PANIC_ABSORPTION)。
        """
        print("    -- [剧本推演] 恐慌投降反转 (全息证据链)...")
        # --- 1. 收集所有相关证据 ---
        # 宏观压力证据
        upward_pressure = self._get_fused_score('FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(lower=0)
        # 瞬时量价证据
        price_rebound_evidence = normalize_score(self._get_atomic_score('dip_absorption_power_D'), self.strategy.df_indicators.index, 55)
        vol_ma55 = self._get_atomic_score('VOL_MA_55_D', 1.0)
        volume_spike = (self._get_atomic_score('volume_D') / vol_ma55.replace(0, 1.0)).fillna(1.0)
        volume_evidence = normalize_score(volume_spike, self.strategy.df_indicators.index, 55)
        # 过程演化证据 (新增)
        process_evidence = self._get_atomic_score('PROCESS_META_LOSER_CAPITULATION', 0.0).clip(lower=0)
        # 微观行为证据 (新增)
        micro_evidence = self._get_atomic_score('SCORE_MICRO_PANIC_ABSORPTION', 0.0).clip(lower=0)
        # --- 2. 计算似然度 P(证据 | 恐慌反转) ---
        evidence_scores = np.stack([
            upward_pressure.values,
            price_rebound_evidence.values,
            volume_evidence.values,
            process_evidence.values,
            micro_evidence.values
        ], axis=0)
        # 为不同证据赋予权重
        evidence_weights = np.array([0.1, 0.2, 0.2, 0.3, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 P(反转) ---
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 (最终信号分) ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL': posterior_prob.astype(np.float32)}

    def _deduce_distribution_at_high(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.0 · 全息证据链版】贝叶斯推演：“高位派发”风险剧本
        - 核心升级: 引入“筹码”和“过程”证据，确认派发的真实性。
        - 证据链:
          1. 核心博弈: 资本对抗中主力处于劣势 (FUSION_BIPOLAR_CAPITAL_CONFRONTATION)。
          2. 瞬时量价: 放量滞涨或高位长上影 (VPA_EFFICIENCY_D, upper_shadow_ratio_D)。
          3. 过程演化: 盈利盘与资金流向发生背离 (PROCESS_META_PROFIT_VS_FLOW)。
          4. 筹码结构: 筹码开始发散 (1 - SCORE_CHIP_AXIOM_CONCENTRATION)。
        """
        print("    -- [剧本推演] 高位派发风险 (全息证据链)...")
        # --- 1. 收集所有相关证据 ---
        # 核心博弈证据
        capital_confrontation = self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs()
        # 瞬时量价证据
        price_at_high = normalize_score(self._get_atomic_score('BIAS_55_D'), self.strategy.df_indicators.index, 55)
        efficiency_evidence = (1 - normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55)).clip(0, 1)
        # 过程演化证据 (新增)
        process_evidence = self._get_atomic_score('PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(lower=0)
        # 筹码结构证据 (新增): 筹码集中度下降，即发散
        chip_evidence = (1 - self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.5)).clip(0, 1)
        # --- 2. 计算似然度 P(证据 | 高位派发) ---
        evidence_scores = np.stack([
            capital_confrontation.values,
            price_at_high.values,
            efficiency_evidence.values,
            process_evidence.values,
            chip_evidence.values
        ], axis=0)
        # 为不同证据赋予权重
        evidence_weights = np.array([0.2, 0.1, 0.2, 0.2, 0.3])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 P(趋势) ---
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 (最终风险分) ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH': posterior_prob.astype(np.float32)}

    def _deduce_leading_dragon_awakening(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """【V1.0 · 新增】贝叶斯推演：“龙头苏醒”剧本"""
        print("    -- [剧本推演] 龙头苏醒...")
        # --- 1. 收集证据 ---
        capital_confrontation = self._get_fused_score('FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0)
        breakout_quality = self._get_atomic_score('breakout_quality_score_D', 0.0)
        sector_sync = self._get_atomic_score('PROCESS_META_STOCK_SECTOR_SYNC', 0.0).clip(lower=0)
        relative_strength = normalize_score(self._get_atomic_score('relative_strength_vs_index_D', 0.0), self.strategy.df_indicators.index, 55)
        # --- 2. 计算似然度 ---
        evidence_scores = np.stack([capital_confrontation.values, breakout_quality.values, sector_sync.values, relative_strength.values], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 ---
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING': posterior_prob.astype(np.float32)}

    def _deduce_trend_exhaustion_risk(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.1 · 信号净化版】贝叶斯推演：“趋势衰竭”风险剧本
        - 核心修改: 重命名方法和输出信号，使用更专业的术语。
        - 证据链:
          1. 价势背离: 价格与动量加速度发生背离 (PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE)。
          2. 信念动摇: 赢家信念开始动摇 (PROCESS_META_WINNER_CONVICTION_DECAY)。
          3. 效率衰竭: 出现放量滞涨 (VPA_EFFICIENCY_D)。
          4. 筹码拐点: 筹码集中度斜率转负，开始发散 (SLOPE_5_long_term_concentration_90pct_D)。
        """
        # [代码修改开始]
        print("    -- [剧本推演] 趋势衰竭风险 (拐点精确打击)...")
        # --- 1. 收集证据 ---
        # 证据1: 价格与动量加速度发生背离
        divergence = self._get_atomic_score('PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0).clip(lower=0)
        # 证据2: 赢家信念开始动摇
        winner_conviction_decay = self._get_atomic_score('PROCESS_META_WINNER_CONVICTION_DECAY', 0.0)
        # 证据3: 出现放量滞涨
        stagnation = (1 - normalize_score(self._get_atomic_score('VPA_EFFICIENCY_D'), self.strategy.df_indicators.index, 55)).clip(0, 1)
        # 证据4: 筹码集中度出现拐点 (逻辑升级)
        concentration_slope = self._get_atomic_score('SLOPE_5_long_term_concentration_90pct_D', 0.0)
        chip_distribution_evidence = normalize_score(concentration_slope.clip(upper=0).abs(), self.strategy.df_indicators.index, 55)
        # --- 2. 计算似然度 ---
        evidence_scores = np.stack([divergence.values, winner_conviction_decay.values, stagnation.values, chip_distribution_evidence.values], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.1, 0.3]) # 提升拐点证据的权重
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 ---
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_RISK_TREND_EXHAUSTION': posterior_prob.astype(np.float32)}
        # [代码修改结束]

    def _deduce_sector_rotation_vanguard(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """【V1.1 · 斐波那契周期版】贝叶斯推演：“板块轮动先锋”剧本"""
        print("    -- [剧本推演] 板块轮动先锋...")
        # --- 1. 收集证据 ---
        # 证据1: 板块整体资金流入
        sector_flow = self._get_atomic_score('FUSION_SECTOR_NET_FLOW_SCORE', 0.0).clip(lower=0)
        # 证据2: 股票处于长期底部 (周期从120修正为144)
        price_position = 1 - normalize_score(self._get_atomic_score('BIAS_144_D', 0.0), self.strategy.df_indicators.index, 144)
        # 证据3: 筹码结构干净
        chip_cleanliness = self._get_atomic_score('SCORE_CHIP_CLEANLINESS', 0.0)
        # 证据4: 前期热点板块资金流出
        hot_sector_cooling = self._get_atomic_score('PROCESS_META_HOT_SECTOR_COOLING', 0.0)
        # --- 2. 计算似然度 ---
        evidence_scores = np.stack([sector_flow.values, price_position.values, chip_cleanliness.values, hot_sector_cooling.values], axis=0)
        evidence_weights = np.array([0.4, 0.2, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 ---
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_SECTOR_ROTATION_VANGUARD': posterior_prob.astype(np.float32)}

    def _deduce_energy_compression_breakout(self, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.2 · 元特征增强版】贝叶斯推演：“能量压缩爆发”剧本
        - 核心升级: 引入样本熵(SAMPLE_ENTROPY)作为核心证据。真正的能量压缩，其物理本质是市场进入了高度有序、可预测的状态（低熵）。
        """
        print("    -- [剧本推演] 能量压缩爆发...")
        # --- 1. 收集证据 ---
        # 证据1: 波动率被压缩到极致
        bbw = self._get_atomic_score('BBW_21_2.0_D', 0.1)
        volatility_compression = 1 - normalize_score(bbw, self.strategy.df_indicators.index, 144)
        # 证据2: 成交量极度萎缩
        volume_atrophy = 1 - normalize_score(self._get_atomic_score('volume_D'), self.strategy.df_indicators.index, 144)
        # 证据3 (新增): 市场进入低熵有序状态
        entropy_col = next((col for col in self.strategy.df_indicators.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            entropy = self._get_atomic_score(entropy_col, 1.0)
            # 熵越低，分数越高
            orderliness_score = 1 - normalize_score(entropy, self.strategy.df_indicators.index, 144)
        else:
            orderliness_score = pd.Series(0.5, index=self.strategy.df_indicators.index)
        # --- 2. 计算似然度 ---
        evidence_scores = np.stack([volatility_compression.values, volume_atrophy.values, orderliness_score.values], axis=0)
        evidence_weights = np.array([0.3, 0.3, 0.4]) # 提升有序性证据的权重
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=self.strategy.df_indicators.index)
        # --- 3. 获取先验概率 ---
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        # --- 4. 计算后验概率 ---
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION': posterior_prob.astype(np.float32)}

    def _perform_dynamic_forging(self, df: pd.DataFrame, raw_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.0 · 时空锻造版】动态锻造工坊
        - 核心升级: 在调用元分析引擎时，传入相关的“协同信号”和“风险信号”，
                      从而激活“空间维度”分析，实现“时空一体”的动态锻造。
        """
        forged_scores = {}
        p_conf = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        playbook_synergies = get_param_value(p_conf.get('playbook_synergies'), {})

        for name, snapshot_score in raw_scores.items():
            # 获取当前剧本的协同与风险信号配置
            synergy_config = playbook_synergies.get(name, {})
            synergistic_signals = synergy_config.get('synergistic', [])
            risk_signals = synergy_config.get('risk', [])

            # 调用“时空一体”的元分析引擎
            forged_score = self._perform_cognitive_relational_meta_analysis(
                df, 
                snapshot_score,
                synergistic_signals=synergistic_signals,
                risk_signals=risk_signals
            )
            
            forged_name = name.replace('COGNITIVE_', 'COGNITIVE_FORGED_')
            forged_scores[forged_name] = forged_score
            
        return forged_scores

    def _perform_cognitive_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, synergistic_signals: List[str] = None, risk_signals: List[str] = None) -> pd.Series:
        """
        【V3.0 · 时空一体锻造版】认知层专用的关系元分析核心引擎
        - 核心升级: 在“多时间维度”分析的基础上，引入“相关性空间维度”分析。
                      通过评估目标信号与“协同信号”和“风险信号”的动态关系，
                      为最终的动态分数引入“协同力”和“抑制力”的考量。
        - 核心逻辑:
          1. 计算内在驱动力: 基于多时间维度的“状态-速度-加速度”分析。
          2. 计算外部协同力: 基于目标信号与协同信号的“滚动相关性斜率”。
          3. 计算风险抑制力: 基于目标信号与风险信号的“滚动相关性斜率”。
          4. 最终分数 = (内在驱动力 + 外部协同力) * (1 - 风险抑制力)
        - 收益: 构建了一个“时空一体”的分析框架，使信号评估不再孤立，而是充分考量
                其在多维信号空间中的相对位置和动态关系，极大提升了判断的准确性和前瞻性。
        """
        p_conf = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        
        # --- 1. 计算内在驱动力 (Internal Driving Force) ---
        # 这部分逻辑与V2.0版本保持一致
        periods = get_param_value(p_meta.get('periods'), [5, 13, 21, 55])
        period_weights = get_param_value(p_meta.get('period_weights'), {"5": 0.4, "13": 0.3, "21": 0.2, "55": 0.1})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
        
        internal_driving_force = pd.Series(0.0, index=df.index)
        total_period_weight = sum(w for w in period_weights.values() if w > 0)

        if total_period_weight > 0:
            for p in periods:
                period_weight = period_weights.get(str(p))
                if period_weight is None or period_weight <= 0:
                    continue
                
                bipolar_snapshot = (snapshot_score.fillna(0.5) * 2 - 1).clip(-1, 1)
                velocity = bipolar_snapshot.diff(p).fillna(0)
                velocity_score = normalize_to_bipolar(velocity, df.index, norm_window)
                acceleration = velocity.diff(1).fillna(0)
                acceleration_score = normalize_to_bipolar(acceleration, df.index, norm_window)
                
                bullish_force = (bipolar_snapshot.clip(lower=0) * w_state + velocity_score.clip(lower=0) * w_velocity + acceleration_score.clip(lower=0) * w_acceleration)
                bearish_force = (bipolar_snapshot.clip(upper=0).abs() * w_state + velocity_score.clip(upper=0).abs() * w_velocity + acceleration_score.clip(upper=0).abs() * w_acceleration)
                net_force_p = (bullish_force - bearish_force).clip(-1, 1)
                internal_driving_force += net_force_p * (period_weight / total_period_weight)

        # --- 2. 计算外部协同力 (External Synergy Force) ---
        external_synergy_force = pd.Series(0.0, index=df.index)
        if synergistic_signals:
            synergy_scores = []
            corr_window = get_param_value(p_meta.get('corr_window'), 21)
            slope_window = get_param_value(p_meta.get('corr_slope_window'), 5)
            
            for sig_name in synergistic_signals:
                synergy_signal = self._get_fused_score(sig_name, 0.5) # 协同信号通常是[0,1]的
                # 计算滚动相关性
                rolling_corr = snapshot_score.rolling(corr_window).corr(synergy_signal).fillna(0)
                # 计算相关性的斜率，判断关系是在增强还是减弱
                corr_slope = rolling_corr.diff(slope_window).fillna(0)
                # 归一化为协同力
                synergy_force = normalize_to_bipolar(corr_slope, df.index, norm_window)
                synergy_scores.append(synergy_force)
            
            if synergy_scores:
                # 对所有协同力取平均值
                external_synergy_force = pd.concat(synergy_scores, axis=1).mean(axis=1)

        # --- 3. 计算风险抑制力 (Risk Suppression Force) ---
        risk_suppression_force = pd.Series(0.0, index=df.index)
        if risk_signals:
            risk_scores = []
            corr_window = get_param_value(p_meta.get('corr_window'), 21)
            slope_window = get_param_value(p_meta.get('corr_slope_window'), 5)

            for sig_name in risk_signals:
                risk_signal = self._get_fused_score(sig_name, 0.0) # 风险信号通常是[0,1]的
                # 我们关心的是目标信号与风险信号的“正相关性”是否在增强
                # 正相关性越强，风险越大，抑制力越强
                rolling_corr = snapshot_score.rolling(corr_window).corr(risk_signal).fillna(0)
                corr_slope = rolling_corr.diff(slope_window).fillna(0)
                # 只取斜率为正的部分，代表风险协同在增强
                risk_force = normalize_score(corr_slope.clip(lower=0), df.index, norm_window)
                risk_scores.append(risk_force)
            
            if risk_scores:
                # 多个风险中，取最强的那个作为抑制力
                risk_suppression_force = pd.concat(risk_scores, axis=1).max(axis=1)

        # --- 4. 最终融合 ---
        # 协同力权重
        w_synergy = get_param_value(p_meta.get('synergy_force_weight'), 0.3)
        
        # 总驱动力 = 内在驱动力 + 外部协同力
        total_driving_force = (internal_driving_force * (1 - w_synergy) + external_synergy_force * w_synergy).clip(-1, 1)
        
        # 应用风险抑制
        final_net_force = total_driving_force * (1 - risk_suppression_force.clip(0, 1))
        
        # 转换回[0,1]区间
        final_dynamic_score = (final_net_force.clip(-1, 1) + 1) / 2.0
        
        return final_dynamic_score.astype(np.float32)
