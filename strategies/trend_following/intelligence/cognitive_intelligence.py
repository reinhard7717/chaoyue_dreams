# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, is_limit_up

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
        self.min_evidence_threshold = 1e-9 # 最小证据阈值，避免对数运算错误
        self.norm_window = 55 # 统一归一化窗口，可根据需要调整

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        【V27.1 · 返回值修复版】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        if column_name not in df.columns:
            print(f"    -> [CognitiveIntelligence情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index) # 移除了末尾的 .index
        return df[column_name]

    def _get_fused_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.4 · 返回值修复版】安全地从原子状态库中获取由融合层提供的态势分数。
        - 【V1.4 修复】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        if name in self.strategy.atomic_states:
            score = self.strategy.atomic_states[name]
            return score
        else:
            print(f"    -> [认知层警告] 融合态势信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=df.index) # 移除了末尾的 .index

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V2.3 · 返回值修复版】安全地从原子状态库或主数据帧中获取信号。
        - 【V2.3 修复】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in df.columns:
            return df[name]
        else:
            print(f"    -> [认知层警告] 原子信号 '{name}' 不存在，无法作为证据！返回默认值 {default}。")
            return pd.Series(default, index=df.index) # 移除了末尾的 .index

    def _get_playbook_score(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        安全地从 playbook_states 获取剧本信号分数。
        【V27.1 · 返回值修复版】修复了在信号不存在时错误返回索引而非Series的问题。
        """
        score = self.strategy.playbook_states.get(signal_name)
        if score is None:
            print(f"    -> [认知层警告] 剧本信号 '{signal_name}' 不存在，无法作为证据！返回默认值 {default_value}。")
            return pd.Series(default_value, index=df.index) # 移除了末尾的 .index
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                if isinstance(score, pd.Series):
                    print(f"    -> [DEBUG _get_playbook_score] 信号 '{signal_name}' 原始值: {score.loc[probe_date_for_loop]:.4f}")
                else:
                    print(f"    -> [DEBUG _get_playbook_score] 信号 '{signal_name}' 原始值: {score:.4f}")
        return score

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: List[str], method_name: str) -> bool:
        """
        【V1.1 · 全域视野版】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        - 核心升级: 增加了对 `self.strategy.playbook_states` 的检查，解决了因校验逻辑无法看到
                      已生成的剧本信号而导致的“情报失明症”，确保了剧本依赖链的正确校验。
        """
        missing_signals = []
        for signal in required_signals:
            # 增加对 playbook_states 的检查
            if signal not in df.columns and signal not in self.strategy.atomic_states and signal not in self.strategy.playbook_states:
                missing_signals.append(signal)
        if missing_signals:
            print(f"    -> [认知情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def synthesize_cognitive_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V28.0 · 指挥链重构版】总指挥
        - 核心重构: 彻底重构了剧本推演的执行顺序，将其改造为有序的、分批次的“依赖推演”流程。
                      确保了有依赖关系的剧本总是在其依赖项计算完成之后才被执行，从根本上
                      解决了因“指挥失序”导致的“循环依赖”崩溃问题。
        """
        print("启动【V28.0 · 指挥链重构版】认知情报分析...")
        playbook_states = {}
        priors = self._establish_prior_beliefs(df)
        self.strategy.atomic_states.update(priors)
        # --- 重构为分批次的依赖推演流程 ---
        # 第1批：机会剧本 (通常无内部依赖)
        print("    -> [认知层] 开始推演 第1批 (机会) 剧本...")
        playbook_states.update(self._deduce_suppressive_accumulation(df, priors))
        playbook_states.update(self._deduce_chasing_accumulation(df, priors))
        playbook_states.update(self._deduce_capitulation_reversal(df, priors))
        playbook_states.update(self._deduce_leading_dragon_awakening(df, priors))
        playbook_states.update(self._deduce_sector_rotation_vanguard(df, priors))
        playbook_states.update(self._deduce_energy_compression_breakout(df, priors))
        playbook_states.update(self._deduce_stealth_bottoming_divergence(df, priors))
        playbook_states.update(self._deduce_micro_absorption_divergence(df, priors))
        # 第2批：无内部依赖的基础风险剧本
        print("    -> [认知层] 开始推演 第2批 (基础风险) 剧本...")
        playbook_states.update(self._deduce_distribution_at_high(df, priors))
        playbook_states.update(self._deduce_retail_fomo_retreat_risk(df, priors))
        playbook_states.update(self._deduce_long_term_profit_distribution_risk(df, priors))
        playbook_states.update(self._deduce_market_uncertainty_risk(df, priors))
        playbook_states.update(self._deduce_liquidity_trap_risk(df, priors))
        playbook_states.update(self._deduce_t0_arbitrage_pressure_risk(df, priors))
        playbook_states.update(self._deduce_key_support_break_risk(df, priors))
        # 在调用依赖剧本之前，将当前已生成的剧本更新到 self.strategy.playbook_states 以供 _get_playbook_score 使用
        self.strategy.playbook_states.update(playbook_states)
        # 第3批：依赖第2批剧本的高级风险剧本
        print("    -> [认知层] 开始推演 第3批 (高级风险) 剧本...")
        playbook_states.update(self._deduce_trend_exhaustion_risk(df, priors))
        self.strategy.playbook_states.update(playbook_states) # 每次更新，确保依赖链条完整
        playbook_states.update(self._deduce_harvest_confirmation_risk(df, priors))
        self.strategy.playbook_states.update(playbook_states)
        playbook_states.update(self._deduce_bull_trap_distribution_risk(df, priors))
        self.strategy.playbook_states.update(playbook_states)
        playbook_states.update(self._deduce_high_level_structural_collapse_risk(df, priors))
        self.strategy.playbook_states.update(playbook_states)
        # 第4批：依赖第3批剧本的机会剧本 (如果有)
        print("    -> [认知层] 开始推演 第4批 (依赖型机会) 剧本...")
        playbook_states.update(self._deduce_divergence_reversal(df, priors))
        print("【V28.0 · 指挥链重构版】所有剧本推演完成。")
        return playbook_states

    def _deduce_suppressive_accumulation(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V9.0 · 诡道吸筹动态协同版】贝叶斯推演：“主力打压吸筹”剧本
        - 核心升级:
            1. 动态 `power_factor`：根据 `volatility_instability` 动态调整非线性放大因子。
            2. 增强“压制”证据的情境放大：在熊市/弱势趋势中，额外放大压制相关证据的权重。
            3. 引入“协同吸筹”证据：通过几何平均融合多个核心吸筹信号，捕捉协同效应。
            4. 调整证据权重：根据新增信号的直接性和重要性，重新分配权重。
        """
        print("    -- [剧本推演] 主力打压吸筹 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 'pct_change_D', 'dip_absorption_power_D',
            'PROCESS_META_STEALTH_ACCUMULATION', 'SCORE_CHIP_STRATEGIC_POSTURE', 'FUSION_BIPOLAR_MARKET_CONTRADICTION',
            'SCORE_BEHAVIOR_DECEPTION_INDEX', 'SCORE_BEHAVIOR_VOLUME_ATROPHY',
            'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 'PROCESS_META_POWER_TRANSFER',
            # 情境和时间序列信号
            'FUSION_BIPOLAR_MARKET_REGIME', 'FUSION_BIPOLAR_TREND_QUALITY', 'market_sentiment_score_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'SLOPE_5_dip_absorption_power_D',
            # V6.0 引入的证据信号
            'suppressive_accumulation_intensity_D', 'covert_accumulation_signal_D',
            'SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION', 'PROCESS_META_DECEPTIVE_ACCUMULATION',
            'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 'PROCESS_META_LOSER_CAPITULATION',
            'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT',
            'SCORE_FUND_FLOW_BULLISH_DIVERGENCE', 'SCORE_CHIP_OPP_ABSORPTION_ECHO',
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER',
            # V7.0 新增证据信号
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 'SLOPE_5_pct_change_D',
            'SCORE_MICRO_STRATEGY_STEALTH_OPS', 'SCORE_MICRO_STRATEGY_COST_CONTROL',
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 'SCORE_FF_AXIOM_CONVICTION',
            'SLOPE_5_suppressive_accumulation_intensity_D', 'SLOPE_5_covert_accumulation_signal_D',
            'SCORE_BEHAVIOR_BULLISH_DIVERGENCE',
            # V8.0 引入的证据信号
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'SCORE_CHIP_TACTICAL_EXCHANGE', 'SCORE_FF_AXIOM_FLOW_MOMENTUM',
            'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT', 'SCORE_STRUCT_AXIOM_TENSION'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_suppressive_accumulation"):
            print("    -> [探针] 信号校验失败，返回默认值。")
            return {'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION': pd.Series(0.0, index=df.index)}

        # 获取调试参数
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date_for_loop = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df.index.tz) if df.index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [探针] 主力打压吸筹 @ {probe_date_for_loop.date()}:")

        # 定义基础权重 (在方法开始处定义，确保作用域)
        base_weights_dict = {
            'capital_confrontation': 0.02, # 资本对抗 (看涨部分)
            'price_falling': 0.02, # 价格下跌证据 (基础下跌)
            'deception': 0.04, # 行为欺骗指数 (正向)
            'volume_atrophy': 0.01, # 成交量萎缩
            'efficiency': 0.02, # 承接效率
            'stealth_accum': 0.04, # 隐秘吸筹过程 (PROCESS_META_STEALTH_ACCUMULATION)
            'split_order_accum': 0.02, # 拆单吸筹强度
            'power_transfer': 0.01, # 权力转移 (正向)
            'chip_strategic_posture': 0.03, # 筹码战略态势 (看涨部分)
            'market_contradiction_bullish': 0.01, # 市场矛盾 (看涨部分)
            # V6.0 引入的证据权重
            'suppressive_accum_intensity': 0.05, # 直接的压制吸筹强度 (原始指标)
            'covert_accum_signal': 0.05, # 直接的隐蔽吸筹信号 (原始指标)
            'shakeout_confirmation': 0.04, # 洗盘确认 (行为层)
            'deceptive_accum': 0.04, # 诡道吸筹 (过程层)
            'panic_washout_accum': 0.04, # 恐慌洗盘吸筹 (过程层)
            'loser_capitulation': 0.03, # 输家投降仪式 (过程层)
            'absorption_strength': 0.03, # 承接强度 (行为层)
            'offensive_absorption_intent': 0.03, # 进攻性承接意图 (行为层)
            'fund_flow_bullish_divergence': 0.02, # 资金流看涨背离
            'chip_opp_absorption_echo': 0.04, # 筹码吸筹回声
            'distribution_intent_negative': 0.03, # 派发意图 (反向)
            'chip_risk_distribution_whisper_negative': 0.03, # 派发诡影 (反向)
            # V7.0 新增证据权重
            'price_downward_momentum': 0.05, # 价格下跌动能 (行为层)
            'pct_change_slope': 0.02, # 价格变化斜率 (动态打压)
            'micro_stealth_ops': 0.06, # 微观隐秘行动 (微观层)
            'micro_cost_control': 0.05, # 微观成本控制 (微观层)
            'chip_holder_sentiment_bullish': 0.04, # 筹码持仓信念韧性 (看涨部分)
            'ff_conviction_bullish': 0.04, # 资金流信念韧性 (看涨部分)
            'suppressive_accum_intensity_slope': 0.03, # 压制吸筹强度斜率 (动态吸筹)
            'covert_accum_signal_slope': 0.03, # 隐蔽吸筹信号斜率 (动态吸筹)
            'behavior_bullish_divergence': 0.03, # 行为看涨背离
            # V8.0 引入的证据权重
            'downward_resistance': 0.03, # 下跌抵抗 (行为层)
            'intraday_vwap_battlefield_negative': 0.02, # 日内VWAP攻防 (负向)
            'chip_tactical_exchange': 0.05, # 战术换手博弈 (筹码层)
            'ff_flow_momentum_bullish': 0.04, # 资金流纯度与动能 (看涨部分)
            'ff_accum_inflection_intent': 0.04, # 资金流吸筹拐点意图 (过程层)
            'structural_tension': 0.02, # 结构张力 (结构层)
            # V9.0 新增证据权重
            'synergistic_accumulation': 0.08 # 协同吸筹 (高权重)
        }
        evidence_names = list(base_weights_dict.keys()) # 确保在整个方法中都可访问

        # --- 1. 获取情境调制信号 ---
        market_regime_score = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_REGIME', 0.0) # [-1, 1]
        trend_quality_score = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0) # [-1, 1]
        sentiment_score = self._get_atomic_score(df, 'market_sentiment_score_D', 0.5) # [0, 1] (理论范围，实际可能超出)
        volatility_instability = self._get_atomic_score(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.5) # [0, 1]

        # 修正 sentiment_score 的范围，确保在 [0, 1] 内进行调制计算
        sentiment_score_clipped = sentiment_score.clip(0, 1)

        if probe_date_for_loop is not None and probe_date_for_loop in df.index:
            print(f"       - 情境信号: 市场状态: {market_regime_score.loc[probe_date_for_loop]:.4f}, 趋势质量: {trend_quality_score.loc[probe_date_for_loop]:.4f}, 情绪(原始): {sentiment_score.loc[probe_date_for_loop]:.4f}, 情绪(裁剪后): {sentiment_score_clipped.loc[probe_date_for_loop]:.4f}, 波动不稳: {volatility_instability.loc[probe_date_for_loop]:.4f}")

        # --- 2. 获取原始证据信号及其时间序列动态 ---
        # 2.1 资本对抗 (看涨部分) - 主力买入意愿
        raw_capital_confrontation = self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        capital_confrontation_evidence = self._forge_dynamic_evidence(df, raw_capital_confrontation.clip(lower=0))

        # 2.2 价格下跌证据 - 打压的表象
        raw_price_change = self._get_atomic_score(df, 'pct_change_D')
        price_change_bipolar = normalize_to_bipolar(raw_price_change, df.index, 21)
        price_falling_evidence = self._forge_dynamic_evidence(df, price_change_bipolar.clip(upper=0).abs())

        # 2.3 行为欺骗指数 (正向) - 主力打压的意图 (无斜率，使用静态证据)
        raw_deception_index = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0)
        deception_evidence = self._forge_dynamic_evidence(df, raw_deception_index.clip(lower=0))
        deception_evidence_dynamic = deception_evidence # 静态证据，无动态调制

        # 2.4 成交量萎缩 - 打压吸筹的背景
        raw_volume_atrophy = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        volume_atrophy_evidence = self._forge_dynamic_evidence(df, raw_volume_atrophy)

        # 2.5 承接效率 - 吸收抛压的能力 (有斜率，进行动态调制)
        raw_efficiency = self._get_atomic_score(df, 'dip_absorption_power_D', 0.0)
        slope_efficiency = self._get_atomic_score(df, 'SLOPE_5_dip_absorption_power_D', 0.0)
        efficiency_evidence = self._forge_dynamic_evidence(df, raw_efficiency)

        # 2.6 隐秘吸筹过程 - 直接的吸筹信号 (无斜率，使用静态证据)
        raw_process_stealth_accum = self._get_atomic_score(df, 'PROCESS_META_STEALTH_ACCUMULATION', 0.0)
        process_stealth_accum_evidence = self._forge_dynamic_evidence(df, raw_process_stealth_accum.clip(lower=0))
        process_stealth_accum_evidence_dynamic = process_stealth_accum_evidence # 静态证据，无动态调制

        # 2.7 拆单吸筹强度 - 微观订单流的吸筹证据
        raw_split_order_accum = self._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 0.0)
        split_order_accum_evidence = self._forge_dynamic_evidence(df, raw_split_order_accum)

        # 2.8 权力转移 (正向) - 筹码向主力转移
        raw_power_transfer = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        power_transfer_evidence = self._forge_dynamic_evidence(df, raw_power_transfer.clip(lower=0))

        # 2.9 筹码战略态势 (看涨部分) - 筹码结构优化
        raw_chip_strategic_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        chip_evidence = self._forge_dynamic_evidence(df, raw_chip_strategic_posture.clip(lower=0))

        # 2.10 市场矛盾 (看涨部分) - 底部背离的宏观信号
        raw_market_contradiction = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0)
        market_contradiction_bullish = self._forge_dynamic_evidence(df, raw_market_contradiction.clip(lower=0))

        # V6.0 引入的证据获取
        raw_suppressive_accum_intensity = self._get_atomic_score(df, 'suppressive_accumulation_intensity_D', 0.0)
        suppressive_accum_intensity_evidence = self._forge_dynamic_evidence(df, raw_suppressive_accum_intensity)

        raw_covert_accum_signal = self._get_atomic_score(df, 'covert_accumulation_signal_D', 0.0)
        covert_accum_signal_evidence = self._forge_dynamic_evidence(df, raw_covert_accum_signal)

        raw_shakeout_confirmation = self._get_atomic_score(df, 'SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION', 0.0)
        shakeout_confirmation_evidence = self._forge_dynamic_evidence(df, raw_shakeout_confirmation)

        raw_deceptive_accum = self._get_atomic_score(df, 'PROCESS_META_DECEPTIVE_ACCUMULATION', 0.0)
        deceptive_accum_evidence = self._forge_dynamic_evidence(df, raw_deceptive_accum)

        raw_panic_washout_accum = self._get_atomic_score(df, 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 0.0)
        panic_washout_accum_evidence = self._forge_dynamic_evidence(df, raw_panic_washout_accum)

        raw_loser_capitulation = self._get_atomic_score(df, 'PROCESS_META_LOSER_CAPITULATION', 0.0)
        loser_capitulation_evidence = self._forge_dynamic_evidence(df, raw_loser_capitulation)

        raw_absorption_strength = self._get_atomic_score(df, 'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 0.0)
        absorption_strength_evidence = self._forge_dynamic_evidence(df, raw_absorption_strength)

        raw_offensive_absorption_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 0.0)
        offensive_absorption_intent_evidence = self._forge_dynamic_evidence(df, raw_offensive_absorption_intent)

        raw_fund_flow_bullish_divergence = self._get_atomic_score(df, 'SCORE_FUND_FLOW_BULLISH_DIVERGENCE', 0.0)
        fund_flow_bullish_divergence_evidence = self._forge_dynamic_evidence(df, raw_fund_flow_bullish_divergence)

        raw_chip_opp_absorption_echo = self._get_atomic_score(df, 'SCORE_CHIP_OPP_ABSORPTION_ECHO', 0.0)
        chip_opp_absorption_echo_evidence = self._forge_dynamic_evidence(df, raw_chip_opp_absorption_echo)

        # 反向证据：派发意图越低，越支持吸筹剧本
        raw_distribution_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0)
        distribution_intent_negative_evidence = self._forge_dynamic_evidence(df, (1 - raw_distribution_intent).clip(0, 1))

        raw_chip_risk_distribution_whisper = self._get_atomic_score(df, 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 0.0)
        chip_risk_distribution_whisper_negative_evidence = self._forge_dynamic_evidence(df, (1 - raw_chip_risk_distribution_whisper).clip(0, 1))

        # V7.0 新增证据获取
        raw_price_downward_momentum = self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0)
        price_downward_momentum_evidence = self._forge_dynamic_evidence(df, raw_price_downward_momentum)

        raw_pct_change_slope = self._get_atomic_score(df, 'SLOPE_5_pct_change_D', 0.0)
        pct_change_slope_evidence = self._forge_dynamic_evidence(df, normalize_score(-raw_pct_change_slope, df.index, 21, ascending=True))

        raw_micro_stealth_ops = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        micro_stealth_ops_evidence = self._forge_dynamic_evidence(df, raw_micro_stealth_ops)

        raw_micro_cost_control = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_COST_CONTROL', 0.0)
        micro_cost_control_evidence = self._forge_dynamic_evidence(df, raw_micro_cost_control)

        raw_chip_holder_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        chip_holder_sentiment_bullish_evidence = self._forge_dynamic_evidence(df, raw_chip_holder_sentiment.clip(lower=0))

        raw_ff_conviction = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONVICTION', 0.0)
        ff_conviction_bullish_evidence = self._forge_dynamic_evidence(df, raw_ff_conviction.clip(lower=0))

        slope_suppressive_accum_intensity = self._get_atomic_score(df, 'SLOPE_5_suppressive_accumulation_intensity_D', 0.0)
        suppressive_accum_intensity_slope_evidence = self._forge_dynamic_evidence(df, normalize_to_bipolar(slope_suppressive_accum_intensity, df.index, 21).clip(lower=0))

        slope_covert_accum_signal = self._get_atomic_score(df, 'SLOPE_5_covert_accumulation_signal_D', 0.0)
        covert_accum_signal_slope_evidence = self._forge_dynamic_evidence(df, normalize_to_bipolar(slope_covert_accum_signal, df.index, 21).clip(lower=0))

        raw_behavior_bullish_divergence = self._get_atomic_score(df, 'SCORE_BEHAVIOR_BULLISH_DIVERGENCE', 0.0)
        behavior_bullish_divergence_evidence = self._forge_dynamic_evidence(df, raw_behavior_bullish_divergence)

        # V8.0 引入的证据获取
        raw_downward_resistance = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE', 0.0)
        downward_resistance_evidence = self._forge_dynamic_evidence(df, raw_downward_resistance)

        raw_intraday_vwap_battlefield = self._get_atomic_score(df, 'SCORE_INTRADAY_VWAP_BATTLEFIELD', 0.0)
        # VWAP攻防是双极性，负值表示卖压，我们希望负值越大越好（作为打压证据）
        intraday_vwap_battlefield_negative_evidence = self._forge_dynamic_evidence(df, raw_intraday_vwap_battlefield.clip(upper=0).abs())

        raw_chip_tactical_exchange = self._get_atomic_score(df, 'SCORE_CHIP_TACTICAL_EXCHANGE', 0.0)
        chip_tactical_exchange_evidence = self._forge_dynamic_evidence(df, raw_chip_tactical_exchange.clip(lower=0))

        raw_ff_flow_momentum = self._get_atomic_score(df, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0)
        ff_flow_momentum_bullish_evidence = self._forge_dynamic_evidence(df, raw_ff_flow_momentum.clip(lower=0))

        raw_ff_accum_inflection_intent = self._get_atomic_score(df, 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT', 0.0)
        ff_accum_inflection_intent_evidence = self._forge_dynamic_evidence(df, raw_ff_accum_inflection_intent)

        raw_structural_tension = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0)
        structural_tension_evidence = self._forge_dynamic_evidence(df, raw_structural_tension)

        # V9.0 新增协同吸筹证据
        # 确保所有组件都是 Series 且索引一致，然后进行几何平均
        synergistic_accumulation_components = [
            covert_accum_signal_evidence,
            chip_tactical_exchange_evidence,
            ff_accum_inflection_intent_evidence
        ]
        # 过滤掉可能为0的项，避免log(0)错误，并确保所有项都是正数
        synergistic_accumulation_safe_components = [np.maximum(s, 1e-9) for s in synergistic_accumulation_components]
        # 几何平均
        synergistic_accumulation_evidence = self._forge_dynamic_evidence(df, pd.Series(
            np.exp(np.mean([np.log(s.values) for s in synergistic_accumulation_safe_components], axis=0)),
            index=df.index
        ))


        # --- 3. 时间序列动态调制 (Time-series Dynamics Modulation) ---
        slope_impact_factor = 0.5 # 斜率对证据的调制强度
        norm_window_slope = 21 # 斜率归一化窗口

        # 3.1 承接效率动态调制
        norm_slope_efficiency = normalize_to_bipolar(slope_efficiency, df.index, norm_window_slope)
        efficiency_evidence_dynamic = efficiency_evidence * (1 + norm_slope_efficiency.clip(lower=0) * slope_impact_factor)
        efficiency_evidence_dynamic = efficiency_evidence_dynamic.clip(0, 1)

        if probe_date_for_loop is not None and probe_date_for_loop in df.index:
            print(f"       - 行为欺骗(原始): {raw_deception_index.loc[probe_date_for_loop]:.4f}, 动态证据: {deception_evidence_dynamic.loc[probe_date_for_loop]:.4f} (无斜率调制)")
            print(f"       - 隐秘吸筹(原始): {raw_process_stealth_accum.loc[probe_date_for_loop]:.4f}, 动态证据: {process_stealth_accum_evidence_dynamic.loc[probe_date_for_loop]:.4f} (无斜率调制)")
            print(f"       - 承接效率(原始): {raw_efficiency.loc[probe_date_for_loop]:.4f}, 斜率: {slope_efficiency.loc[probe_date_for_loop]:.4f}, 动态证据: {efficiency_evidence_dynamic.loc[probe_date_for_loop]:.4f}")
            print(f"       - 压制吸筹强度(原始): {raw_suppressive_accum_intensity.loc[probe_date_for_loop]:.4f}, 斜率: {slope_suppressive_accum_intensity.loc[probe_date_for_loop]:.4f}, 证据: {suppressive_accum_intensity_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 压制吸筹强度斜率证据: {suppressive_accum_intensity_slope_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 隐蔽吸筹信号(原始): {raw_covert_accum_signal.loc[probe_date_for_loop]:.4f}, 斜率: {slope_covert_accum_signal.loc[probe_date_for_loop]:.4f}, 证据: {covert_accum_signal_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 隐蔽吸筹信号斜率证据: {covert_accum_signal_slope_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 洗盘确认(原始): {raw_shakeout_confirmation.loc[probe_date_for_loop]:.4f}, 证据: {shakeout_confirmation_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 派发意图(原始): {raw_distribution_intent.loc[probe_date_for_loop]:.4f}, 反向证据: {distribution_intent_negative_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 价格下跌动能(原始): {raw_price_downward_momentum.loc[probe_date_for_loop]:.4f}, 证据: {price_downward_momentum_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 价格变化斜率(原始): {raw_pct_change_slope.loc[probe_date_for_loop]:.4f}, 证据: {pct_change_slope_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 下跌抵抗(原始): {raw_downward_resistance.loc[probe_date_for_loop]:.4f}, 证据: {downward_resistance_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 日内VWAP攻防(原始): {raw_intraday_vwap_battlefield.loc[probe_date_for_loop]:.4f}, 负向证据: {intraday_vwap_battlefield_negative_evidence.loc[probe_date_for_loop]:.4f}")
            print(f"       - 协同吸筹证据: {synergistic_accumulation_evidence.loc[probe_date_for_loop]:.4f}")


        # --- 4. 情境自适应权重 (Context-adaptive weights) ---
        # 初始化 adaptive_weights_per_date 为一个字典，每个值都是一个 Series，索引与 df.index 相同
        adaptive_weights_per_date = {
            name: pd.Series(base_weight, index=df.index, dtype=np.float32)
            for name, base_weight in base_weights_dict.items()
        }

        # 调制因子：将情境信号映射到权重调整系数
        # 市场状态：越熊市/震荡，打压吸筹越合理，增加相关证据权重
        market_regime_mod = (1 - market_regime_score.clip(lower=0)) * 0.2 # 熊市/震荡时为正，牛市时为0
        # 趋势质量：趋势越差，打压吸筹越合理
        trend_quality_mod = (1 - trend_quality_score.clip(lower=0)) * 0.15
        # 市场情绪：情绪越低迷，吸筹越隐蔽，增加吸筹证据权重
        sentiment_mod = (1 - sentiment_score_clipped) * 0.1 # 使用裁剪后的情绪分数
        # 波动不稳定性：波动越大，打压吸筹可能越剧烈，或吸筹难度越大
        volatility_mod = volatility_instability * 0.05 # 波动越大，略微增加打压证据权重

        # V9.0 增强“压制”证据的情境放大
        # 当市场状态和趋势质量偏负时，压制证据的权重应更高
        suppression_context_amplification = (market_regime_score.clip(upper=0).abs() + trend_quality_score.clip(upper=0).abs()) * 0.1
        
        # 应用调制：直接将调制 Series 加到对应的权重 Series 上
        # 打压证据权重增加 (price_falling, deception, volume_atrophy, suppressive_accum_intensity, shakeout_confirmation, deceptive_accum, panic_washout_accum, price_downward_momentum, pct_change_slope, downward_resistance, intraday_vwap_battlefield_negative)
        adaptive_weights_per_date['price_falling'] += market_regime_mod + trend_quality_mod + volatility_mod + suppression_context_amplification
        adaptive_weights_per_date['deception'] += market_regime_mod + trend_quality_mod + sentiment_mod + suppression_context_amplification
        adaptive_weights_per_date['volume_atrophy'] += market_regime_mod + trend_quality_mod + sentiment_mod + suppression_context_amplification
        adaptive_weights_per_date['suppressive_accum_intensity'] += market_regime_mod + trend_quality_mod + volatility_mod + suppression_context_amplification
        adaptive_weights_per_date['suppressive_accum_intensity_slope'] += market_regime_mod + trend_quality_mod + volatility_mod + suppression_context_amplification
        adaptive_weights_per_date['shakeout_confirmation'] += market_regime_mod + trend_quality_mod + volatility_mod + suppression_context_amplification
        adaptive_weights_per_date['deceptive_accum'] += market_regime_mod + trend_quality_mod + sentiment_mod + suppression_context_amplification
        adaptive_weights_per_date['panic_washout_accum'] += market_regime_mod + trend_quality_mod + sentiment_mod + suppression_context_amplification
        adaptive_weights_per_date['price_downward_momentum'] += market_regime_mod + trend_quality_mod + volatility_mod + suppression_context_amplification
        adaptive_weights_per_date['pct_change_slope'] += market_regime_mod + trend_quality_mod + volatility_mod + suppression_context_amplification
        adaptive_weights_per_date['downward_resistance'] += market_regime_mod + trend_quality_mod + volatility_mod + suppression_context_amplification
        adaptive_weights_per_date['intraday_vwap_battlefield_negative'] += market_regime_mod + trend_quality_mod + volatility_mod + suppression_context_amplification

        # 吸筹证据权重增加 (capital_confrontation, efficiency, stealth_accum, split_order_accum, power_transfer, chip_strategic_posture, loser_capitulation, absorption_strength, offensive_absorption_intent, chip_opp_absorption_echo, covert_accum_signal, fund_flow_bullish_divergence, micro_stealth_ops, micro_cost_control, chip_holder_sentiment_bullish, ff_conviction_bullish, chip_tactical_exchange, ff_flow_momentum_bullish, ff_accum_inflection_intent, synergistic_accumulation)
        adaptive_weights_per_date['capital_confrontation'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['efficiency'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['stealth_accum'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['split_order_accum'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['power_transfer'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['chip_strategic_posture'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['loser_capitulation'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['absorption_strength'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['offensive_absorption_intent'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['chip_opp_absorption_echo'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['covert_accum_signal'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['covert_accum_signal_slope'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['fund_flow_bullish_divergence'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['micro_stealth_ops'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['micro_cost_control'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['chip_holder_sentiment_bullish'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['ff_conviction_bullish'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['chip_tactical_exchange'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['ff_flow_momentum_bullish'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['ff_accum_inflection_intent'] += market_regime_mod + sentiment_mod + trend_quality_mod
        adaptive_weights_per_date['synergistic_accumulation'] += market_regime_mod + sentiment_mod + trend_quality_mod # 修改行: 协同吸筹调制

        # 市场矛盾、行为看涨背离和结构张力权重相对稳定，略受趋势质量影响
        adaptive_weights_per_date['market_contradiction_bullish'] += (1 - trend_quality_mod) * 0.05
        adaptive_weights_per_date['behavior_bullish_divergence'] += (1 - trend_quality_mod) * 0.05
        adaptive_weights_per_date['structural_tension'] += (1 - trend_quality_mod) * 0.05

        # 反向证据权重：在市场情绪低迷或趋势差时，反向证据的缺失更重要
        adaptive_weights_per_date['distribution_intent_negative'] += market_regime_mod + sentiment_mod
        adaptive_weights_per_date['chip_risk_distribution_whisper_negative'] += market_regime_mod + sentiment_mod

        # 确保权重非负
        for name in adaptive_weights_per_date:
            adaptive_weights_per_date[name] = adaptive_weights_per_date[name].clip(lower=0)

        # 将字典转换为 DataFrame，并按行（日期）归一化
        weights_df = pd.DataFrame(adaptive_weights_per_date)
        weights_sum_per_date = weights_df.sum(axis=1)
        # 避免除以零，如果和为零，则均匀分配权重
        weights_sum_per_date = weights_sum_per_date.replace(0, weights_df.shape[1])
        weights_df = weights_df.div(weights_sum_per_date, axis=0)

        if probe_date_for_loop is not None and probe_date_for_loop in df.index:
            print(f"       - 市场状态调制: {market_regime_mod.loc[probe_date_for_loop]:.4f}, 趋势质量调制: {trend_quality_mod.loc[probe_date_for_loop]:.4f}, 情绪调制: {sentiment_mod.loc[probe_date_for_loop]:.4f}")
            # 打印特定日期的自适应权重
            print(f"       - 自适应权重 (2025-11-28):")
            for name in evidence_names: # 使用已经定义好的 evidence_names
                print(f"         - {name}: {weights_df.loc[probe_date_for_loop, name]:.4f}")

        # --- 5. 非线性变换 (Power Transformation) ---
        # V9.0 动态 power_factor
        # power_factor 范围从 1.0 (波动性为0) 到 1.5 (波动性为1)
        power_factor_dynamic = 1.0 + volatility_instability * 0.5 

        evidence_list = [
            capital_confrontation_evidence,
            price_falling_evidence,
            deception_evidence_dynamic, # 使用静态证据
            volume_atrophy_evidence,
            efficiency_evidence_dynamic, # 使用动态调制后的证据
            process_stealth_accum_evidence_dynamic, # 使用静态证据
            split_order_accum_evidence,
            power_transfer_evidence,
            chip_evidence, # chip_strategic_posture
            market_contradiction_bullish,
            # V6.0 引入的证据
            suppressive_accum_intensity_evidence,
            covert_accum_signal_evidence,
            shakeout_confirmation_evidence,
            deceptive_accum_evidence,
            panic_washout_accum_evidence,
            loser_capitulation_evidence,
            absorption_strength_evidence,
            offensive_absorption_intent_evidence,
            fund_flow_bullish_divergence_evidence,
            chip_opp_absorption_echo_evidence,
            distribution_intent_negative_evidence,
            chip_risk_distribution_whisper_negative_evidence,
            # V7.0 新增证据
            price_downward_momentum_evidence,
            pct_change_slope_evidence,
            micro_stealth_ops_evidence,
            micro_cost_control_evidence,
            chip_holder_sentiment_bullish_evidence,
            ff_conviction_bullish_evidence,
            suppressive_accum_intensity_slope_evidence,
            covert_accum_signal_slope_evidence,
            behavior_bullish_divergence_evidence,
            # V8.0 新增证据
            downward_resistance_evidence,
            intraday_vwap_battlefield_negative_evidence,
            chip_tactical_exchange_evidence,
            ff_flow_momentum_bullish_evidence,
            ff_accum_inflection_intent_evidence,
            structural_tension_evidence,
            # V9.0 新增证据
            synergistic_accumulation_evidence
        ]
        # evidence_names 已经定义在方法顶部，无需重复定义

        transformed_evidence_scores = []
        for i, evidence_series in enumerate(evidence_list):
            # 修改行: 使用动态 power_factor_dynamic
            transformed_score = evidence_series.pow(power_factor_dynamic)
            transformed_evidence_scores.append(transformed_score.values)
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"       - {evidence_names[i]} (原始证据): {evidence_series.loc[probe_date_for_loop]:.4f}, 变换后: {transformed_score.loc[probe_date_for_loop]:.4f}")

        stacked_transformed_scores = np.stack(transformed_evidence_scores, axis=0)
        safe_scores = np.maximum(stacked_transformed_scores, 1e-9) # 避免对数运算错误

        # --- 6. 计算似然度 (Likelihood) ---
        # 使用归一化后的 weights_df.values.T (转置) 来与 evidence_scores 进行元素乘法
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * weights_df.values.T, axis=0))
        likelihood = pd.Series(likelihood_values, index=df.index)

        # --- 7. 反事实推理代理 (Unexpected Accumulation Bonus) ---
        unexpected_bonus_factor = 0.3 # 意外吸筹奖励强度
        # 如果趋势质量不是非常差（即不是极端熊市），但仍有吸筹行为，则给予奖励
        # (1 - trend_quality_score.clip(lower=0)) 意味着在趋势质量为正时，该项接近0，在趋势质量为负时，该项接近1
        # 我们想要的是在趋势质量“不那么差”的时候，吸筹才算“意外”
        # 所以，如果 trend_quality_score > -0.5 (即不是非常熊市)，则 bonus 越高
        unexpected_context_multiplier = (trend_quality_score + 0.5).clip(0, 1) # 趋势质量从-0.5到1，乘数从0到1.5
        
        # 意外吸筹奖励 = (隐秘吸筹 + 拆单吸筹 + 诡道吸筹 + 恐慌洗盘吸筹 + 隐蔽吸筹信号 + 微观隐秘行动 + 资金流吸筹拐点意图 + 协同吸筹) * 意外情境乘数 * 奖励因子
        unexpected_accumulation_bonus = (
            process_stealth_accum_evidence_dynamic + split_order_accum_evidence +
            deceptive_accum_evidence + panic_washout_accum_evidence +
            covert_accum_signal_evidence + micro_stealth_ops_evidence +
            ff_accum_inflection_intent_evidence + synergistic_accumulation_evidence # 修改行: 添加 synergistic_accumulation_evidence 到奖励计算
        ) * unexpected_context_multiplier * unexpected_bonus_factor
        
        # 将奖励加到似然度上，并确保不超过1
        likelihood = (likelihood + unexpected_accumulation_bonus).clip(0, 1)

        if probe_date_for_loop is not None and probe_date_for_loop in df.index:
            print(f"       - 意外情境乘数: {unexpected_context_multiplier.loc[probe_date_for_loop]:.4f}")
            print(f"       - 意外吸筹奖励: {unexpected_accumulation_bonus.loc[probe_date_for_loop]:.4f}")
            print(f"       - 似然度 (含奖励): {likelihood.loc[probe_date_for_loop]:.4f}")

        # --- 8. 计算后验概率 (Posterior Probability) ---
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)

        if probe_date_for_loop is not None and probe_date_for_loop in df.index:
            print(f"       - 先验概率(prior_prob): {prior_prob.loc[probe_date_for_loop]:.4f}")
            print(f"       - 最终后验概率(posterior_prob): {posterior_prob.loc[probe_date_for_loop]:.4f}")

        return {'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_distribution_at_high(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.12 · 依赖净化版】贝叶斯推演：“高位派发”风险剧本
        - 核心修复: 将对已废弃的 `FUSION_BIPOLAR_UPPER_SHADOW_INTENT` 的依赖，
                    升级为对行为层更权威的 `SCORE_BEHAVIOR_DISTRIBUTION_INTENT` 信号的依赖。
        """
        print("    -- [剧本推演] 高位派发风险 (动态证据)...")
        # 更新信号校验列表
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 'PROCESS_META_PROFIT_VS_FLOW', 'SCORE_CHIP_STRATEGIC_POSTURE',
            'FUSION_BIPOLAR_MARKET_CONTRADICTION', 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT',
            'SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_distribution_at_high"):
            return {'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_distribution_at_high").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        capital_confrontation_bearish = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        price_overextension_risk = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0).clip(upper=0).abs())
        low_upward_efficiency = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5)).clip(0, 1))
        profit_vs_flow_bearish = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        chip_dispersion_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(upper=0).abs())
        market_contradiction_bearish = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0).clip(upper=0).abs())
        # 替换为更权威的派发意图信号
        distribution_intent_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0))
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        chip_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            capital_confrontation_bearish.values,
            price_overextension_risk.values,
            low_upward_efficiency.values,
            profit_vs_flow_bearish.values,
            chip_dispersion_evidence.values,
            market_contradiction_bearish.values,
            distribution_intent_evidence.values, # 使用新变量
            fund_flow_bearish_divergence.values,
            chip_bearish_divergence.values,
            dip_absorption_inverse.values,
            main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.12, 0.08, 0.08, 0.12, 0.12, 0.08, 0.12, 0.04, 0.04, 0.10, 0.10])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH': posterior_prob.astype(np.float32)}

    def _deduce_trend_exhaustion_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.5 · 依赖净化版】贝叶斯推演：“趋势衰竭”风险剧本
        - 核心修复: 将对已废弃的 `FUSION_BIPOLAR_UPPER_SHADOW_INTENT` 的依赖，
                    升级为对行为层更权威的 `SCORE_BEHAVIOR_DISTRIBUTION_INTENT` 信号的依赖。
        """
        print("    -- [剧本推演] 趋势衰竭风险 (动态证据)...")
        # 更新信号校验列表
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 'PROCESS_META_WINNER_CONVICTION_DECAY',
            'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 'SCORE_FUND_FLOW_BEARISH_DIVERGENCE',
            'COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 'SCORE_CHIP_STRATEGIC_POSTURE',
            'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION',
            'FUSION_BIPOLAR_MARKET_CONTRADICTION', 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT',
            'COGNITIVE_RISK_CYCLICAL_TOP', 'CONTEXT_NEW_HIGH_STRENGTH', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_trend_exhaustion_risk"):
            return {'COGNITIVE_RISK_TREND_EXHAUSTION': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_trend_exhaustion_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        price_momentum_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0).clip(lower=0))
        stagnation_evidence = self._forge_dynamic_evidence(df, (1 - self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5)).clip(0, 1))
        raw_price_overextension_score = self._get_fused_score(df, 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0)
        price_overextension_risk = self._forge_dynamic_evidence(df, raw_price_overextension_score.clip(upper=0).abs())
        winner_conviction_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        raw_capital_confrontation_score = self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        capital_retreat_evidence = self._forge_dynamic_evidence(df, raw_capital_confrontation_score.clip(upper=0).abs())
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        retail_fomo_retreat_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        chip_dispersion_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(upper=0).abs())
        chip_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 0.0))
        long_term_profit_distribution_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 0.0), is_probability=True)
        raw_structural_trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        structural_deterioration = self._forge_dynamic_evidence(df, raw_structural_trend_form_score.clip(upper=0).abs())
        raw_market_contradiction_score = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0)
        market_contradiction_bearish = self._forge_dynamic_evidence(df, raw_market_contradiction_score.clip(upper=0).abs())
        # 替换为更权威的派发意图信号
        distribution_intent_risk = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0))
        cyclical_top_risk = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'COGNITIVE_RISK_CYCLICAL_TOP', 0.0))
        trend_quality_inverse = self._forge_dynamic_evidence(df, 1 - self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0).clip(lower=0))
        new_high_strength_inverse = self._forge_dynamic_evidence(df, 1 - self._get_atomic_score(df, 'CONTEXT_NEW_HIGH_STRENGTH', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            price_momentum_divergence.values, winner_conviction_decay.values, stagnation_evidence.values,
            chip_dispersion_evidence.values, fund_flow_bearish_divergence.values, structural_deterioration.values,
            capital_retreat_evidence.values, cyclical_top_risk.values, price_overextension_risk.values,
            distribution_intent_risk.values, market_contradiction_bearish.values, retail_fomo_retreat_risk.values,
            chip_bearish_divergence.values, long_term_profit_distribution_risk.values, trend_quality_inverse.values,
            new_high_strength_inverse.values, dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([
            0.07, 0.06, 0.02, 0.06, 0.05, 0.04, 0.07, 0.05, 0.02, 0.03, 0.03, 0.03, 0.02, 0.02, 0.08, 0.08, 0.07, 0.09
        ])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [趋势衰竭风险探针] @ {probe_date_for_loop.date()}:")
                print(f"       - posterior_prob: {posterior_prob.loc[probe_date_for_loop]:.4f}")
        return {'COGNITIVE_RISK_TREND_EXHAUSTION': posterior_prob.astype(np.float32)}

    def _establish_prior_beliefs(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.8 · 数据帧上下文修复版】建立先验信念
        - 【V1.8 修复】接收并使用 df 参数，确保索引上下文统一。
        """
        # 增加信号校验
        required_signals = [
            'FUSION_BIPOLAR_MARKET_REGIME', 'FUSION_BIPOLAR_TREND_QUALITY', 'FUSION_BIPOLAR_TREND_STRUCTURE_SCORE',
            'FUSION_BIPOLAR_FUND_FLOW_TREND', 'FUSION_BIPOLAR_CHIP_TREND', 'SCORE_CHIP_COHERENT_DRIVE',
            'FUSION_BIPOLAR_MARKET_PRESSURE', 'CONTEXT_TREND_CONFIRMED'
        ]
        if not self._validate_required_signals(df, required_signals, "_establish_prior_beliefs"):
            # 如果关键先验信号缺失，返回一个中性的先验概率
            default_prob = pd.Series(0.5, index=df.index)
            return {
                'COGNITIVE_PRIOR_TREND_PROB': default_prob,
                'COGNITIVE_PRIOR_REVERSAL_PROB': default_prob
            }
        states = {}
        df_index = df.index
        market_regime = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_REGIME', 0.0)
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        trend_structure_score = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_STRUCTURE_SCORE', 0.0)
        fund_flow_trend = self._get_fused_score(df, 'FUSION_BIPOLAR_FUND_FLOW_TREND', 0.0)
        chip_trend = self._get_fused_score(df, 'FUSION_BIPOLAR_CHIP_TREND', 0.0)
        structural_consensus = self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0)
        market_regime_prob = (market_regime + 1) / 2
        trend_quality_prob = (trend_quality + 1) / 2
        trend_structure_prob = (trend_structure_score + 1) / 2
        fund_flow_trend_prob = (fund_flow_trend + 1) / 2
        chip_trend_prob = (chip_trend + 1) / 2
        structural_consensus_prob = structural_consensus
        regime_weight = 0.15
        quality_weight = 0.15
        structure_weight = 0.15
        fund_flow_weight = 0.15
        chip_trend_weight = 0.15
        structural_consensus_weight = 0.25
        prior_trend = (
            market_regime_prob * regime_weight +
            trend_quality_prob * quality_weight +
            trend_structure_prob * structure_weight +
            fund_flow_trend_prob * fund_flow_weight +
            chip_trend_prob * chip_trend_weight +
            structural_consensus_prob * structural_consensus_weight
        ).clip(0, 1)
        states['COGNITIVE_PRIOR_TREND_PROB'] = prior_trend.astype(np.float32)
        market_pressure = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_PRESSURE', 0.0)
        reversal_pressure_weight = 0.6
        reversal_regime_strength_weight = 0.4
        trend_confirmed = self._get_atomic_score(df, 'CONTEXT_TREND_CONFIRMED', 0.0)
        suppression_factor = (1 - trend_confirmed).clip(0, 1)
        prior_reversal_raw = (market_pressure.abs() * reversal_pressure_weight + market_regime.abs() * reversal_regime_strength_weight).clip(0, 1)
        prior_reversal = (prior_reversal_raw * suppression_factor).clip(0, 1)
        states['COGNITIVE_PRIOR_REVERSAL_PROB'] = prior_reversal.astype(np.float32)
        return states

    def _fuse_and_adjudicate_playbooks(self, df: pd.DataFrame, playbook_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.6 · 数据帧上下文修复版】融合与裁决模块
        - 核心升级: 将新的认知剧本 `COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE` 集成到看涨剧本列表中。
        - 【V3.6 修复】接收并使用 df 参数，确保索引上下文统一。
        """
        states = {}
        df_index = df.index
        bullish_playbooks = [
            'COGNITIVE_PLAYBOOK_SUPPRESSIVE_ACCUMULATION', 'COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION',
            'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL', 'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING',
            'COGNITIVE_PLAYBOOK_SECTOR_ROTATION_VANGUARD', 'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION',
            'COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE', 'COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE',
        ]
        bullish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bullish_playbooks]
        cognitive_bullish_score = np.maximum.reduce([s.values for s in bullish_scores])
        states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df_index, dtype=np.float32)
        bearish_playbooks = [
            'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 'COGNITIVE_RISK_TREND_EXHAUSTION',
            'COGNITIVE_RISK_MARKET_UNCERTAINTY', 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT',
            'COGNITIVE_RISK_HARVEST_CONFIRMATION', 'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION',
            'COGNITIVE_RISK_LIQUIDITY_TRAP', 'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE',
            'COGNITIVE_RISK_KEY_SUPPORT_BREAK', 'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE',
            'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION'
        ]
        bearish_scores = [playbook_scores.get(name, pd.Series(0.0, index=df_index)) for name in bearish_playbooks]
        cognitive_bearish_score = np.maximum.reduce([s.values for s in bearish_scores])
        states['COGNITIVE_BEARISH_SCORE'] = pd.Series(cognitive_bearish_score, index=df_index, dtype=np.float32)
        return states

    def _deduce_chasing_accumulation(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V3.8 · 调用修复版】贝叶斯推演：“主力拉升抢筹”剧本
        - 【V3.8 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 主力拉升抢筹 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 'pct_change_D', 'VPA_EFFICIENCY_D',
            'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 'PROCESS_META_WINNER_CONVICTION',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_CHIP_COHERENT_DRIVE',
            'SCORE_PATTERN_PULLBACK_CONFIRMATION', 'SCORE_PATTERN_DUOFANGPAO'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_chasing_accumulation"):
            return {'COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION': pd.Series(0.0, index=df.index)}
        capital_confrontation = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        price_change_bipolar = normalize_to_bipolar(self._get_atomic_score(df, 'pct_change_D'), df.index, 21)
        price_rising_evidence = self._forge_dynamic_evidence(df, price_change_bipolar.clip(lower=0))
        efficiency_evidence = self._forge_dynamic_evidence(df, normalize_score(self._get_atomic_score(df, 'VPA_EFFICIENCY_D'), df.index, 55))
        rally_intent_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0).clip(lower=0))
        conviction_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION', 0.0).clip(lower=0))
        process_evidence = (rally_intent_evidence * conviction_evidence).pow(0.5)
        # 将 SCORE_CHIP_AXIOM_CONCENTRATION 替换为 SCORE_CHIP_STRATEGIC_POSTURE
        chip_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(lower=0))
        structural_consensus_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0))
        pullback_confirmation_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_PATTERN_PULLBACK_CONFIRMATION', 0.0))
        duofangpao_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_PATTERN_DUOFANGPAO', 0.0))
        evidence_scores = np.stack([
            capital_confrontation.values, price_rising_evidence.values, efficiency_evidence.values,
            process_evidence.values, chip_evidence.values, structural_consensus_evidence.values,
            pullback_confirmation_evidence.values, duofangpao_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.12, 0.08, 0.08, 0.18, 0.12, 0.12, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CHASING_ACCUMULATION': posterior_prob.astype(np.float32)}

    def _deduce_capitulation_reversal(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.3 · 情报同步版】贝叶斯推演：“恐慌投降反转”机会剧本
        - 核心修复: 将对已废弃的微观行为反转信号的依赖，同步为对过程层最新的
                    “微观策略”反转信号的依赖，完成情报链路的最终同步。
        """
        print("    -- [剧本推演] 恐慌投降反转 (动态证据)...")
        # 更新信号校验列表
        required_signals = [
            'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION',
            'PROCESS_META_PRICE_VS_RETAIL_CAPITULATION', 'PROCESS_META_LOSER_CAPITULATION',
            'PROCESS_META_POWER_TRANSFER', 'PROCESS_META_MICRO_STRATEGY_BOTTOM_REVERSAL',
            'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 'SCORE_CHIP_STRATEGIC_POSTURE'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_capitulation_reversal"):
            return {'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL': pd.Series(0.0, index=df.index)}
        df_index = df.index
        panic_washout_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 0.0))
        lower_shadow_absorption = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0))
        price_vs_capitulation_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PRICE_VS_RETAIL_CAPITULATION', 0.0))
        loser_capitulation_consensus = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_LOSER_CAPITULATION', 0.0))
        power_transfer_to_main_force = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0))
        # 获取新的微观策略反转信号
        micro_reversal_confirmation = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_MICRO_STRATEGY_BOTTOM_REVERSAL', 0.0))
        chip_geography_support = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 0.0).clip(lower=0))
        chip_posture_improvement = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(lower=0))
        evidence_scores = np.stack([
            panic_washout_evidence.values, lower_shadow_absorption.values,
            price_vs_capitulation_divergence.values, loser_capitulation_consensus.values,
            power_transfer_to_main_force.values, micro_reversal_confirmation.values,
            chip_geography_support.values, chip_posture_improvement.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.15, 0.15, 0.10, 0.15, 0.10, 0.05, 0.10])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_CAPITULATION_REVERSAL': posterior_prob.astype(np.float32)}

    def _deduce_leading_dragon_awakening(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 调用修复版】贝叶斯推演：“龙头苏醒”剧本
        - 核心修复: 修正信号名称。
        - 【V1.6 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 龙头苏醒 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 'breakout_quality_score_D', 'PROCESS_META_STOCK_SECTOR_SYNC',
            'industry_strength_rank_D', 'IS_BAZHAN_D', 'SCORE_CHIP_COHERENT_DRIVE'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_leading_dragon_awakening"):
            return {'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING': pd.Series(0.0, index=df.index)}
        capital_confrontation = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(lower=0))
        breakout_quality = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'breakout_quality_score_D', 0.0))
        sector_sync = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_STOCK_SECTOR_SYNC', 0.0).clip(lower=0))
        relative_strength = self._forge_dynamic_evidence(df, normalize_score(self._get_atomic_score(df, 'industry_strength_rank_D', 0.5), df.index, 55))
        bazhan_mode = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'IS_BAZHAN_D', 0.0).astype(float))
        structural_consensus_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0))
        evidence_scores = np.stack([
            capital_confrontation.values, breakout_quality.values, sector_sync.values,
            relative_strength.values, bazhan_mode.values,
            structural_consensus_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=df.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_TREND_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_LEADING_DRAGON_AWAKENING': posterior_prob.astype(np.float32)}

    def _deduce_divergence_reversal(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V2.2 · 调用修复版】贝叶斯推演：“背离反转”剧本
        - 核心逻辑: 捕捉价格与关键指标的背离。
        - 【V2.2 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 背离反转 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 'SCORE_FUND_FLOW_BEARISH_DIVERGENCE',
            'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 'COGNITIVE_RISK_TREND_EXHAUSTION',
            'FUSION_BIPOLAR_MARKET_CONTRADICTION', 'PROCESS_META_WINNER_CONVICTION_DECAY'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_divergence_reversal"):
            return {'COGNITIVE_PLAYBOOK_DIVERGENCE_REVERSAL': pd.Series(0.0, index=df.index)}
        df_index = df.index
        price_momentum_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE', 0.0))
        fund_flow_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FUND_FLOW_BEARISH_DIVERGENCE', 0.0))
        # 将 SCORE_CHIP_BEARISH_DIVERGENCE 替换为 SCORE_CHIP_RISK_DISTRIBUTION_WHISPER
        chip_bearish_divergence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 0.0))
        trend_exhaustion_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_TREND_EXHAUSTION', 0.0), is_probability=True)
        raw_market_contradiction_score = self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_CONTRADICTION', 0.0)
        market_contradiction_bearish = self._forge_dynamic_evidence(df, raw_market_contradiction_score.clip(upper=0).abs())
        winner_conviction_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        evidence_scores = np.stack([
            price_momentum_divergence.values, fund_flow_bearish_divergence.values,
            chip_bearish_divergence.values, trend_exhaustion_risk.values,
            market_contradiction_bearish.values, winner_conviction_decay.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_DIVERGENCE_REVERSAL': posterior_prob.astype(np.float32)}

    def _deduce_sector_rotation_vanguard(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.7 · 调用修复版】贝叶斯推演：“板块轮动先锋”剧本
        - 核心修复: 修正信号名称。
        - 【V1.7 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 板块轮动先锋 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'PROCESS_META_FUND_FLOW_BOTTOM_REVERSAL', 'BIAS_144_D', 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY',
            'PROCESS_META_HOT_SECTOR_COOLING'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_sector_rotation_vanguard"):
            return {'COGNITIVE_PLAYBOOK_SECTOR_ROTATION_VANGUARD': pd.Series(0.0, index=df.index)}
        sector_flow = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_FUND_FLOW_BOTTOM_REVERSAL', 0.0).clip(lower=0))
        price_position = self._forge_dynamic_evidence(df, 1 - normalize_score(self._get_atomic_score(df, 'BIAS_144_D', 0.0), df.index, 144))
        # 将 SCORE_CHIP_CLEANLINESS 替换为 SCORE_CHIP_BATTLEFIELD_GEOGRAPHY
        chip_cleanliness = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 0.0).clip(lower=0))
        hot_sector_cooling = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_HOT_SECTOR_COOLING', 0.0))
        evidence_scores = np.stack([sector_flow.values, price_position.values, chip_cleanliness.values, hot_sector_cooling.values], axis=0)
        evidence_weights = np.array([0.4, 0.2, 0.2, 0.2])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood = pd.Series(np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0)), index=df.index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_SECTOR_ROTATION_VANGUARD': posterior_prob.astype(np.float32)}

    def _deduce_energy_compression_breakout(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.9 · 调用修复版】贝叶斯推演：“能量压缩爆发”剧本
        - 核心升级: 替换为更精确的信号。
        - 【V1.9 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 能量压缩爆发 (动态证据)...")
        # 增加信号校验 (注意：SAMPLE_ENTROPY_* 是动态查找的，不加入校验)
        required_signals = [
            'BBW_21_2.0_D', 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 'pct_change_D', 'SCORE_BEHAVIOR_VOLUME_BURST'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_energy_compression_breakout"):
            return {'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION': pd.Series(0.0, index=df.index)}
        df_index = df.index
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        bbw = self._get_atomic_score(df, 'BBW_21_2.0_D', 0.1)
        volatility_compression_raw_score = normalize_score(1 - bbw, df_index, 144, ascending=True)
        volatility_compression = self._forge_dynamic_evidence(df, 1 - normalize_score(bbw, df_index, 144))
        volume_atrophy_raw_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        volume_atrophy = self._forge_dynamic_evidence(df, volume_atrophy_raw_score)
        entropy_col = next((col for col in df.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            entropy = self._get_atomic_score(df, entropy_col, 1.0)
            orderliness_score = self._forge_dynamic_evidence(df, 1 - normalize_score(entropy, df_index, 144))
        else:
            orderliness_score = pd.Series(0.5, index=df.index)
        pct_change_raw = self._get_atomic_score(df, 'pct_change_D', 0.0)
        price_burst_evidence = self._forge_dynamic_evidence(df, pct_change_raw.clip(lower=0), is_probability=False)
        volume_burst_raw = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_BURST', 0.0)
        volume_burst_evidence = self._forge_dynamic_evidence(df, volume_burst_raw, is_probability=False)
        volatility_compression_final = volatility_compression.mask(is_limit_up_day, volatility_compression_raw_score)
        volume_atrophy_final = volume_atrophy.mask(is_limit_up_day, volume_atrophy_raw_score)
        evidence_scores = np.stack([
            volatility_compression_final.values, volume_atrophy_final.values,
            orderliness_score.values, price_burst_evidence.values, volume_burst_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.15, 0.15, 0.15, 0.3, 0.25])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood.mask(is_limit_up_day, likelihood + 0.3).clip(0, 1)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_ENERGY_COMPRESSION': posterior_prob.astype(np.float32)}

    def _forge_dynamic_evidence(self, df: pd.DataFrame, evidence: pd.Series, is_probability: bool = False) -> pd.Series:
        """
        【V2.2 · 返回值修复版】动态证据锻造
        - 【V2.2 修复】修复了方法内部多处错误返回索引而非Series的问题，确保返回值始终是数值型Series。
        """
        if not isinstance(evidence, pd.Series):
            evidence = pd.Series(evidence, index=df.index) # 移除了末尾的 .index
        evidence = evidence.fillna(self.min_evidence_threshold)
        evidence = evidence.mask(evidence < self.min_evidence_threshold, self.min_evidence_threshold)
        if not is_probability:
            evidence = normalize_score(evidence, df.index, window=self.norm_window, ascending=True) # 移除了末尾的 .index
        return evidence

    def _deduce_long_term_profit_distribution_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 调用修复版】贝叶斯推演：“长期获利盘派发”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.5 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 长期获利盘派发风险 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'PROCESS_META_WINNER_CONVICTION_DECAY', 'SCORE_CHIP_STRATEGIC_POSTURE',
            'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_long_term_profit_distribution_risk"):
            return {'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_long_term_profit_distribution_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        long_term_profit_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        # 将 SCORE_CHIP_AXIOM_CONCENTRATION 替换为 SCORE_CHIP_STRATEGIC_POSTURE 的负向表现
        chip_dispersion = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(upper=0).abs())
        capital_outflow = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            long_term_profit_decay.values, chip_dispersion.values, capital_outflow.values,
            dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION': posterior_prob.astype(np.float32)}

    def _deduce_market_uncertainty_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.5 · 调用修复版】贝叶斯推演：“市场方向不明”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.5 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 市场方向不明风险 (动态证据)...")
        # 增加信号校验 (注意：SAMPLE_ENTROPY_* 是动态查找的，不加入校验)
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'FUSION_BIPOLAR_MARKET_REGIME', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_market_uncertainty_risk"):
            return {'COGNITIVE_RISK_MARKET_UNCERTAINTY': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_market_uncertainty_risk").shift(1).fillna(False)
        regime_neutrality = self._forge_dynamic_evidence(df, 1 - self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_REGIME', 0.0).abs())
        low_trend_quality = self._forge_dynamic_evidence(df, 1 - self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0).abs())
        entropy_col = next((col for col in df.columns if col.startswith('SAMPLE_ENTROPY_')), None)
        if entropy_col:
            high_entropy = self._forge_dynamic_evidence(df, self._get_atomic_score(df, entropy_col, 0.5))
        else:
            high_entropy = pd.Series(0.5, index=df.index)
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            regime_neutrality.values, low_trend_quality.values,
            high_entropy.values, dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_MARKET_UNCERTAINTY': posterior_prob.astype(np.float32)}

    def _deduce_retail_fomo_retreat_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.10 · 调用修复版】贝叶斯推演：“散户狂热主力撤退”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.10 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 散户狂热主力撤退风险 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'SCORE_FF_AXIOM_CONSENSUS', 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 'pct_change_D',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_retail_fomo_retreat_risk"):
            return {'COGNITIVE_RISK_RETAIL_FOMO_RETREAT': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_retail_fomo_retreat_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        raw_retail_inflow_score = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0)
        retail_inflow = self._forge_dynamic_evidence(df, raw_retail_inflow_score.clip(lower=0))
        raw_mf_confrontation_score = self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        main_force_outflow = self._forge_dynamic_evidence(df, raw_mf_confrontation_score.clip(upper=0).abs())
        raw_price_rising_score = normalize_to_bipolar(self._get_atomic_score(df, 'pct_change_D'), df.index, 21)
        price_rising = self._forge_dynamic_evidence(df, raw_price_rising_score.clip(lower=0))
        # 将 SCORE_CHIP_AXIOM_CONCENTRATION 替换为 SCORE_CHIP_STRATEGIC_POSTURE 的负向表现
        chip_dispersion = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(upper=0).abs())
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            retail_inflow.values, main_force_outflow.values, price_rising.values,
            chip_dispersion.values, dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.10, 0.25, 0.07, 0.25, 0.18, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_RETAIL_FOMO_RETREAT': posterior_prob.astype(np.float32)}

    def _deduce_harvest_confirmation_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 调用修复版】贝叶斯推演：“收割确认”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.6 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 收割确认风险 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 'PROCESS_META_PROFIT_VS_FLOW',
            'PROCESS_META_WINNER_CONVICTION_DECAY', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_harvest_confirmation_risk"):
            return {'COGNITIVE_RISK_HARVEST_CONFIRMATION': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_harvest_confirmation_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        high_distribution_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0), is_probability=True)
        high_t0_efficiency = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        winner_conviction_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_distribution_risk.values, high_t0_efficiency.values, winner_conviction_decay.values,
            dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_HARVEST_CONFIRMATION': posterior_prob.astype(np.float32)}

    def _deduce_bull_trap_distribution_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.12 · 依赖净化版】贝叶斯推演：“主力诱多派发”风险剧本
        - 核心修复: 将对已废弃的 `FUSION_BIPOLAR_UPPER_SHADOW_INTENT` 的依赖，
                    升级为对行为层更权威的 `SCORE_BEHAVIOR_DISTRIBUTION_INTENT` 信号的依赖。
        """
        print("    -- [剧本推演] 主力诱多派发风险 (动态证据)...")
        # 更新信号校验列表
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'pct_change_D', 'SCORE_MICRO_STRATEGY_SHOCK_AND_AWE', 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION',
            'PROCESS_META_PROFIT_VS_FLOW', 'PROCESS_META_WINNER_CONVICTION_DECAY',
            'COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT',
            'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_bull_trap_distribution_risk"):
            return {'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_bull_trap_distribution_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        price_rising = self._forge_dynamic_evidence(df, normalize_to_bipolar(self._get_atomic_score(df, 'pct_change_D'), df.index, 21).clip(lower=0))
        micro_shock_awe_bearish = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_SHOCK_AND_AWE', 0.0).clip(upper=0).abs())
        # 替换为更权威的派发意图信号
        distribution_intent_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0))
        chip_dispersion = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(upper=0).abs())
        raw_mf_confrontation_score = self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0)
        main_force_outflow = self._forge_dynamic_evidence(df, raw_mf_confrontation_score.clip(upper=0).abs())
        raw_profit_vs_flow_score = self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0)
        profit_vs_flow_bearish = self._forge_dynamic_evidence(df, raw_profit_vs_flow_score.clip(upper=0).abs())
        winner_conviction_decay = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION_DECAY', 0.0))
        retail_fomo_retreat_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        raw_price_overextension_score = self._get_fused_score(df, 'FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0)
        price_overextension_risk = self._forge_dynamic_evidence(df, raw_price_overextension_score.clip(upper=0).abs())
        long_term_profit_distribution_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION', 0.0), is_probability=True)
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            price_rising.values, micro_shock_awe_bearish.values, distribution_intent_evidence.values,
            chip_dispersion.values, main_force_outflow.values, profit_vs_flow_bearish.values,
            winner_conviction_decay.values, retail_fomo_retreat_risk.values, price_overextension_risk.values,
            long_term_profit_distribution_risk.values, dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.02, 0.08, 0.08, 0.12, 0.12, 0.05, 0.05, 0.08, 0.07, 0.10, 0.12, 0.11])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION': posterior_prob.astype(np.float32)}

    def _deduce_liquidity_trap_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 基础层升维同步版】贝叶斯推演：“流动性陷阱”风险剧本
        - 核心升级: 将对旧“波动”公理的依赖，替换为对新“市场张力”公理的依赖，
                      以更精准地刻画能量压缩状态。
        """
        print("    -- [剧本推演] 流动性陷阱风险 (动态证据)...")
        # 更新依赖信号，用“市场张力”替换旧的“波动率”
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 'SCORE_BEHAVIOR_VOLUME_ATROPHY',
            'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_liquidity_trap_risk"):
            return {'COGNITIVE_RISK_LIQUIDITY_TRAP': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_liquidity_trap_risk").shift(1).fillna(False)
        capital_outflow = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        volume_apathy = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        # 直接使用“市场张力”分作为能量压缩的证据
        market_tension = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            capital_outflow.values, volume_apathy.values,
            market_tension.values, dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_LIQUIDITY_TRAP': posterior_prob.astype(np.float32)}

    def _deduce_t0_arbitrage_pressure_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 情报同步版】贝叶斯推演：“T+0套利压力”风险剧本
        - 核心修复: 将对废弃信号 `SCORE_MICRO_AXIOM_DECEPTION` 的依赖，替换为对
                    新信号 `SCORE_MICRO_STRATEGY_SHOCK_AND_AWE` 的依赖，完成情报代际同步。
        """
        print("    -- [剧本推演] T+0套利压力风险 (动态证据)...")
        # 更新信号校验列表
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'PROCESS_META_PROFIT_VS_FLOW', 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION',
            'SCORE_MICRO_STRATEGY_SHOCK_AND_AWE', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_t0_arbitrage_pressure_risk"):
            return {'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_t0_arbitrage_pressure_risk").shift(1).fillna(False)
        high_t0_efficiency = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_PROFIT_VS_FLOW', 0.0).clip(upper=0).abs())
        capital_outflow = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_CAPITAL_CONFRONTATION', 0.0).clip(upper=0).abs())
        # 替换为新的微观信号
        micro_shock_awe_bearish = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_SHOCK_AND_AWE', 0.0).clip(upper=0).abs())
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_t0_efficiency.values, capital_outflow.values,
            micro_shock_awe_bearish.values, dip_absorption_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.25, 0.25, 0.20, 0.30])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE': posterior_prob.astype(np.float32)}

    def _deduce_key_support_break_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 基础层升维同步版】贝叶斯推演：“关键支撑破位”风险剧本
        - 核心升级: 将对旧“趋势”公理的依赖，替换为对新“市场体质”公理的依赖，
                      以更全面地评估趋势的“羸弱”程度。
        """
        print("    -- [剧本推演] 关键支撑破位风险 (动态证据)...")
        # 更新依赖信号，用“市场体质”替换旧的“趋势”
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'FUSION_BIPOLAR_MARKET_PRESSURE', 'SCORE_STRUCT_AXIOM_STABILITY', 'SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION',
            'PROCESS_META_LOSER_CAPITULATION', 'dip_absorption_power_D', 'close_D', 'EMA_21_D', 'EMA_55_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_key_support_break_risk"):
            return {'COGNITIVE_RISK_KEY_SUPPORT_BREAK': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_key_support_break_risk").shift(1).fillna(False)
        downward_pressure = self._forge_dynamic_evidence(df, self._get_fused_score(df, 'FUSION_BIPOLAR_MARKET_PRESSURE', 0.0).clip(upper=0).abs())
        low_structural_stability = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_STABILITY', 0.0).clip(upper=0).abs())
        # 使用“市场体质”的负向部分作为“羸弱”的证据
        weak_market_constitution = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION', 0.0).clip(upper=0).abs())
        loser_capitulation = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_LOSER_CAPITULATION', 0.0))
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        close_price = self._get_safe_series(df, 'close_D', method_name="_deduce_key_support_break_risk")
        ema21 = self._get_safe_series(df, 'EMA_21_D', method_name="_deduce_key_support_break_risk")
        ema55 = self._get_safe_series(df, 'EMA_55_D', method_name="_deduce_key_support_break_risk")
        price_above_ma_score = normalize_score(
            (close_price - ema21).clip(lower=0) + (close_price - ema55).clip(lower=0),
            df_index, window=55, ascending=True
        )
        price_above_ma_inverse = self._forge_dynamic_evidence(df, 1 - price_above_ma_score)
        evidence_scores = np.stack([
            downward_pressure.values, low_structural_stability.values, weak_market_constitution.values,
            loser_capitulation.values, dip_absorption_inverse.values, price_above_ma_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_KEY_SUPPORT_BREAK': posterior_prob.astype(np.float32)}

    def _deduce_high_level_structural_collapse_risk(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.6 · 调用修复版】贝叶斯推演：“高位结构瓦解”风险剧本
        - 核心升级: 引入“趋势背景调制因子”。
        - 【V1.6 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 高位结构瓦解风险 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'FUSION_BIPOLAR_TREND_QUALITY', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'IS_LIMIT_UP_D',
            'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'dip_absorption_power_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_high_level_structural_collapse_risk"):
            return {'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE': pd.Series(0.0, index=df.index)}
        df_index = df.index
        trend_quality = self._get_fused_score(df, 'FUSION_BIPOLAR_TREND_QUALITY', 0.0)
        structural_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        trend_modulator = pd.Series(1.0, index=df_index)
        positive_trend_mask = (trend_quality > 0) & (structural_trend_form > 0)
        trend_modulator[positive_trend_mask] = (1 - (trend_quality[positive_trend_mask] + structural_trend_form[positive_trend_mask]) / 2 * 1.0).clip(0.5, 1.0)
        is_limit_up_yesterday = self._get_safe_series(df, 'IS_LIMIT_UP_D', False, method_name="_deduce_high_level_structural_collapse_risk").shift(1).fillna(False)
        main_force_holding_strength = self._get_main_force_holding_strength(df)
        main_force_holding_inverse = self._forge_dynamic_evidence(df, 1 - main_force_holding_strength)
        high_distribution_risk = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_DISTRIBUTION_AT_HIGH', 0.0), is_probability=True)
        structural_trend_deterioration = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0).clip(upper=0).abs())
        retail_fomo_retreat = self._forge_dynamic_evidence(df, self._get_playbook_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_RETREAT', 0.0), is_probability=True)
        # 将 SCORE_CHIP_AXIOM_CONCENTRATION 替换为 SCORE_CHIP_STRATEGIC_POSTURE 的负向表现
        chip_dispersion = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(upper=0).abs())
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        dip_absorption_inverse = (1 - dip_absorption_power).clip(0, 1)
        evidence_scores = np.stack([
            high_distribution_risk.values, structural_trend_deterioration.values, retail_fomo_retreat.values,
            chip_dispersion.values, dip_absorption_inverse.values, main_force_holding_inverse.values
        ], axis=0)
        evidence_weights = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        likelihood = likelihood * trend_modulator
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        posterior_prob = posterior_prob.mask(is_limit_up_yesterday, posterior_prob * 0.5).clip(0, 1)
        return {'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE': posterior_prob.astype(np.float32)}

    def _deduce_stealth_bottoming_divergence(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 调用修复版】贝叶斯推演：“隐秘筑底背离”剧本
        - 核心逻辑: 识别底部背离信号。
        - 【V1.3 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 隐秘筑底背离 (动态证据)...")
        # 增加信号校验
        required_signals = [
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 'ACCEL_5_close_D', 'PROCESS_META_BEHAVIOR_BOTTOM_REVERSAL',
            'SCORE_BEHAVIOR_VOLUME_ATROPHY', 'SCORE_FF_AXIOM_CONSENSUS', 'PROCESS_META_POWER_TRANSFER',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'PROCESS_META_STEALTH_ACCUMULATION',
            'PROCESS_META_COST_ADVANTAGE_TREND', 'PROCESS_META_LOSER_CAPITULATION', 'SCORE_CHIP_COHERENT_DRIVE'
        ]
        if not self._validate_required_signals(df, required_signals, "_deduce_stealth_bottoming_divergence"):
            return {'COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE': pd.Series(0.0, index=df.index)}
        df_index = df.index
        downward_momentum_decay = self._forge_dynamic_evidence(df, 1 - self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0))
        price_accel_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'ACCEL_5_close_D', 0.0).clip(lower=0))
        behavior_bottom_reversal = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_BEHAVIOR_BOTTOM_REVERSAL', 0.0))
        volume_atrophy_strong = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        fund_flow_consensus_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(lower=0))
        power_transfer_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0))
        # 将 SCORE_CHIP_AXIOM_CONCENTRATION 替换为 SCORE_CHIP_STRATEGIC_POSTURE
        chip_concentration_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(lower=0))
        stealth_accumulation_process = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_STEALTH_ACCUMULATION', 0.0))
        cost_advantage_trend_positive = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_COST_ADVANTAGE_TREND', 0.0).clip(lower=0))
        loser_capitulation_process = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'PROCESS_META_LOSER_CAPITULATION', 0.0))
        structural_consensus_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0))
        evidence_scores = np.stack([
            downward_momentum_decay.values, price_accel_positive.values, behavior_bottom_reversal.values,
            volume_atrophy_strong.values, fund_flow_consensus_positive.values, power_transfer_positive.values,
            chip_concentration_positive.values, stealth_accumulation_process.values, cost_advantage_trend_positive.values,
            loser_capitulation_process.values, structural_consensus_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.08, 0.08, 0.04, 0.15, 0.12, 0.04, 0.12, 0.08, 0.04, 0.05, 0.20])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_STEALTH_BOTTOMING_DIVERGENCE': posterior_prob.astype(np.float32)}

    def _deduce_micro_absorption_divergence(self, df: pd.DataFrame, priors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.3 · 调用修复版】贝叶斯推演：“微观承接背离”剧本
        - 核心逻辑: 识别微观层面的底部背离。
        - 【V1.3 修复】修正对 _forge_dynamic_evidence 的调用，传入 df 参数。
        """
        print("    -- [剧本推演] 微观承接背离 (动态证据)...")
        df_index = df.index
        price_down_momentum_high = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0))
        price_stabilization = self._forge_dynamic_evidence(df, 1 - self._get_atomic_score(df, 'pct_change_D', 0.0).abs())
        price_weak_or_stable_context = np.maximum(price_down_momentum_high, price_stabilization)
        volume_atrophy_context = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0))
        counterparty_exhaustion = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'counterparty_exhaustion_index_D', 0.0))
        selling_pressure_decreasing = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SLOPE_5_active_selling_pressure_D', 0.0).clip(upper=0).abs())
        dip_absorption_power = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'dip_absorption_power_D', 0.0))
        buying_support_increasing = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SLOPE_5_active_buying_support_D', 0.0).clip(lower=0))
        structural_consensus_evidence = self._forge_dynamic_evidence(df, self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0))
        evidence_scores = np.stack([
            price_weak_or_stable_context.values, volume_atrophy_context.values, counterparty_exhaustion.values,
            selling_pressure_decreasing.values, dip_absorption_power.values, buying_support_increasing.values,
            structural_consensus_evidence.values
        ], axis=0)
        evidence_weights = np.array([0.08, 0.08, 0.20, 0.12, 0.20, 0.12, 0.20])
        evidence_weights /= evidence_weights.sum()
        safe_scores = np.maximum(evidence_scores, 1e-9)
        likelihood_values = np.exp(np.sum(np.log(safe_scores) * evidence_weights[:, np.newaxis], axis=0))
        likelihood = pd.Series(likelihood_values, index=df_index)
        prior_prob = priors.get('COGNITIVE_PRIOR_REVERSAL_PROB', pd.Series(0.0, index=likelihood.index))
        posterior_prob = (likelihood * prior_prob).clip(0, 1)
        return {'COGNITIVE_PLAYBOOK_MICRO_ABSORPTION_DIVERGENCE': posterior_prob.astype(np.float32)}

    def _get_main_force_holding_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · 数据帧上下文修复版】计算主力持仓信念强度。
        - 核心逻辑: 融合筹码集中度、资金流信念、主力控盘和成本优势趋势，评估主力当前对股票的持有信念。
        - 【V1.1 修复】接收并使用 df 参数，确保索引上下文统一。
        """
        # 增加信号校验
        required_signals = [
            'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_FF_AXIOM_CONVICTION',
            'PROCESS_META_MAIN_FORCE_CONTROL', 'PROCESS_META_COST_ADVANTAGE_TREND'
        ]
        if not self._validate_required_signals(df, required_signals, "_get_main_force_holding_strength"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        # 将 SCORE_CHIP_AXIOM_CONCENTRATION 替换为 SCORE_CHIP_STRATEGIC_POSTURE
        chip_concentration = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(lower=0)
        fund_flow_conviction = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONVICTION', 0.0).clip(lower=0)
        main_force_control = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_CONTROL', 0.0).clip(lower=0)
        cost_advantage_trend = self._get_atomic_score(df, 'PROCESS_META_COST_ADVANTAGE_TREND', 0.0).clip(lower=0)
        components = [chip_concentration, fund_flow_conviction, main_force_control, cost_advantage_trend]
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        main_force_holding_strength = (
            aligned_components[0] * weights[0] +
            aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] +
            aligned_components[3] * weights[3]
        ).clip(0, 1)
        return main_force_holding_strength

