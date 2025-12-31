# 文件: strategies/trend_following/intelligence/fusion_intelligence.py
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, get_adaptive_mtf_normalized_score, normalize_score, normalize_to_bipolar

class FusionIntelligence:
    """
    【V3.0 · 战场态势引擎】
    - 核心重构: 遵循“联合情报部”职责，废弃所有旧方法。不再消费原始指标，
                  只消费各原子情报层输出的“公理级”信号。
    - 核心职责: 将各领域情报“冶炼”成四大客观战场态势：
                  1. 市场政权 (Market Regime): 判断趋势市 vs 震荡市。
                  2. 趋势质量 (Trend Quality): 评估趋势的健康度与共识度。
                  3. 市场压力 (Market Pressure): 衡量向上与向下的反转压力。
                  4. 资本对抗 (Capital Confrontation): 洞察主力与散户的博弈格局。
    - 定位: 连接“感知”与“认知”的关键桥梁，为认知层提供决策依据。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        # 新增：从独立的 fusion.json 文件加载配置
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..'))
        fusion_config_path = os.path.join(project_root, 'config', 'intelligence', 'fusion.json')
        try:
            with open(fusion_config_path, 'r', encoding='utf-8') as f:
                self.params = json.load(f).get('fusion_intelligence_params', {})
        except FileNotFoundError:
            print(f"警告: 未找到融合情报配置文件 {fusion_config_path}，使用默认空配置。")
            self.params = {}
        except json.JSONDecodeError:
            print(f"警告: 融合情报配置文件 {fusion_config_path} 解析失败，使用默认空配置。")
            self.params = {}

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回一个包含NaN的Series。
        """
        if column_name not in df.columns:
            print(f"    -> [融合情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，将返回NaN Series。")
            # 返回一个包含NaN的Series，以便后续计算传播NaN，更容易发现问题
            return pd.Series(np.nan, index=df.index, dtype=np.float32)
        return df[column_name].astype(np.float32)

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> pd.Series:
        """
        【V1.5 · NaN传播版】安全地从原子状态库或主数据帧中获取分数。
        - 核心职责: 统一信号获取路径，优先从 self.strategy.atomic_states 获取，
                      若无则从 self.strategy.df_indicators 获取，最后返回一个包含NaN的Series，
                      确保数据流的稳定性，并暴露缺失问题。
        - 【V1.5 修复】接收 df 参数，并使用其索引创建默认 Series，确保上下文一致。
        - 【V1.5 变更】当信号不存在时，返回NaN Series，而不是默认值0.0，以暴露问题。
        """
        is_debug_enabled, probe_ts, method_name = debug_info if debug_info else (False, None, "未知方法")
        score_series = None
        if name in self.strategy.atomic_states:
            score_series = self.strategy.atomic_states[name]
        elif name in self.strategy.df_indicators.columns:
            score_series = self.strategy.df_indicators[name]
        if score_series is not None:
            if not score_series.index.equals(df.index):
                if is_debug_enabled and probe_ts and probe_ts in df.index:
                    print(f"      [融合层-原子信号调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 信号 '{name}' 索引不一致，正在重新对齐。")
                score_series = score_series.reindex(df.index).fillna(np.nan) # 重新对齐后用NaN填充
            return score_series.astype(np.float32)
        else:
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层-原子信号调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 预期原子信号 '{name}' 在 atomic_states 和 df_indicators 中均不存在，返回NaN Series。")
            return pd.Series(np.nan, index=df.index, dtype=np.float32)

    def _get_normalized_risk_score(self, df: pd.DataFrame, signal_name: str, norm_window: int, default_value: float = 0.0, mtf_norm_weights: Dict = None, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> pd.Series:
        """
        【V1.3 · NaN传播与MTF归一化版】获取并归一化风险信号。处理双极性信号的正负部分。
        - 【V1.3 变更】当原始信号为NaN时，归一化后也为NaN，以暴露问题。
        """
        is_debug_enabled, probe_ts, method_name = debug_info if debug_info else (False, None, "未知方法")
        is_positive_part = False
        is_negative_part = False
        base_signal_name = signal_name
        if signal_name.endswith('_POSITIVE'):
            is_positive_part = True
            base_signal_name = signal_name.replace('_POSITIVE', '')
        elif signal_name.endswith('_NEGATIVE'):
            is_negative_part = True
            base_signal_name = signal_name.replace('_NEGATIVE', '')
        original_signal = self._get_atomic_score(df, base_signal_name, default_value, debug_info)
        if is_positive_part:
            processed_signal = original_signal.clip(lower=0)
        elif is_negative_part:
            processed_signal = original_signal.clip(upper=0).abs()
        else:
            processed_signal = original_signal
        # 确保NaN值在归一化前被保留，而不是被fillna(0.0)掩盖
        processed_signal_no_nan_fill = processed_signal
        if mtf_norm_weights and mtf_norm_weights.get('enabled', False):
            normalized_score = get_adaptive_mtf_normalized_score(processed_signal_no_nan_fill, df.index, ascending=True, tf_weights=mtf_norm_weights.get('weights'))
        else:
            normalized_score = normalize_score(processed_signal_no_nan_fill, df.index, windows=norm_window, ascending=True)
        # 归一化后，如果原始信号为NaN，则归一化结果也应为NaN
        normalized_score = normalized_score.where(~original_signal.isna(), np.nan)
        # 只有当原始信号非常接近0时，才强制归零，避免浮点数误差导致微小值被放大
        zero_processed_mask = (processed_signal.abs() < 1e-6)
        normalized_score.loc[zero_processed_mask] = 0.0
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"        [融合层-风险归一化调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 信号 '{signal_name}' (原始: {original_signal.loc[probe_ts]:.4f}, 处理后: {processed_signal.loc[probe_ts]:.4f}) -> 归一化分数: {normalized_score.loc[probe_ts]:.4f}")
        return normalized_score.astype(np.float32)

    def run_fusion_diagnostics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.0 · 依赖彻底解耦与探针增强版】融合情报分析总指挥
        - 核心重构: 遵循“联合情报部”职责，废弃所有旧方法。不再消费原始指标，
                    只消费各原子情报层输出的“公理级”信号。
        - 核心职责: 将各领域情报“冶炼”成四大客观战场态势：
                    1. 市场政权 (Market Regime): 判断趋势市 vs 震荡市。
                    2. 趋势质量 (Trend Quality): 评估趋势的健康度与共识度。
                    3. 市场压力 (Market Pressure): 衡量向上与向下的反转压力。
                    4. 资本对抗 (Capital Confrontation): 洞察主力与散户的博弈格局。
        - 定位: 连接“感知”与“认知”的关键桥梁，为认知层提供决策依据。
        - 【V7.0 增强】所有 `_synthesize_` 方法内部对 `FUSION_*` 信号的依赖已彻底解耦，
                      替换为基础信号。因此，`run_fusion_diagnostics` 中的调用顺序可以简化，
                      不再需要复杂的阶段划分来解决循环依赖。
                      引入调试探针机制，将调试信息传递给所有内部调用的方法。
        """
        print("启动【V7.0 · 依赖彻底解耦与探针增强版】融合情报分析...")
        all_fusion_states = {}
        # 修正：debug_params 应该从 self.strategy 中获取，而不是 self.params
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        probe_ts = None
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            if valid_probe_dates:
                probe_ts = valid_probe_dates[0]
        debug_info = (is_debug_enabled, probe_ts, "run_fusion_diagnostics")
        # --- 融合信号计算 (现在所有融合信号都只依赖基础信号，因此可以按任意顺序调用) ---
        # 建议按逻辑分组，例如先计算环境类，再计算风险类，最后机会类
        # 环境类融合信号
        regime_states = self._synthesize_market_regime(df, debug_info)
        all_fusion_states.update(regime_states)
        self.strategy.atomic_states.update(regime_states)
        confrontation_states = self._synthesize_capital_confrontation(df, debug_info)
        all_fusion_states.update(confrontation_states)
        self.strategy.atomic_states.update(confrontation_states)
        trend_structure_states = self._synthesize_trend_structure_score(df, debug_info)
        all_fusion_states.update(trend_structure_states)
        self.strategy.atomic_states.update(trend_structure_states)
        chip_trend_states = self._synthesize_chip_trend(df, debug_info)
        all_fusion_states.update(chip_trend_states)
        self.strategy.atomic_states.update(chip_trend_states)
        market_contradiction_states = self._synthesize_market_contradiction(df, debug_info)
        all_fusion_states.update(market_contradiction_states)
        self.strategy.atomic_states.update(market_contradiction_states)
        # 风险类融合信号
        stagnation_risk_states = self._synthesize_stagnation_risk(df, debug_info)
        all_fusion_states.update(stagnation_risk_states)
        self.strategy.atomic_states.update(stagnation_risk_states)
        distribution_pressure_states = self._synthesize_distribution_pressure(df, debug_info)
        all_fusion_states.update(distribution_pressure_states)
        self.strategy.atomic_states.update(distribution_pressure_states)
        price_overextension_intent_states = self._synthesize_price_overextension_intent(df, debug_info)
        all_fusion_states.update(price_overextension_intent_states)
        self.strategy.atomic_states.update(price_overextension_intent_states)
        trend_exhaustion_states = self._synthesize_trend_exhaustion_syndrome(df, debug_info)
        all_fusion_states.update(trend_exhaustion_states)
        self.strategy.atomic_states.update(trend_exhaustion_states)
        # 机会类融合信号
        micro_conviction_states = self._synthesize_micro_conviction(df, debug_info)
        all_fusion_states.update(micro_conviction_states)
        self.strategy.atomic_states.update(micro_conviction_states)
        contested_accumulation_states = self._synthesize_contested_accumulation(df, debug_info)
        all_fusion_states.update(contested_accumulation_states)
        self.strategy.atomic_states.update(contested_accumulation_states)
        accumulation_inflection_states = self._synthesize_accumulation_inflection(df, debug_info)
        all_fusion_states.update(accumulation_inflection_states)
        self.strategy.atomic_states.update(accumulation_inflection_states)
        accumulation_playbook_states = self._synthesize_accumulation_playbook(df, debug_info)
        all_fusion_states.update(accumulation_playbook_states)
        self.strategy.atomic_states.update(accumulation_playbook_states)
        # 综合类融合信号 (可能依赖其他融合信号，但现在这些依赖都已是基础信号)
        fund_flow_trend_states = self._synthesize_fund_flow_trend(df, debug_info)
        all_fusion_states.update(fund_flow_trend_states)
        self.strategy.atomic_states.update(fund_flow_trend_states)
        market_pressure_states = self._synthesize_market_pressure(df, debug_info)
        all_fusion_states.update(market_pressure_states)
        self.strategy.atomic_states.update(market_pressure_states)
        liquidity_dynamics_states = self._synthesize_liquidity_dynamics(df, debug_info)
        all_fusion_states.update(liquidity_dynamics_states)
        self.strategy.atomic_states.update(liquidity_dynamics_states)
        # 趋势质量 (Trend Quality) - 依赖多个风险信号和微观信念，现在这些都已计算完毕
        quality_states = self._synthesize_trend_quality(df, debug_info)
        all_fusion_states.update(quality_states)
        self.strategy.atomic_states.update(quality_states)
        # --- 新增的融合信号调用 ---
        multi_layer_reversal_states = self._synthesize_multi_layer_reversal_confirmation(df, debug_info)
        all_fusion_states.update(multi_layer_reversal_states)
        self.strategy.atomic_states.update(multi_layer_reversal_states)
        breakout_sustainability_states = self._synthesize_breakout_sustainability(df, debug_info)
        all_fusion_states.update(breakout_sustainability_states)
        self.strategy.atomic_states.update(breakout_sustainability_states)
        intraday_strategic_alignment_states = self._synthesize_intraday_strategic_alignment(df, debug_info)
        all_fusion_states.update(intraday_strategic_alignment_states)
        self.strategy.atomic_states.update(intraday_strategic_alignment_states)
        capital_dominance_shift_states = self._synthesize_capital_dominance_shift(df, debug_info)
        all_fusion_states.update(capital_dominance_shift_states)
        self.strategy.atomic_states.update(capital_dominance_shift_states)
        hidden_accumulation_quality_states = self._synthesize_hidden_accumulation_quality(df, debug_info)
        all_fusion_states.update(hidden_accumulation_quality_states)
        self.strategy.atomic_states.update(hidden_accumulation_quality_states)
        market_structure_health_states = self._synthesize_market_structure_health(df, debug_info)
        all_fusion_states.update(market_structure_health_states)
        self.strategy.atomic_states.update(market_structure_health_states)
        sector_leadership_dynamics_states = self._synthesize_sector_leadership_dynamics(df, debug_info)
        all_fusion_states.update(sector_leadership_dynamics_states)
        self.strategy.atomic_states.update(sector_leadership_dynamics_states)
        print(f"【V7.0 · 依赖彻底解耦与探针增强版】分析完成，生成 {len(all_fusion_states)} 个融合态势信号。")
        return all_fusion_states

    def _synthesize_market_contradiction(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V5.2 · 情境审判强化版 - 依赖解耦】冶炼“市场矛盾” (Market Contradiction)
        - 核心升华: 引入“趋势质量”作为上下文法官，对“原始矛盾”进行非线性审判。
        - 终章心法: 矛盾的威力，取决于其与趋势的冲突程度。逆势之兆，罪加一等。
                      此法之后，再无增益。
        - 【V5.2 增强】将对融合信号 `FUSION_BIPOLAR_TREND_QUALITY` 的依赖替换为基础信号 `SCORE_STRUCT_AXIOM_TREND_FORM`，以解耦依赖。
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_market_contradiction"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“市场矛盾”...")
        states = {}
        df_index = df.index
        # 1. [V4.0 核心保留] 计算原始矛盾分 (迭代共振)
        divergence_sources = {
            'CHIP': 0.30, 'FUND_FLOW': 0.25, 'BEHAVIOR': 0.15,
            'DYNAMIC_MECHANICS': 0.10, 'STRUCTURE': 0.10,
            'PATTERN': 0.05, 'MICRO_BEHAVIOR': 0.05,
        }
        bullish_resonance_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        bearish_resonance_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # 特殊处理双极性的筹码背离信号
        chip_divergence = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_DIVERGENCE', 0.0, debug_info)
        chip_weight = divergence_sources.pop('CHIP')
        weighted_bullish_chip = chip_divergence.clip(lower=0) * chip_weight
        weighted_bearish_chip = chip_divergence.clip(upper=0).abs() * chip_weight
        bullish_resonance_score = bullish_resonance_score + weighted_bullish_chip - (bullish_resonance_score * weighted_bullish_chip)
        bearish_resonance_score = bearish_resonance_score + weighted_bearish_chip - (bearish_resonance_score * weighted_bearish_chip)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 筹码背离 (原始: {chip_divergence.loc[probe_ts]:.4f}) -> 看涨贡献: {weighted_bullish_chip.loc[probe_ts]:.4f}, 看跌贡献: {weighted_bearish_chip.loc[probe_ts]:.4f}")
        # 迭代处理其他单极性背离信号
        for source, weight in divergence_sources.items():
            if source == 'BEHAVIOR':
                bull_signal_name = 'SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY'
                bear_signal_name = 'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY'
            else:
                bull_signal_name = f'SCORE_{source}_BULLISH_DIVERGENCE'
                bear_signal_name = f'SCORE_{source}_BEARISH_DIVERGENCE'
            bull_signal = self._get_atomic_score(df, bull_signal_name, 0.0, debug_info)
            bear_signal = self._get_atomic_score(df, bear_signal_name, 0.0, debug_info)
            weighted_bull_signal = bull_signal * weight
            bullish_resonance_score = bullish_resonance_score + weighted_bull_signal - (bullish_resonance_score * weighted_bull_signal)
            weighted_bear_signal = bear_signal * weight
            bearish_resonance_score = bearish_resonance_score + weighted_bear_signal - (bearish_resonance_score * weighted_bear_signal)
        raw_bipolar_contradiction = (bullish_resonance_score - bearish_resonance_score).clip(-1, 1)
        # 2. 核心数学逻辑 - 情境审判
        # 2.1 获取情境法官 - 趋势质量 (替换为基础信号)
        trend_quality = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0, debug_info)
        # 2.2 计算冲突度 (只有当矛盾与趋势方向相反时，才产生冲突)
        # np.sign(raw_bipolar_contradiction) * trend_quality 得到的是矛盾与趋势的同向性
        # 负号表示当矛盾与趋势方向相反时，冲突度为正
        conflict_score = (-np.sign(raw_bipolar_contradiction) * trend_quality).clip(lower=0)
        # 2.3 构建情境调节器
        modulation_factor = 0.5 # 冲突调节系数
        contextual_modulator = 1 + (conflict_score * modulation_factor)
        # 2.4 最终审判
        final_score = (raw_bipolar_contradiction * contextual_modulator).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_CONTRADICTION'] = final_score.astype(np.float32)
        print(f"  -- [融合层] “市场矛盾”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_market_regime(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.3 · 上下文修复与探针增强版】冶炼“市场政权” (Market Regime)
        - 【V1.3 修复】修正方法签名以接受 debug_info 参数。
        - 【V1.3 增强】增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_market_regime"
        # 修正此处：检查 debug_info 是否为 None
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        states = {}
        df_index = df.index
        # 获取原始信号，并传递 debug_info
        hurst_memory = self._get_atomic_score(df, 'SCORE_CYCLICAL_HURST_MEMORY', 0.0, debug_info)
        inertia = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_INERTIA', 0.0, debug_info)
        stability = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0, debug_info)
        # 趋势证据融合
        trend_evidence_weights = {'hurst': 0.4, 'inertia': 0.4, 'stability': 0.2}
        trend_evidence = (
            hurst_memory * trend_evidence_weights['hurst'] +
            inertia * trend_evidence_weights['inertia'] +
            stability * trend_evidence_weights['stability']
        ).clip(-1, 1)
        # 均值回归证据融合
        # 均值回归证据：当Hurst记忆性为负（均值回归倾向），且惯性为负（趋势难以维持）时，均值回归证据增强
        reversion_evidence = (hurst_memory.clip(upper=0).abs() * inertia.clip(upper=0).abs()).pow(0.5)
        # 双极性政权分数：趋势证据减去均值回归证据
        bipolar_regime = (trend_evidence - reversion_evidence).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_REGIME'] = bipolar_regime.astype(np.float32)
        print(f"  -- [融合层] “市场政权”冶炼完成，最新分值: {bipolar_regime.iloc[-1]:.4f}")
        return states

    def _synthesize_stagnation_risk(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V3.7 · 内腐外强版 - 依赖解耦与强化】冶炼“滞涨风险” (FUSION_RISK_STAGNATION)
        - 核心重构: 废弃V2.1的“症状清单”模型，引入“内腐外强”背离审判模型。
        - 核心公式: 滞涨风险 = (内部腐化度 × 外部强势幻象)^(1/2)
        - 诡道哲学: 最大的风险，源于内部趋势质量的腐化与外部价格强势的假象之间的
                      致命背离，此乃“温水煮蛙”之局。
        - 【V3.7 增强】将对融合信号 `FUSION_BIPOLAR_TREND_QUALITY` 和 `FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT` 的依赖替换为基础信号 `SCORE_STRUCT_AXIOM_TREND_FORM` 和 `INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW`，以解耦依赖。
                      增加 `SCORE_CHIP_RETAIL_VULNERABILITY` 和 `SCORE_RISK_LIQUIDITY_DRAIN` 信号，提高内部腐化度的准确性。
                      调整 `internal_decay_components` 权重，确保高风险信号能有效贡献，避免被过多零值信号稀释。
                      **将 `internal_decay_score` 的计算方式从加权几何平均改为加权算术平均，以更准确地聚合内部腐化信号。**
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_stagnation_risk"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“滞涨风险”...")
        states = {}
        df_index = df.index
        # --- 1. 信号升维：定义“内腐”与“外强”两大阵营 ---
        # 阵营一：内部腐化度 (Internal Decay) - 趋势的内在病根
        # 替换 FUSION_BIPOLAR_TREND_QUALITY 为 SCORE_STRUCT_AXIOM_TREND_FORM
        trend_quality_base = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0, debug_info)
        trend_quality_decay = -trend_quality_base.diff(1).fillna(0.0).clip(upper=0) # 核心病根：趋势质量衰减
        distribution_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0, debug_info)
        fund_flow_bearish = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0, debug_info).clip(upper=0).abs()
        chip_dispersion = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info).clip(upper=0).abs()
        stagnation_evidence = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0, debug_info)
        # 新增信号：散户筹码脆弱性和流动性枯竭风险
        retail_vulnerability = self._get_atomic_score(df, 'SCORE_CHIP_RETAIL_VULNERABILITY', 0.0, debug_info)
        liquidity_drain_risk = self._get_atomic_score(df, 'SCORE_RISK_LIQUIDITY_DRAIN', 0.0, debug_info)
        # 阵营二：外部强势幻象 (External Strength Illusion) - 迷惑性的表象
        # 替换 FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT 为 INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW
        price_overextension = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.0, debug_info).clip(lower=0) # 仅取正向超买部分
        # 修正：替换 rally_distribution_pressure_D 为 SCORE_BEHAVIOR_DISTRIBUTION_INTENT
        profit_taking_supply = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0, debug_info).clip(0, 1)
        # 修正：替换 retail_fomo_premium_index_D 为 SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM 的正向部分
        retail_fomo = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0, debug_info).clip(lower=0).clip(0, 1)
        # --- 2. 核心数学逻辑 - 背离审判 ---
        # 2.1 计算“内部腐化度” (加权算术平均，体现症状累加)
        # 调整权重，提高散户筹码脆弱性和流动性枯竭风险的权重
        internal_decay_components = {
            '趋势质量衰减': (trend_quality_decay, 0.15), # 调整权重
            '派发意图': (distribution_intent, 0.15),
            '资金流出': (fund_flow_bearish, 0.10),
            '筹码分散': (chip_dispersion, 0.10),
            '微观滞涨': (stagnation_evidence, 0.10),
            '散户筹码脆弱性': (retail_vulnerability, 0.20), # 提高权重
            '流动性枯竭风险': (liquidity_drain_risk, 0.20), # 提高权重
        }
        internal_decay_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_weight_sum = 0.0
        for name, (comp, weight) in internal_decay_components.items():
            # 确保每个组件在参与算术平均前，填充NaN为0，并裁剪到 [0, 1] 范围
            comp_processed = comp.fillna(0.0).clip(0, 1)
            internal_decay_score += comp_processed * weight
            total_weight_sum += weight
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"        [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 内部腐化组件 '{name}' (值: {comp_processed.loc[probe_ts]:.4f}, 权重: {weight})")
        
        if total_weight_sum > 0:
            internal_decay_score /= total_weight_sum
        else:
            internal_decay_score = pd.Series(0.0, index=df_index, dtype=np.float32) # 避免除以零
        
        internal_decay_score = internal_decay_score.clip(0, 1) # 确保最终分数在 [0, 1] 范围内

        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 内部腐化度 (internal_decay_score): {internal_decay_score.loc[probe_ts]:.4f}")
        # 2.2 计算“外部强势幻象” (加权平均)
        external_illusion_score = (
            price_overextension * 0.4 +
            retail_fomo * 0.4 +
            profit_taking_supply * 0.2
        ).clip(0, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 外部强势幻象 (external_illusion_score): {external_illusion_score.loc[probe_ts]:.4f}")
        # 2.3 最终融合：内腐 × 外强
        raw_stagnation_risk = (internal_decay_score * external_illusion_score).pow(0.5).fillna(0.0)
        # 【V3.1 变更】移除 is_price_stagnant_or_rising 过滤器
        final_stagnation_risk = raw_stagnation_risk.clip(0, 1)
        states['FUSION_RISK_STAGNATION'] = final_stagnation_risk.astype(np.float32)
        print(f"  -- [融合层] “滞涨风险”冶炼完成，最新分值: {final_stagnation_risk.iloc[-1]:.4f}")
        return states

    def _synthesize_capital_confrontation(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V5.6 · 意志与环境版 - 风险强化与强化】冶炼“资本对抗” (Capital Confrontation)
        - 核心升华: 彻底分离“内因”(主力意志)与“外缘”(对手盘环境)，构建“战役意志 ×
                      环境调节器”的“天人合一”终极模型。
        - 终章心法: 以我心，应天时。内因驱动，外缘催化，方为博弈大道。此法之后，再无增益。
        - 【V5.6 增强】增加 `SCORE_CHIP_MAIN_FORCE_COST_INTENT` 和 `SCORE_INTRADAY_DOMINANCE_CONSENSUS` 信号，提高资本对抗的准确性。
                      **根据 `SCORE_BEHAVIOR_DECEPTION_INDEX` 的正负值（诱空/诱多）明确应用奖励或惩罚，以更精准地反映主力意图。**
                      进一步增强 `environment_modulator` 对散户贪婪的惩罚力度。
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_capital_confrontation"
        # 修正此处：检查 debug_info 是否为 None
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“资本对抗”...")
        states = {}
        df_index = df.index
        # 1. 信号升维：定义“内因”与“外缘”
        # --- 内因 (Internal Cause): 主力的“战役意志” ---
        ff_posture = self._get_atomic_score(df, 'SCORE_FF_STRATEGIC_POSTURE', 0.0, debug_info)
        chip_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info)
        main_force_cost_intent = self._get_atomic_score(df, 'SCORE_CHIP_MAIN_FORCE_COST_INTENT', 0.0, debug_info).clip(lower=0) # 新增主力成本区攻防意图正向
        main_force_intent = (ff_posture * 0.4 + chip_posture * 0.4 + main_force_cost_intent * 0.2).clip(-1, 1) # 调整权重
        tactical_execution = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0, debug_info)
        intraday_dominance = self._get_atomic_score(df, 'SCORE_INTRADAY_DOMINANCE_CONSENSUS', 0.0, debug_info).clip(lower=0) # 新增日内支配共识正向
        # 引入欺骗指数作为对主力意志的直接惩罚或奖励
        deception_index = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0, debug_info)
        # 根据用户明确的定义：
        # deception_index 负值是诱多，代表主力出货 -> 惩罚
        # deception_index 正值是诱空，代表主力意图拉升 -> 奖励
        deception_penalty = deception_index.clip(upper=0).abs() * 1.5 # 诱多（负值）时产生惩罚，系数1.5，可调
        deception_reward = deception_index.clip(lower=0) * 0.5 # 诱空（正值）时产生奖励，系数0.5，可调
        # --- 外缘 (External Condition): 对手盘的“环境” ---
        sentiment_pendulum = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0, debug_info)
        counterparty_state = -sentiment_pendulum # 散户情绪越狂热（sentiment_pendulum为正），对手盘状态越负面（风险越高）
        # 2. 核心数学逻辑 - 天人合一
        # 2.1 融合“内因”，计算“战役意志”
        mapped_intent = main_force_intent + 1 # 映射到 [0, 2]
        mapped_execution = (tactical_execution * 0.7 + intraday_dominance * 0.3) + 1 # 融合日内支配共识，映射到 [0, 2]
        will_mapped = (mapped_intent.clip(lower=1e-9) * mapped_execution.clip(lower=1e-9)).pow(1/2)
        campaign_will = (will_mapped - 1).clip(-1, 1)
        # 应用欺骗惩罚和奖励到战役意志
        # 惩罚是乘法削弱，奖励是加法增强
        campaign_will_modulated = (campaign_will * (1 - deception_penalty) + deception_reward).clip(-1, 1)
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力意图 (main_force_intent): {main_force_intent.loc[probe_ts]:.4f}, 战役意志 (campaign_will): {campaign_will.loc[probe_ts]:.4f}")
        #     print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 欺骗指数 (deception_index): {deception_index.loc[probe_ts]:.4f}, 欺骗惩罚 (deception_penalty): {deception_penalty.loc[probe_ts]:.4f}, 欺骗奖励 (deception_reward): {deception_reward.loc[probe_ts]:.4f}")
        #     print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 调制后战役意志 (campaign_will_modulated): {campaign_will_modulated.loc[probe_ts]:.4f}")
        # 2.2 构建“环境调节器”
        # 进一步提高环境调节器的敏感度，使散户贪婪的惩罚更显著
        modulation_factor = 1.8 # 再次提高环境调节系数，从1.5提高到1.8
        # 当散户情绪狂热（sentiment_pendulum > 0），counterparty_state < 0，modulator < 1，惩罚
        # 当散户情绪恐慌（sentiment_pendulum < 0），counterparty_state > 0，modulator > 1，奖励
        environment_modulator = (1 + counterparty_state * modulation_factor).clip(0, 2)
        # if is_debug_enabled and probe_ts and probe_ts in df.index:
        #     print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 对手盘状态 (counterparty_state): {counterparty_state.loc[probe_ts]:.4f}, 环境调节器 (environment_modulator): {environment_modulator.loc[probe_ts]:.4f}")
        # 2.3 最终决断: 战役意志(我) × 环境调节器(天)
        final_score = (campaign_will_modulated * environment_modulator).clip(-1, 1)
        states['FUSION_BIPOLAR_CAPITAL_CONFRONTATION'] = final_score.astype(np.float32)
        print(f"  -- [融合层] “资本对抗”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_price_overextension_intent(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V4.5 · 天道裁决版 - 依赖解耦与强化】冶炼“价格超买意图” (Price Overextension Intent)
        - 核心升华: 引入“趋势质量”作为天道背景，对“三位一体”的初审判决进行终极裁决。
        - 终章心法: 罪罚需与时势相符。逆势之罪，天道加诛。此法之后，再无增益。
        - 【V4.5 增强】将对融合信号 `FUSION_BIPOLAR_TREND_QUALITY` 的依赖替换为基础信号 `SCORE_STRUCT_AXIOM_TREND_FORM`，以解耦依赖。
                      增强 `overextension_bearish` 对高位诱多行为的敏感度。
                      增加 `SCORE_CHIP_RISK_DISTRIBUTION_WHISPER` 和 `SCORE_RISK_BREAKOUT_FAILURE_CASCADE` 信号，提高风险识别的准确性。
                      调整 `bearish_pressure_sources` 权重，确保风险信号能有效贡献。
                      修正 `bearish_pressure` 计算中 `NaN` 传播问题，将缺失信号贡献视为 0。
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_price_overextension_intent"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“价格超买意图”...")
        states = {}
        df_index = df.index
        # --- 第1步: 初审法庭 (V3.0核心保留) ---
        # 1.1 定义“天时、人和、地利”的原料
        overbought_state_sources = {'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM': 0.6, 'SCORE_STRUCT_AXIOM_MTF_COHESION': 0.4}
        # 增强 bearish_intent_sources
        bearish_intent_sources = {
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT': 0.3, # 调整权重
            'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY': 0.2, # 调整权重
            'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW': 0.2, # 调整权重
            'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER': 0.3 # 新增筹码派发诡影
        }
        bullish_intent_sources = {'SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK': 0.4, 'SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY': 0.3, 'SCORE_BEHAVIOR_ABSORPTION_STRENGTH': 0.3}
        # 增强 bearish_pressure_sources，调整权重
        # 修正：将信号名称改为不带 _POSITIVE 或 _NEGATIVE 后缀的原始信号名称
        bearish_pressure_sources = {
            'SCORE_FF_AXIOM_CONSENSUS': {'weight': 0.3, 'part': 'negative'}, # 资金流共识负向
            'SCORE_CHIP_STRATEGIC_POSTURE': {'weight': 0.3, 'part': 'negative'}, # 筹码战略态势负向
            'SCORE_RISK_BREAKOUT_FAILURE_CASCADE': {'weight': 0.2, 'part': 'positive'}, # 突破失败级联风险 (本身是风险，取正向)
            'SCORE_BEHAVIOR_DECEPTION_INDEX': {'weight': 0.2, 'part': 'positive'} # 欺骗指数正向，作为诱多压力
        }
        bullish_pressure_sources = {'SCORE_OPPORTUNITY_SELLING_EXHAUSTION': 0.6, 'SCORE_FF_AXIOM_CONSENSUS': 0.4}
        # 1.2 计算“天时”分 (状态)
        sentiment_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0, debug_info)
        cohesion_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_MTF_COHESION', 0.0, debug_info)
        # 情绪钟摆和MTF协同都为正时，代表市场狂热且结构协同，是超买状态
        overbought_state = (sentiment_score.clip(lower=0) * overbought_state_sources['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'] + cohesion_score.clip(lower=0) * overbought_state_sources['SCORE_STRUCT_AXIOM_MTF_COHESION']).clip(0,1)
        # 情绪钟摆和MTF协同都为负时，代表市场恐慌且结构混乱，是超卖状态
        oversold_state = (sentiment_score.clip(upper=0).abs() * overbought_state_sources['SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'] + cohesion_score.clip(upper=0).abs() * overbought_state_sources['SCORE_STRUCT_AXIOM_MTF_COHESION']).clip(0,1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 情绪钟摆: {sentiment_score.loc[probe_ts]:.4f}, MTF协同: {cohesion_score.loc[probe_ts]:.4f}")
            print(f"        超买状态 (overbought_state): {overbought_state.loc[probe_ts]:.4f}, 超卖状态 (oversold_state): {oversold_state.loc[probe_ts]:.4f}")
        # 1.3 计算“人和”分 (意图)
        bearish_intent = pd.Series(0.0, index=df_index, dtype=np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 熊市意图原始信号 ---")
        for signal, weight in bearish_intent_sources.items():
            score = self._get_atomic_score(df, signal, 0.0, debug_info).fillna(0.0) # 缺失信号贡献为0
            bearish_intent += score * weight
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"        {signal}: {score.loc[probe_ts]:.4f}")
        bearish_intent = bearish_intent.clip(0,1)
        bullish_intent = pd.Series(0.0, index=df_index, dtype=np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 牛市意图原始信号 ---")
        for signal, weight in bullish_intent_sources.items():
            score = self._get_atomic_score(df, signal, 0.0, debug_info).fillna(0.0) # 缺失信号贡献为0
            bullish_intent += score * weight
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"        {signal}: {score.loc[probe_ts]:.4f}")
        bullish_intent = bullish_intent.clip(0,1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 派发意图 (bearish_intent): {bearish_intent.loc[probe_ts]:.4f}, 吸筹意图 (bullish_intent): {bullish_intent.loc[probe_ts]:.4f}")
        # 1.4 计算“地利”分 (压力)
        # 资金流共识负向和筹码战略态势负向代表熊市压力
        bearish_pressure = pd.Series(0.0, index=df_index, dtype=np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 熊市压力原始信号 ---")
        for signal_name, config in bearish_pressure_sources.items(): # 修正：遍历新的字典结构
            weight = config['weight']
            part = config['part']
            score = self._get_atomic_score(df, signal_name, 0.0, debug_info).fillna(0.0) # 获取原始信号
            if part == 'negative':
                bearish_pressure += score.clip(upper=0).abs() * weight
            elif part == 'positive':
                bearish_pressure += score.clip(lower=0) * weight
            else: # 默认情况，直接使用
                bearish_pressure += score * weight
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"        {signal_name} ({part}): {score.loc[probe_ts]:.4f}")
        bearish_pressure = bearish_pressure.clip(0,1)
        # 卖盘衰竭和资金流共识正向代表牛市压力（即卖压小，买压大）
        bullish_pressure = pd.Series(0.0, index=df_index, dtype=np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 牛市压力原始信号 ---")
        # 确保 SCORE_OPPORTUNITY_SELLING_EXHAUSTION 和 SCORE_FF_AXIOM_CONSENSUS 存在且 NaN 填充为 0
        selling_exhaustion_score = self._get_atomic_score(df, 'SCORE_OPPORTUNITY_SELLING_EXHAUSTION', 0.0, debug_info).fillna(0.0)
        ff_consensus_score = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0, debug_info).fillna(0.0)
        bullish_pressure = (selling_exhaustion_score * bullish_pressure_sources['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] + ff_consensus_score.clip(lower=0) * bullish_pressure_sources['SCORE_FF_AXIOM_CONSENSUS']).clip(0,1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"        SCORE_OPPORTUNITY_SELLING_EXHAUSTION: {selling_exhaustion_score.loc[probe_ts]:.4f}")
            print(f"        SCORE_FF_AXIOM_CONSENSUS: {ff_consensus_score.loc[probe_ts]:.4f}")
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 熊市压力 (bearish_pressure): {bearish_pressure.loc[probe_ts]:.4f}, 牛市压力 (bullish_pressure): {bullish_pressure.loc[probe_ts]:.4f}")
        # 1.5 三位一体融合，得出“初审判决”
        # 增强 overextension_bearish 对高位诱多行为的敏感度
        overextension_bearish = (overbought_state * bearish_intent * bearish_pressure).pow(1/3).fillna(0.0).clip(0, 1)
        overextension_bullish = (oversold_state * bullish_intent * bullish_pressure).pow(1/3).fillna(0.0).clip(0, 1)
        raw_bipolar_intent = (overextension_bullish - overextension_bearish).clip(-1, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 初审判决 - 熊市超买: {overextension_bearish.loc[probe_ts]:.4f}, 牛市超卖: {overextension_bullish.loc[probe_ts]:.4f} -> 原始双极意图: {raw_bipolar_intent.loc[probe_ts]:.4f}")
        # --- 第2步: 天道裁决 ---
        # 2.1 获取天道背景 - 趋势质量 (替换为基础信号)
        trend_quality = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0, debug_info).fillna(0.0) # 缺失信号贡献为0
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 天道背景 - 趋势质量 (基础信号): {trend_quality.loc[probe_ts]:.4f}")
        # 2.2 计算冲突度 (只有当反转意图与趋势大势相反时，才产生冲突)
        conflict_score = (-np.sign(raw_bipolar_intent) * trend_quality).clip(lower=0)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 冲突分数 (矛盾与趋势反向): {conflict_score.loc[probe_ts]:.4f}")
        # 2.3 构建情境调节器
        modulation_factor = 0.5 # 冲突调节系数
        contextual_modulator = 1 + (conflict_score * modulation_factor)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 情境调节器: {contextual_modulator.loc[probe_ts]:.4f}")
        # 2.4 终审裁决
        final_score = (raw_bipolar_intent * contextual_modulator).clip(-1, 1)
        states['FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “价格超买意图”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “价格超买意图”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_structure_score(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V2.2 · 四象共振与探针增强版 - 强化】冶炼“趋势结构分” (FUSION_BIPOLAR_TREND_STRUCTURE_SCORE)
        - 核心重构: 废弃V1.x基于底层技术指标的算术模型，引入基于四大情报域顶层信号的
                      “四象共振”非线性融合模型。
        - 诡道哲学: 基于“木桶效应”，采用几何平均融合。一个健康的趋势结构，必须是结构、
                      力学、筹码、资金流四大支柱的共振，任何一环的缺失都将导致整体崩塌。
        - 【V2.2 增强】增加 `SCORE_FOUNDATION_STRATEGIC_POSTURE` 信号，作为第五个支柱，提高趋势结构分的准确性。
                      修正方法签名以接受 debug_info 参数，并增加详细探针。
        """
        method_name = "_synthesize_trend_structure_score"
        # 修正此处：检查 debug_info 是否为 None
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“趋势结构分”...")
        states = {}
        df_index = df.index
        # 1. 信号升维：定义五大支柱，引用各情报域的顶层信号
        five_pillars = {
            'structure': 'SCORE_STRUCT_STRATEGIC_POSTURE', # 结构支柱 (骨)
            'dynamics': 'SCORE_DYN_GRAND_UNIFICATION',     # 力学支柱 (势)
            'chip': 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY',    # 筹码支柱 (基)
            'fund_flow': 'SCORE_FF_STRATEGIC_POSTURE',      # 资金支柱 (血)
            'foundation': 'SCORE_FOUNDATION_STRATEGIC_POSTURE' # 新增基础支柱 (根)
        }
        # 2. 获取各支柱的原子信号分
        pillar_scores = {}
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始支柱信号 ---")
        for pillar, signal_name in five_pillars.items():
            score = self._get_atomic_score(df, signal_name, 0.0, debug_info)
            pillar_scores[pillar] = score
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"        {pillar} ({signal_name}): {score.loc[probe_ts]:.4f}")
        # 3. 核心数学逻辑 - 五象共振 (几何平均)
        # 为避免负数开方，先将所有[-1, 1]的信号映射到[0, 2]区间进行计算
        # (score + 1) 将 [-1, 1] 映射到 [0, 2]
        mapped_scores = [score + 1 for score in pillar_scores.values()]
        # 几何平均，体现“木桶效应”
        # 为防止0值导致结果恒为0，加入一个极小值
        product_of_scores = pd.Series(1.0, index=df_index, dtype=np.float32)
        for score in mapped_scores:
            product_of_scores *= (score.clip(lower=1e-9))
        resonance_score_mapped = product_of_scores.pow(1 / len(five_pillars))
        # 将结果从[0, 2]区间映射回[-1, 1]
        final_score = (resonance_score_mapped - 1).clip(-1, 1)
        states['FUSION_BIPOLAR_TREND_STRUCTURE_SCORE'] = final_score.astype(np.float32)
        print(f"  -- [融合层] “趋势结构分”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_fund_flow_trend(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V3.2 · 君臣共振与探针增强版 - 依赖解耦】冶炼“资金趋势” (FUSION_BIPOLAR_FUND_FLOW_TREND)
        - 核心重构: 废弃V2.x混合低阶信号的攻防模型，引入基于顶层信号的“君臣共振”模型。
        - 架构戒律: 严格遵守融合层职责，不再消费任何原始数据或低阶公理，只融合最高阶的战略信号。
        - 诡道哲学: 最终趋势 = 战略态势(君) × (1 + 微观信念(臣) × 确认系数)。宏观趋势必须
                      得到微观意图的确认，否则即为陷阱。
        - 【V3.2 增强】将对融合信号 `FUSION_BIPOLAR_MICRO_CONVICTION` 的依赖替换为基础信号 `SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT`，以解耦依赖。
                      修正方法签名以接受 debug_info 参数，并增加详细探针。
        """
        method_name = "_synthesize_fund_flow_trend"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        states = {}
        # 1. 信号升维：定义“君”与“臣”
        # 君：资金流情报引擎的最高战略判断
        strategic_posture = self._get_atomic_score(df, 'SCORE_FF_STRATEGIC_POSTURE', 0.0, debug_info)
        # 臣：盘口最真实的微观意图，作为现实检验器 (替换为基础信号)
        micro_conviction = self._get_atomic_score(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0, debug_info)
        # 2. 核心数学逻辑 - 君臣共振模型
        confirmation_factor = 0.5 # 确认系数，控制微观信念的影响力
        # 共振调节器：当微观信念与战略态势同向时 > 1 (放大)，反向时 < 1 (抑制)
        resonance_modulator = (1 + micro_conviction * confirmation_factor).clip(0, 2)
        # 非线性融合
        final_score = (strategic_posture * resonance_modulator).clip(-1, 1)
        states['FUSION_BIPOLAR_FUND_FLOW_TREND'] = final_score.astype(np.float32)
        print(f"  -- [融合层] “资金趋势”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_chip_trend(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V4.3 · 神魂根基与探针增强版 - 强化】冶炼“筹码趋势” (FUSION_BIPOLAR_CHIP_TREND)
        - 核心重构: 废弃V3.0晦涩的“静态/动态”模型，引入更符合博弈哲学的“神魂与根基”非线性调制模型。
        - 诡道哲学: 最终趋势 = 根基(客观结构) × (1 + 神魂(主观意愿) × 调制系数)。
                      坚实的筹码结构若无持股信心注入，亦是“死城”一座；反之，强大的信心能为
                      尚在构建的结构注入无穷潜力。此法旨在捕捉“体用合一”与“貌合神离”。
        - 【V4.3 增强】增加 `SCORE_CHIP_COHERENT_DRIVE` 和 `SCORE_CHIP_AXIOM_TREND_MOMENTUM` 信号，提高筹码趋势的准确性。
                      引入 `SCORE_CHIP_RISK_DISTRIBUTION_WHISPER` 作为对筹码趋势的直接惩罚，以应对“拉高诱多”情境。
                      修正方法签名以接受 debug_info 参数，并增加详细探针。
        """
        method_name = "_synthesize_chip_trend"
        # 修正此处：检查 debug_info 是否为 None
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“筹码趋势”...")
        states = {}
        df_index = df.index
        # 1. 重组信号，划分为“根基”与“神魂”两大阵营
        # --- 根基 (Foundation) - 客观的物理结构与趋势 ---
        battlefield_geography = self._get_atomic_score(df, 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 0.0, debug_info)
        strategic_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info)
        chip_trend_momentum = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_TREND_MOMENTUM', 0.0, debug_info) # 新增结构性推力
        # --- 神魂 (Soul) - 主观的持股意愿与战局变数 ---
        holder_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0, debug_info)
        divergence = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_DIVERGENCE', 0.0, debug_info)
        coherent_drive = self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0, debug_info) # 新增筹码同调驱动力
        # 引入派发诡影作为对筹码趋势的直接惩罚
        distribution_whisper = self._get_atomic_score(df, 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 0.0, debug_info)
        # 当派发诡影为正时，惩罚筹码趋势
        distribution_penalty = distribution_whisper.clip(lower=0) * 1.0 # 惩罚系数1.0，可调
        # 2. 核心数学逻辑 - 神魂调制模型
        # 2.1 融合“根基分” (Foundation Score)
        # 地形学(静态)与态势(动态)同等重要，并引入结构性推力
        foundation_score = (battlefield_geography * 0.4 + strategic_posture * 0.4 + chip_trend_momentum * 0.2).clip(-1, 1) # 调整权重
        # 2.2 融合“神魂分” (Soul Score)
        # 持股心态是主导，背离作为修正，并引入筹码同调驱动力
        soul_score = (holder_sentiment * 0.6 + divergence * 0.2 + coherent_drive * 0.2).clip(-1, 1) # 调整权重
        # 2.3 构建“神魂调制器” (Soul Modulator)
        modulation_factor = 0.5 # 调制系数，控制神魂的影响力
        soul_modulator = (1 + soul_score * modulation_factor).clip(0, 2)
        # 2.4 应用派发惩罚到最终分数
        # 将 foundation_score 映射到 [0, 2] 区间，应用惩罚，再映射回 [-1, 1]
        final_score_base = (foundation_score * soul_modulator).clip(-1, 1)
        final_score = (final_score_base * (1 - distribution_penalty)).clip(-1, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 根基分 (foundation_score): {foundation_score.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 神魂分 (soul_score): {soul_score.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 神魂调制器 (soul_modulator): {soul_modulator.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 派发诡影 (distribution_whisper): {distribution_whisper.loc[probe_ts]:.4f}, 派发惩罚 (distribution_penalty): {distribution_penalty.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 最终分数 (final_score): {final_score.loc[probe_ts]:.4f}")
        states['FUSION_BIPOLAR_CHIP_TREND'] = final_score.astype(np.float32)
        print(f"  -- [融合层] “筹码趋势”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_accumulation_inflection(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V5.4 · 厚積薄發与探针增强版 - 依赖解耦与强化 (終章)】冶炼“吸籌拐點信號” (FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT)
        - 核心升华: “势”的来源从单日的瞬时变化，升华为基于EMA的平滑趋势变化，旨在
                      滤除噪波，洞察“地利”持续改善或恶化的真实“势”之厚度。
        - 终章心法: 真正的拐点，是和谐之态与厚積之勢的共鸣。此法之后，再无增益。
        - 诡道哲学: 终极拐点 = 和谐之态(根基) × 厚積之勢(趨勢)。
        - 【V5.4 增强】将对融合信号 `FUSION_BIPOLAR_CHIP_TREND` 和 `FUSION_BIPOLAR_MARKET_PRESSURE` 的依赖替换为基础信号 `SCORE_CHIP_STRATEGIC_POSTURE` 和 `SCORE_RISK_UNRESOLVED_PRESSURE`，以解耦依赖。
                      增加 `SCORE_CHIP_MAIN_FORCE_COST_INTENT` 和 `SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION` 信号，提高吸筹拐点信号的准确性。
                      调整 `tian_shi_raw` 和 `di_li_raw` 的融合方式，使其对新增信号的贡献更平滑。
                      修正方法签名以接受 debug_info 参数，并增加详细探针。
        """
        method_name = "_synthesize_accumulation_inflection"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“吸筹拐点信号”...")
        states = {}
        fusion_intelligence_params = get_params_block(self.strategy, 'fusion_intelligence_params', {})
        params = fusion_intelligence_params.get('fusion_accumulation_inflection_params', {})
        tian_shi_raw_base = self._get_atomic_score(df, 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT', 0.0, debug_info)
        main_force_cost_intent = self._get_atomic_score(df, 'SCORE_CHIP_MAIN_FORCE_COST_INTENT', 0.0, debug_info).clip(lower=0) # 新增主力成本区攻防意图正向
        # 调整融合方式，使用加权平均，避免 0 值完全抑制
        tian_shi_raw = (tian_shi_raw_base * 0.7 + main_force_cost_intent * 0.3).clip(0,1) # 融合天时
        # 替换 FUSION_BIPOLAR_CHIP_TREND 为 SCORE_CHIP_STRATEGIC_POSTURE
        di_li_raw_base = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info)
        turnover_purity = self._get_atomic_score(df, 'SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION', 0.0, debug_info).clip(lower=0) # 新增换手纯度与成本优化正向
        # 调整融合方式，使用加权平均
        di_li_raw = (di_li_raw_base * 0.7 + turnover_purity * 0.3).clip(-1,1) # 融合地利
        # 替换 FUSION_BIPOLAR_MARKET_PRESSURE 为 SCORE_RISK_UNRESOLVED_PRESSURE
        ren_he_raw = self._get_atomic_score(df, 'SCORE_RISK_UNRESOLVED_PRESSURE', 0.0, debug_info)
        tian_shi_score = tian_shi_raw.clip(0, 1)
        di_li_score = di_li_raw.clip(lower=0) # 筹码趋势只取正向，代表地利
        ren_he_score = (1 - ren_he_raw) # 市场压力映射到 [0, 1]，压力越大，人和分越低
        # 和谐之态：天时、地利、人和的几何平均
        harmony_state_score = (tian_shi_score.clip(lower=1e-9) * di_li_score.clip(lower=1e-9) * ren_he_score.clip(lower=1e-9)).pow(1/3).fillna(0.0)
        # 厚積之勢：地利（筹码趋势）的变化率，EMA平滑
        di_li_change_raw = di_li_raw.diff(1).fillna(0.0)
        smoothed_di_li_change = di_li_change_raw.ewm(span=3, adjust=False).mean()
        amplification_factor = 0.5
        # 势能调节器：当 smoothed_di_li_change 为正时，放大；为负时，抑制
        potential_energy_modulator = (1 + smoothed_di_li_change * amplification_factor).clip(0, 2)
        final_score = (harmony_state_score * potential_energy_modulator).clip(0, 1)
        states['FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT'] = final_score.astype(np.float32)
        print(f"  -- [融合层] “吸筹拐点信号”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_contested_accumulation(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V2.2 · 吸收裁决与探针增强版 - 依赖解耦】冶炼“博弈吸筹” (Contested Accumulation)
        - 核心重构: 废弃V1.0“盲目相乘”模型，引入“吸收裁决”作为核心裁决维度，
                      构建“战场识别 × 吸收裁决 × 战略背景”的三位一体审判模型。
        - 核心公式: 博弈吸筹分 = 战场识别分 × 吸收裁决分 × 战略背景分
        - 诡道哲学: 真正的权力交接，不仅要看“博弈”的激烈程度，更要看“吸收”的最终战果。
        - 【V2.2 增强】将对融合信号 `FUSION_BIPOLAR_TREND_QUALITY` 的依赖替换为基础信号 `SCORE_STRUCT_AXIOM_TREND_FORM`，以解耦依赖。
                      修正方法签名以接受 debug_info 参数，并增加详细探针。
        """
        method_name = "_synthesize_contested_accumulation"
        # 修正此处：检查 debug_info 是否为 None
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“博弈吸筹”...")
        states = {}
        df_index = df.index
        # 1. 信号升维：定义“战场”、“裁决”、“背景”三大支柱
        # 支柱一：战场识别 (识别权力交接的战场)
        stealth_ops = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0, debug_info)
        distribution_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0, debug_info)
        # 支柱二：吸收裁决 (审判新主力是否成功吸收抛压)
        downward_resistance = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE', 0.0, debug_info)
        absorption_strength = self._get_atomic_score(df, 'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 0.0, debug_info)
        # 支柱三：战略背景 (确保战术服务于战略) (替换为基础信号)
        trend_quality = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0, debug_info).clip(lower=0)
        # 2. 核心数学逻辑 - 三位一体审判
        # 2.1 计算“战场识别分”
        # 当隐秘行动和派发意图同时存在时，才认为是博弈战场
        battlefield_score = (stealth_ops * distribution_intent).pow(0.5).fillna(0.0).clip(0, 1)
        # 2.2 计算“吸收裁决分”
        # 下跌抵抗和承接强度共同决定吸收品质
        absorption_verdict = (downward_resistance * 0.5 + absorption_strength * 0.5).clip(0, 1)
        # 2.3 最终融合：三者相乘，体现“缺一不可”的严苛逻辑
        final_score = (battlefield_score * absorption_verdict * trend_quality).clip(0, 1)
        output_name = 'FUSION_OPPORTUNITY_CONTESTED_ACCUMULATION'
        states[output_name] = final_score.astype(np.float32)
        print(f"  -- [融合层] “博弈吸筹”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_micro_conviction(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V3.3 · 战术品质版 - 依赖解耦与强化】冶炼“微观信念” (Micro Conviction)
        - 核心重构: 在V2.0“意图-确认”模型基础上，引入“战术品质”作为第三裁决维度。
        - 核心公式: 最终信念 = (瞬时意图 × 确认调节器) × 品质调节器
        - 诡道哲学: 信念的含金量，不仅在于意图的强度与持续性，更在于其执行的战术
                      品质。此法旨在区分“匹夫之勇”与“运筹帷幄”。
        - 【V3.3 增强】将对融合信号 `FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT` 的依赖替换为基础信号 `INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW`，以解耦依赖。
                      增加 `SCORE_BEHAVIOR_DISTRIBUTION_INTENT` 和 `SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL` 信号，提高战术品质的准确性。
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_micro_conviction"
        # 修正此处：检查 debug_info 是否为 None
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“微观信念”...")
        states = {}
        df_index = df.index
        # 1. 信号升维：定义“意图”、“趋势”、“品质”三大支柱
        # 支柱一：瞬时意图
        micro_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0, debug_info)
        # 支柱二：意图趋势 (用于确认)
        micro_divergence = self._get_atomic_score(df, 'SCORE_MICRO_AXIOM_DIVERGENCE', 0.0, debug_info)
        # 支柱三：战术品质 (用于裁决)
        cost_control = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_COST_CONTROL', 0.0, debug_info)
        offensive_purity = self._get_atomic_score(df, 'SCORE_INTRADAY_OFFENSIVE_PURITY', 0.0, debug_info)
        # 引入价格超买意图风险 (替换为基础信号)
        price_overextension_risk = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.0, debug_info).clip(lower=0)
        # 新增信号：派发意图和日内多头控制
        distribution_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0, debug_info)
        intraday_bull_control = self._get_atomic_score(df, 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', 0.0, debug_info)
        # 2. 核心数学逻辑 - “意图-趋势-品质”三位一体裁决
        # 2.1 计算“确认后的意图” (V2.0核心保留)
        confirmation_factor = 0.5 # 确认系数
        confirmation_modulator = (1 + micro_divergence * confirmation_factor)
        confirmed_intent = (micro_intent * confirmation_modulator).clip(-1, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 确认调节器: {confirmation_modulator.loc[probe_ts]:.4f}, 确认后的意图: {confirmed_intent.loc[probe_ts]:.4f}")
        # 2.2 计算“战术品质”
        # 成本控制(防守)与进攻纯度(进攻)同等重要，并引入派发意图和日内多头控制进行调节
        tactical_quality_base = (cost_control * 0.5 + offensive_purity * 0.5).clip(-1, 1)
        # 派发意图作为负向调节，日内多头控制作为正向调节
        tactical_quality = (tactical_quality_base * (1 - distribution_intent * 0.5) + intraday_bull_control * 0.2).clip(-1, 1)
        # 2.3 构建“品质调节器”
        quality_factor = 0.3 # 品质影响系数
        quality_modulator = (1 + tactical_quality * quality_factor).clip(0.7, 1.3)
        # 【V3.1 增强】引入价格超买意图风险作为负向调节器
        quality_modulator = (quality_modulator * (1 - price_overextension_risk * 0.5)).clip(0.5, 1.3) # 0.5为惩罚系数
        # 2.4 最终裁决：(确认后的意图) × 品质调节器
        final_conviction_score = (confirmed_intent * quality_modulator).clip(-1, 1)
        output_name = 'FUSION_BIPOLAR_MICRO_CONVICTION'
        states[output_name] = final_conviction_score.astype(np.float32)
        print(f"  -- [融合层] “微观信念”冶炼完成，最新分值: {final_conviction_score.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_quality(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V5.3 · 态势合一版 - 依赖解耦与强化】冶炼“趋势质量” (Trend Quality)
        - 核心重构: 在V4.0“四象支柱”模型基础上，引入质量自身的“变化率”作为“势”，
                      构建“静态之态 × 动态之势”的双核驱动模型。
        - 核心公式: 最终质量 = 静态质量(态) × (1 + 质量变化率(势) × 调节系数)
        - 诡道哲学: 既要洞察当下的强弱(态)，更要审度其演化的方向(势)，方为趋势之本源。
        - 【V5.3 增强】将对融合信号的依赖替换为基础信号，以解耦依赖。
                      引入风险信号作为负向调节器，增强对高位风险的敏感度。
                      增加 `SCORE_STRUCT_AXIOM_TENSION` 和 `SCORE_FF_AXIOM_FLOW_MOMENTUM` 信号，提高支柱得分的准确性。
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_trend_quality"
        # 修正此处：检查 debug_info 是否为 None
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“趋势质量”...")
        states = {}
        df_index = df.index
        # --- 1. 态之诊断 (State Diagnosis) ---
        # 1.1 信号原料库 (严格筛选公理级信号)
        struct_posture = self._get_atomic_score(df, 'SCORE_STRUCT_STRATEGIC_POSTURE', 0.0, debug_info)
        struct_geography = self._get_atomic_score(df, 'SCORE_CHIP_BATTLEFIELD_GEOGRAPHY', 0.0, debug_info)
        struct_stability = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_STABILITY', 0.0, debug_info)
        struct_tension = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0, debug_info) # 新增结构张力
        dyn_momentum = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_MOMENTUM', 0.0, debug_info)
        dyn_inertia = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_INERTIA', 0.0, debug_info)
        behavior_upward_momentum = self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM', 0.0, debug_info)
        ff_posture = self._get_atomic_score(df, 'SCORE_FF_STRATEGIC_POSTURE', 0.0, debug_info)
        ff_flow_momentum = self._get_atomic_score(df, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0, debug_info) # 新增资金流纯度与动能
        chip_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info)
        chip_sentiment = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0, debug_info)
        found_posture = self._get_atomic_score(df, 'SCORE_FOUNDATION_STRATEGIC_POSTURE', 0.0, debug_info)
        found_constitution = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION', 0.0, debug_info)
        found_tide = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 0.0, debug_info)
        # 替换 FUSION_BIPOLAR_MICRO_CONVICTION 为 SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT
        micro_conviction = self._get_atomic_score(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0, debug_info)
        # 引入风险信号作为负向调节器 (替换为基础信号)
        stagnation_risk = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0, debug_info)
        distribution_pressure = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0, debug_info)
        price_overextension_risk = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.0, debug_info).clip(lower=0)
        # 1.2 计算各支柱得分 (映射到[0,2]区间)
        pillar_structure = ((struct_posture + 1) * 0.4 + (struct_geography + 1) * 0.2 + (struct_stability + 1) * 0.2 + (struct_tension + 1) * 0.2) # 调整权重
        pillar_momentum = ((dyn_momentum + 1) * 0.4 + (dyn_inertia + 1) * 0.3 + (behavior_upward_momentum + 1) * 0.3)
        pillar_conviction = ((ff_posture + 1) * 0.3 + (ff_flow_momentum + 1) * 0.3 + (chip_posture + 1) * 0.2 + (chip_sentiment + 1) * 0.2) # 调整权重
        pillar_foundation = ((found_posture + 1) * 0.5 + (found_constitution + 1) * 0.3 + (found_tide + 1) * 0.2)
        # 1.3 四象共振，得出“静态质量分”
        raw_quality_mapped = (pillar_structure.clip(lower=1e-9) * pillar_momentum.clip(lower=1e-9) * pillar_conviction.clip(lower=1e-9) * pillar_foundation.clip(lower=1e-9)).pow(1/4)
        state_quality_score = (raw_quality_mapped - 1).clip(-1, 1)
        # 1.4 微观信念作为真实性检验
        micro_conviction_regulator = (1 + micro_conviction * 0.3).clip(0.7, 1.3)
        state_quality_score_final = (state_quality_score * micro_conviction_regulator).clip(-1, 1)
        # 【V5.1 增强】引入风险信号作为负向调节器
        risk_penalty_factor = (stagnation_risk * 0.4 + distribution_pressure * 0.4 + price_overextension_risk * 0.2).clip(0,1)
        state_quality_score_final = (state_quality_score_final * (1 - risk_penalty_factor)).clip(-1, 1)
        # --- 2. 势之诊断 (Potential Diagnosis) ---
        # 2.1 计算“质量势能分”，即静态质量分的变化率 (EMA平滑)
        quality_change = state_quality_score_final.diff(1).fillna(0.0)
        quality_potential_score = quality_change.ewm(span=3, adjust=False).mean()
        # 2.2 构建“势能调节器”
        potential_modulation_factor = 0.5 # 势能调节系数
        potential_modulator = (1 + quality_potential_score * potential_modulation_factor).clip(0.5, 1.5)
        # --- 3. 态势合一，终极裁决 ---
        final_quality = (state_quality_score_final * potential_modulator).clip(-1, 1)
        states['FUSION_BIPOLAR_TREND_QUALITY'] = final_quality.astype(np.float32)
        print(f"  -- [融合层] “趋势质量”冶炼完成，最新分值: {final_quality.iloc[-1]:.4f}")
        return states

    def _synthesize_market_pressure(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V3.3 · 态势裁决版 - 依赖解耦与强化】冶炼“市场压力” (Market Pressure)
        - 核心重构: 废弃V2.0“独立共振后相减”模型，引入“态势裁决”模型。
        - 核心公式: 最终压力 = 战术净压力(臣) × 战场态势调节器(君)
        - 诡道哲学: 压力之强弱，不在其本身，而在其是否顺应大势。以“趋势质量”为君，
                      裁决战术压力之臣，方能洞察顺势强攻与逆势反抽之别。
        - 【V3.3 增强】将对融合信号的依赖替换为基础信号，以解耦依赖。
                      增强 `risk_signals` 对高位风险的敏感度。
                      调整 `battlefield_modulator` 的计算，使其在趋势良好但战术压力大时放大负向压力。
                      增加 `SCORE_RISK_LIQUIDITY_DRAIN` 和 `SCORE_FOUNDATION_AXIOM_MARKET_FRICTION` 信号，提高压力识别的准确性。
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_market_pressure"
        # 修正此处：检查 debug_info 是否为 None
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“市场压力”...")
        states = {}
        df_index = df.index
        # --- 1. [修改] 信号升维并计算“战术净压力” (臣) ---
        opportunity_signals = {
            'SCORE_CHIP_HARMONY_INFLECTION': 0.3,
            'SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK': 0.25,
            'SCORE_OPPORTUNITY_SELLING_EXHAUSTION': 0.2,
            'SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY': 0.15,
            'SCORE_BEHAVIOR_ABSORPTION_STRENGTH': 0.1, # 替换 FUSION_OPPORTUNITY_CONTESTED_ACCUMULATION
        }
        # 替换 PROCESS_FUSION_TREND_EXHAUSTION_SYNDROME 为基础信号组合
        # trend_exhaustion_proxy = (self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0, debug_info) * 0.5 + self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0, debug_info) * 0.5)
        # 修正：将信号名称改为不带 _POSITIVE 或 _NEGATIVE 后缀的原始信号名称，并在获取时处理正负部分
        risk_signals_config = {
            'SCORE_RISK_BREAKOUT_FAILURE_CASCADE': {'weight': 0.25, 'part': 'positive'}, # 突破失败级联风险 (本身是风险，取正向)
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT': {'weight': 0.2, 'part': 'positive'}, # 派发意图 (本身是风险，取正向)
            'PROCESS_META_WINNER_CONVICTION_DECAY': {'weight': 0.15, 'part': 'positive'}, # 赢家信念衰减 (本身是风险，取正向)
            'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY': {'weight': 0.15, 'part': 'positive'}, # 熊市背离品质 (本身是风险，取正向)
            'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW': {'weight': 0.1, 'part': 'positive'}, # 滞涨证据 (本身是风险，取正向)
            'SCORE_RISK_LIQUIDITY_DRAIN': {'weight': 0.05, 'part': 'positive'}, # 流动性枯竭风险 (本身是风险，取正向)
            'SCORE_FF_AXIOM_CONSENSUS': {'weight': 0.05, 'part': 'negative'}, # 资金流共识负向
            'SCORE_CHIP_STRATEGIC_POSTURE': {'weight': 0.05, 'part': 'negative'} # 筹码战略态势负向
        }
        tactical_upward_pressure = pd.Series(0.0, index=df_index, dtype=np.float32)
        for signal, weight in opportunity_signals.items():
            score = self._get_atomic_score(df, signal, 0.0, debug_info)
            tactical_upward_pressure += score * weight
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"        [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 机会信号 '{signal}' (值: {score.loc[probe_ts]:.4f}, 权重: {weight})")
        tactical_upward_pressure = tactical_upward_pressure.clip(0,1)
        tactical_downward_pressure = pd.Series(0.0, index=df_index, dtype=np.float32)
        for signal_name, config in risk_signals_config.items(): # 修正：遍历新的字典结构
            weight = config['weight']
            part = config['part']
            score = self._get_atomic_score(df, signal_name, 0.0, debug_info).fillna(0.0) # 获取原始信号
            if part == 'negative':
                tactical_downward_pressure += score.clip(upper=0).abs() * weight
            elif part == 'positive':
                tactical_downward_pressure += score.clip(lower=0) * weight
            else: # 默认情况，直接使用
                tactical_downward_pressure += score * weight
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"        [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 风险信号 '{signal_name}' ({part} part) (值: {score.loc[probe_ts]:.4f}, 权重: {weight})")
        tactical_downward_pressure = tactical_downward_pressure.clip(0,1)
        net_tactical_pressure = (tactical_upward_pressure - tactical_downward_pressure).clip(-1, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 战术向上压力: {tactical_upward_pressure.loc[probe_ts]:.4f}, 战术向下压力: {tactical_downward_pressure.loc[probe_ts]:.4f} -> 净战术压力: {net_tactical_pressure.loc[probe_ts]:.4f}")
        # --- 2. 获取“战场态势” (君) --- (替换为基础信号)
        battlefield_context = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0, debug_info)
        # 引入市场摩擦作为调节器
        market_friction = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_FRICTION', 0.0, debug_info)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 战场态势 (趋势形态 - 基础信号): {battlefield_context.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 市场摩擦 (新增): {market_friction.loc[probe_ts]:.4f}")
        # --- 3. 核心数学逻辑 - 态势裁决 ---
        # 3.1 构建“战场态势调节器”
        modulation_factor = 0.5 # 态势影响系数
        # 当趋势质量为正，净压力为负时，modulator应该放大负向压力
        # 当趋势质量为负，净压力为正时，modulator应该放大正向压力
        # 调整为：当净压力与趋势质量方向相反时，放大其绝对值
        battlefield_modulator_base = (1 + np.sign(net_tactical_pressure) * (-battlefield_context) * modulation_factor).clip(0.5, 1.5)
        # 市场摩擦作为额外的调节器，如果摩擦大且利于下跌，则进一步放大负向压力
        battlefield_modulator = battlefield_modulator_base * (1 + market_friction.clip(upper=0).abs() * 0.2) # 0.2为摩擦影响系数
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 战场态势调节器: {battlefield_modulator.loc[probe_ts]:.4f}")
        # 3.2 最终裁决：战术净压力 × 战场态势调节器
        final_pressure = (net_tactical_pressure * battlefield_modulator).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_PRESSURE'] = final_pressure.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “市场压力”冶炼完成，最终分值: {final_pressure.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “市场压力”冶炼完成，最新分值: {final_pressure.iloc[-1]:.4f}")
        return states

    def _synthesize_accumulation_playbook(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V5.4 · 道法合一与探针增强版 - 依赖解耦与强化 (終章)】冶炼“吸筹剧本” (FUSION_ACCUMULATION_PLAYBOOK)
        - 核心升华: 引入“王霸并济”二元法则。根据“点火器”强度，动态切换“王者之道”
                      (薪火相加)与“霸王之道”(状态重置)，以区分“量变积累”与“质变突破”。
        - 终章心法: 王者，积寸功；霸者，定乾坤。此法之后，再无增益。
        - 【V5.4 增强】将对融合信号 `FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT`、`FUSION_BIPOLAR_CHIP_TREND` 和 `FUSION_BIPOLAR_MARKET_PRESSURE` 的依赖替换为基础信号，以解耦依赖。
                      增加 `SCORE_CHIP_OPP_ABSORPTION_ECHO` 和 `SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL` 信号，提高吸筹剧本的准确性。
                      调整 `igniter_signal` 和 `di_li` 的融合方式，使其对新增信号的贡献更平滑。
                      修正方法签名以接受 debug_info 参数，并增加详细探针。
        """
        method_name = "_synthesize_accumulation_playbook"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“吸筹剧本”...")
        states = {}
        # 1. 信号升维：定义“点火器”与“根基”
        # 替换 FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT 为 PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT
        igniter_signal_base = self._get_atomic_score(df, 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT', 0.0, debug_info)
        absorption_echo = self._get_atomic_score(df, 'SCORE_CHIP_OPP_ABSORPTION_ECHO', 0.0, debug_info) # 新增吸筹回声
        # 调整融合方式，使用加权平均，避免 0 值完全抑制
        igniter_signal = (igniter_signal_base * 0.6 + absorption_echo * 0.4).clip(0,1) # 融合点火器
        # 替换 FUSION_BIPOLAR_CHIP_TREND 为 SCORE_CHIP_STRATEGIC_POSTURE
        di_li_base = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info).clip(lower=0)
        chip_potential = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0, debug_info).clip(lower=0) # 新增筹码势能
        # 调整融合方式，使用加权平均
        di_li = (di_li_base * 0.6 + chip_potential * 0.4).clip(lower=0) # 融合地利
        # 替换 FUSION_BIPOLAR_MARKET_PRESSURE 为 SCORE_RISK_UNRESOLVED_PRESSURE
        ren_he = (1 - self._get_atomic_score(df, 'SCORE_RISK_UNRESOLVED_PRESSURE', 0.0, debug_info)) # 市场压力越大，人和分越低
        foundation_sustain_factor = (di_li.clip(lower=1e-9) * ren_he.clip(lower=1e-9)).pow(1/2).fillna(0.0)
        # 2. 核心数学逻辑 - 王霸并济，道法合一
        playbook_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        hegemon_threshold = 0.75 # 定义“霸王门槛”，区分“突破”与“胶着”
        # 确保循环从 df 的第二个元素开始，因为需要 previous_score
        for i in range(1, len(df)):
            previous_score = playbook_score.iloc[i-1]
            decay_modulator = foundation_sustain_factor.iloc[i]
            decayed_score = previous_score * decay_modulator
            current_igniter = igniter_signal.iloc[i]
            if current_igniter > hegemon_threshold:
                playbook_score.iloc[i] = current_igniter
            else:
                playbook_score.iloc[i] = decayed_score + current_igniter - (decayed_score * current_igniter)
        states['FUSION_ACCUMULATION_PLAYBOOK'] = playbook_score.astype(np.float32)
        print(f"  -- [融合层] “吸筹剧本”冶炼完成，最新分值: {playbook_score.iloc[-1] if not playbook_score.empty else 0.0:.4f}")
        return states

    def _synthesize_trend_exhaustion_syndrome(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V2.4 · 意志与压力版 - 依赖解耦与强化】冶炼“趋势衰竭综合征”
        - 核心重构: 废弃V1.0基于陈旧过程信号的“症状叠加”模型，引入基于高阶信号的
                      “意志vs压力”二元博弈模型。
        - 核心公式: 衰竭风险 = (上涨意志衰竭度 × 反转压力增强度)^(1/2)
        - 诡道哲学: 趋势之终结，非无故自崩，乃上涨意志之衰竭，恰逢反转压力之增强，
                      两相博弈，天平倾覆之必然结果。
        - 【V2.4 增强】将对融合信号的依赖替换为基础信号，以解耦依赖。
                      增强 `weakening_will_score` 和 `intensifying_pressure_score` 对高位风险的敏感度。
                      调整 `will_components` 权重，确保风险信号能有效贡献。
                      **将 `weakening_will_score` 和 `intensifying_pressure_score` 的计算方式从加权几何平均改为加权算术平均，以更准确地聚合内部风险信号。**
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_trend_exhaustion_syndrome"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“趋势衰竭综合征”...")
        states = {}
        df_index = df.index
        fusion_intelligence_params = get_params_block(self.strategy, 'fusion_intelligence_params', {})
        fusion_playbook_params = fusion_intelligence_params.get('fusion_playbook_params', {})
        tes_params = fusion_playbook_params.get('trend_exhaustion_syndrome', {})
        # --- 1. 上涨意志衰竭度 (Weakening Will) ---
        # 替换 FUSION_BIPOLAR_TREND_QUALITY 为 SCORE_STRUCT_AXIOM_TREND_FORM
        trend_quality = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0, debug_info)
        trend_quality_decay = -trend_quality.diff(1).fillna(0.0).clip(upper=0) # 核心病根：趋势质量衰减
        distribution_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0, debug_info)
        chip_posture_decay = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info).clip(upper=0).abs() # 筹码战略态势恶化
        # 引入价格超买意图风险作为意志衰竭的证据 (替换为基础信号)
        price_overextension_risk = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.0, debug_info).clip(lower=0)
        # 调整 will_components 的权重，确保即使部分信号为 0，其他非 0 信号也能有效贡献
        will_components_with_weights = [
            (trend_quality_decay, 0.25),
            (distribution_intent, 0.25),
            (chip_posture_decay, 0.2),
            (price_overextension_risk, 0.3)
        ]
        weakening_will_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_will_weight = sum(w for _, w in will_components_with_weights)
        if total_will_weight > 0:
            for comp, weight in will_components_with_weights:
                comp_processed = comp.fillna(0.0).clip(0, 1)
                weakening_will_score += comp_processed * weight
            weakening_will_score /= total_will_weight
        else:
            weakening_will_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        weakening_will_score = weakening_will_score.clip(0, 1)
        # --- 2. 反转压力增强度 (Intensifying Pressure) ---
        # 替换 FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT 为 INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW
        overextension_risk = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.0, debug_info).clip(lower=0) # 价格超买意图正向部分
        bearish_divergence = self._get_atomic_score(df, 'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY', 0.0, debug_info)
        # 替换 FUSION_RISK_STAGNATION 为 INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW
        stagnation_risk = self._get_atomic_score(df, 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0, debug_info)
        # 替换 FUSION_RISK_DISTRIBUTION_PRESSURE 为 SCORE_BEHAVIOR_DISTRIBUTION_INTENT
        distribution_pressure = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0, debug_info) # 新增派发压力
        # 调整 pressure_components 的权重
        pressure_components_with_weights = [
            (overextension_risk, 0.3),
            (bearish_divergence, 0.2),
            (stagnation_risk, 0.25),
            (distribution_pressure, 0.25)
        ]
        intensifying_pressure_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_pressure_weight = sum(w for _, w in pressure_components_with_weights)
        if total_pressure_weight > 0:
            for comp, weight in pressure_components_with_weights:
                comp_processed = comp.fillna(0.0).clip(0, 1)
                intensifying_pressure_score += comp_processed * weight
            intensifying_pressure_score /= total_pressure_weight
        else:
            intensifying_pressure_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        intensifying_pressure_score = intensifying_pressure_score.clip(0, 1)
        # --- 3. 最终融合 ---
        syndrome_score = (weakening_will_score * intensifying_pressure_score).pow(0.5).fillna(0.0).clip(0, 1)
        output_name = 'PROCESS_FUSION_TREND_EXHAUSTION_SYNDROME'
        states[output_name] = syndrome_score.astype(np.float32)
        print(f"  -- [融合层] “趋势衰竭综合征”冶炼完成，最新分值: {syndrome_score.iloc[-1]:.4f}")
        return states

    def _synthesize_liquidity_dynamics(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V5.6 · 诡道穿透版 - 依赖解耦与强化】冶炼“流动性博弈动态” (FUSION_BIPOLAR_LIQUIDITY_DYNAMICS)
        - 核心重构: 引入“诡道因子非对称调制”，深化流动性“质量”与“纯度”量化，优化动态激活敏感度，强化维度间协同/冲突裁决。
        - 核心目标: 融合“价量效能”、“权势转移”、“流动性状态”三大维度，输出[-1, 1]的双极性分数。
        - 诡道哲学: 流动性是市场博弈的血液。健康的流动性动态，是主力主导、价量协同、权力稳固的体现；
                      混乱的流动性动态，则预示着主力派发、权力失衡、市场混乱。
                      其判断重心与反应速度，应随市场情境（趋势质量、市场政权、波动率、情绪、资金流可信度）
                      动态调整，并能识别维度间的协同与冲突，穿透主力诡道。
                      其判断重心与反应速度，应随市场情境（趋势质量、市场政权、波动率、情绪、资金流可信度）
                      动态调整，并能识别维度间协同与冲突，穿透主力诡道。
        - 融合模型: 最终得分 = tanh( (动态权重_pve*PVE + 动态权重_pt*PT + 动态权重_ls*LS) * 协同/冲突因子 * 诡道穿透因子)
        - 【V5.6 增强】将对融合信号的依赖替换为基础信号，以解耦依赖。
                      增强对风险信号的敏感度，确保诡道因子在诱多情境下能有效惩罚。
                      增加 `SCORE_FF_AXIOM_INTENT_PURITY` 和 `SCORE_FOUNDATION_AXIOM_MARKET_FRICTION` 信号，提高流动性博弈动态的准确性。
                      调整 `LS激活敏感度` 的计算，避免输出过于极端的 1.0000 或 0.0000。
                      **将 `flow_credibility_index_D` 归一化到 0-1 范围，以确保其在计算中的行为符合预期。**
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_liquidity_dynamics"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“流动性博弈动态”...")
        states = {}
        df_index = df.index
        fusion_intelligence_params = get_params_block(self.strategy, 'fusion_intelligence_params', {})
        fusion_playbook_params = fusion_intelligence_params.get('fusion_playbook_params', {})
        ld_params = fusion_playbook_params.get('liquidity_dynamics', {})
        def weighted_sum_with_activation_series(components_with_weights, index, activation_sensitivity=1.0,
                                                deception_index=None, wash_trade_intensity=None, flow_credibility=None,
                                                deception_mod_params=None, is_bullish=True, debug_info=None, component_type=""):
            is_debug_enabled_inner, probe_ts_inner, method_name_inner = debug_info if debug_info else (False, None, "未知方法")
            if not components_with_weights:
                return pd.Series(0.0, index=index, dtype=np.float32)
            raw_sum = pd.Series(0.0, index=index, dtype=np.float32)
            total_possible_weight = 0.0
            if is_debug_enabled_inner and probe_ts_inner and probe_ts_inner in index:
                print(f"        [融合层调试] {method_name_inner} @ {probe_ts_inner.strftime('%Y-%m-%d')}: --- {component_type}原始信号 (is_bullish={is_bullish}) ---")
            
            for comp_info in components_with_weights:
                # comp_info 可以是 (signal_name_str, weight, part) 或 (pd.Series_obj, weight, part)
                if isinstance(comp_info[0], str):
                    signal_name, weight, part = comp_info
                    # 获取原始信号
                    comp_raw = self._get_atomic_score(df, signal_name, 0.0, debug_info)
                elif isinstance(comp_info[0], pd.Series):
                    comp_raw, weight, part = comp_info
                    signal_name = comp_raw.name if comp_raw.name else "Unnamed_Series" # 用于调试输出
                else:
                    raise ValueError(f"Invalid component type in components_with_weights: {type(comp_info[0])}")
                # 根据 part 提取正向或负向部分
                if part == 'positive':
                    comp = comp_raw.clip(lower=0)
                elif part == 'negative':
                    comp = comp_raw.clip(upper=0).abs()
                else: # 默认情况，直接使用
                    comp = comp_raw
                
                # 修正：确保所有组件在贡献给加权和之前都被裁剪到 [0, 1] 范围
                comp = comp.clip(0, 1) # 添加此行
                
                # 确保组件是Series且索引对齐，并填充NaN为0，避免NaN传播
                comp_aligned = comp.reindex(index).fillna(0.0)
                raw_sum += comp_aligned * weight
                total_possible_weight += weight
                if is_debug_enabled_inner and probe_ts_inner and probe_ts_inner in index:
                    print(f"          [融合层调试] {method_name_inner} @ {probe_ts_inner.strftime('%Y-%m-%d')}: 组件 '{signal_name}' ({part} part) (值: {comp_aligned.loc[probe_ts_inner]:.4f}, 权重: {weight})")
            
            if total_possible_weight > 0:
                normalized_sum = raw_sum / total_possible_weight
                # 激活函数，将分数映射到 [0, 1]
                activated_score = (np.tanh(normalized_sum * activation_sensitivity) + 1) / 2
                if deception_index is not None and wash_trade_intensity is not None and flow_credibility is not None and deception_mod_params is not None:
                    deception_penalty_factor = get_param_value(deception_mod_params.get('deception_penalty_factor'), 0.2)
                    wash_trade_penalty_factor = get_param_value(deception_mod_params.get('wash_trade_penalty_factor'), 0.1)
                    deception_reward_factor = get_param_value(deception_mod_params.get('deception_reward_factor'), 0.1)
                    credibility_influence = get_param_value(deception_mod_params.get('credibility_influence'), 0.5)
                    # 资金流可信度对诡道因子的影响
                    # flow_credibility 已经归一化到 0-1
                    credibility_mod = (1 - flow_credibility * credibility_influence).clip(0, 1)
                    modulated_deception = deception_index * credibility_mod
                    modulated_wash_trade = wash_trade_intensity * credibility_mod
                    if is_bullish:
                        # 看涨情境下，诱多和对倒都是惩罚
                        penalty = (modulated_deception.clip(lower=0) * deception_penalty_factor +
                                   modulated_wash_trade * wash_trade_penalty_factor)
                        activated_score = (activated_score - penalty).clip(0, 1)
                        if is_debug_enabled_inner and probe_ts_inner and probe_ts_inner in index:
                            print(f"          [融合层调试] {method_name_inner} @ {probe_ts_inner.strftime('%Y-%m-%d')}: 看涨诡道惩罚 (欺骗: {modulated_deception.loc[probe_ts_inner]:.4f}, 对倒: {modulated_wash_trade.loc[probe_ts_inner]:.4f}) -> 惩罚: {penalty.loc[probe_ts_inner]:.4f}")
                    else: # is_bearish
                        # 看跌情境下，诱空是奖励，诱多是惩罚
                        reward = (modulated_deception.clip(upper=0).abs() * deception_reward_factor) # 诱空奖励
                        activated_score = (activated_score + reward).clip(0, 1)
                        penalty = (modulated_deception.clip(lower=0) * deception_penalty_factor) # 诱多惩罚
                        activated_score = (activated_score - penalty).clip(0, 1)
                        if is_debug_enabled_inner and probe_ts_inner and probe_ts_inner in index:
                            print(f"          [融合层调试] {method_name_inner} @ {probe_ts_inner.strftime('%Y-%m-%d')}: 看跌诡道调制 (诱空奖励: {reward.loc[probe_ts_inner]:.4f}, 诱多惩罚: {penalty.loc[probe_ts_inner]:.4f})")
                return activated_score
            else:
                return pd.Series(0.0, index=index, dtype=np.float32)
        # --- 预计算缺失或需要特殊处理的信号 ---
        # price_volume_entropy_D_INVERSE
        # 修正：替换 price_volume_entropy_D 为 SCORE_BEHAVIOR_VOLUME_BURST
        price_volume_entropy_D_proxy = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_BURST', 0.0, debug_info).fillna(0.0).clip(0, 1)
        price_volume_entropy_D_INVERSE = (1 - price_volume_entropy_D_proxy).rename('price_volume_entropy_D_INVERSE')
        # CAPITAL_CONFRONTATION_PROXY_BULLISH 和 CAPITAL_CONFRONTATION_PROXY_BEARISH
        capital_confrontation_proxy_bullish = (self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0, debug_info).clip(lower=0) * 0.5 + 
                                               self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info).clip(lower=0) * 0.5).rename('CAPITAL_CONFRONTATION_PROXY_BULLISH')
        capital_confrontation_proxy_bearish = (self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONSENSUS', 0.0, debug_info).clip(upper=0).abs() * 0.5 + 
                                               self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info).clip(upper=0).abs() * 0.5).rename('CAPITAL_CONFRONTATION_PROXY_BEARISH')
        # main_force_flow_gini_D_INVERSE
        # 修正：替换 main_force_flow_gini_D 为 SCORE_FF_AXIOM_FLOW_MOMENTUM 的正向部分
        main_force_flow_gini_D_proxy = self._get_atomic_score(df, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0, debug_info).clip(lower=0).fillna(0.0).clip(0, 1)
        main_force_flow_gini_D_INVERSE = (1 - main_force_flow_gini_D_proxy).rename('main_force_flow_gini_D_INVERSE')
        # retail_flow_dominance_index_D_INVERSE
        # 修正：替换 retail_flow_dominance_index_D 为 SCORE_CHIP_RETAIL_VULNERABILITY
        retail_flow_dominance_index_D_proxy = self._get_atomic_score(df, 'SCORE_CHIP_RETAIL_VULNERABILITY', 0.0, debug_info).fillna(0.0).clip(0, 1)
        retail_flow_dominance_index_D_INVERSE = (1 - retail_flow_dominance_index_D_proxy).rename('retail_flow_dominance_index_D_INVERSE')
        # --- 原始信号获取 ---
        # PVE (价量效能)
        bullish_pve_components_with_weights = [
            ('SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM', 0.25, 'positive'),
            ('SCORE_BEHAVIOR_VOLUME_BURST', 0.2, 'positive'),
            ('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.2, 'positive'),
            ('SCORE_DYN_AXIOM_MOMENTUM', 0.2, 'positive'), # 从双极性信号中取正向
            (price_volume_entropy_D_INVERSE, 0.15, 'positive') # 使用预计算的 Series
        ]
        bearish_pve_components_with_weights = [
            ('SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.3, 'positive'),
            ('PROCESS_RISK_VPA_EFFICIENCY_DECAY', 0.3, 'positive'),
            ('SCORE_DYN_AXIOM_MOMENTUM', 0.2, 'negative'), # 从双极性信号中取负向绝对值
            (price_volume_entropy_D_proxy.rename('price_volume_entropy_D'), 0.2, 'positive') # 使用预计算的 Series
        ]
        # PT (权势转移)
        bullish_pt_components_with_weights = [
            ('PROCESS_META_POWER_TRANSFER', 0.25, 'positive'),
            ('SCORE_CHIP_TACTICAL_EXCHANGE', 0.2, 'positive'),
            ('SCORE_FF_AXIOM_CONSENSUS', 0.2, 'positive'),
            (capital_confrontation_proxy_bullish, 0.2, 'positive'), # 使用预计算的 Series
            ('SCORE_FF_AXIOM_INTENT_PURITY', 0.15, 'positive')
        ]
        bearish_pt_components_with_weights = [
            ('PROCESS_META_POWER_TRANSFER', 0.25, 'negative'),
            ('SCORE_CHIP_TACTICAL_EXCHANGE', 0.2, 'negative'),
            ('SCORE_FF_AXIOM_CONSENSUS', 0.2, 'negative'),
            (capital_confrontation_proxy_bearish, 0.2, 'positive'), # 使用预计算的 Series
            ('SCORE_FF_AXIOM_INTENT_PURITY', 0.15, 'negative')
        ]
        # LS (流动性状态)
        bullish_ls_components_with_weights = [
            ('SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 0.2, 'positive'),
            ('SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.15, 'positive'),
            ('SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH', 0.15, 'positive'),
            ('SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.15, 'positive'),
            (main_force_flow_gini_D_INVERSE, 0.1, 'positive'), # 使用预计算的 Series
            (retail_flow_dominance_index_D_INVERSE, 0.1, 'positive'), # 使用预计算的 Series
            ('SCORE_FOUNDATION_AXIOM_MARKET_FRICTION', 0.15, 'negative') # 从双极性信号中取负向绝对值
        ]
        bearish_ls_components_with_weights = [
            ('SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 0.2, 'negative'),
            ('SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.15, 'negative'),
            ('SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH', 0.15, 'negative'),
            ('INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.15, 'positive'), # 使用基础信号，本身是风险，取正向
            (main_force_flow_gini_D_proxy.rename('main_force_flow_gini_D'), 0.1, 'positive'), # 使用预计算的 Series
            (retail_flow_dominance_index_D_proxy.rename('retail_flow_dominance_index_D'), 0.1, 'positive'), # 使用预计算的 Series
            ('SCORE_FOUNDATION_AXIOM_MARKET_FRICTION', 0.15, 'positive') # 从双极性信号中取正向
        ]
        # 修正：替换 VOLATILITY_INSTABILITY_INDEX_21d_D 为 SCORE_STRUCT_AXIOM_TENSION
        volatility_instability = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0, debug_info).fillna(0.0).clip(0, 1)
        # 修正：替换 market_sentiment_score_D 为 SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM
        market_sentiment = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0, debug_info).fillna(0.0).clip(-1, 1)
        # 修正：替换 flow_credibility_index_D 为 SCORE_FF_AXIOM_INTENT_PURITY 的正向部分
        flow_credibility_raw = self._get_atomic_score(df, 'SCORE_FF_AXIOM_INTENT_PURITY', 0.0, debug_info).clip(lower=0).fillna(0.0)
        flow_credibility = flow_credibility_raw.clip(0, 1) # 已经是归一化信号，直接裁剪
        # 修正：替换 deception_index_D 为 SCORE_BEHAVIOR_DECEPTION_INDEX
        deception_index = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0, debug_info).fillna(0.0).clip(-1, 1)
        # 修正：替换 wash_trade_intensity_D 为 SCORE_BEHAVIOR_DECEPTION_INDEX 的绝对值作为代理
        wash_trade_intensity = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0, debug_info).abs().fillna(0.0).clip(0, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号 ---")
            print(f"        波动不稳定性 (SCORE_STRUCT_AXIOM_TENSION): {volatility_instability.loc[probe_ts]:.4f}, 市场情绪 (SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM): {market_sentiment.loc[probe_ts]:.4f}")
            print(f"        资金流可信度 (SCORE_FF_AXIOM_INTENT_PURITY): {flow_credibility.loc[probe_ts]:.4f}")
            print(f"        欺骗指数 (SCORE_BEHAVIOR_DECEPTION_INDEX): {deception_index.loc[probe_ts]:.4f}, 对倒强度 (SCORE_BEHAVIOR_DECEPTION_INDEX.abs()): {wash_trade_intensity.loc[probe_ts]:.4f}")
        pve_sens_params = ld_params.get('pve_activation_sensitivity_params', {})
        pve_base_sens = get_param_value(pve_sens_params.get('base_sensitivity'), 2.0)
        pve_mod_factor = get_param_value(pve_sens_params.get('modulator_factor'), 0.5)
        pve_activation_sensitivity = pve_base_sens * (1 + np.tanh(volatility_instability * pve_mod_factor)).clip(0.5, 1.5)
        pt_sens_params = ld_params.get('pt_activation_sensitivity_params', {})
        pt_base_sens = get_param_value(pt_sens_params.get('base_sensitivity'), 2.0)
        pt_mod_factor = get_param_value(pt_sens_params.get('modulator_factor'), 0.5)
        pt_activation_sensitivity = pt_base_sens * (1 + np.tanh(market_sentiment * pt_mod_factor)).clip(0.5, 1.5)
        ls_sens_params = ld_params.get('ls_activation_sensitivity_params', {})
        ls_base_sens = get_param_value(ls_sens_params.get('base_sensitivity'), 2.0)
        ls_mod_factor = get_param_value(ls_sens_params.get('modulator_factor'), 0.5)
        # 调整 LS 激活敏感度，使其在极端情况下不会过高，避免 tanh 迅速饱和
        ls_activation_sensitivity = ls_base_sens * (1 + np.tanh(flow_credibility * ls_mod_factor * 0.5)).clip(0.5, 1.5) # 增加一个 0.5 的缩放因子
        deception_mod_params = ld_params.get('deception_modulation_params', {})
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: PVE激活敏感度: {pve_activation_sensitivity.loc[probe_ts]:.4f}, PT激活敏感度: {pt_activation_sensitivity.loc[probe_ts]:.4f}, LS激活敏感度: {ls_activation_sensitivity.loc[probe_ts]:.4f}")
        # 修正：调用 weighted_sum_with_activation_series 时，传递正确的 components_with_weights 结构
        bullish_pve_fused = weighted_sum_with_activation_series(bullish_pve_components_with_weights, df_index, pve_activation_sensitivity,
                                                                deception_index, wash_trade_intensity, flow_credibility, deception_mod_params, is_bullish=True, debug_info=debug_info, component_type="PVE看涨")
        bearish_pve_fused = weighted_sum_with_activation_series(bearish_pve_components_with_weights, df_index, pve_activation_sensitivity,
                                                                deception_index, wash_trade_intensity, flow_credibility, deception_mod_params, is_bullish=False, debug_info=debug_info, component_type="PVE看跌")
        pve_score = (bullish_pve_fused - bearish_pve_fused).clip(-1, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: PVE看涨融合: {bullish_pve_fused.loc[probe_ts]:.4f}, PVE看跌融合: {bearish_pve_fused.loc[probe_ts]:.4f} -> PVE分数: {pve_score.loc[probe_ts]:.4f}")
        # 修正：调用 weighted_sum_with_activation_series 时，传递正确的 components_with_weights 结构
        bullish_pt_fused = weighted_sum_with_activation_series(bullish_pt_components_with_weights, df_index, pt_activation_sensitivity,
                                                               deception_index, wash_trade_intensity, flow_credibility, deception_mod_params, is_bullish=True, debug_info=debug_info, component_type="PT看涨")
        bearish_pt_fused = weighted_sum_with_activation_series(bearish_pt_components_with_weights, df_index, pt_activation_sensitivity,
                                                               deception_index, wash_trade_intensity, flow_credibility, deception_mod_params, is_bullish=False, debug_info=debug_info, component_type="PT看跌")
        pt_score = (bullish_pt_fused - bearish_pt_fused).clip(-1, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: PT看涨融合: {bullish_pt_fused.loc[probe_ts]:.4f}, PT看跌融合: {bearish_pt_fused.loc[probe_ts]:.4f} -> PT分数: {pt_score.loc[probe_ts]:.4f}")
        # 修正：调用 weighted_sum_with_activation_series 时，传递正确的 components_with_weights 结构
        bullish_ls_fused = weighted_sum_with_activation_series(bullish_ls_components_with_weights, df_index, ls_activation_sensitivity,
                                                               deception_index, wash_trade_intensity, flow_credibility, deception_mod_params, is_bullish=True, debug_info=debug_info, component_type="LS看涨")
        bearish_ls_fused = weighted_sum_with_activation_series(bearish_ls_components_with_weights, df_index, ls_activation_sensitivity,
                                                               deception_index, wash_trade_intensity, flow_credibility, deception_mod_params, is_bullish=False, debug_info=debug_info, component_type="LS看跌")
        ls_score = (bullish_ls_fused - bearish_ls_fused).clip(-1, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: LS看涨融合: {bullish_ls_fused.loc[probe_ts]:.4f}, LS看跌融合: {bearish_ls_fused.loc[probe_ts]:.4f} -> LS分数: {ls_score.loc[probe_ts]:.4f}")
        # 替换 FUSION_BIPOLAR_TREND_QUALITY 为 SCORE_STRUCT_AXIOM_TREND_FORM
        trend_quality = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0, debug_info).fillna(0.0)
        # 替换 FUSION_BIPOLAR_MARKET_REGIME 为 SCORE_CYCLICAL_HURST_MEMORY
        market_regime = self._get_atomic_score(df, 'SCORE_CYCLICAL_HURST_MEMORY', 0.0, debug_info).fillna(0.0)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 趋势形态 (基础信号): {trend_quality.loc[probe_ts]:.4f}, Hurst记忆性 (基础信号): {market_regime.loc[probe_ts]:.4f}")
        base_weights = ld_params.get('base_weights', {'pve': 0.33, 'pt': 0.34, 'ls': 0.33})
        context_mod_params = ld_params.get('final_fusion_context_modulation_params', {})
        tq_pve_pt_boost_factor = get_param_value(context_mod_params.get('tq_pve_pt_boost_factor'), 0.3)
        tq_ls_boost_factor = get_param_value(context_mod_params.get('tq_ls_boost_factor'), 0.4)
        mr_pve_pt_boost_factor = get_param_value(context_mod_params.get('mr_pve_pt_boost_factor'), 0.2)
        mr_ls_boost_factor = get_param_value(context_mod_params.get('mr_ls_boost_factor'), 0.2)
        tq_non_linear_sensitivity = get_param_value(context_mod_params.get('tq_non_linear_sensitivity'), 2.0)
        mr_non_linear_sensitivity = get_param_value(context_mod_params.get('mr_non_linear_sensitivity'), 2.0)
        modulated_tq = np.tanh(trend_quality * tq_non_linear_sensitivity)
        modulated_mr = np.tanh(market_regime * mr_non_linear_sensitivity)
        dynamic_weights_pve = base_weights['pve'] * (1 + modulated_tq.clip(lower=0) * tq_pve_pt_boost_factor + modulated_mr.clip(lower=0) * mr_pve_pt_boost_factor)
        dynamic_weights_pt = base_weights['pt'] * (1 + modulated_tq.clip(lower=0) * tq_pve_pt_boost_factor + modulated_mr.clip(lower=0) * mr_pve_pt_boost_factor)
        dynamic_weights_ls = base_weights['ls'] * (1 + modulated_tq.clip(upper=0).abs() * tq_ls_boost_factor + modulated_mr.clip(upper=0).abs() * mr_ls_boost_factor)
        total_dynamic_weight = dynamic_weights_pve + dynamic_weights_pt + dynamic_weights_ls
        total_dynamic_weight = total_dynamic_weight.replace(0, 1.0) # 避免除以零
        normalized_weights_pve = dynamic_weights_pve / total_dynamic_weight
        normalized_weights_pt = dynamic_weights_pt / total_dynamic_weight
        normalized_weights_ls = dynamic_weights_ls / total_dynamic_weight
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 动态权重 (PVE: {normalized_weights_pve.loc[probe_ts]:.4f}, PT: {normalized_weights_pt.loc[probe_ts]:.4f}, LS: {normalized_weights_ls.loc[probe_ts]:.4f})")
        raw_fusion_score = (
            pve_score * normalized_weights_pve +
            pt_score * normalized_weights_pt +
            ls_score * normalized_weights_ls
        )
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 原始融合分数: {raw_fusion_score.loc[probe_ts]:.4f}")
        synergy_conflict_params = ld_params.get('synergy_conflict_params', {})
        synergy_threshold = get_param_value(synergy_conflict_params.get('synergy_threshold'), 0.5)
        conflict_threshold = get_param_value(synergy_conflict_params.get('conflict_threshold'), -0.5)
        synergy_bonus_factor = get_param_value(synergy_conflict_params.get('synergy_bonus_factor'), 0.1)
        conflict_penalty_factor = get_param_value(synergy_conflict_params.get('conflict_penalty_factor'), 0.1)
        deception_impact_on_synergy = get_param_value(synergy_conflict_params.get('deception_impact_on_synergy'), 0.5)
        synergy_conflict_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        # 强看涨协同
        strong_bullish_synergy = (pve_score > synergy_threshold) & (pt_score > synergy_threshold) & (ls_score > synergy_threshold)
        synergy_conflict_modulator.loc[strong_bullish_synergy] += synergy_bonus_factor * (1 - deception_index.clip(lower=0) * deception_impact_on_synergy)
        # 强看跌协同
        strong_bearish_synergy = (pve_score < conflict_threshold) & (pt_score < conflict_threshold) & (ls_score < conflict_threshold)
        synergy_conflict_modulator.loc[strong_bearish_synergy] += synergy_bonus_factor * (1 - deception_index.clip(upper=0).abs() * deception_impact_on_synergy)
        # 冲突情境：PVE看涨，但PT和LS看跌
        bullish_pve_bearish_pt_ls = (pve_score > synergy_threshold) & (pt_score < conflict_threshold) & (ls_score < conflict_threshold)
        synergy_conflict_modulator.loc[bullish_pve_bearish_pt_ls] -= conflict_penalty_factor * (1 + deception_index.clip(lower=0) * deception_impact_on_synergy)
        # 冲突情境：PVE看跌，但PT和LS看涨
        bearish_pve_bullish_pt_ls = (pve_score < conflict_threshold) & (pt_score > synergy_threshold) & (ls_score > synergy_threshold)
        synergy_conflict_modulator.loc[bearish_pve_bullish_pt_ls] -= conflict_penalty_factor * (1 + deception_index.clip(upper=0).abs() * deception_impact_on_synergy)
        synergy_conflict_modulator = synergy_conflict_modulator.clip(0.5, 1.5)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 协同/冲突调制器: {synergy_conflict_modulator.loc[probe_ts]:.4f}")
        final_score = np.tanh(raw_fusion_score * 2.0 * synergy_conflict_modulator)
        states['FUSION_BIPOLAR_LIQUIDITY_DYNAMICS'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “流动性博弈动态”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “流动性博弈动态”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_distribution_pressure(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.8 · 诡道穿透与动态博弈版 - 依赖解耦与强化】冶炼“派发压力” (FUSION_RISK_DISTRIBUTION_PRESSURE)
        - 核心重构: 基于“主力派发意图、散户不愿承接度、市场结构脆弱性”三大维度，量化主力在高位派发筹码的风险。
        - 诡道哲学: 派发风险的本质是主力与散户的博弈，以及市场结构对这种博弈的承载能力。
                      当主力意图派发，散户却狂热承接，且市场结构脆弱时，风险达到极致。
        - 融合模型: 最终风险 = tanh( (MFDI_score^w_mfdi * RAW_unwillingness_score^w_raw * MSF_score^w_msf)^(1/sum_weights) * non_linear_sensitivity )
        - 升级说明: 维度内部子信号聚合方式调整为加权算术平均，三大维度之间聚合保持加权几何平均，以更好地体现风险的“木桶效应”。
                    强化了对原始零值信号的归一化处理。此版本增加了详细探针，用于调试和检查每一步计算。
                    引入动态权重、诡道调制、情境放大器和协同/冲突裁决，以更精准地捕捉派发风险。
        - 【V1.8 增强】将对融合信号的依赖替换为基础信号，以解耦依赖。
                      修正 `raw_score` (散户承接意愿) 的逻辑，使其在高散户狂热时放大派发风险。
                      增加 `SCORE_BEHAVIOR_DECEPTION_INDEX` (正向) 和 `SCORE_CHIP_HOLLOWING_OUT_RISK` 信号，提高派发压力的准确性。
                      修正 `RAW恐慌抑制` 逻辑，避免 `raw_score` 被错误归零。
                      增加详细探针，输出所有原料数据、关键计算节点和结果的值。
        """
        method_name = "_synthesize_distribution_pressure"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“派发压力”...")
        states = {}
        df_index = df.index
        fusion_intelligence_params = get_params_block(self.strategy, 'fusion_intelligence_params', {})
        params = fusion_intelligence_params.get('fusion_risk_distribution_pressure_params', {})
        body_weights = get_param_value(params.get('body_weights'), {})
        mfdi_signal_weights = get_param_value(params.get('mfdi_signal_weights'), {})
        raw_signal_weights = get_param_value(params.get('raw_signal_weights'), {})
        msf_signal_weights = get_param_value(params.get('msf_signal_weights'), {})
        non_linear_sensitivity = get_param_value(params.get('non_linear_sensitivity'), 2.0)
        norm_window = get_param_value(params.get('norm_window'), 55)
        mtf_norm_weights = get_param_value(params.get('mtf_norm_weights'), {})
        # --- 1. MFDI (主力派发意图) ---
        # 新增博弈欺骗指数正向部分，作为主力派发意图的证据
        mfdi_signal_weights['SCORE_BEHAVIOR_DECEPTION_INDEX_POSITIVE'] = mfdi_signal_weights.get('SCORE_BEHAVIOR_DECEPTION_INDEX_POSITIVE', 0.05) # 假设权重为0.05
        mfdi_weighted_sum = pd.Series(0.0, index=df_index, dtype=np.float32)
        mfdi_total_weight = sum(mfdi_signal_weights.values())
        if mfdi_total_weight > 0:
            for signal, weight in mfdi_signal_weights.items():
                score = self._get_normalized_risk_score(df, signal, norm_window, mtf_norm_weights=mtf_norm_weights, debug_info=debug_info)
                mfdi_weighted_sum += score * weight
            mfdi_score = mfdi_weighted_sum / mfdi_total_weight
        else:
            mfdi_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # MFDI 动态权重与诡道调制
        mfdi_dynamic_weights_params = get_param_value(params.get('mfdi_dynamic_weights'), {})
        if get_param_value(mfdi_dynamic_weights_params.get('enabled'), False):
            # 替换 conviction_signal 的来源为基础信号
            conviction_signal = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CONVICTION', 0.0, debug_info)
            credibility_signal = self._get_atomic_score(df, get_param_value(mfdi_dynamic_weights_params.get('credibility_signal')), 0.0, debug_info)
            conviction_sensitivity = get_param_value(mfdi_dynamic_weights_params.get('conviction_sensitivity'), 0.5)
            credibility_sensitivity = get_param_value(mfdi_dynamic_weights_params.get('credibility_sensitivity'), 0.3)
            base_weight = get_param_value(mfdi_dynamic_weights_params.get('base_weight'), 1.0)
            dynamic_mfdi_weight_mod = (base_weight + conviction_signal.clip(lower=0) * conviction_sensitivity + credibility_signal.clip(lower=0) * credibility_sensitivity).clip(0.5, 1.5)
            mfdi_score = mfdi_score * dynamic_mfdi_weight_mod
        mfdi_deception_modulation_params = get_param_value(params.get('mfdi_deception_modulation'), {})
        if get_param_value(mfdi_deception_modulation_params.get('enabled'), False):
            deception_signal = self._get_atomic_score(df, get_param_value(mfdi_deception_modulation_params.get('deception_signal')), 0.0, debug_info)
            amplifier_factor = get_param_value(mfdi_deception_modulation_params.get('amplifier_factor'), 0.5)
            threshold = get_param_value(mfdi_deception_modulation_params.get('threshold'), 0.1)
            deception_amplifier = (deception_signal.clip(lower=threshold) * amplifier_factor).clip(0, 1)
            mfdi_score = mfdi_score * (1 + deception_amplifier)
        # --- 2. RAW (散户承接意愿) ---
        raw_weighted_sum = pd.Series(0.0, index=df_index, dtype=np.float32)
        raw_total_weight = sum(raw_signal_weights.values())
        if raw_total_weight > 0:
            for signal, weight in raw_signal_weights.items():
                score = self._get_normalized_risk_score(df, signal, norm_window, mtf_norm_weights=mtf_norm_weights, debug_info=debug_info)
                raw_weighted_sum += score * weight
            raw_score = raw_weighted_sum / raw_total_weight
        else:
            raw_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # RAW 动态权重与恐慌抑制
        raw_dynamic_weights_params = get_param_value(params.get('raw_dynamic_weights'), {})
        if get_param_value(raw_dynamic_weights_params.get('enabled'), False):
            fomo_signal = self._get_atomic_score(df, get_param_value(raw_dynamic_weights_params.get('fomo_signal')), 0.0, debug_info)
            sentiment_signal = self._get_atomic_score(df, get_param_value(raw_dynamic_weights_params.get('sentiment_signal')), 0.0, debug_info)
            fomo_sensitivity = get_param_value(raw_dynamic_weights_params.get('fomo_sensitivity'), 0.5)
            sentiment_sensitivity = get_param_value(raw_dynamic_weights_params.get('sentiment_sensitivity'), 0.3)
            base_weight = get_param_value(raw_dynamic_weights_params.get('base_weight'), 1.0)
            dynamic_raw_weight_mod = (base_weight + fomo_signal.clip(lower=0) * fomo_sensitivity + sentiment_signal.clip(lower=0) * sentiment_sensitivity).clip(0.5, 1.5)
            raw_score = raw_score * dynamic_raw_weight_mod
        raw_panic_dampener_params = get_param_value(params.get('raw_panic_dampener'), {})
        if get_param_value(raw_panic_dampener_params.get('enabled'), False):
            panic_signal = self._get_atomic_score(df, get_param_value(raw_panic_dampener_params.get('panic_signal')), 0.0, debug_info)
            dampener_factor = get_param_value(raw_panic_dampener_params.get('dampener_factor'), 0.5)
            # 修正恐慌抑制逻辑：使用 tanh 激活函数，使其平滑抑制，而不是直接归零
            # panic_dampener = (panic_signal.clip(lower=threshold) * dampener_factor).clip(0, 1)
            panic_dampener = (np.tanh(panic_signal * dampener_factor)).clip(0, 1) # 使用 tanh 平滑抑制
            raw_score = raw_score * (1 - panic_dampener)
        # --- 3. MSF (市场结构脆弱性) ---
        # 新增筹码空心化风险，作为市场结构脆弱性的证据
        msf_signal_weights['SCORE_CHIP_HOLLOWING_OUT_RISK'] = msf_signal_weights.get('SCORE_CHIP_HOLLOWING_OUT_RISK', 0.05) # 假设权重为0.05
        msf_weighted_sum = pd.Series(0.0, index=df_index, dtype=np.float32)
        msf_total_weight = sum(msf_signal_weights.values())
        if msf_total_weight > 0:
            for signal, weight in msf_signal_weights.items():
                score = self._get_normalized_risk_score(df, signal, norm_window, mtf_norm_weights=mtf_norm_weights, debug_info=debug_info)
                msf_weighted_sum += score * weight
                if is_debug_enabled and probe_ts and probe_ts in df.index:
                    print(f"        [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: MSF组件 '{signal}' (归一化: {score.loc[probe_ts]:.4f}, 权重: {weight})")
            msf_score = msf_weighted_sum / msf_total_weight
        else:
            msf_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # MSF 动态权重与流动性陷阱放大
        msf_dynamic_weights_params = get_param_value(params.get('msf_dynamic_weights'), {})
        if get_param_value(msf_dynamic_weights_params.get('enabled'), False):
            volatility_signal = self._get_atomic_score(df, get_param_value(msf_dynamic_weights_params.get('volatility_signal')), 0.0, debug_info)
            tension_signal = self._get_atomic_score(df, get_param_value(msf_dynamic_weights_params.get('tension_signal')), 0.0, debug_info)
            volatility_sensitivity = get_param_value(msf_dynamic_weights_params.get('volatility_sensitivity'), 0.5)
            tension_sensitivity = get_param_value(msf_dynamic_weights_params.get('tension_sensitivity'), 0.3)
            base_weight = get_param_value(msf_dynamic_weights_params.get('base_weight'), 1.0)
            dynamic_msf_weight_mod = (base_weight + volatility_signal.clip(lower=0) * volatility_sensitivity + tension_signal.clip(lower=0) * tension_sensitivity).clip(0.5, 1.5)
            msf_score = msf_score * dynamic_msf_weight_mod
        msf_liquidity_trap_amplifier_params = get_param_value(params.get('msf_liquidity_trap_amplifier'), {})
        if get_param_value(msf_liquidity_trap_amplifier_params.get('enabled'), False):
            efficiency_signal = self._get_atomic_score(df, get_param_value(msf_liquidity_trap_amplifier_params.get('efficiency_signal')), 0.0, debug_info)
            amplifier_factor = get_param_value(msf_liquidity_trap_amplifier_params.get('amplifier_factor'), 0.5)
            threshold = get_param_value(msf_liquidity_trap_amplifier_params.get('threshold'), 0.3)
            liquidity_trap_amplifier = (1 - efficiency_signal.clip(upper=threshold)) * amplifier_factor
            msf_score = msf_score * (1 + liquidity_trap_amplifier)
        # --- 4. 融合三体分数 (加权几何平均) ---
        # 【V1.5 变更】修正 `retail_unwillingness_score` 的逻辑，使其在高散户狂热时放大派发风险
        retail_unwillingness_score = raw_score # 直接使用 raw_score，高分代表散户承接意愿强，即派发风险高
        final_log_sum = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_body_weight = sum(body_weights.values())
        if total_body_weight > 0:
            mfdi_score_clipped = mfdi_score.clip(lower=1e-9)
            retail_unwillingness_score_clipped = retail_unwillingness_score.clip(lower=1e-9)
            msf_score_clipped = msf_score.clip(lower=1e-9)
            mfdi_log_contribution = np.log(mfdi_score_clipped) * (body_weights.get('main_force_distribution_intent', 0.0) / total_body_weight)
            raw_log_contribution = np.log(retail_unwillingness_score_clipped) * (body_weights.get('retail_absorption_willingness', 0.0) / total_body_weight)
            msf_log_contribution = np.log(msf_score_clipped) * (body_weights.get('market_structural_fragility', 0.0) / total_body_weight)
            final_log_sum += mfdi_log_contribution + raw_log_contribution + msf_log_contribution
            geometric_mean_score = np.exp(final_log_sum)
        else:
            geometric_mean_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # 动态非线性敏感度
        final_fusion_dynamic_exponent_params = get_param_value(params.get('final_fusion_dynamic_exponent'), {})
        dynamic_non_linear_sensitivity = pd.Series(non_linear_sensitivity, index=df_index, dtype=np.float32)
        if get_param_value(final_fusion_dynamic_exponent_params.get('enabled'), False):
            # 替换 trend_quality_signal 和 market_regime_signal 的来源为基础信号
            trend_quality_signal = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0, debug_info)
            market_regime_signal = self._get_atomic_score(df, 'SCORE_CYCLICAL_HURST_MEMORY', 0.0, debug_info)
            tq_sensitivity = get_param_value(final_fusion_dynamic_exponent_params.get('tq_sensitivity'), 0.5)
            mr_sensitivity = get_param_value(final_fusion_dynamic_exponent_params.get('mr_sensitivity'), 0.3)
            base_exponent = get_param_value(final_fusion_dynamic_exponent_params.get('base_exponent'), 2.0)
            min_exponent = get_param_value(final_fusion_dynamic_exponent_params.get('min_exponent'), 1.0)
            max_exponent = get_param_value(final_fusion_dynamic_exponent_params.get('max_exponent'), 3.0)
            dynamic_exponent_mod = (1 - trend_quality_signal * tq_sensitivity - market_regime_signal * mr_sensitivity).clip(-1, 1)
            dynamic_non_linear_sensitivity = (base_exponent * (1 + dynamic_exponent_mod)).clip(min_exponent, max_exponent)
        # 协同/冲突调制
        synergy_conflict_params = get_param_value(params.get('final_fusion_synergy_conflict'), {})
        synergy_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        if get_param_value(synergy_conflict_params.get('enabled'), False):
            synergy_threshold = get_param_value(synergy_conflict_params.get('synergy_threshold'), 0.7)
            conflict_threshold = get_param_value(synergy_conflict_params.get('conflict_threshold'), 0.3)
            synergy_bonus_factor = get_param_value(synergy_conflict_params.get('synergy_bonus_factor'), 0.2)
            conflict_penalty_factor = get_param_value(synergy_conflict_params.get('conflict_penalty_factor'), 0.2)
            strong_synergy_mask = (mfdi_score > synergy_threshold) & (retail_unwillingness_score > synergy_threshold) & (msf_score > synergy_threshold)
            synergy_modulator.loc[strong_synergy_mask] += synergy_bonus_factor
            conflict_mask = (mfdi_score > synergy_threshold) & ((retail_unwillingness_score < conflict_threshold) | (msf_score < conflict_threshold))
            synergy_modulator.loc[conflict_mask] -= conflict_penalty_factor
            synergy_modulator = synergy_modulator.clip(0.5, 1.5)
        tanh_input = geometric_mean_score * dynamic_non_linear_sensitivity * synergy_modulator
        final_distribution_pressure = (np.tanh(tanh_input) + 1) / 2
        final_distribution_pressure = final_distribution_pressure.clip(0, 1).astype(np.float32)
        states['FUSION_RISK_DISTRIBUTION_PRESSURE'] = final_distribution_pressure
        print(f"  -- [融合层] “派发压力”冶炼完成，最新分值: {final_distribution_pressure.iloc[-1]:.4f}")
        return states




    def _synthesize_multi_layer_reversal_confirmation(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.0 · 万法归宗版】冶炼“多层反转共振确认” (FUSION_BIPOLAR_MULTI_LAYER_REVERSAL_CONFIRMATION)
        - 核心思路: 融合各情报层（行为、结构、形态、力学、筹码、资金流、微观行为）的底部/顶部反转证据。
                    通过非线性加权聚合，量化反转信号的广度、深度和强度，从而提供一个更可靠的趋势转折预警。
        - 融合哲学: “万法归宗，反转共鸣”。当各情报层都指向同一方向的反转时，其可靠性远超单一信号。
        """
        method_name = "_synthesize_multi_layer_reversal_confirmation"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“多层反转共振确认”...")
        states = {}
        df_index = df.index
        # 从 fusion.json 中获取参数
        fusion_params = self.params.get('fusion_multi_layer_reversal_confirmation_params', {})
        bullish_weights_config = get_param_value(fusion_params.get('bullish_weights'), {})
        bearish_weights_config = get_param_value(fusion_params.get('bearish_weights'), {})
        non_linear_exponent = get_param_value(fusion_params.get('non_linear_exponent'), 2.0)
        # 1. 收集看涨反转信号
        bullish_reversal_components = []
        for signal_name, weight in bullish_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'SCORE_PATTERN_AXIOM_REVERSAL_POSITIVE':
                # SCORE_PATTERN_AXIOM_REVERSAL 是双极性信号，取正向部分
                original_score = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_REVERSAL', 0.0, debug_info)
                score = original_score.clip(lower=0)
            elif signal_name == 'PROCESS_META_MICRO_BEHAVIOR_BOTTOM_REVERSAL':
                # 暂时缺失，返回0
                score = pd.Series(0.0, index=df_index, dtype=np.float32)
                if is_debug_enabled and probe_ts and probe_ts in df.index:
                    print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 警告：看涨信号 '{signal_name}' 缺失，使用0.0填充。")
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info).clip(0, 1)
            bullish_reversal_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 看涨信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 2. 收集看跌反转信号
        bearish_reversal_components = []
        for signal_name, weight in bearish_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'SCORE_PATTERN_AXIOM_REVERSAL_NEGATIVE':
                # SCORE_PATTERN_AXIOM_REVERSAL 是双极性信号，取负向部分的绝对值
                original_score = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_REVERSAL', 0.0, debug_info)
                score = original_score.clip(upper=0).abs()
            elif signal_name == 'PROCESS_META_MICRO_BEHAVIOR_TOP_REVERSAL':
                # 暂时缺失，返回0
                score = pd.Series(0.0, index=df_index, dtype=np.float32)
                if is_debug_enabled and probe_ts and probe_ts in df.index:
                    print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 警告：看跌信号 '{signal_name}' 缺失，使用0.0填充。")
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info).clip(0, 1)
            bearish_reversal_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 看跌信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 3. 融合看涨反转信号 (加权几何平均)
        bullish_score = pd.Series(1.0, index=df_index, dtype=np.float32)
        total_bullish_weight = sum(w for _, w, _ in bullish_reversal_components)
        if total_bullish_weight > 0:
            for comp, weight, _ in bullish_reversal_components:
                # 修正：先fillna(0.0)再clip，确保NaN和0都贡献0
                bullish_score *= (comp.fillna(0.0).clip(lower=1e-9)) ** (weight / total_bullish_weight)
        else:
            bullish_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        bullish_score = bullish_score.fillna(0.0).clip(0, 1)
        # 4. 融合看跌反转信号 (加权几何平均)
        bearish_score = pd.Series(1.0, index=df_index, dtype=np.float32)
        total_bearish_weight = sum(w for _, w, _ in bearish_reversal_components)
        if total_bearish_weight > 0:
            for comp, weight, _ in bearish_reversal_components:
                # 修正：先fillna(0.0)再clip，确保NaN和0都贡献0
                bearish_score *= (comp.fillna(0.0).clip(lower=1e-9)) ** (weight / total_bearish_weight)
        else:
            bearish_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        bearish_score = bearish_score.fillna(0.0).clip(0, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 融合看涨分数: {bullish_score.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 融合看跌分数: {bearish_score.loc[probe_ts]:.4f}")
        # 5. 计算最终双极性分数
        # 使用非线性指数放大，增强极端情况下的信号强度
        final_score = (bullish_score**non_linear_exponent - bearish_score**non_linear_exponent).clip(-1, 1)
        states['FUSION_BIPOLAR_MULTI_LAYER_REVERSAL_CONFIRMATION'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “多层反转共振确认”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “多层反转共振确认”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_breakout_sustainability(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.0 · 穿透假象版】冶炼“突破持续性品质” (FUSION_BIPOLAR_BREAKOUT_SUSTAINABILITY)
        - 核心思路: 融合结构层的突破准备度、行为层的新高强度和量能爆发、力学层的结构动量以及形态层的突破公理。
                    旨在区分“昙花一现”的诱多突破和“厚积薄发”的真突破。
        - 融合哲学: “突破真伪，结构为王”。一个健康的突破，必须有坚实的结构基础、充沛的动能和明确的行为支持。
        """
        method_name = "_synthesize_breakout_sustainability"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“突破持续性品质”...")
        states = {}
        df_index = df.index
        # 从 fusion.json 中获取参数
        fusion_params = self.params.get('fusion_breakout_sustainability_params', {})
        bullish_weights_config = get_param_value(fusion_params.get('bullish_weights'), {})
        bearish_weights_config = get_param_value(fusion_params.get('bearish_weights'), {})
        non_linear_exponent = get_param_value(fusion_params.get('non_linear_exponent'), 2.0)
        # 1. 收集看涨突破信号
        bullish_breakout_components = []
        for signal_name, weight in bullish_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'SCORE_STRUCT_MOMENTUM_POSITIVE':
                original_score = self._get_atomic_score(df, 'SCORE_STRUCT_MOMENTUM', 0.0, debug_info)
                score = original_score.clip(lower=0)
            elif signal_name == 'SCORE_PATTERN_AXIOM_BREAKOUT_POSITIVE':
                original_score = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_BREAKOUT', 0.0, debug_info)
                score = original_score.clip(lower=0)
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info).clip(0, 1)
            bullish_breakout_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 看涨突破信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 2. 收集看跌突破信号
        bearish_breakout_components = []
        for signal_name, weight in bearish_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'SCORE_STRUCT_MOMENTUM_NEGATIVE':
                original_score = self._get_atomic_score(df, 'SCORE_STRUCT_MOMENTUM', 0.0, debug_info)
                score = original_score.clip(upper=0).abs()
            elif signal_name == 'SCORE_PATTERN_AXIOM_BREAKOUT_NEGATIVE':
                original_score = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_BREAKOUT', 0.0, debug_info)
                score = original_score.clip(upper=0).abs()
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info).clip(0, 1)
            bearish_breakout_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 看跌突破信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 3. 融合看涨突破信号 (加权几何平均)
        bullish_score = pd.Series(1.0, index=df_index, dtype=np.float32)
        total_bullish_weight = sum(w for _, w, _ in bullish_breakout_components)
        if total_bullish_weight > 0:
            for comp, weight, _ in bullish_breakout_components:
                # 修正：先fillna(0.0)再clip，确保NaN和0都贡献0
                bullish_score *= (comp.fillna(0.0).clip(lower=1e-9)) ** (weight / total_bullish_weight)
        else:
            bullish_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        bullish_score = bullish_score.fillna(0.0).clip(0, 1)
        # 4. 融合看跌突破信号 (加权几何平均)
        bearish_score = pd.Series(1.0, index=df_index, dtype=np.float32)
        total_bearish_weight = sum(w for _, w, _ in bearish_breakout_components)
        if total_bearish_weight > 0:
            for comp, weight, _ in bearish_breakout_components:
                # 修正：先fillna(0.0)再clip，确保NaN和0都贡献0
                bearish_score *= (comp.fillna(0.0).clip(lower=1e-9)) ** (weight / total_bearish_weight)
        else:
            bearish_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        bearish_score = bearish_score.fillna(0.0).clip(0, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 融合看涨分数: {bullish_score.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 融合看跌分数: {bearish_score.loc[probe_ts]:.4f}")
        # 5. 计算最终双极性分数
        final_score = (bullish_score**non_linear_exponent - bearish_score**non_linear_exponent).clip(-1, 1)
        states['FUSION_BIPOLAR_BREAKOUT_SUSTAINABILITY'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “突破持续性品质”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “突破持续性品质”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_intraday_strategic_alignment(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.0 · 战术意图版】冶炼“日内战略协同度” (FUSION_BIPOLAR_INTRADAY_STRATEGIC_ALIGNMENT)
        - 核心思路: 融合日内行为K线质量、战术弧线、竞价意图等日内信号，与微观战略意图进行协同评估。
                    高分代表日内战术与微观战略高度一致，主力控盘坚决；低分或负分则可能预示日内行为与战略意图存在冲突。
        - 融合哲学: “日内战术，宏观意图”。微观行为是宏观战略的缩影，其协同性决定了趋势的纯度。
        """
        method_name = "_synthesize_intraday_strategic_alignment"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“日内战略协同度”...")
        states = {}
        df_index = df.index
        # 从 fusion.json 中获取参数
        fusion_params = self.params.get('fusion_intraday_strategic_alignment_params', {})
        intraday_weights_config = get_param_value(fusion_params.get('intraday_weights'), {})
        micro_strategic_intent_weight = get_param_value(fusion_params.get('micro_strategic_intent_weight'), 0.5)
        synergy_factor = get_param_value(fusion_params.get('synergy_factor'), 0.5)
        non_linear_exponent = get_param_value(fusion_params.get('non_linear_exponent'), 1.5)
        # 1. 收集日内行为信号
        intraday_components = []
        for signal_name, weight in intraday_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'BIPOLAR_BEHAVIORAL_DAY_QUALITY':
                # 暂时缺失，返回0
                score = pd.Series(0.0, index=df_index, dtype=np.float32)
                if is_debug_enabled and probe_ts and probe_ts in df.index:
                    print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 警告：日内信号 '{signal_name}' 缺失，使用0.0填充。")
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info)
            intraday_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 日内信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 2. 融合日内行为信号 (加权平均)
        intraday_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_intraday_weight = sum(w for _, w, _ in intraday_components)
        if total_intraday_weight > 0:
            for comp, weight, _ in intraday_components:
                # 对于加权平均，NaN会传播，所以这里需要fillna(0.0)
                intraday_score += comp.fillna(0.0) * weight
            intraday_score /= total_intraday_weight
        intraday_score = intraday_score.fillna(0.0).clip(-1, 1)
        # 3. 获取微观战略意图
        micro_strategic_intent = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGIC_INTENT', 0.0, debug_info).fillna(0.0).clip(-1, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 微观战略意图: {micro_strategic_intent.loc[probe_ts]:.4f}")
        # 4. 计算协同度
        # 当日内行为和微观战略意图方向一致时，协同度高；方向相反时，协同度低
        synergy_score = (intraday_score * micro_strategic_intent).clip(-1, 1)
        # 5. 最终融合：日内行为与微观战略意图的加权平均，并由协同度进行非线性调制
        # 协同度越高，最终分数越接近两者平均值；协同度越低，分数越被拉向0
        # 使用 tanh 激活函数将协同度映射到 [0, 1] 范围，作为调制因子
        synergy_modulator = (np.tanh(synergy_score * synergy_factor) + 1) / 2
        # 最终分数 = (日内行为分数 * (1 - micro_strategic_intent_weight) + 微观战略意图 * micro_strategic_intent_weight) * 协同调制器
        final_score_base = (intraday_score * (1 - micro_strategic_intent_weight) + micro_strategic_intent * micro_strategic_intent_weight).clip(-1, 1)
        final_score = (final_score_base * synergy_modulator**non_linear_exponent).clip(-1, 1)
        states['FUSION_BIPOLAR_INTRADAY_STRATEGIC_ALIGNMENT'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 日内融合分数: {intraday_score.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 协同分数: {synergy_score.loc[probe_ts]:.4f}, 协同调制器: {synergy_modulator.loc[probe_ts]:.4f}")
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “日内战略协同度”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “日内战略协同度”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_capital_dominance_shift(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.0 · 谁主沉浮版】冶炼“资本主导力转移” (FUSION_BIPOLAR_CAPITAL_DOMINANCE_SHIFT)
        - 核心思路: 融合资金流的资本属性、资金流纯度与动能、筹码的战略态势以及市场情绪钟摆。
                    量化当前市场由哪种资本主导，以及这种主导力是否正在发生转移。
        - 融合哲学: “资本属性，市场主导”。谁在控盘，决定了市场的脾气和走向。
        """
        method_name = "_synthesize_capital_dominance_shift"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“资本主导力转移”...")
        states = {}
        df_index = df.index
        # 从 fusion.json 中获取参数
        fusion_params = self.params.get('fusion_capital_dominance_shift_params', {})
        capital_attribute_weight = get_param_value(fusion_params.get('capital_attribute_weight'), 0.4)
        flow_momentum_weight = get_param_value(fusion_params.get('flow_momentum_weight'), 0.3)
        chip_posture_weight = get_param_value(fusion_params.get('chip_posture_weight'), 0.2)
        sentiment_pendulum_weight = get_param_value(fusion_params.get('sentiment_pendulum_weight'), 0.1)
        non_linear_exponent = get_param_value(fusion_params.get('non_linear_exponent'), 1.5)
        credibility_mod_factor = get_param_value(fusion_params.get('credibility_mod_factor'), 0.5)
        # 1. 获取核心输入信号
        capital_attribute = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CAPITAL_SIGNATURE', 0.0, debug_info).fillna(0.0).clip(-1, 1)
        flow_momentum = self._get_atomic_score(df, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0, debug_info).fillna(0.0).clip(-1, 1)
        chip_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0, debug_info).fillna(0.0).clip(-1, 1)
        sentiment_pendulum = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0, debug_info).fillna(0.0).clip(-1, 1)
        # 修正：替换 flow_credibility_index_D 为 SCORE_FF_AXIOM_INTENT_PURITY 的正向部分
        flow_credibility = self._get_atomic_score(df, 'SCORE_FF_AXIOM_INTENT_PURITY', 0.0, debug_info).clip(lower=0).fillna(0.0).clip(0, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资本属性: {capital_attribute.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资金流纯度与动能: {flow_momentum.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 筹码战略态势: {chip_posture.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 情绪钟摆: {sentiment_pendulum.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {probe_ts.strftime('%Y-%m-%d')}: 资金流可信度 (SCORE_FF_AXIOM_INTENT_PURITY): {flow_credibility.loc[probe_ts]:.4f}")
        # 2. 融合核心维度 (加权平均)
        raw_dominance_score = (
            capital_attribute * capital_attribute_weight +
            flow_momentum * flow_momentum_weight +
            chip_posture * chip_posture_weight +
            sentiment_pendulum * sentiment_pendulum_weight
        ) / (capital_attribute_weight + flow_momentum_weight + chip_posture_weight + sentiment_pendulum_weight)
        raw_dominance_score = raw_dominance_score.clip(-1, 1)
        # 3. 资金流可信度作为调制器，增强或削弱信号
        # 可信度越高，信号越可靠，调制器越接近1；可信度越低，调制器越接近0，削弱信号
        credibility_modulator = (flow_credibility * credibility_mod_factor + (1 - credibility_mod_factor)).clip(0, 1) # 确保在0到1之间
        # 4. 最终分数：原始主导力分数 × 可信度调制器，并进行非线性放大
        final_score = (raw_dominance_score * credibility_modulator)**non_linear_exponent
        final_score = final_score.clip(-1, 1)
        states['FUSION_BIPOLAR_CAPITAL_DOMINANCE_SHIFT'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 原始主导力分数: {raw_dominance_score.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 可信度调制器: {credibility_modulator.loc[probe_ts]:.4f}")
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “资本主导力转移”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “资本主导力转移”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_hidden_accumulation_quality(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.0 · 暗流涌动版】冶炼“隐蔽吸筹品质” (FUSION_OPPORTUNITY_HIDDEN_ACCUMULATION_QUALITY)
        - 核心思路: 融合各种隐蔽吸筹的过程信号（如隐蔽吸筹、诡道吸筹、恐慌洗盘吸筹、上冲回落洗盘），
                    并结合成交量萎缩、筹码持仓信念韧性等品质因子，评估隐蔽吸筹的真实性、效率和可持续性。
        - 融合哲学: “暗流涌动，蓄势待发”。真正的底部，往往在寂静中酝酿。
        """
        method_name = "_synthesize_hidden_accumulation_quality"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“隐蔽吸筹品质”...")
        states = {}
        df_index = df.index
        # 从 fusion.json 中获取参数
        fusion_params = self.params.get('fusion_hidden_accumulation_quality_params', {})
        accumulation_process_weights_config = get_param_value(fusion_params.get('accumulation_process_weights'), {})
        quality_factor_weights_config = get_param_value(fusion_params.get('quality_factor_weights'), {})
        non_linear_exponent = get_param_value(fusion_params.get('non_linear_exponent'), 1.5)
        # 1. 收集吸筹过程信号
        process_components = []
        for signal_name, weight in accumulation_process_weights_config.items():
            score = self._get_atomic_score(df, signal_name, 0.0, debug_info).clip(0, 1)
            process_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 吸筹过程信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 2. 收集品质因子信号
        quality_components = []
        for signal_name, weight in quality_factor_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT_POSITIVE':
                original_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0, debug_info)
                score = original_score.clip(lower=0)
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info).clip(0, 1)
            quality_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 品质因子信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 3. 融合吸筹过程信号 (加权几何平均)
        process_score = pd.Series(1.0, index=df_index, dtype=np.float32)
        total_process_weight = sum(w for _, w, _ in process_components)
        if total_process_weight > 0:
            for comp, weight, _ in process_components:
                # 修正：先fillna(0.0)再clip，确保NaN和0都贡献0
                process_score *= (comp.fillna(0.0).clip(lower=1e-9)) ** (weight / total_process_weight)
        else:
            process_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        process_score = process_score.fillna(0.0).clip(0, 1)
        # 4. 融合品质因子信号 (加权几何平均)
        quality_score = pd.Series(1.0, index=df_index, dtype=np.float32)
        total_quality_weight = sum(w for _, w, _ in quality_components)
        if total_quality_weight > 0:
            for comp, weight, _ in quality_components:
                # 修正：先fillna(0.0)再clip，确保NaN和0都贡献0
                quality_score *= (comp.fillna(0.0).clip(lower=1e-9)) ** (weight / total_quality_weight)
        else:
            quality_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        quality_score = quality_score.fillna(0.0).clip(0, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 融合过程分数: {process_score.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 融合品质分数: {quality_score.loc[probe_ts]:.4f}")
        # 5. 最终融合：过程分数与品质分数的几何平均，并进行非线性放大
        final_score = (process_score.clip(lower=1e-9) * quality_score.clip(lower=1e-9)).pow(0.5)**non_linear_exponent
        final_score = final_score.clip(0, 1)
        states['FUSION_OPPORTUNITY_HIDDEN_ACCUMULATION_QUALITY'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “隐蔽吸筹品质”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “隐蔽吸筹品质”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_market_structure_health(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.0 · 骨架诊断版】冶炼“市场结构健康度” (FUSION_BIPOLAR_MARKET_STRUCTURE_HEALTH)
        - 核心思路: 融合平台基石品质、动态平台高低点、结构稳定性、宏观趋势健康度以及结构张力等信号。
                    提供一个对市场“骨架”健康程度的综合评估。
        - 融合哲学: “市场骨架，健康为本”。坚实的结构是趋势持续的基础，脆弱的结构则暗藏风险。
        """
        method_name = "_synthesize_market_structure_health"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“市场结构健康度”...")
        states = {}
        df_index = df.index
        # 从 fusion.json 中获取参数
        fusion_params = self.params.get('fusion_market_structure_health_params', {})
        health_components_weights_config = get_param_value(fusion_params.get('health_components_weights'), {})
        non_linear_exponent = get_param_value(fusion_params.get('non_linear_exponent'), 1.5)
        # 1. 收集结构健康度相关信号
        health_components = []
        for signal_name, weight in health_components_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'STRUCT_PLATFORM_DYNAMIC_HIGH_INVERSE':
                # 假设这是一个从 STRUCT_PLATFORM_DYNAMIC_HIGH 派生出的反向指标，暂时缺失
                score = pd.Series(0.0, index=df_index, dtype=np.float32)
                if is_debug_enabled and probe_ts and probe_ts in df.index:
                    print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 警告：结构健康组件 '{signal_name}' 缺失，使用0.0填充。")
            elif signal_name == 'STRUCT_PLATFORM_DYNAMIC_LOW_INVERSE':
                # 假设这是一个从 STRUCT_PLATFORM_DYNAMIC_LOW 派生出的反向指标，暂时缺失
                score = pd.Series(0.0, index=df_index, dtype=np.float32)
                if is_debug_enabled and probe_ts and probe_ts in df.index:
                    print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 警告：结构健康组件 '{signal_name}' 缺失，使用0.0填充。")
            elif signal_name == 'STRUCT_PLATFORM_VPOC_STABILITY':
                # 假设这是一个从 STRUCT_PLATFORM_VPOC 派生出的稳定性指标，暂时缺失
                score = pd.Series(0.0, index=df_index, dtype=np.float32)
                if is_debug_enabled and probe_ts and probe_ts in df.index:
                    print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 警告：结构健康组件 '{signal_name}' 缺失，使用0.0填充。")
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info).fillna(0.0)
            health_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 结构健康组件 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 2. 融合结构健康度信号 (加权平均)
        raw_health_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_weight = sum(w for _, w, _ in health_components)
        if total_weight > 0:
            for comp, weight, _ in health_components:
                # 对于加权平均，NaN会传播，所以这里需要fillna(0.0)
                raw_health_score += comp.fillna(0.0) * weight
            raw_health_score /= total_weight
        raw_health_score = raw_health_score.fillna(0.0).clip(-1, 1)
        # 3. 最终分数：非线性放大
        final_score = (np.sign(raw_health_score) * (raw_health_score.abs()**non_linear_exponent)).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_STRUCTURE_HEALTH'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 原始健康分数: {raw_health_score.loc[probe_ts]:.4f}")
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “市场结构健康度”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “市场结构健康度”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states

    def _synthesize_sector_leadership_dynamics(self, df: pd.DataFrame, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Dict[str, pd.Series]:
        """
        【V1.0 · 谁是真龙版】冶炼“板块龙头动态” (FUSION_BIPOLAR_SECTOR_LEADERSHIP_DYNAMICS)
        - 核心思路: 融合个股与板块的协同性、热门板块冷却迹象、结构层的龙头潜力以及基础层的相对强度。
                    评估个股在板块中的地位、板块的整体健康度以及潜在的龙头切换。
        - 融合哲学: “板块轮动，龙头先行”。洞察板块内部力量对比和资金流向，是捕捉市场热点的关键。
        """
        method_name = "_synthesize_sector_leadership_dynamics"
        is_debug_enabled, probe_ts, _ = debug_info if debug_info else (False, None, method_name)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在冶炼“板块龙头动态”...")
        states = {}
        df_index = df.index
        # 修正：将 positive_score 和 negative_score 的初始化提前，确保它们始终被定义
        positive_score = pd.Series(1.0, index=df_index, dtype=np.float32)
        negative_score = pd.Series(1.0, index=df_index, dtype=np.float32)
        # 从 fusion.json 中获取参数
        fusion_params = self.params.get('fusion_sector_leadership_dynamics_params', {})
        positive_weights_config = get_param_value(fusion_params.get('positive_weights'), {})
        negative_weights_config = get_param_value(fusion_params.get('negative_weights'), {})
        non_linear_exponent = get_param_value(fusion_params.get('non_linear_exponent'), 1.5)
        # 1. 收集正向（看涨）信号
        positive_components = []
        for signal_name, weight in positive_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'PROCESS_META_STOCK_SECTOR_SYNC_POSITIVE':
                original_score = self._get_atomic_score(df, 'PROCESS_META_STOCK_SECTOR_SYNC', 0.0, debug_info)
                score = original_score.clip(lower=0)
            elif signal_name == 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH_POSITIVE':
                original_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH', 0.0, debug_info)
                score = original_score.clip(lower=0)
            elif signal_name == 'SCORE_STRUCT_AXIOM_ENVIRONMENT_POSITIVE':
                original_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_ENVIRONMENT', 0.0, debug_info)
                score = original_score.clip(lower=0)
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info).clip(0, 1)
            positive_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正向信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 2. 收集负向（看跌）信号
        negative_components = []
        for signal_name, weight in negative_weights_config.items():
            score = pd.Series(0.0, index=df_index, dtype=np.float32) # 默认值
            if signal_name == 'PROCESS_META_STOCK_SECTOR_SYNC_NEGATIVE':
                original_score = self._get_atomic_score(df, 'PROCESS_META_STOCK_SECTOR_SYNC', 0.0, debug_info)
                score = original_score.clip(upper=0).abs()
            elif signal_name == 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH_NEGATIVE':
                original_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH', 0.0, debug_info)
                score = original_score.clip(upper=0).abs()
            elif signal_name == 'SCORE_STRUCT_AXIOM_ENVIRONMENT_NEGATIVE':
                original_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_ENVIRONMENT', 0.0, debug_info)
                score = original_score.clip(upper=0).abs()
            else:
                score = self._get_atomic_score(df, signal_name, 0.0, debug_info).clip(0, 1)
            negative_components.append((score, weight, signal_name))
            if is_debug_enabled and probe_ts and probe_ts in df.index:
                print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 负向信号 '{signal_name}': {score.loc[probe_ts]:.4f} (权重: {weight})")
        # 3. 融合正向信号 (加权几何平均)
        total_positive_weight = sum(w for _, w, _ in positive_components)
        if total_positive_weight > 0:
            for comp, weight, _ in positive_components:
                positive_score *= (comp.fillna(0.0).clip(lower=1e-9)) ** (weight / total_positive_weight)
        else:
            positive_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        positive_score = positive_score.fillna(0.0).clip(0, 1)
        # 4. 融合负向信号 (加权几何平均)
        total_negative_weight = sum(w for _, w, _ in negative_components)
        if total_negative_weight > 0:
            for comp, weight, _ in negative_components:
                negative_score *= (comp.fillna(0.0).clip(lower=1e-9)) ** (weight / total_negative_weight)
        else:
            negative_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        negative_score = negative_score.fillna(0.0).clip(0, 1)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 融合正向分数: {positive_score.loc[probe_ts]:.4f}")
            print(f"      [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 融合负向分数: {negative_score.loc[probe_ts]:.4f}")
        # 5. 计算最终双极性分数
        final_score = (positive_score**non_linear_exponent - negative_score**non_linear_exponent).clip(-1, 1)
        states['FUSION_BIPOLAR_SECTOR_LEADERSHIP_DYNAMICS'] = final_score.astype(np.float32)
        if is_debug_enabled and probe_ts and probe_ts in df.index:
            print(f"  -- [融合层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: “板块龙头动态”冶炼完成，最终分值: {final_score.loc[probe_ts]:.4f}")
        else:
            print(f"  -- [融合层] “板块龙头动态”冶炼完成，最新分值: {final_score.iloc[-1]:.4f}")
        return states















