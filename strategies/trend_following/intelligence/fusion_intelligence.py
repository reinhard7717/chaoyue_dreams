# 文件: strategies/trend_following/intelligence/fusion_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, Any
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

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [融合情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _get_atomic_score(self, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.3 · 严格原子信号获取与数据层回退版】安全地从原子状态库或主数据帧中获取分数，处理缺失情况。
        - 核心修复: 增加了对 `self.strategy.df_indicators` 的回退逻辑。
                      原子信号优先从 `self.strategy.atomic_states` 获取，
                      若无则从 `self.strategy.df_indicators` 获取，最后提供默认值。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in self.strategy.df_indicators.columns: # 增加从 df_indicators 获取的逻辑
            return self.strategy.df_indicators[name]
        else:
            print(f"    -> [融合层-原子信号警告] 预期原子信号 '{name}' 在 atomic_states 和 df_indicators 中均不存在，使用默认值 {default}。")
            return pd.Series(default, index=self.strategy.df_indicators.index)

    def run_fusion_diagnostics(self) -> Dict[str, pd.Series]:
        print("启动【V3.3 · 结构势能版】融合情报分析...") # 修改版本号
        all_fusion_states = {}
        # 每次合成后立即更新到 self.strategy.atomic_states，确保后续方法能获取到
        regime_states = self._synthesize_market_regime()
        all_fusion_states.update(regime_states)
        self.strategy.atomic_states.update(regime_states)
        quality_states = self._synthesize_trend_quality()
        all_fusion_states.update(quality_states)
        self.strategy.atomic_states.update(quality_states)
        pressure_states = self._synthesize_market_pressure()
        all_fusion_states.update(pressure_states)
        self.strategy.atomic_states.update(pressure_states)
        confrontation_states = self._synthesize_capital_confrontation()
        all_fusion_states.update(confrontation_states)
        self.strategy.atomic_states.update(confrontation_states)
        contradiction_states = self._synthesize_market_contradiction()
        all_fusion_states.update(contradiction_states)
        self.strategy.atomic_states.update(contradiction_states)
        overextension_intent_states = self._synthesize_price_overextension_intent()
        all_fusion_states.update(overextension_intent_states)
        self.strategy.atomic_states.update(overextension_intent_states)
        upper_shadow_intent_states = self._synthesize_upper_shadow_intent()
        all_fusion_states.update(upper_shadow_intent_states)
        self.strategy.atomic_states.update(upper_shadow_intent_states)
        # 冶炼“滞涨风险”
        stagnation_risk_states = self._synthesize_stagnation_risk()
        all_fusion_states.update(stagnation_risk_states)
        self.strategy.atomic_states.update(stagnation_risk_states)
        # 冶炼“趋势结构分”
        trend_structure_states = self._synthesize_trend_structure_score()
        all_fusion_states.update(trend_structure_states)
        self.strategy.atomic_states.update(trend_structure_states)
        # 新增：冶炼“资金趋势”和“筹码趋势”
        fund_flow_trend_states = self._synthesize_fund_flow_trend()
        all_fusion_states.update(fund_flow_trend_states)
        self.strategy.atomic_states.update(fund_flow_trend_states)
        chip_trend_states = self._synthesize_chip_trend()
        all_fusion_states.update(chip_trend_states)
        self.strategy.atomic_states.update(chip_trend_states)
        # 新增代码行：冶炼“筹码结构势能”
        chip_potential_states = self._synthesize_chip_structural_potential()
        all_fusion_states.update(chip_potential_states)
        self.strategy.atomic_states.update(chip_potential_states)
        # 新增：冶炼“吸筹拐点信号”
        accumulation_inflection_states = self._synthesize_accumulation_inflection()
        all_fusion_states.update(accumulation_inflection_states)
        self.strategy.atomic_states.update(accumulation_inflection_states)
        print(f"【V3.3 · 结构势能版】分析完成，生成 {len(all_fusion_states)} 个融合态势信号。") # 修改版本号
        return all_fusion_states

    def _synthesize_market_contradiction(self) -> Dict[str, pd.Series]:
        """
        【V1.0】冶炼“市场矛盾” (Market Contradiction)
        - 核心思想: 融合各情报领域（行为、筹码、资金流、结构、力学、形态、微观）的背离信号。
        - 证据链: 收集所有领域的看涨背离和看跌背离信号，进行加权融合。
        """
        print("  -- [融合层] 正在冶炼“市场矛盾”...")
        states = {}
        # 定义所有领域的背离信号源
        divergence_sources = [
            'FOUNDATION', 'STRUCTURE', 'PATTERN', 'DYNAMIC_MECHANICS',
            'CHIP', 'FUND_FLOW', 'MICRO_BEHAVIOR', 'BEHAVIOR'
        ]
        bullish_divergence_scores = []
        bearish_divergence_scores = []
        for source in divergence_sources:
            bull_signal_name = f'SCORE_{source}_BULLISH_DIVERGENCE'
            bear_signal_name = f'SCORE_{source}_BEARISH_DIVERGENCE'
            bullish_divergence_scores.append(self._get_atomic_score(bull_signal_name, 0.0).values)
            bearish_divergence_scores.append(self._get_atomic_score(bear_signal_name, 0.0).values)
        net_bullish_divergence = np.maximum.reduce(bullish_divergence_scores)
        net_bearish_divergence = np.maximum.reduce(bearish_divergence_scores)
        bipolar_contradiction = (pd.Series(net_bullish_divergence, index=self.strategy.df_indicators.index) -
                                 pd.Series(net_bearish_divergence, index=self.strategy.df_indicators.index)).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_CONTRADICTION'] = bipolar_contradiction.astype(np.float32)
        print(f"  -- [融合层] “市场矛盾”冶炼完成，最新分值: {bipolar_contradiction.iloc[-1]:.4f}")
        return states

    def _synthesize_market_regime(self) -> Dict[str, pd.Series]:
        """
        【V1.1 · 柔性评估重构版】冶炼“市场政权” (Market Regime)
        - 核心重构: 废弃了对 `hurst_memory`, `inertia`, `stability` 进行 `clip(lower=0)` 的“一票否决”逻辑。
                      改为直接使用它们的原始双极性分数进行加权平均，得到一个更柔性的 `trend_evidence`。
                      这避免了在趋势酝酿期，因单一公理暂时为负而导致整体趋势证据归零的问题。
        - 探针植入: 新增探针，打印 `hurst_memory`, `inertia`, `stability` 的原始值，以及计算出的 `trend_evidence` 和 `reversion_evidence`。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        print("  -- [融合层] 正在冶炼“市场政权”...")
        df = self.strategy.df_indicators
        states = {}
        hurst_memory = self._get_atomic_score('SCORE_CYCLICAL_HURST_MEMORY', 0.0)
        inertia = self._get_atomic_score('SCORE_DYN_AXIOM_INERTIA', 0.0)
        stability = self._get_atomic_score('SCORE_DYN_AXIOM_STABILITY', 0.0)
        trend_evidence_weights = {'hurst': 0.4, 'inertia': 0.4, 'stability': 0.2}
        trend_evidence = (
            hurst_memory * trend_evidence_weights['hurst'] +
            inertia * trend_evidence_weights['inertia'] +
            stability * trend_evidence_weights['stability']
        ).clip(-1, 1)
        reversion_evidence = (hurst_memory.clip(upper=0).abs() * inertia.clip(upper=0).abs()).pow(0.5)
        bipolar_regime = (trend_evidence - reversion_evidence).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_REGIME'] = bipolar_regime.astype(np.float32)
        return states

    def _synthesize_market_pressure(self) -> Dict[str, pd.Series]:
        """
        【V1.2 · 底分型集成版】冶炼“市场压力” (Market Pressure)
        - 核心思想: 衡量市场中“向上反转”与“向下回调”两股力量的净压力。
        - 证据链: 融合所有原子情报层和过程情报层的“底部反转”和“顶部反转”信号。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【新增】将 `SCORE_STRUCT_BOTTOM_FRACTAL` 信号作为底部反转的证据之一。
        """
        print("  -- [融合层] 正在冶炼“市场压力”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        # 定义所有领域的反转信号源 (现在从 ProcessIntelligence 获取)
        reversal_sources = [
            'PROCESS_META_FOUNDATION_BOTTOM_REVERSAL', 'PROCESS_META_FOUNDATION_TOP_REVERSAL',
            'PROCESS_META_STRUCTURE_BOTTOM_REVERSAL', 'PROCESS_META_STRUCTURE_TOP_REVERSAL',
            'PROCESS_META_PATTERN_BOTTOM_REVERSAL', 'PROCESS_META_PATTERN_TOP_REVERSAL',
            'PROCESS_META_DYNAMIC_MECHANICS_BOTTOM_REVERSAL', 'PROCESS_META_DYNAMIC_MECHANICS_TOP_REVERSAL',
            'PROCESS_META_CHIP_BOTTOM_REVERSAL', 'PROCESS_META_CHIP_TOP_REVERSAL',
            'PROCESS_META_FUND_FLOW_BOTTOM_REVERSAL', 'PROCESS_META_FUND_FLOW_TOP_REVERSAL',
            'PROCESS_META_MICRO_BEHAVIOR_BOTTOM_REVERSAL', 'PROCESS_META_MICRO_BEHAVIOR_TOP_REVERSAL',
            'PROCESS_META_BEHAVIOR_BOTTOM_REVERSAL', 'PROCESS_META_BEHAVIOR_TOP_REVERSAL'
        ]
        upward_pressure_scores = []
        downward_pressure_scores = []
        for signal_name in reversal_sources:
            if 'BOTTOM_REVERSAL' in signal_name:
                upward_pressure_scores.append(self._get_atomic_score(signal_name, 0.0).values)
            elif 'TOP_REVERSAL' in signal_name:
                downward_pressure_scores.append(self._get_atomic_score(signal_name, 0.0).values)
        # 【新增代码行】将结构底分型信号添加到向上压力证据中
        structural_bottom_fractal = self._get_atomic_score('SCORE_STRUCT_BOTTOM_FRACTAL', 0.0)
        upward_pressure_scores.append(structural_bottom_fractal.values)
        net_upward_pressure = np.maximum.reduce(upward_pressure_scores)
        net_downward_pressure = np.maximum.reduce(downward_pressure_scores)
        bipolar_pressure = (pd.Series(net_upward_pressure, index=self.strategy.df_indicators.index) -
                            pd.Series(net_downward_pressure, index=self.strategy.df_indicators.index)).clip(-1, 1)
        states['FUSION_BIPOLAR_MARKET_PRESSURE'] = bipolar_pressure.astype(np.float32)
        # 调试信息
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [市场压力探针] @ {probe_date_for_loop.date()}:")
                print(f"       - SCORE_STRUCT_BOTTOM_FRACTAL: {structural_bottom_fractal.loc[probe_date_for_loop]:.4f}")
                print(f"       - net_upward_pressure: {net_upward_pressure[df_index.get_loc(probe_date_for_loop)]:.4f}")
                print(f"       - net_downward_pressure: {net_downward_pressure[df_index.get_loc(probe_date_for_loop)]:.4f}")
                print(f"       - FUSION_BIPOLAR_MARKET_PRESSURE: {bipolar_pressure.loc[probe_date_for_loop]:.4f}")
        print(f"  -- [融合层] “市场压力”冶炼完成，最新分值: {bipolar_pressure.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_quality(self) -> Dict[str, pd.Series]:
        """
        【V2.0 · 结构动力学增强版】冶炼“趋势质量” (Trend Quality)
        - 核心增强: 引入新一代筹码高级指标 `breakout_readiness_score_D` 和 `trend_vitality_index_D` 作为趋势质量的直接证据，
                      前者衡量突破的准备程度，后者评估趋势的内在生命力。
        - 核心修复: 遵循原有融合逻辑，将各领域公理的双极性分数进行加权，形成整体判断。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        print("  -- [融合层] 正在冶炼“趋势质量”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date_for_loop = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop not in df_index:
                probe_date_for_loop = None
        foundation_trend = self._get_atomic_score('SCORE_FOUNDATION_AXIOM_TREND', 0.0)
        foundation_oscillator = self._get_atomic_score('SCORE_FOUNDATION_AXIOM_OSCILLATOR', 0.0)
        foundation_flow = self._get_atomic_score('SCORE_FOUNDATION_AXIOM_FLOW', 0.0)
        foundation_volatility = self._get_atomic_score('SCORE_FOUNDATION_AXIOM_VOLATILITY', 0.0)
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        structural_mtf_cohesion = self._get_atomic_score('SCORE_STRUCT_AXIOM_MTF_COHESION', 0.0)
        structural_stability = self._get_atomic_score('SCORE_STRUCT_AXIOM_STABILITY', 0.0)
        dynamic_momentum = self._get_atomic_score('SCORE_DYN_AXIOM_MOMENTUM', 0.0)
        dynamic_inertia = self._get_atomic_score('SCORE_DYN_AXIOM_INERTIA', 0.0)
        dynamic_stability = self._get_atomic_score('SCORE_DYN_AXIOM_STABILITY', 0.0)
        dynamic_energy = self._get_atomic_score('SCORE_DYN_AXIOM_ENERGY', 0.0)
        dynamic_ma_acceleration = self._get_atomic_score('SCORE_DYN_AXIOM_MA_ACCELERATION', 0.0)
        fund_flow_consensus = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        fund_flow_conviction = self._get_atomic_score('SCORE_FF_AXIOM_CONVICTION', 0.0)
        fund_flow_increment = self._get_atomic_score('SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0)
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        chip_cost_structure = self._get_atomic_score('SCORE_CHIP_AXIOM_COST_STRUCTURE', 0.0)
        chip_holder_sentiment = self._get_atomic_score('SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        chip_peak_integrity = self._get_atomic_score('SCORE_CHIP_AXIOM_PEAK_INTEGRITY', 0.0)
        micro_deception = self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        micro_probe = self._get_atomic_score('SCORE_MICRO_AXIOM_PROBE', 0.0)
        micro_efficiency = self._get_atomic_score('SCORE_MICRO_AXIOM_EFFICIENCY', 0.0)
        behavior_upward_efficiency = self._get_atomic_score('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.0)
        behavior_downward_resistance = self._get_atomic_score('SCORE_BEHAVIOR_DOWNWARD_RESISTANCE', 0.0)
        behavior_intraday_bull_control = self._get_atomic_score('SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', 0.0)
        pattern_reversal = self._get_atomic_score('SCORE_PATTERN_AXIOM_REVERSAL', 0.0)
        pattern_breakout = self._get_atomic_score('SCORE_PATTERN_AXIOM_BREAKOUT', 0.0)
        main_force_on_peak_flow = self._get_atomic_score('main_force_on_peak_flow_D', 0.0)
        aaa_raw = self._get_safe_series(self.strategy.df_indicators, 'AAA_D', 0.0, method_name="_synthesize_trend_quality")
        aaa_score = normalize_to_bipolar(aaa_raw * -1, df_index, window=55)
        pdi_raw = self._get_safe_series(self.strategy.df_indicators, 'PDI_14_D', 0.0, method_name="_synthesize_trend_quality")
        pdi_score = normalize_to_bipolar(pdi_raw, df_index, window=55)
        structural_bottom_fractal = self._get_atomic_score('SCORE_STRUCT_BOTTOM_FRACTAL', 0.0)
        # 新增代码行：获取新一代筹码高级指标并进行归一化
        breakout_readiness_raw = self._get_safe_series(self.strategy.df_indicators, 'breakout_readiness_score_D', 0.0, method_name="_synthesize_trend_quality")
        breakout_readiness_score = normalize_to_bipolar(breakout_readiness_raw, df_index, window=55, sensitivity=20)
        trend_vitality_raw = self._get_safe_series(self.strategy.df_indicators, 'trend_vitality_index_D', 0.0, method_name="_synthesize_trend_quality")
        trend_vitality_score = normalize_to_bipolar(trend_vitality_raw, df_index, window=55, sensitivity=0.5)
        components_and_weights = {
            'foundation_trend': (foundation_trend, 0.08),
            'foundation_oscillator': (foundation_oscillator, -0.02),
            'foundation_flow': (foundation_flow, 0.03),
            'foundation_volatility': (foundation_volatility, 0.02),
            'structural_trend_form': (structural_trend_form, 0.10),
            'structural_mtf_cohesion': (structural_mtf_cohesion, 0.05),
            'structural_stability': (structural_stability, 0.05),
            'dynamic_momentum': (dynamic_momentum, 0.08),
            'dynamic_inertia': (dynamic_inertia, 0.05),
            'dynamic_stability': (dynamic_stability, 0.02),
            'dynamic_energy': (dynamic_energy, 0.02),
            'dynamic_ma_acceleration': (dynamic_ma_acceleration, 0.03),
            'fund_flow_consensus': (fund_flow_consensus, 0.03),
            'fund_flow_conviction': (fund_flow_conviction, 0.03),
            'fund_flow_increment': (fund_flow_increment, 0.03),
            'chip_concentration': (chip_concentration, 0.03),
            'chip_cost_structure': (chip_cost_structure, 0.03),
            'chip_holder_sentiment': (chip_holder_sentiment, 0.03),
            'chip_peak_integrity': (chip_peak_integrity, 0.03),
            'micro_deception': (micro_deception, 0.01),
            'micro_probe': (micro_probe, 0.01),
            'micro_efficiency': (micro_efficiency, 0.01),
            'behavior_upward_efficiency': (behavior_upward_efficiency, 0.02),
            'behavior_downward_resistance': (behavior_downward_resistance, 0.02),
            'behavior_intraday_bull_control': (behavior_intraday_bull_control, 0.01),
            'pattern_reversal': (pattern_reversal, 0.01),
            'pattern_breakout': (pattern_breakout, 0.02),
            'main_force_on_peak_flow': (main_force_on_peak_flow, 0.01),
            'aaa_score': (aaa_score, 0.02),
            'pdi_score': (pdi_score, 0.03),
            'structural_bottom_fractal': (structural_bottom_fractal, 0.02),
            'breakout_readiness_score': (breakout_readiness_score, 0.10), # 新增代码行：突破就绪分及其权重
            'trend_vitality_score': (trend_vitality_score, 0.10) # 新增代码行：趋势生命力及其权重
        }
        bipolar_quality = pd.Series(0.0, index=df_index)
        if probe_date_for_loop is not None and probe_date_for_loop in df_index:
            print(f"    -> [趋势质量探针] @ {probe_date_for_loop.date()}: 组成公理贡献明细")
        total_weight = sum(w for _, w in components_and_weights.values()) # 新增代码行：计算总权重
        for name, (series, weight) in components_and_weights.items():
            if not series.empty:
                contribution = series * (weight / total_weight) # 修改代码行：使用归一化后的权重
                bipolar_quality += contribution
                if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                    print(f"       - {name:<30}: 原始值={series.loc[probe_date_for_loop]:.4f}, 权重={weight:.2f}, 贡献={contribution.loc[probe_date_for_loop]:.4f}")
        bipolar_quality = bipolar_quality.clip(-1, 1)
        states['FUSION_BIPOLAR_TREND_QUALITY'] = bipolar_quality.astype(np.float32)
        print(f"  -- [融合层] “趋势质量”冶炼完成，最新分值: {bipolar_quality.iloc[-1]:.4f}")
        return states

    def _synthesize_stagnation_risk(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 深度博弈版】冶炼“滞涨风险” (FUSION_RISK_STAGNATION)
        - 核心思想: 综合多维度证据，精确识别多头力量衰竭与空头隐秘积蓄的滞涨风险。
        - 证据链:
          1. 行为层滞涨证据原始分 (INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW)
          2. 价格超买意图 (FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT 的负向部分)
          3. 上影线抛压意图 (FUSION_BIPOLAR_UPPER_SHADOW_INTENT 的负向部分)
          4. 主力资金流出背离 (SCORE_FF_AXIOM_CONSENSUS 的负向部分)
          5. 筹码集中度下降 (SCORE_CHIP_AXIOM_CONCENTRATION 的负向部分)
          6. 获利盘供给压力 (imminent_profit_taking_supply_D)
          7. 趋势确认度下降 (CONTEXT_TREND_CONFIRMED 的反向)
          8. 市场情绪亢奋 (retail_fomo_premium_index_D)
        - 输出: [0, 1] 的风险分数，0表示无风险，1表示最大风险。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        print("  -- [融合层] 正在冶炼“滞涨风险”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        # 1. 行为层滞涨证据原始分 (来自行为层)
        stagnation_evidence_raw = self._get_atomic_score('INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', 0.0)
        # 2. 价格超买意图 (来自融合层，负向部分代表风险)
        price_overextension_risk = self._get_atomic_score('FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT', 0.0).clip(upper=0).abs()
        # 3. 上影线抛压意图 (来自融合层，负向部分代表风险)
        upper_shadow_pressure_risk = self._get_atomic_score('FUSION_BIPOLAR_UPPER_SHADOW_INTENT', 0.0).clip(upper=0).abs()
        # 4. 主力资金流出背离 (来自资金流层，负向共识代表风险)
        fund_flow_bearish_risk = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0).clip(upper=0).abs()
        # 5. 筹码集中度下降 (来自筹码层，负向集中度代表风险)
        chip_dispersion_risk = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0).clip(upper=0).abs()
        # 6. 获利盘供给压力 (来自筹码高级指标，需要归一化)
        profit_taking_supply_risk = normalize_score(self._get_safe_series(self.strategy.df_indicators, 'imminent_profit_taking_supply_D', 0.0, method_name="_synthesize_stagnation_risk"), df_index, window=55, ascending=True).clip(0, 1)
        # 7. 趋势确认度下降 (来自基础层上下文，反向代表风险)
        trend_confirmation_risk = (1 - self._get_atomic_score('CONTEXT_TREND_CONFIRMED', 0.0)).clip(0, 1)
        # 8. 市场情绪亢奋 (来自资金流高级指标，需要归一化)
        retail_fomo_risk = normalize_score(self._get_safe_series(self.strategy.df_indicators, 'retail_fomo_premium_index_D', 0.0, method_name="_synthesize_stagnation_risk"), df_index, window=55, ascending=True).clip(0, 1)
        # 9. 价格上涨或横盘的前提条件 (滞涨风险的前提)
        is_price_stagnant_or_rising = (self._get_safe_series(self.strategy.df_indicators, 'pct_change_D', method_name="_synthesize_stagnation_risk") >= -0.005).astype(float)
        # 融合所有风险证据 (加权几何平均)
        # 权重分配需要根据回测结果进行优化，这里给出示例权重
        risk_components = [
            stagnation_evidence_raw,        # 行为层原始证据 (权重高)
            price_overextension_risk,       # 价格超买意图 (权重中高)
            upper_shadow_pressure_risk,     # 上影线抛压意图 (权重高)
            fund_flow_bearish_risk,         # 主力资金流出背离 (权重高)
            chip_dispersion_risk,           # 筹码集中度下降 (权重高)
            profit_taking_supply_risk,      # 获利盘供给压力 (权重中)
            trend_confirmation_risk,        # 趋势确认度下降 (权重中低)
            retail_fomo_risk                # 市场情绪亢奋 (权重中)
        ]
        weights = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05, 0.10])
        # 确保所有风险分量都是 Series，并且索引对齐
        aligned_risk_components = [comp.reindex(df_index, fill_value=0.0) for comp in risk_components]
        # 避免 log(0) 错误，将所有分量加上一个极小值
        safe_risk_components = [comp + 1e-9 for comp in aligned_risk_components]
        # 计算加权几何平均
        stagnation_risk_score = pd.Series(np.prod([comp.values ** w for comp, w in zip(safe_risk_components, weights)], axis=0), index=df_index)
        # 最终风险分只在价格上涨或横盘时有效，否则为0
        final_stagnation_risk = (stagnation_risk_score * is_price_stagnant_or_rising).clip(0, 1)
        states['FUSION_RISK_STAGNATION'] = final_stagnation_risk.astype(np.float32)
        print(f"  -- [融合层] “滞涨风险”冶炼完成，最新分值: {final_stagnation_risk.iloc[-1]:.4f}")
        return states

    def _synthesize_capital_confrontation(self) -> Dict[str, pd.Series]:
        """
        【V1.2 · 探针增强与打印修复版】冶炼“资本对抗” (Capital Confrontation)
        - 核心思想: 深度洞察A股的博弈核心——主力与散户的对抗。
        - 证据链:
          1. 资金流对抗 (FundFlow): 主力与散户的资金流方向是否相反。
          2. 筹码转移 (Chip): 筹码是在集中还是在发散。
          3. 微观欺骗 (MicroBehavior): 是否存在“伪装成散户吸筹”等欺骗行为。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【新增】增加详细探针输出，追踪计算过程。
        - 【修复】修正最终分值打印，使其与探针日期一致。
        """
        print("  -- [融合层] 正在冶炼“资本对抗”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date_for_loop = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop not in df_index:
                probe_date_for_loop = None
        flow_confrontation = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        chip_transfer = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        deception = self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0)
        bipolar_confrontation = (flow_confrontation * 0.5 + chip_transfer * 0.3 + deception * 0.2).clip(-1, 1)
        states['FUSION_BIPOLAR_CAPITAL_CONFRONTATION'] = bipolar_confrontation.astype(np.float32)
        if probe_date_for_loop is not None and probe_date_for_loop in df_index:
            print(f"    -> [资本对抗探针] @ {probe_date_for_loop.date()}:")
            print(f"       - SCORE_FF_AXIOM_CONSENSUS: {flow_confrontation.loc[probe_date_for_loop]:.4f}")
            print(f"       - SCORE_CHIP_AXIOM_CONCENTRATION: {chip_transfer.loc[probe_date_for_loop]:.4f}")
            print(f"       - SCORE_MICRO_AXIOM_DECEPTION: {deception.loc[probe_date_for_loop]:.4f}")
            print(f"       - Calculated bipolar_confrontation: {bipolar_confrontation.loc[probe_date_for_loop]:.4f}")
            print(f"  -- [融合层] “资本对抗”冶炼完成，最新分值: {bipolar_confrontation.loc[probe_date_for_loop]:.4f}")
        else:
            print(f"  -- [融合层] “资本对抗”冶炼完成，最新分值: {bipolar_confrontation.iloc[-1]:.4f}")
        return states

    def _synthesize_price_overextension_intent(self) -> Dict[str, pd.Series]:
        """
        【V3.0 · 深度博弈版】冶炼“价格超买意图” (Price Overextension Intent)
        - 核心思想: 综合判断价格偏离均线是强力进攻（抢筹）还是真实超买风险。
        - 核心逻辑: 将超买证据与趋势健康证据分离，通过加权求和与减法操作，
                      实现对“抢筹”与“超买”的量化区分。
        - 证据链:
          1. 核心超买证据: 行为层原始超买、乖离率、RSI、获利盘比例。
          2. 趋势健康证据: 资金流共识、筹码集中度、结构趋势形态、微观效率、K线实体与影线。
        - 输出: [-1, 1] 的双极性分数，正分代表超买风险，负分代表健康上涨（抢筹）。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        print("  -- [融合层] 正在冶炼“价格超买意图”...")
        states = {}
        df = self.strategy.df_indicators
        df_index = df.index
        norm_window = 55 # 统一归一化窗口
        # --- Debugging setup ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date_for_loop = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop not in df_index:
                probe_date_for_loop = None # Reset if not in index
        # 1. 核心超买证据 (越高越超买)
        # 行为层价格超买原始分 (0到1，越高越超买) 转换为双极性
        overextension_raw_bipolar = (self._get_atomic_score('INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 0.5) * 2 - 1).clip(-1, 1)
        # 乖离率 (BIAS_21_D) 越高，超买程度越高
        bias_raw = self._get_safe_series(df, 'BIAS_21_D', pd.Series(0.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        bias_score = normalize_to_bipolar(bias_raw, df_index, window=norm_window, sensitivity=0.05)
        # 获利盘比例 (total_winner_rate_D) 越高，超买风险越高
        winner_rate_raw = self._get_safe_series(df, 'total_winner_rate_D', pd.Series(0.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        winner_rate_score = normalize_to_bipolar(winner_rate_raw, df_index, window=norm_window, sensitivity=0.1, default_value=-1.0)
        # RSI (RSI_13_D) 越高，超买程度越高
        rsi_raw = self._get_safe_series(df, 'RSI_13_D', pd.Series(50.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        rsi_score = normalize_to_bipolar(rsi_raw, df_index, window=norm_window, sensitivity=10.0)
        # 核心超买证据的加权和
        core_overextension_sum = (
            overextension_raw_bipolar * 0.2 +
            bias_score * 0.2 +
            rsi_score * 0.15 +
            winner_rate_score * 0.15
        )
        # 2. 趋势健康证据 (越高越健康，越支持抢筹，越降低超买风险)
        # 资金流共识 (SCORE_FF_AXIOM_CONSENSUS) [-1, 1]
        fund_flow_consensus = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        # 筹码集中度 (SCORE_CHIP_AXIOM_CONCENTRATION) [-1, 1]
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        # 结构趋势形态 (SCORE_STRUCT_AXIOM_TREND_FORM) [-1, 1]
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        # 微观效率 (SCORE_MICRO_AXIOM_EFFICIENCY) [-1, 1]
        micro_efficiency = (self._get_atomic_score('SCORE_MICRO_AXIOM_EFFICIENCY', 0.5) * 2 - 1).clip(-1, 1)
        # K线实体与影线 (body_ratio_D, upper_shadow_ratio_D)
        # 实体饱满，上影线短 -> 健康
        # 替换 body_ratio_D 为 closing_price_deviation_score_D
        body_ratio_raw = self._get_safe_series(df, 'closing_price_deviation_score_D', pd.Series(0.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        body_score = normalize_to_bipolar(body_ratio_raw, df_index, window=norm_window, sensitivity=0.2)
        upper_shadow_ratio_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', pd.Series(0.0, index=df_index), method_name="_synthesize_price_overextension_intent")
        upper_shadow_score = normalize_to_bipolar(upper_shadow_ratio_raw, df_index, window=norm_window, sensitivity=0.2) * -1 # 上影线越短越好，所以反向
        # 趋势健康证据的加权和
        health_sum = (
            fund_flow_consensus * 0.1 +
            chip_concentration * 0.05 +
            structural_trend_form * 0.05 +
            micro_efficiency * 0.03 +
            body_score * 0.04 +
            upper_shadow_score * 0.03
        )
        # 最终融合: 核心超买证据 - 趋势健康证据
        # 如果健康证据强，则会降低超买意图；如果健康证据弱（甚至负），则会增加超买意图。
        final_overextension_intent = (core_overextension_sum - health_sum).clip(-1, 1)
        # --- Debugging output for probe date ---
        if probe_date_for_loop is not None and probe_date_for_loop in df_index:
            print(f"    -> [价格超买意图探针] @ {probe_date_for_loop.date()}:")
            print(f"       - overextension_raw_bipolar: {overextension_raw_bipolar.loc[probe_date_for_loop]:.4f}")
            print(f"       - bias_score: {bias_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - winner_rate_score: {winner_rate_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - rsi_score: {rsi_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - core_overextension_sum: {core_overextension_sum.loc[probe_date_for_loop]:.4f}")
            print(f"       - fund_flow_consensus: {fund_flow_consensus.loc[probe_date_for_loop]:.4f}")
            print(f"       - chip_concentration: {chip_concentration.loc[probe_date_for_loop]:.4f}")
            print(f"       - structural_trend_form: {structural_trend_form.loc[probe_date_for_loop]:.4f}")
            print(f"       - micro_efficiency: {micro_efficiency.loc[probe_date_for_loop]:.4f}")
            print(f"       - body_score: {body_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - upper_shadow_score: {upper_shadow_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - health_sum: {health_sum.loc[probe_date_for_loop]:.4f}")
            print(f"       - final_overextension_intent: {final_overextension_intent.loc[probe_date_for_loop]:.4f}")
        states['FUSION_BIPOLAR_PRICE_OVEREXTENSION_INTENT'] = final_overextension_intent.astype(np.float32)
        print(f"  -- [融合层] “价格超买意图”冶炼完成，最新分值: {final_overextension_intent.iloc[-1]:.4f}")
        return states

    def _synthesize_upper_shadow_intent(self) -> Dict[str, pd.Series]: # 修改方法
        """
        【V2.0 · 深度博弈版】冶炼“上影线意图” (Upper Shadow Intent)
        - 核心思想: 综合判断上影线是真抛压（空头意图）还是洗盘/试探（多头意图）。
        - 证据链:
          1. 行为层上影线原始分 (INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW)
          2. 价格涨跌 (pct_change_D)
          3. 主力资金流向 (main_force_net_flow_calibrated_D)
          4. 筹码集中度 (SCORE_CHIP_AXIOM_CONCENTRATION)
          5. 微观欺骗 (SCORE_MICRO_AXIOM_DECEPTION)
          6. 上涨效率 (SCORE_BEHAVIOR_UPWARD_EFFICIENCY)
          7. 结构趋势形态 (SCORE_STRUCT_AXIOM_TREND_FORM)
          8. 成交量爆发 (SCORE_BEHAVIOR_VOLUME_BURST)
        - 输出: [-1, 1] 的双极性分数，负分代表真抛压风险，正分代表偏向多头意图（如洗盘、试探）。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        print("  -- [融合层] 正在冶炼“上影线意图” (深度博弈版)...")
        states = {}
        df = self.strategy.df_indicators
        df_index = df.index
        norm_window = 55 # 统一归一化窗口
        # 1. 获取行为层上影线原始分 (0到1，越高上影线越强)
        upper_shadow_normalized = self._get_atomic_score('INTERNAL_BEHAVIOR_UPPER_SHADOW_RAW', 0.0)
        # 2. 获取其他维度的支持/抑制信号 (转换为 [-1, 1] 双极性或 [0, 1] 单极性)
        pct_change = self._get_safe_series(df, 'pct_change_D', pd.Series(0.0, index=df_index), method_name="_synthesize_upper_shadow_intent")
        main_force_flow = normalize_to_bipolar(self._get_safe_series(df, 'main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index), method_name="_synthesize_upper_shadow_intent"), df_index, norm_window)
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0) # [-1, 1]
        micro_deception = self._get_atomic_score('SCORE_MICRO_AXIOM_DECEPTION', 0.0) # [-1, 1]
        upward_efficiency = (self._get_atomic_score('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.5) * 2 - 1).clip(-1, 1) # 转换为双极性
        structural_trend_form = self._get_atomic_score('SCORE_STRUCT_AXIOM_TREND_FORM', 0.0) # [-1, 1]
        volume_burst = (self._get_atomic_score('SCORE_BEHAVIOR_VOLUME_BURST', 0.5) * 2 - 1).clip(-1, 1) # 转换为双极性
        # 3. 价格背景判断
        is_up_day = (pct_change > 0).astype(float)
        is_down_day = (pct_change < 0).astype(float)
        is_flat_day = ((pct_change == 0) | (pct_change.abs() < 0.005)).astype(float) # 定义平盘日
        # 4. 计算多头意图分数 (Bullish Intent Score) - 范围 [0, 1]
        # 这些因素，当为正时，表明上影线背后是多头意图（洗盘、试探）
        bullish_intent_components = [
            main_force_flow.clip(lower=0), # 主力净流入
            chip_concentration.clip(lower=0), # 筹码集中
            micro_deception.clip(lower=0), # 微观欺骗为伪装派发（实为吸筹）
            upward_efficiency.clip(lower=0), # 上涨效率高
            structural_trend_form.clip(lower=0) # 结构趋势向上
        ]
        # 权重分配：主力资金和筹码集中度最为关键
        bullish_weights = np.array([0.35, 0.25, 0.2, 0.1, 0.1]) # 权重和为1
        bullish_intent_score = sum(w * s for w, s in zip(bullish_weights, bullish_intent_components)).clip(0, 1)
        # 5. 计算空头意图分数 (Bearish Intent Score) - 范围 [0, 1]
        # 这些因素，当为正时，表明上影线背后是空头意图（派发、诱多）
        bearish_intent_components = [
            main_force_flow.clip(upper=0).abs(), # 主力净流出
            chip_concentration.clip(upper=0).abs(), # 筹码分散
            micro_deception.clip(upper=0).abs(), # 微观欺骗为伪装吸筹（实为派发）
            (1 - upward_efficiency.clip(lower=0)), # 上涨效率低 (1-效率高)
            structural_trend_form.clip(upper=0).abs() # 结构趋势向下
        ]
        # 权重分配：同样主力资金和筹码分散度最为关键
        bearish_weights = np.array([0.35, 0.25, 0.2, 0.1, 0.1]) # 权重和为1
        bearish_intent_score = sum(w * s for w, s in zip(bearish_weights, bearish_intent_components)).clip(0, 1)
        # 6. 综合净意图方向 (Net Intent Direction) - 范围 [-1, 1]
        # 正值代表多头意图占优，负值代表空头意图占优
        net_intent_direction = (bullish_intent_score - bearish_intent_score).clip(-1, 1)
        # 7. 成交量信念乘数 (Volume Conviction Multiplier) - 范围 [0, 1]
        # 高量能 (volume_burst=1) 意味着信念乘数为1，低量能 (volume_burst=-1) 意味着信念乘数为0
        # 这样，低量能的上影线，无论意图如何，其最终信号强度都会被削弱
        volume_conviction_multiplier = (volume_burst + 1) / 2 # 将 [-1, 1] 映射到 [0, 1]
        # 8. 基础意图分数 (Base Intent Score)
        # 原始上影线强度 * 净意图方向 * 成交量信念乘数
        base_intent_score = upper_shadow_normalized * net_intent_direction * volume_conviction_multiplier
        # 9. 价格背景的条件调整
        final_intent = pd.Series(0.0, index=df_index)
        # 调整1: 上涨日 (is_up_day)
        # 上涨日的上影线，如果净意图为正（洗盘/试探），则放大其积极性；如果净意图为负（诱多/派发），则放大其消极性。
        up_day_adjusted_intent = base_intent_score * (1 + base_intent_score.abs() * 0.5) # 意图越明确，放大效果越强
        final_intent = final_intent.mask(is_up_day.astype(bool), up_day_adjusted_intent)
        # 调整2: 下跌日 (is_down_day)
        # 下跌日的上影线，通常是抛压。即使有少量多头证据，也应大幅削弱其积极性，并强化其消极性。
        # 基础惩罚：-upper_shadow_normalized * 0.8 (即使没有其他空头证据，也至少是较强的抛压)
        # 叠加空头意图：-upper_shadow_normalized * bearish_intent_score * 0.5
        # 少量多头缓解：+upper_shadow_normalized * bullish_intent_score * 0.2 (缓解作用有限)
        down_day_adjusted_intent = (
            -upper_shadow_normalized * 0.8 # 基础抛压
            - upper_shadow_normalized * bearish_intent_score * 0.5 # 叠加空头意图
            + upper_shadow_normalized * bullish_intent_score * 0.2 # 少量多头缓解
        ) * volume_conviction_multiplier # 依然受量能信念影响
        final_intent = final_intent.mask(is_down_day.astype(bool), down_day_adjusted_intent)
        # 调整3: 平盘日 (is_flat_day)
        # 平盘日的上影线，意图相对模糊，信号强度减半。
        flat_day_adjusted_intent = base_intent_score * 0.5
        final_intent = final_intent.mask(is_flat_day.astype(bool), flat_day_adjusted_intent)
        final_intent = final_intent.clip(-1, 1)
        states['FUSION_BIPOLAR_UPPER_SHADOW_INTENT'] = final_intent.astype(np.float32)
        print(f"  -- [融合层] “上影线意图”冶炼完成，最新分值: {final_intent.iloc[-1]:.4f}")
        return states

    def _synthesize_trend_structure_score(self) -> Dict[str, pd.Series]:
        """
        【V1.4 · 趋势结构分优化版】冶炼“趋势结构分” (FUSION_BIPOLAR_TREND_STRUCTURE_SCORE)
        - 核心优化:
          1. 修正 `divergence_score` 逻辑，改为加权算术平均，避免乘法带来的反直觉结果。
          2. 修正最终融合方式，从加权几何平均改为加权算术平均，减少单个极端负值对整体分数的“一票否决”效应。
          3. 增加调试探针，输出关键中间计算结果。
          4. 调整 `alignment_score` 的 `normalize_to_bipolar` 敏感度，避免微小均线交叉被过度放大。
        - 【新增】引入 `DMA_D` (均线差) 和 `ZIG_5_5.0_D` (ZIGZAG趋势) 作为趋势结构分的组成部分。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        print("  -- [融合层] 正在冶炼“趋势结构分”...")
        states = {}
        df = self.strategy.df_indicators
        df_index = df.index
        norm_window = 55 # 统一归一化窗口
        # --- Debugging setup ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date_for_loop = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            # 确保探针日期与df_index的时区一致
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop not in df_index:
                probe_date_for_loop = None # Reset if not in index
        # 1. 均线排列分 (EMA5 vs EMA21)
        ema5 = self._get_safe_series(df, 'EMA_5_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        ema21 = self._get_safe_series(df, 'EMA_21_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        # 确保EMA数据存在且有效
        if ema5.isnull().all() or ema21.isnull().all():
            print("    -> [趋势结构分] 警告: 缺少EMA5或EMA21数据，均线排列分将为0。")
            alignment_score = pd.Series(0.0, index=df_index)
        else:
            # EMA5 > EMA21 为正，反之为负，归一化到 [-1, 1]
            # 使用一个小的epsilon防止除以零
            raw_alignment = (ema5 - ema21) / (ema21.abs().replace(0, 1e-9)) # 相对距离
            # 增加 normalize_to_bipolar 的 sensitivity 参数
            alignment_score = normalize_to_bipolar(raw_alignment, df_index, window=norm_window, sensitivity=5.0) # 敏感度调整
        # 2. 均线斜率分 (SLOPE_5_EMA_5_D 和 SLOPE_5_EMA_21_D)
        slope_ema5 = self._get_safe_series(df, 'SLOPE_5_EMA_5_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        slope_ema21 = self._get_safe_series(df, 'SLOPE_5_EMA_21_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score") # 修正默认值
        if slope_ema5.isnull().all() or slope_ema21.isnull().all():
            print("    -> [趋势结构分] 警告: 缺少EMA斜率数据，均线斜率分将为0。")
            slope_score = pd.Series(0.0, index=df_index)
        else:
            # 归一化斜率到双极性分数
            norm_slope_ema5 = normalize_to_bipolar(slope_ema5, df_index, window=norm_window, sensitivity=0.005)
            norm_slope_ema21 = normalize_to_bipolar(slope_ema21, df_index, window=norm_window, sensitivity=0.005)
            # 加权平均归一化后的斜率
            slope_score = (norm_slope_ema5 * 0.6 + norm_slope_ema21 * 0.4).clip(-1, 1) # 5日斜率权重更高
        # 3. 均线发散/收敛分 (EMA5与EMA21的乖离率及其变化率)
        # 乖离率 = (EMA5 - EMA21) / EMA21
        if ema5.isnull().all() or ema21.isnull().all():
            print("    -> [趋势结构分] 警告: 缺少EMA数据，均线发散/收敛分将为0。")
            divergence_score = pd.Series(0.0, index=df_index)
        else:
            ma_bias_raw = (ema5 - ema21) / (ema21.abs().replace(0, 1e-9))
            # 乖离率的斜率，反映发散/收敛的速度
            ma_bias_slope_raw = ma_bias_raw.diff(1).fillna(0)
            # 归一化乖离率和乖离率斜率
            norm_ma_bias = normalize_to_bipolar(ma_bias_raw, df_index, window=norm_window, sensitivity=0.02)
            norm_ma_bias_slope = normalize_to_bipolar(ma_bias_slope_raw, df_index, window=norm_window, sensitivity=0.001)
            # 修正 divergence_score 逻辑为加权算术平均
            # 目标：当乖离率为正且乖离率斜率为正时，为强正分；当乖离率为负且乖离率斜率为负时，为强负分。
            # 当乖离率为负但乖离率斜率为正（负乖离缩小，结构改善）时，应为正分。
            # 当乖离率为正但乖离率斜率为负（正乖离缩小，结构恶化）时，应为负分。
            # 简单的加权平均可以更好地捕捉这种关系。
            divergence_score = (norm_ma_bias * 0.7 + norm_ma_bias_slope * 0.3).clip(-1, 1)
        # 4. DMA线 (Difference of Moving Averages)
        dma_raw = self._get_safe_series(df, 'DMA_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        dma_score = normalize_to_bipolar(dma_raw, df_index, window=norm_window)
        # 5. ZIGZAG趋势 (ZIG_5_5.0_D)
        # ZIGZAG趋势向上为正，向下为负，直接归一化
        zigzag_raw = self._get_safe_series(df, 'ZIG_5_5.0_D', pd.Series(0.0, index=df_index), method_name="_synthesize_trend_structure_score")
        zigzag_score = normalize_to_bipolar(zigzag_raw, df_index, window=norm_window, sensitivity=0.05)
        # --- Debugging output for probe date ---
        if probe_date_for_loop is not None and probe_date_for_loop in df.index:
            print(f"    -> [趋势结构分探针] @ {probe_date_for_loop.date()}:")
            print(f"       - EMA5: {ema5.loc[probe_date_for_loop]:.4f}")
            print(f"       - EMA21: {ema21.loc[probe_date_for_loop]:.4f}")
            print(f"       - SLOPE_5_EMA_5_D: {slope_ema5.loc[probe_date_for_loop]:.4f}")
            print(f"       - SLOPE_5_EMA_21_D: {slope_ema21.loc[probe_date_for_loop]:.4f}")
            print(f"       - raw_alignment: {raw_alignment.loc[probe_date_for_loop]:.4f}")
            print(f"       - alignment_score: {alignment_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - norm_slope_ema5: {norm_slope_ema5.loc[probe_date_for_loop]:.4f}")
            print(f"       - norm_slope_ema21: {norm_slope_ema21.loc[probe_date_for_loop]:.4f}")
            print(f"       - slope_score: {slope_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - ma_bias_raw: {ma_bias_raw.loc[probe_date_for_loop]:.4f}")
            print(f"       - ma_bias_slope_raw: {ma_bias_slope_raw.loc[probe_date_for_loop]:.4f}")
            print(f"       - norm_ma_bias: {norm_ma_bias.loc[probe_date_for_loop]:.4f}")
            print(f"       - norm_ma_bias_slope: {norm_ma_bias_slope.loc[probe_date_for_loop]:.4f}")
            print(f"       - divergence_score: {divergence_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - dma_score: {dma_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - zigzag_score: {zigzag_score.loc[probe_date_for_loop]:.4f}")
        # 6. 融合所有子分数
        # 权重分配 (示例，需优化)
        weights = np.array([0.3, 0.3, 0.15, 0.15, 0.1]) # 排列和斜率依然重要，DMA和ZIGZAG提供额外确认
        components = [alignment_score, slope_score, divergence_score, dma_score, zigzag_score]
        # 确保所有分量都是 Series，并且索引对齐
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        # 从加权几何平均改为加权算术平均，避免“一票否决”效应
        final_trend_structure_score = (
            aligned_components[0] * weights[0] +
            aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] +
            aligned_components[3] * weights[3] +
            aligned_components[4] * weights[4]
        ).clip(-1, 1) # 确保最终分数在 [-1, 1] 范围内
        states['FUSION_BIPOLAR_TREND_STRUCTURE_SCORE'] = final_trend_structure_score.astype(np.float32)
        print(f"  -- [融合层] “趋势结构分”冶炼完成，最新分值: {final_trend_structure_score.iloc[-1]:.4f}")
        return states

    def _synthesize_fund_flow_trend(self) -> Dict[str, pd.Series]:
        """
        【V1.1 · 资金流动量版】冶炼“资金趋势” (FUSION_BIPOLAR_FUND_FLOW_TREND)
        - 核心思想: 综合判断主力资金的真实意图、信念和市场活跃度，形成资金流的整体趋势判断。
        - 证据链:
          1. 资金流共识 (SCORE_FF_AXIOM_CONSENSUS): 主力与散户的资金流博弈结果。
          2. 资金流信念 (SCORE_FF_AXIOM_CONVICTION): 主力资金的持仓决心和成本优势。
          3. 资金流动量 (SCORE_FF_AXIOM_FLOW_MOMENTUM): 主力资金净流量的相对强度和趋势动量。
          4. 资金流背离 (SCORE_FF_AXIOM_DIVERGENCE): 资金流与价格的背离情况。
        - 输出: [-1, 1] 的双极性分数，正分代表资金趋势向好，负分代表资金趋势恶化。
        """
        print("  -- [融合层] 正在冶炼“资金趋势”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        ff_consensus = self._get_atomic_score('SCORE_FF_AXIOM_CONSENSUS', 0.0)
        ff_conviction = self._get_atomic_score('SCORE_FF_AXIOM_CONVICTION', 0.0)
        ff_flow_momentum = self._get_atomic_score('SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0) # 获取新的资金流动量公理
        ff_divergence = self._get_atomic_score('SCORE_FF_AXIOM_DIVERGENCE', 0.0)
        # 融合所有资金流证据，采用加权平均
        # 权重分配：共识和信念是核心，动量和背离是辅助确认
        components = [ff_consensus, ff_conviction, ff_flow_momentum, ff_divergence]
        weights = np.array([0.35, 0.30, 0.25, 0.10]) # 调整权重
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        fund_flow_trend_score = (
            aligned_components[0] * weights[0] +
            aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] +
            aligned_components[3] * weights[3]
        ).clip(-1, 1)
        states['FUSION_BIPOLAR_FUND_FLOW_TREND'] = fund_flow_trend_score.astype(np.float32)
        print(f"  -- [融合层] “资金趋势”冶炼完成，最新分值: {fund_flow_trend_score.iloc[-1]:.4f}")
        return states

    def _synthesize_chip_trend(self) -> Dict[str, pd.Series]:
        """
        【V1.6 · 筹码趋势动量强化与缺失信号严格处理及探针版】冶炼“筹码趋势” (FUSION_BIPOLAR_CHIP_TREND)
        - 核心思想: 综合判断市场筹码的集中度、成本结构、持股心态和峰形态，形成筹码的整体趋势判断。
        - 证据链:
          1. 筹码集中度 (SCORE_CHIP_AXIOM_CONCENTRATION): 筹码是集中还是分散。
          2. 筹码成本结构 (SCORE_CHIP_AXIOM_COST_STRUCTURE): 成本结构是否健康，获利盘和套牢盘分布。
          3. 持股心态 (SCORE_CHIP_AXIOM_HOLDER_SENTIMENT): 市场参与者的持股信心。
          4. 筹码峰完整性 (SCORE_CHIP_AXIOM_PEAK_INTEGRITY): 筹码峰的支撑或压力作用。
          5. 筹码背离 (SCORE_CHIP_AXIOM_DIVERGENCE): 筹码与价格的背离情况。
          6. 筹码干净度 (SCORE_CHIP_CLEANLINESS): 市场浮筹的多少。
          7. 筹码锁定度 (SCORE_CHIP_LOCKDOWN_DEGREE): 筹码被锁定的程度。
          8. 筹码趋势动量 (SCORE_CHIP_AXIOM_TREND_MOMENTUM): 整体筹码健康度的变化速度和方向。
        - 输出: [-1, 1] 的双极性分数，正分代表筹码趋势向好，负分代表筹码趋势恶化。
        - 【修正】调整权重，显著提高 `SCORE_CHIP_AXIOM_TREND_MOMENTUM` 的权重，并对负面信号更敏感。
        - 【新增】在计算前验证所有所需信号是否存在，如果缺失则打印警告，并对缺失信号进行更保守的处理。
        - 【新增】加入探针，打印每个组成信号的原始值、权重和贡献。
        - 【核心修复】由于 `_get_atomic_score` 的严格化，缺失信号将严格使用 `required_chip_signals_meta` 中定义的默认值。
        """
        print("  -- [融合层] 正在冶炼“筹码趋势”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        # --- Debugging setup ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date_for_loop = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop not in df_index:
                probe_date_for_loop = None # Reset if not in index
        # 明确定义所有所需信号及其在融合中的预期极性（正向贡献、负向贡献或双极性）
        # 预期极性用于在信号缺失时，决定使用何种默认值，以避免乐观偏置
        # 0: 双极性, 1: 正向贡献, -1: 负向贡献
        required_chip_signals_meta = {
            'SCORE_CHIP_AXIOM_CONCENTRATION': {'polarity': 0, 'default_on_missing': 0.0}, # 双极性，缺失时中性
            'SCORE_CHIP_AXIOM_COST_STRUCTURE': {'polarity': 0, 'default_on_missing': 0.0}, # 双极性，缺失时中性
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT': {'polarity': 0, 'default_on_missing': 0.0}, # 双极性，缺失时中性
            'SCORE_CHIP_AXIOM_PEAK_INTEGRITY': {'polarity': 0, 'default_on_missing': 0.0}, # 双极性，缺失时中性
            'SCORE_CHIP_AXIOM_DIVERGENCE': {'polarity': 0, 'default_on_missing': 0.0}, # 双极性，缺失时中性
            'SCORE_CHIP_CLEANLINESS': {'polarity': 1, 'default_on_missing': 0.0}, # 正向贡献，缺失时中性
            'SCORE_CHIP_LOCKDOWN_DEGREE': {'polarity': 1, 'default_on_missing': 0.0}, # 正向贡献，缺失时中性
            'SCORE_CHIP_AXIOM_TREND_MOMENTUM': {'polarity': 0, 'default_on_missing': 0.0} # 双极性，缺失时中性
        }
        # 检查哪些信号是缺失的
        missing_signals_in_atomic_states = [sig for sig in required_chip_signals_meta.keys() if sig not in self.strategy.atomic_states]
        if missing_signals_in_atomic_states:
            print(f"    -> [融合层-筹码趋势警告] 缺少以下关键筹码信号，将使用其定义默认值进行计算: {', '.join(missing_signals_in_atomic_states)}")
        # 获取所有信号，并根据缺失情况调整其值
        # 注意：这里调用的是 FusionIntelligence 自己的 _get_atomic_score，它现在已经严格化
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', required_chip_signals_meta['SCORE_CHIP_AXIOM_CONCENTRATION']['default_on_missing'])
        chip_cost_structure = self._get_atomic_score('SCORE_CHIP_AXIOM_COST_STRUCTURE', required_chip_signals_meta['SCORE_CHIP_AXIOM_COST_STRUCTURE']['default_on_missing'])
        chip_holder_sentiment = self._get_atomic_score('SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', required_chip_signals_meta['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT']['default_on_missing'])
        chip_peak_integrity = self._get_atomic_score('SCORE_CHIP_AXIOM_PEAK_INTEGRITY', required_chip_signals_meta['SCORE_CHIP_AXIOM_PEAK_INTEGRITY']['default_on_missing'])
        chip_divergence = self._get_atomic_score('SCORE_CHIP_AXIOM_DIVERGENCE', required_chip_signals_meta['SCORE_CHIP_AXIOM_DIVERGENCE']['default_on_missing'])
        chip_cleanliness = self._get_atomic_score('SCORE_CHIP_CLEANLINESS', required_chip_signals_meta['SCORE_CHIP_CLEANLINESS']['default_on_missing'])
        chip_lockdown_degree = self._get_atomic_score('SCORE_CHIP_LOCKDOWN_DEGREE', required_chip_signals_meta['SCORE_CHIP_LOCKDOWN_DEGREE']['default_on_missing'])
        chip_trend_momentum = self._get_atomic_score('SCORE_CHIP_AXIOM_TREND_MOMENTUM', required_chip_signals_meta['SCORE_CHIP_AXIOM_TREND_MOMENTUM']['default_on_missing'])
        # 融合所有筹码证据，采用加权平均
        # 权重分配：筹码趋势动量作为OCH_D的直接反映，应具有最高权重。
        # 集中度、成本结构、持股心态次之。其他为辅助。
        components = [
            chip_concentration, chip_cost_structure, chip_holder_sentiment,
            chip_peak_integrity, chip_divergence, chip_cleanliness, chip_lockdown_degree,
            chip_trend_momentum
        ]
        # 调整权重，显著提高 `SCORE_CHIP_AXIOM_TREND_MOMENTUM` 的权重
        # 确保总和为1
        # 进一步调整权重，增加对负面信号的敏感度
        weights = np.array([0.12, 0.12, 0.12, 0.08, 0.08, 0.05, 0.05, 0.38]) # 调整权重，趋势动量权重进一步提高
        # aligned_components 填充值现在应该使用每个信号元数据中定义的 default_on_missing
        aligned_components = []
        for comp, meta_key in zip(components, required_chip_signals_meta.keys()):
            aligned_components.append(comp.reindex(df_index, fill_value=required_chip_signals_meta[meta_key]['default_on_missing']))
        chip_trend_score = pd.Series(0.0, index=df_index)
        if probe_date_for_loop is not None and probe_date_for_loop in df_index:
            print(f"    -> [筹码趋势探针] @ {probe_date_for_loop.date()}: 组成公理贡献明细")
        for i, (comp_series, weight) in enumerate(zip(aligned_components, weights)):
            signal_name = list(required_chip_signals_meta.keys())[i]
            contribution = comp_series * weight
            chip_trend_score += contribution
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"       - {signal_name:<35}: 原始值={comp_series.loc[probe_date_for_loop]:.4f}, 权重={weight:.2f}, 贡献={contribution.loc[probe_date_for_loop]:.4f}")
        chip_trend_score = chip_trend_score.clip(-1, 1)
        states['FUSION_BIPOLAR_CHIP_TREND'] = chip_trend_score.astype(np.float32)
        print(f"  -- [融合层] “筹码趋势”冶炼完成，最新分值: {chip_trend_score.iloc[-1]:.4f}")
        return states

    def _synthesize_accumulation_inflection(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 资金筹码融合版】冶炼“吸筹拐点信号” (FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT)
        - 核心思想: 融合资金流吸筹拐点意图与筹码结构支持，识别主力完成隐蔽吸筹，开始加大吸筹力度，甚至转向抢筹的时间点。
        - 证据链:
          1. 资金流吸筹拐点意图 (PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT)
          2. 筹码集中度 (SCORE_CHIP_AXIOM_CONCENTRATION)
          3. 筹码成本结构 (SCORE_CHIP_AXIOM_COST_STRUCTURE)
          4. 持股心态 (SCORE_CHIP_AXIOM_HOLDER_SENTIMENT)
          5. 筹码趋势动量 (SCORE_CHIP_AXIOM_TREND_MOMENTUM)
        - 输出: [0, 1] 的单极性分数，0表示无拐点迹象，1表示强烈的拐点信号。
        """
        print("  -- [融合层] 正在冶炼“吸筹拐点信号”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        inflection_score = pd.Series(0.0, index=df_index)
        # --- Debugging setup ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_date_for_loop = None
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop not in df_index:
                probe_date_for_loop = None
        # 1. 获取资金流吸筹拐点意图 (来自 FundFlowIntelligence)
        ff_inflection_intent = self._get_atomic_score('PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT', 0.0)
        # 2. 获取核心筹码信号 (来自 ChipIntelligence)
        chip_concentration = self._get_atomic_score('SCORE_CHIP_AXIOM_CONCENTRATION', 0.0)
        chip_cost_structure = self._get_atomic_score('SCORE_CHIP_AXIOM_COST_STRUCTURE', 0.0)
        chip_holder_sentiment = self._get_atomic_score('SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        chip_trend_momentum = self._get_atomic_score('SCORE_CHIP_AXIOM_TREND_MOMENTUM', 0.0)
        # 3. 定义参数 (可配置，用于调整信号敏感度)
        p_conf_fusion_inflection = get_params_block(self.strategy, 'fusion_accumulation_inflection_params', {})
        ff_inflection_threshold = get_param_value(p_conf_fusion_inflection.get('ff_inflection_threshold'), 0.6) # 资金流拐点意图阈值
        chip_concentration_threshold = get_param_value(p_conf_fusion_inflection.get('chip_concentration_threshold'), 0.5) # 筹码集中度高阈值
        chip_cost_structure_favorable_threshold = get_param_value(p_conf_fusion_inflection.get('chip_cost_structure_favorable_threshold'), 0.0) # 筹码成本结构有利阈值
        chip_holder_sentiment_strong_threshold = get_param_value(p_conf_fusion_inflection.get('chip_holder_sentiment_strong_threshold'), 0.5) # 持股心态强阈值
        chip_trend_momentum_positive_threshold = get_param_value(p_conf_fusion_inflection.get('chip_trend_momentum_positive_threshold'), 0.0) # 筹码趋势动量转正阈值
        # 4. 核心条件判断
        # 条件A: 资金流吸筹拐点意图强烈
        cond_ff_inflection_strong = (ff_inflection_intent > ff_inflection_threshold)
        # 条件B: 筹码结构支持 (筹码集中、成本有利、持股稳定、趋势动量向上，为拉升提供基础)
        cond_chip_supportive = (chip_concentration > chip_concentration_threshold) & \
                               (chip_cost_structure > chip_cost_structure_favorable_threshold) & \
                               (chip_holder_sentiment > chip_holder_sentiment_strong_threshold) & \
                               (chip_trend_momentum > chip_trend_momentum_positive_threshold)
        # 5. 融合条件，计算最终拐点分数
        # 基础分数：当资金流拐点意图强烈时
        inflection_score.loc[cond_ff_inflection_strong] = ff_inflection_intent.loc[cond_ff_inflection_strong] * 0.6 # 资金流意图是核心
        # 如果筹码结构也支持，则进一步增强信号
        inflection_score.loc[cond_ff_inflection_strong & cond_chip_supportive] += \
            (chip_concentration.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.1 +
             chip_cost_structure.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.1 +
             chip_holder_sentiment.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.1 +
             chip_trend_momentum.loc[cond_ff_inflection_strong & cond_chip_supportive] * 0.1) # 筹码各方面加权
        # 确保分数在 [0, 1] 之间
        inflection_score = inflection_score.clip(0, 1)
        # 6. 多时间维度归一化 (平滑信号，使其更具趋势性)
        tf_weights_fusion_inflection = get_param_value(p_conf_fusion_inflection.get('tf_fusion_weights'), {5: 0.6, 13: 0.3, 21: 0.1})
        inflection_score_normalized = get_adaptive_mtf_normalized_score(inflection_score, df_index, ascending=True, tf_weights=tf_weights_fusion_inflection).clip(0, 1)
        states['FUSION_BIPOLAR_ACCUMULATION_INFLECTION_POINT'] = inflection_score_normalized
        # --- Debugging output for probe date ---
        if probe_date_for_loop is not None and probe_date_for_loop in df_index:
            print(f"    -> [吸筹拐点信号探针] @ {probe_date_for_loop.date()}:")
            print(f"       - ff_inflection_intent: {ff_inflection_intent.loc[probe_date_for_loop]:.4f}")
            print(f"       - chip_concentration: {chip_concentration.loc[probe_date_for_loop]:.4f}")
            print(f"       - chip_cost_structure: {chip_cost_structure.loc[probe_date_for_loop]:.4f}")
            print(f"       - chip_holder_sentiment: {chip_holder_sentiment.loc[probe_date_for_loop]:.4f}")
            print(f"       - chip_trend_momentum: {chip_trend_momentum.loc[probe_date_for_loop]:.4f}")
            print(f"       - cond_ff_inflection_strong: {cond_ff_inflection_strong.loc[probe_date_for_loop]}")
            print(f"       - cond_chip_supportive: {cond_chip_supportive.loc[probe_date_for_loop]}")
            print(f"       - inflection_score (raw): {inflection_score.loc[probe_date_for_loop]:.4f}")
            print(f"       - inflection_score (normalized): {inflection_score_normalized.loc[probe_date_for_loop]:.4f}")
        print(f"  -- [融合层] “吸筹拐点信号”冶炼完成，最新分值: {inflection_score_normalized.iloc[-1]:.4f}")
        return states

    def _synthesize_chip_structural_potential(self) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】冶炼“筹码结构势能” (FUSION_BIPOLAR_CHIP_STRUCTURAL_POTENTIAL)
        - 核心思想: 融合新一代筹码指标中具备“势能”和“潜力”属性的信号，专门用于量化当前筹码结构中蕴含的、
                      未来可能爆发的能量。旨在为认知层的“突破”、“加速”等剧本提供关键的先验信念。
        - 证据链:
          1. 结构势能分 (structural_potential_score_D): 官方综合势能评分。
          2. 突破就绪分 (breakout_readiness_score_D): 对即将突破的量化评估。
          3. 结构张力指数 (structural_tension_index_D): 结构内部积蓄的能量。
          4. 结构杠杆 (structural_leverage_D): 势能释放的效率。
          5. 真空区量级 (vacuum_zone_magnitude_D): 价格运动的潜在空间。
        - 输出: [-1, 1] 的双极性分数，正分代表向上的结构势能强大，负分代表结构松散或存在向下的势能。
        """
        print("  -- [融合层] 正在冶炼“筹码结构势能”...")
        states = {}
        df_index = self.strategy.df_indicators.index
        # 获取所有势能相关指标
        potential_score_raw = self._get_safe_series(self.strategy.df_indicators, 'structural_potential_score_D', 50.0, method_name="_synthesize_chip_structural_potential")
        breakout_readiness_raw = self._get_safe_series(self.strategy.df_indicators, 'breakout_readiness_score_D', 50.0, method_name="_synthesize_chip_structural_potential")
        tension_raw = self._get_safe_series(self.strategy.df_indicators, 'structural_tension_index_D', 0.0, method_name="_synthesize_chip_structural_potential")
        leverage_raw = self._get_safe_series(self.strategy.df_indicators, 'structural_leverage_D', 0.0, method_name="_synthesize_chip_structural_potential")
        vacuum_raw = self._get_safe_series(self.strategy.df_indicators, 'vacuum_zone_magnitude_D', 0.0, method_name="_synthesize_chip_structural_potential")
        # 归一化为双极性分数
        potential_score = normalize_to_bipolar(potential_score_raw, df_index, window=55, sensitivity=20)
        breakout_readiness_score = normalize_to_bipolar(breakout_readiness_raw, df_index, window=55, sensitivity=20)
        tension_score = normalize_to_bipolar(tension_raw, df_index, window=55, sensitivity=0.5)
        leverage_score = normalize_to_bipolar(leverage_raw, df_index, window=55, sensitivity=0.5)
        vacuum_score = normalize_to_bipolar(vacuum_raw, df_index, window=55, sensitivity=0.5)
        # 定义权重并进行加权融合
        components = [potential_score, breakout_readiness_score, tension_score, leverage_score, vacuum_score]
        weights = np.array([0.3, 0.3, 0.15, 0.15, 0.1])
        aligned_components = [comp.reindex(df_index, fill_value=0.0) for comp in components]
        structural_potential_score = (
            aligned_components[0] * weights[0] +
            aligned_components[1] * weights[1] +
            aligned_components[2] * weights[2] +
            aligned_components[3] * weights[3] +
            aligned_components[4] * weights[4]
        ).clip(-1, 1)
        states['FUSION_BIPOLAR_CHIP_STRUCTURAL_POTENTIAL'] = structural_potential_score.astype(np.float32)
        print(f"  -- [融合层] “筹码结构势能”冶炼完成，最新分值: {structural_potential_score.iloc[-1]:.4f}")
        return states

