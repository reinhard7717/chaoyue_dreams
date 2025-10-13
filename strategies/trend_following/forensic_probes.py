# 文件: strategies/trend_following/forensic_probes.py
# 法医探针集合
import pandas as pd
import numpy as np
import pandas_ta as ta
import json
from typing import Dict
from strategies.trend_following.utils import get_params_block, calculate_holographic_dynamics, get_param_value, calculate_context_scores, normalize_score, normalize_to_bipolar, _calculate_gaia_bedrock_support, _calculate_historical_low_support, get_unified_score

class ForensicProbes:
    """
    【V1.1 · 依赖重铸版】法医探针集合
    - 核心修正: 构造函数现在接收 IntelligenceLayer 实例，解决了因错误的对象引用导致的 AttributeError。
    - 架构意义: 建立了清晰的依赖关系，探针集合现在是情报层的直接附属，可以访问其所有子模块。
    """
    def __init__(self, intelligence_layer_instance):
        # 接收 intelligence_layer_instance 而非 strategy_instance
        self.strategy = intelligence_layer_instance.strategy
        # 探针可能需要访问认知引擎等子模块，通过 intelligence_layer_instance 传递
        self.cognitive_intel = intelligence_layer_instance.cognitive_intel
        # 为新的筹码探针获取 chip_intel 引用
        self.chip_intel = intelligence_layer_instance.chip_intel

    def _deploy_prophet_entry_probe(self, probe_date: pd.Timestamp):
        """
        【V2.5 · 赫淮斯托斯重铸协议同步版】“先知入场神谕”专属法医探针
        - 核心革命: 探针的重算逻辑已与主引擎的“赫淮斯托斯重铸协议”版本完全同步。
        - 新核心公式: intraday_low_pct_change 被 clip(upper=0)，确保只有真下跌才会计分。
        - 收益: 确保探针能够正确解剖和验证“只买真恐慌”的最终哲学。
        """
        print("\n--- [探针] 正在解剖: 【神谕 · 先知入场】 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators

        def get_val(name, date, default=np.nan):
            series = atomic.get(name)
            if series is None: return default
            return series.get(date, default)

        print("\n  [链路层 1] 解剖 -> 最终预测机会 (PREDICTIVE_OPP_CAPITULATION_REVERSAL)")
        final_opp_score = get_val('PREDICTIVE_OPP_CAPITULATION_REVERSAL', probe_date, 0.0)
        print(f"    - 【最终预测值】: {final_opp_score:.4f}")
        print(f"    - [核心公式]: 预测机会 = 恐慌战备分 (SCORE_SETUP_PANIC_SELLING)")

        print("\n  [链路层 2] 解剖 -> 核心输入: 恐慌战备分 (SCORE_SETUP_PANIC_SELLING)")
        panic_setup_score = get_val('SCORE_SETUP_PANIC_SELLING', probe_date, 0.0)
        print(f"    - 【恐慌战备分】: {panic_setup_score:.4f}")
        print(f"    - [核心公式]: (六大支柱加权和 * 静谧度 * 反弹强度) * 赫尔墨斯调节器 (当满足价格暴跌门槛时)")

        print("\n  [链路层 3] 钻透 -> 六大支柱 & 调节器")
        
        p_panic = get_params_block(self.strategy, 'panic_selling_setup_params', {})
        pillar_weights = get_param_value(p_panic.get('pillar_weights'), {})
        min_price_drop_pct = get_param_value(p_panic.get('min_price_drop_pct'), -0.025)

        # 同步“赫淮斯托斯重铸协议”
        intraday_low_pct_change_raw = (df.at[probe_date, 'low_D'] - df.at[probe_date, 'pre_close_D']) / df.at[probe_date, 'pre_close_D'] if df.at[probe_date, 'pre_close_D'] > 0 else 0.0
        intraday_low_pct_change_series = ((df['low_D'] - df['pre_close_D']) / df['pre_close_D'].replace(0, np.nan)).clip(upper=0)
        
        price_drop_score_recalc = normalize_score(intraday_low_pct_change_series, df.index, window=60, ascending=False).get(probe_date, 0.0)
        print(f"    --- 支柱一: 价格暴跌 (权重: {pillar_weights.get('price_drop', 0):.2f}) ---")
        print(f"      - 当日盘中最大跌幅 (原始值): {intraday_low_pct_change_raw:.2%}")
        print(f"      - [探针重算] 价格暴跌分 (经clip修正): {price_drop_score_recalc:.4f}")

        # ... 其他支柱和调节器的解剖逻辑保持不变 ...
        volume_spike_score_recalc = normalize_score(df['volume_D'] / df['VOL_MA_21_D'], df.index, window=60, ascending=True).get(probe_date, 0.0)
        print(f"    --- 支柱二: 成交天量 (权重: {pillar_weights.get('volume_spike', 0):.2f}) ---")
        print(f"      - [探针重算] 成交天量分: {volume_spike_score_recalc:.4f}")
        from .utils import get_unified_score
        chip_breakdown_score_recalc = get_unified_score(atomic, df.index, 'CHIP_BEARISH_RESONANCE').get(probe_date, 0.0)
        chip_integrity_score_recalc = 1.0 - chip_breakdown_score_recalc
        print(f"    --- 支柱三: 结构完整度 (权重: {pillar_weights.get('chip_integrity', 0):.2f}) ---")
        print(f"      - [探针重算] 结构完整度分: {chip_integrity_score_recalc:.4f}")
        tactic_engine_probe = self.cognitive_intel.tactic_engine
        despair_context_score_recalc = tactic_engine_probe._calculate_despair_context_score(df, p_panic).get(probe_date, 0.0)
        print(f"    --- 支柱四: 绝望背景 (权重: {pillar_weights.get('despair_context', 0):.2f}) ---")
        print(f"      - [探针重算] 绝望背景分: {despair_context_score_recalc:.4f}")
        structural_test_score_recalc = tactic_engine_probe.calculate_structural_test_score(df, p_panic).get(probe_date, 0.0)
        print(f"    --- 支柱五: 结构支撑测试 (权重: {pillar_weights.get('structural_test', 0):.2f}) ---")
        print(f"      - [探针重算] 结构支撑测试分: {structural_test_score_recalc:.4f}")
        ma_structure_score_recalc = tactic_engine_probe._calculate_ma_trend_context(df, [5, 13, 21, 55]).get(probe_date, 0.5)
        print(f"    --- 支柱六: 均线结构 (权重: {pillar_weights.get('ma_structure', 0):.2f}) ---")
        print(f"      - [探针重算] 均线结构分: {ma_structure_score_recalc:.4f}")
        print(f"    --- 调节器 I: 成交量静谧度 ---")
        logic_params = get_param_value(p_panic.get('volume_calmness_logic'), {})
        lifeline_ma_period = get_param_value(logic_params.get('lifeline_ma_period'), 5)
        lifeline_base_score = get_param_value(logic_params.get('lifeline_base_score'), 1.0)
        p_depth_bonus = get_param_value(logic_params.get('absolute_depth_bonus'), {})
        p_gradient_bonus = get_param_value(logic_params.get('structural_gradient_bonus'), {})
        raw_volume_calmness_score_recalc = 0.0
        lifeline_ma_col = f'VOL_MA_{lifeline_ma_period}_D'
        if lifeline_ma_col in df.columns and df.at[probe_date, 'volume_D'] < df.at[probe_date, lifeline_ma_col]:
            raw_volume_calmness_score_recalc = lifeline_base_score
            for p_str, weight in p_depth_bonus.items():
                ma_col = f'VOL_MA_{p_str}_D'
                if ma_col in df.columns and df.at[probe_date, 'volume_D'] < df.at[probe_date, ma_col]:
                    raw_volume_calmness_score_recalc += weight
            if get_param_value(p_gradient_bonus.get('enabled'), False):
                level_weights = get_param_value(p_gradient_bonus.get('level_weights'), {})
                ma5, ma13, ma21, ma55 = df.get(f'VOL_MA_5_D').at[probe_date], df.get(f'VOL_MA_13_D').at[probe_date], df.get(f'VOL_MA_21_D').at[probe_date], df.get(f'VOL_MA_55_D').at[probe_date]
                is_level_1, is_level_2, is_level_3 = (ma5 < ma13), (ma5 < ma13) and (ma13 < ma21), (ma5 < ma13) and (ma13 < ma21) and (ma21 < ma55)
                if is_level_1: raw_volume_calmness_score_recalc += level_weights.get('level_1', 0.0)
                if is_level_2: raw_volume_calmness_score_recalc += level_weights.get('level_2', 0.0)
                if is_level_3: raw_volume_calmness_score_recalc += level_weights.get('level_3', 0.0)
        final_calmness_score_recalc = raw_volume_calmness_score_recalc
        print(f"      - [探针重算] 最终静谧度分: {final_calmness_score_recalc:.4f}")
        print(f"    --- 调节器 II: 反弹强度 ---")
        day_range_raw = df.at[probe_date, 'high_D'] - df.at[probe_date, 'low_D']
        rebound_strength_score_recalc = ((df.at[probe_date, 'close_D'] - df.at[probe_date, 'low_D']) / day_range_raw) if day_range_raw > 0 else 0.5
        print(f"      - [探针重算] 反弹强度分: {rebound_strength_score_recalc:.4f}")
        print(f"    --- 调节器 III: 赫尔墨斯信使 (日内博弈) ---")
        upper_shadow_raw = df.at[probe_date, 'high_D'] - max(df.at[probe_date, 'open_D'], df.at[probe_date, 'close_D'])
        lower_shadow_raw = min(df.at[probe_date, 'open_D'], df.at[probe_date, 'close_D']) - df.at[probe_date, 'low_D']
        hermes_score_raw = ((lower_shadow_raw - upper_shadow_raw) / day_range_raw) if day_range_raw > 0 else 0.0
        hermes_regulator_recalc = (hermes_score_raw + 1) / 2.0
        print(f"      - [探针重算] 赫尔墨斯调节器: {hermes_regulator_recalc:.4f}")

        print("\n  [链路层 4] 最终验证")
        snapshot_panic_recalc = (
            price_drop_score_recalc * pillar_weights.get('price_drop', 0) +
            volume_spike_score_recalc * pillar_weights.get('volume_spike', 0) +
            chip_integrity_score_recalc * pillar_weights.get('chip_integrity', 0) +
            despair_context_score_recalc * pillar_weights.get('despair_context', 0) +
            structural_test_score_recalc * pillar_weights.get('structural_test', 0) +
            ma_structure_score_recalc * pillar_weights.get('ma_structure', 0)
        )
        print(f"    - [探针重算] 六大支柱加权和 (瞬时恐慌快照分): {snapshot_panic_recalc:.4f}")
        
        is_significant_drop = intraday_low_pct_change_raw < min_price_drop_pct
        print(f"    - [探针检查] 价格暴跌门槛 ({min_price_drop_pct:.2%}) 是否满足? {'✅ 是' if is_significant_drop else '❌ 否'}")

        base_recalculated_score = snapshot_panic_recalc * final_calmness_score_recalc * rebound_strength_score_recalc
        final_recalculated_score = base_recalculated_score * hermes_regulator_recalc if is_significant_drop else 0
        
        print(f"    - [探针重算恐慌战备分]: ({snapshot_panic_recalc:.4f} * {final_calmness_score_recalc:.4f} * {rebound_strength_score_recalc:.4f}) * {hermes_regulator_recalc:.4f} = {final_recalculated_score:.4f}")
        print(f"    - [对比]: 实际值 {panic_setup_score:.4f} vs 重算值 {final_recalculated_score:.4f}")
        print("--- 先知入场神谕探针解剖完毕 ---")

    # 部署“德尔菲神谕-离场探针协议”
    def _deploy_prophet_exit_probe(self, probe_date: pd.Timestamp):
        """
        【V1.0 · 新增】“德尔菲神谕-离场探针”
        - 核心职责: 钻透式解剖“高潮衰竭”风险信号 (PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION) 及其警报触发逻辑。
        """
        print("\n--- [探针] 正在解剖: 【神谕 · 先知离场】 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators

        def get_val(name, date, default=np.nan):
            series = atomic.get(name)
            if series is None: return default
            return series.get(date, default)

        print("\n  [链路层 1] 解剖 -> 最终预测风险 (PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION)")
        final_risk_score = get_val('PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION', probe_date, 0.0)
        print(f"    - 【最终风险值】: {final_risk_score:.4f}")
        print(f"    - [核心公式]: (亢奋分 * 天量分 * K线疲弱分) ^ (1/3)")

        print("\n  [链路层 2] 钻透 -> 风险三位一体")
        p_pred = get_params_block(self.strategy, 'predictive_intelligence_params', {})
        fusion_weights = get_param_value(p_pred.get('trinity_fusion_weights'), {})

        # 2.1 亢奋分
        euphoria_score_recalc = get_val('COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION', probe_date, 0.0)
        print(f"    --- 支柱一: 亢奋分 (权重: {fusion_weights.get('euphoria', 0):.2f}) ---")
        print(f"      - [探针获取] 亢奋分: {euphoria_score_recalc:.4f}")

        # 2.2 天量分
        vol_spike_series = df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)
        volume_spike_score_recalc = normalize_score(vol_spike_series, df.index, window=60, ascending=True).get(probe_date, 0.0)
        print(f"    --- 支柱二: 天量分 (权重: {fusion_weights.get('volume', 0):.2f}) ---")
        print(f"      - [探针重算] 天量分: {volume_spike_score_recalc:.4f}")

        # 2.3 K线疲弱分
        day_range = df.at[probe_date, 'high_D'] - df.at[probe_date, 'low_D']
        upper_shadow = df.at[probe_date, 'high_D'] - max(df.at[probe_date, 'open_D'], df.at[probe_date, 'close_D'])
        upper_shadow_ratio = (upper_shadow / day_range) if day_range > 0 else 0.0
        is_negative_close = df.at[probe_date, 'close_D'] < df.at[probe_date, 'open_D']
        kline_weakness_score_recalc = upper_shadow_ratio * float(is_negative_close)
        print(f"    --- 支柱三: K线疲弱分 (权重: {fusion_weights.get('kline', 0):.2f}) ---")
        print(f"      - [探针重算] K线疲弱分: {kline_weakness_score_recalc:.4f} (上影线率: {upper_shadow_ratio:.2f}, 是否阴线: {is_negative_close})")

        print("\n  [链路层 3] 最终验证 -> 风险融合")
        recalculated_risk_score = (
            (euphoria_score_recalc ** fusion_weights.get('euphoria', 0.33)) *
            (volume_spike_score_recalc ** fusion_weights.get('volume', 0.33)) *
            (kline_weakness_score_recalc ** fusion_weights.get('kline', 0.33))
        )
        print(f"    - [探针重算风险值]: ({euphoria_score_recalc:.4f}^{fusion_weights.get('euphoria', 0.33):.2f} * {volume_spike_score_recalc:.4f}^{fusion_weights.get('volume', 0.33):.2f} * {kline_weakness_score_recalc:.4f}^{fusion_weights.get('kline', 0.33):.2f}) = {recalculated_risk_score:.4f}")
        print(f"    - [对比]: 实际值 {final_risk_score:.4f} vs 重算值 {recalculated_risk_score:.4f}")

        print("\n  [链路层 4] 最终验证 -> 警报触发逻辑")
        p_judge = get_params_block(self.strategy, 'judgment_params', {})
        prophet_threshold = get_param_value(p_judge.get('prophet_alert_threshold'), 0.7)
        is_uptrend_context = df.at[probe_date, 'close_D'] > df.at[probe_date, 'EMA_5_D']
        
        alert_level = get_val('ALERT_LEVEL', probe_date, 0)
        alert_reason = get_val('ALERT_REASON', probe_date, '')

        print(f"    - [条件一] 风险分 > 阈值?  ({final_risk_score:.4f} > {prophet_threshold}) -> {'✅ 是' if final_risk_score > prophet_threshold else '❌ 否'}")
        print(f"    - [条件二] 处于上升趋势 (close > EMA5)? -> {'✅ 是' if is_uptrend_context else '❌ 否'}")
        
        recalculated_alert = (final_risk_score > prophet_threshold) and is_uptrend_context
        print(f"    - [探针重算警报]: {'触发' if recalculated_alert else '不触发'}")
        print(f"    - [实际警报]: Level {alert_level} ({alert_reason})")
        print("--- 先知离场神谕探针解剖完毕 ---")

    def _deploy_hephaestus_forge_probe(self, probe_date: pd.Timestamp):
        """
        【V2.2 · 牛顿第二定律校准版】“赫淮斯托斯熔炉”探针
        - 核心修复: 修正了加速度计算的致命错误，确保其是对“速度”求导，而非重复对“位移”求导。
        - 收益: 彻底解决了探针重算结果与主引擎之间的微小偏差，实现了完美的逻辑统一。
        """
        print("\n--- [探针] 正在启用: 🔥【赫淮斯托斯熔炉 · 风险融合解剖 V2.2】🔥 ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        p_fused_risk = get_params_block(self.strategy, 'fused_risk_scoring')
        if not get_param_value(p_fused_risk.get('enabled'), True):
            print("    - 风险融合模块未启用，跳过解剖。")
            return
        print("  --- [阶段1] 信号输入审查 ---")
        risk_categories = p_fused_risk.get('risk_categories', {})
        all_required_signals = {s for signals in risk_categories.values() if isinstance(signals, dict) for s in signals if s != "说明"}
        signal_value_cache = {}
        for sig_name in sorted(list(all_required_signals)):
            val = atomic.get(sig_name, pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
            signal_value_cache[sig_name] = val
            if "ARCHANGEL" in sig_name:
                print(f"    - 关键输入信号 [{sig_name}]: {val:.4f}  <-- 风险源头")
            else:
                print(f"    - 输入信号 [{sig_name}]: {val:.4f}")
        print("\n  --- [阶段2] 维度内融合解剖 (修正为 max() 算法) ---")
        fused_dimension_scores = {}
        for category_name, signals in risk_categories.items():
            if category_name == "说明": continue
            print(f"\n    -> 正在处理维度: [{category_name.upper()}]")
            category_signal_scores = []
            for signal_name, signal_params in signals.items():
                if signal_name == "说明": continue
                atomic_score = signal_value_cache.get(signal_name, 0.0)
                processed_score = 1.0 - atomic_score if signal_params.get('inverse', False) else atomic_score
                weighted_score = processed_score * signal_params.get('weight', 1.0)
                category_signal_scores.append(weighted_score)
                print(f"      - 信号 '{signal_name}':")
                print(f"        - 原始值: {atomic_score:.4f} -> 处理后: {processed_score:.4f} -> 加权后: {weighted_score:.4f} (权重: {signal_params.get('weight', 1.0)})")
            if category_signal_scores:
                dimension_risk = max(category_signal_scores)
                fused_dimension_scores[category_name] = dimension_risk
                print(f"      - 维度内融合计算 (max() 算法):")
                print(f"        - 维度总风险 = max({[f'{s:.4f}' for s in category_signal_scores]}) = {dimension_risk:.4f}")
            else:
                fused_dimension_scores[category_name] = 0.0
                print(f"      - 维度内无信号，总风险为 0.0")
        print("\n  --- [阶段3] 跨维度融合解剖 ---")
        valid_scores = list(fused_dimension_scores.values())
        if valid_scores:
            total_fused_risk = max(valid_scores)
            print(f"    - 所有维度风险分: { {k: f'{v:.4f}' for k, v in fused_dimension_scores.items()} }")
            print(f"    - 裁决: 取最大值 -> {total_fused_risk:.4f}")
        else:
            total_fused_risk = 0.0
            print(f"    - 无有效维度风险分，总分为 0.0")
        print("\n  --- [阶段4] 共振惩罚解剖 ---")
        p_resonance = p_fused_risk.get('resonance_penalty_params', {})
        if get_param_value(p_resonance.get('enabled'), True):
            core_dims = p_resonance.get('core_risk_dimensions', [])
            min_dims = p_resonance.get('min_dimensions_for_resonance', 2)
            threshold = p_resonance.get('risk_score_threshold', 0.6)
            penalty_multiplier = p_resonance.get('penalty_multiplier', 1.2)
            high_risk_dimension_count = sum(1 for dim in core_dims if fused_dimension_scores.get(dim, 0.0) > threshold)
            is_resonance_triggered = (high_risk_dimension_count >= min_dims)
            print(f"    - 共振诊断: {high_risk_dimension_count}个核心维度 > {threshold} (要求: {min_dims}个) -> 触发: {is_resonance_triggered}")
            if is_resonance_triggered:
                final_risk_score = total_fused_risk * penalty_multiplier
                print(f"    - 共振惩罚: {total_fused_risk:.4f} * {penalty_multiplier} = {final_risk_score:.4f}")
            else:
                final_risk_score = total_fused_risk
                print(f"    - 未触发共振惩罚。")
        else:
            final_risk_score = total_fused_risk
            print(f"    - 共振惩罚模块未启用。")
        print("\n  --- [阶段5] 神盾协议 & 动态锻造解剖 ---")
        trend_quality_score = atomic.get('COGNITIVE_SCORE_TREND_QUALITY', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        healthy_pullback_score = atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        aegis_shield_strength = max(trend_quality_score, healthy_pullback_score)
        suppression_factor = 1.0 - aegis_shield_strength
        risk_snapshot_score_recalc = final_risk_score * suppression_factor
        print(f"    - 神盾强度 (Aegis Strength): max(趋势质量:{trend_quality_score:.2f}, 健康回踩:{healthy_pullback_score:.2f}) = {aegis_shield_strength:.4f}")
        print(f"    - 风险抑制因子 (Suppression): 1.0 - {aegis_shield_strength:.2f} = {suppression_factor:.4f}")
        print(f"    - 风险快照分 (经神盾调节): {final_risk_score:.4f} * {suppression_factor:.2f} = {risk_snapshot_score_recalc:.4f}")
        risk_snapshot_series = atomic.get('COGNITIVE_INTERNAL_RISK_SNAPSHOT', pd.Series(0.0, index=df.index))
        p_meta_cog = get_params_block(self.strategy, 'cognitive_intelligence_params', {}).get('relational_meta_analysis_params', {})
        w_state = get_param_value(p_meta_cog.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta_cog.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta_cog.get('acceleration_weight'), 0.4)
        state_val = risk_snapshot_series.clip(0, 2.0).get(probe_date, 0.0)
        # 修正开始: 确保加速度是对速度求导
        relationship_trend = risk_snapshot_series.diff(5).fillna(0)
        vel_series = normalize_to_bipolar(relationship_trend, df.index, 55)
        vel_val = vel_series.get(probe_date, 0.0)
        relationship_accel = relationship_trend.diff(5).fillna(0) # 修正: 加速度是速度的变化率
        accel_series = normalize_to_bipolar(relationship_accel, df.index, 55)
        accel_val = accel_series.get(probe_date, 0.0)
        # 修正结束
        final_dynamic_risk_recalc = (state_val * w_state + vel_val * w_velocity + accel_val * w_acceleration).clip(0, 1)
        final_dynamic_risk_actual = atomic.get('COGNITIVE_FUSED_RISK_SCORE', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        print(f"    - 动态锻造: State({state_val:.2f})*w + Velocity({vel_val:.2f})*w + Accel({accel_val:.2f})*w = {final_dynamic_risk_recalc:.4f}")
        print(f"\n  --- [最终裁决] ---")
        print(f"    - 🔥 熔炉产物 (COGNITIVE_FUSED_RISK_SCORE): {final_dynamic_risk_actual:.4f}")
        print(f"    - [对比]: 实际值 {final_dynamic_risk_actual:.4f} vs 重算值 {final_dynamic_risk_recalc:.4f}")
        print("--- “赫淮斯托斯熔炉”探针运行完毕 ---\n")

    def _deploy_themis_scales_probe(self, probe_date: pd.Timestamp):
        """
        【V1.26 · 神盾协议探针版】“忒弥斯天平”上下文解剖探针
        - 核心升级: 历史高点显微镜现在调用通用的阿波罗之箭探针，并展示“质量x重要性”的融合逻辑。
        """
        print("\n--- [探针] 正在启用: ⚖️【忒弥斯天平 · 上下文解剖】⚖️ ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        strategy_instance_ref = self.strategy
        p_synthesis = get_params_block(strategy_instance_ref, 'ultimate_signal_synthesis_params', {})
        gaia_params = get_param_value(p_synthesis.get('gaia_bedrock_params'), {})
        cooldown_reset_volume_ma_period = get_param_value(gaia_params.get('cooldown_reset_volume_ma_period'), 55)
        ares_vol_ma_col = 'VOL_MA_5_D'
        cooldown_vol_ma_col = f'VOL_MA_{cooldown_reset_volume_ma_period}_D'
        print("\n  --- [结构性支撑审查] 关键均线系统快照 ---")
        ma_periods_to_probe = [5, 55, 144, 233, 377]
        close_price = df.get('close_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        open_price = df.get('open_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        low_price = df.get('low_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        high_price = df.get('high_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        volume = df.get('volume_D', pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        ares_volume_ma = df.get(ares_vol_ma_col, pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        cooldown_volume_ma = df.get(cooldown_vol_ma_col, pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
        if isinstance(close_price, (float, np.floating)):
            print(f"    - {'high_D':<12}: {high_price:.2f}  (当日最高价)")
            print(f"    - {'open_D':<12}: {open_price:.2f}  (当日开盘价)")
            print(f"    - {'close_D':<12}: {close_price:.2f}  (当日收盘价)")
            print(f"    - {'low_D':<12}: {low_price:.2f}  (当日最低价)")
            print(f"    - {'volume_D':<12}: {volume:,.0f}  (当日成交量)")
            print(f"    - {ares_vol_ma_col:<12}: {ares_volume_ma:,.0f}  (阿瑞斯之矛-成交量均线)")
            print(f"    - {cooldown_vol_ma_col:<12}: {cooldown_volume_ma:,.0f}  (冷却重置-成交量均线)")
        else:
            print(f"    - {'high_D':<12}: {high_price}")
            print(f"    - {'open_D':<12}: {open_price}")
            print(f"    - {'close_D':<12}: {close_price}")
            print(f"    - {'low_D':<12}: {low_price}")
            print(f"    - {'volume_D':<12}: {volume}")
            print(f"    - {ares_vol_ma_col:<12}: {ares_volume_ma}")
            print(f"    - {cooldown_vol_ma_col:<12}: {cooldown_volume_ma}")
        for period in ma_periods_to_probe:
            col_name = f'MA_{period}_D'
            ma_value = df.get(col_name, pd.Series(np.nan, index=df.index)).get(probe_date, 'N/A')
            if isinstance(ma_value, (float, np.floating)):
                print(f"    - {col_name:<12}: {ma_value:.2f}")
            else:
                print(f"    - {col_name:<12}: {ma_value}")
        print("\n  --- [天平左侧] 底部上下文解剖 ---")
        depth_threshold = get_param_value(p_synthesis.get('deep_bearish_threshold'), 0.05)
        ma55_lifeline = df.get('MA_55_D', df['close_D'])
        is_deep_bearish_zone = (df['close_D'] < ma55_lifeline * (1 - depth_threshold)).astype(float)
        ma55_slope = ma55_lifeline.diff(3).fillna(0)
        slope_moderator = (0.5 + 0.5 * np.tanh(ma55_slope * 100)).fillna(0.5)
        distance_from_ma55 = (df['close_D'] - ma55_lifeline) / ma55_lifeline.replace(0, np.nan)
        lifeline_support_score_raw = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2).fillna(0.0)
        lifeline_support_score = lifeline_support_score_raw * slope_moderator
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        absolute_value_zone_score = 1.0 - price_pos_yearly
        deep_bottom_context_score_values = np.maximum.reduce([
            lifeline_support_score.values,
            absolute_value_zone_score.values
        ])
        deep_bottom_context_score = pd.Series(deep_bottom_context_score_values, index=df.index, dtype=np.float32)
        rsi_w_col = 'RSI_13_W'
        rsi_w_oversold_score = normalize_score(df.get(rsi_w_col, pd.Series(50, index=df.index)), df.index, window=52, ascending=False, default_value=0.5)
        cycle_phase = atomic.get('DOMINANT_CYCLE_PHASE', pd.Series(0.0, index=df.index)).fillna(0.0)
        cycle_trough_score = (1 - cycle_phase) / 2.0
        context_weights = get_param_value(p_synthesis.get('bottom_context_weights'), {'price_pos': 0.5, 'rsi_w': 0.3, 'cycle': 0.2})
        score_components = {'price_pos': deep_bottom_context_score, 'rsi_w': rsi_w_oversold_score, 'cycle': cycle_trough_score}
        valid_scores, valid_weights = [], []
        for name, weight in context_weights.items():
            if name in score_components and weight > 0:
                valid_scores.append(score_components[name].values)
                valid_weights.append(weight)
        if not valid_scores:
            bottom_context_score_raw = pd.Series(0.5, index=df.index, dtype=np.float32)
        else:
            weights_array = np.array(valid_weights)
            total_weight = weights_array.sum()
            normalized_weights = weights_array / total_weight if total_weight > 0 else np.full_like(weights_array, 1.0 / len(weights_array))
            stacked_scores = np.stack(valid_scores, axis=0)
            safe_scores = np.maximum(stacked_scores, 1e-9)
            weighted_log_sum = np.sum(np.log(safe_scores) * normalized_weights[:, np.newaxis], axis=0)
            bottom_context_score_raw = pd.Series(np.exp(weighted_log_sum), index=df.index, dtype=np.float32)
        conventional_bottom_score = bottom_context_score_raw * is_deep_bearish_zone
        print(f"    - [组件1] 常规底部得分 (经深度熊市过滤): {conventional_bottom_score.get(probe_date, 0.0):.4f}")
        # 在探针调用时，传入 atomic_states
        gaia_bedrock_support_score = _calculate_gaia_bedrock_support(df, gaia_params, atomic)
        
        print(f"    - [组件2] 盖亚基石支撑分: {gaia_bedrock_support_score.get(probe_date, 0.0):.4f}")
        print("      --- [盖亚显微镜] 深入解剖 ---")
        support_levels = get_param_value(gaia_params.get('support_levels'), [55, 89, 144, 233, 377])
        confirmation_window = get_param_value(gaia_params.get('confirmation_window'), 3)
        aegis_lookback_window = get_param_value(gaia_params.get('aegis_lookback_window'), 5)
        confirmation_cooldown_period = get_param_value(gaia_params.get('confirmation_cooldown_period'), 10)
        influence_zone_pct = get_param_value(gaia_params.get('influence_zone_pct'), 0.03)
        defense_base_score = get_param_value(gaia_params.get('defense_base_score'), 0.4)
        defense_yang_line_weight = get_param_value(gaia_params.get('defense_yang_line_weight'), 0.1)
        defense_dominance_weight = get_param_value(gaia_params.get('defense_dominance_weight'), 0.2)
        defense_volume_weight = get_param_value(gaia_params.get('defense_volume_weight'), 0.3)
        confirmation_score = get_param_value(gaia_params.get('confirmation_score'), 0.8)
        aegis_quality_bonus_factor = get_param_value(gaia_params.get('aegis_quality_bonus_factor'), 0.25)
        g_ma_cols = [f'MA_{p}_D' for p in support_levels if f'MA_{p}_D' in df.columns]
        g_ma_df = df[g_ma_cols]
        g_ma_df_below_price = g_ma_df.where(g_ma_df.le(df['close_D'], axis=0))
        g_acting_lifeline = g_ma_df_below_price.max(axis=1).ffill()
        g_is_in_influence_zone = pd.Series(False, index=df.index)
        g_valid_indices = g_acting_lifeline.dropna().index
        g_upper_bound = g_acting_lifeline[g_valid_indices] * (1 + influence_zone_pct)
        g_is_in_influence_zone.loc[g_valid_indices] = df.loc[g_valid_indices, 'close_D'].between(g_acting_lifeline[g_valid_indices], g_upper_bound)
        g_base_defense_condition = (df['low_D'] < g_acting_lifeline) & g_is_in_influence_zone & (df['close_D'] > df['low_D'])
        g_is_yang_line = df['close_D'] > df['open_D']
        g_upper_shadow = df['high_D'] - np.maximum(df['open_D'], df['close_D'])
        g_lower_shadow = np.minimum(df['open_D'], df['close_D']) - df['low_D']
        g_has_dominance = g_lower_shadow > g_upper_shadow
        g_has_volume_spike = df['volume_D'] > df[ares_vol_ma_col]
        g_is_cassandra_warning = (g_upper_shadow > g_lower_shadow) & g_has_volume_spike
        g_defense_quality_score = pd.Series(0.0, index=df.index)
        g_defense_quality_score.loc[g_base_defense_condition] = defense_base_score
        g_defense_quality_score.loc[g_base_defense_condition & g_is_yang_line] += defense_yang_line_weight
        g_defense_quality_score.loc[g_base_defense_condition & g_has_dominance] += defense_dominance_weight
        g_defense_quality_score.loc[g_base_defense_condition & g_has_dominance & g_has_volume_spike] += defense_volume_weight
        g_defense_quality_score.loc[g_is_in_influence_zone & g_is_cassandra_warning] = 0.0
        g_defense_quality_score = g_defense_quality_score.clip(0, 1.0)
        g_max_recent_defense_quality = g_defense_quality_score.rolling(window=aegis_lookback_window, min_periods=1).max()
        g_is_standing_firm_in_zone = (df['close_D'] > g_acting_lifeline) & g_is_in_influence_zone
        g_is_confirmed_base = g_is_standing_firm_in_zone.rolling(window=confirmation_window, min_periods=confirmation_window).sum() >= confirmation_window
        g_is_cooldown_reset_signal = (g_upper_shadow > g_lower_shadow) & (df['volume_D'] > df[cooldown_vol_ma_col])
        g_confirmation_score_series = pd.Series(0.0, index=df.index)
        g_last_confirmation_date = pd.NaT
        g_is_in_cooldown_on_probe_date = False
        for idx in df.index:
            if idx > probe_date: break
            if pd.notna(g_last_confirmation_date) and (idx - g_last_confirmation_date).days < confirmation_cooldown_period:
                if idx == probe_date: g_is_in_cooldown_on_probe_date = True
                if g_is_cooldown_reset_signal.get(idx, False): g_last_confirmation_date = pd.NaT
                continue
            if g_is_confirmed_base.get(idx, False):
                recent_quality = g_max_recent_defense_quality.get(idx, 0.0)
                if recent_quality > 0:
                    aegis_score = confirmation_score + recent_quality * aegis_quality_bonus_factor
                    g_confirmation_score_series.loc[idx] = min(aegis_score, 1.0)
                else:
                    g_confirmation_score_series.loc[idx] = confirmation_score
                g_last_confirmation_date = idx
        print(f"        - acting_lifeline (代理总指挥): {g_acting_lifeline.get(probe_date, np.nan):.4f}")
        print("        --- [防守质量解剖 (赫尔墨斯信使)] ---")
        base_cond_val, yang_line_val, dominance_val, volume_val, cassandra_val, in_zone_val = (
            g_base_defense_condition.get(probe_date, False), g_is_yang_line.get(probe_date, False),
            g_has_dominance.get(probe_date, False), g_has_volume_spike.get(probe_date, False),
            g_is_cassandra_warning.get(probe_date, False), g_is_in_influence_zone.get(probe_date, False)
        )
        defense_score_today = g_defense_quality_score.get(probe_date, 0.0)
        print(f"          - 预警判定 (卡珊德拉): (在影响区内 {in_zone_val}) AND (上影>下影 AND 放量) -> {cassandra_val and in_zone_val}")
        if cassandra_val and in_zone_val:
            print(f"          - 裁决: 触发卡珊德拉预警，防守质量分强制归零。")
        else:
            print(f"          - 基础条件 (触线+下影): {base_cond_val} -> 基础分 {defense_base_score if base_cond_val else 0.0:.2f}")
            print(f"          - 权重1 (主权宣告-收阳): {yang_line_val} -> 加分 {defense_yang_line_weight if yang_line_val and base_cond_val else 0.0:.2f}")
            print(f"          - 权重2 (韧性胜利-下影优势): {dominance_val} -> 加分 {defense_dominance_weight if dominance_val and base_cond_val else 0.0:.2f}")
            print(f"          - 权重3 (主力参战-放量): {volume_val and dominance_val} (需下影优势) -> 加分 {defense_volume_weight if volume_val and dominance_val and base_cond_val else 0.0:.2f}")
        print(f"          - 当日独立防守分: {defense_score_today:.4f}")
        print("        --- [确认质量评估 (所罗门审判)] ---")
        is_confirmed_val = g_is_confirmed_base.get(probe_date, False)
        recent_quality_val = g_max_recent_defense_quality.get(probe_date, 0.0)
        confirmation_score_today = g_confirmation_score_series.get(probe_date, 0.0)
        print(f"          - is_confirmed_base (是否满足站稳天数): {is_confirmed_val}")
        print(f"          - is_in_cooldown (是否处于确认冷却期): {g_is_in_cooldown_on_probe_date}")
        print(f"          - 近期最高防守质量分 (lookback={aegis_lookback_window}d): {recent_quality_val:.4f}")
        if not g_is_in_cooldown_on_probe_date and is_confirmed_val:
            if recent_quality_val > 0:
                print(f"          - 审判类型: 神盾构筑 (基础分 {confirmation_score:.2f} + 质量奖励 {recent_quality_val:.2f} * {aegis_quality_bonus_factor:.2f})")
            else:
                print(f"          - 审判类型: 常规确认 (固定分 {confirmation_score:.2f})")
        else:
            print(f"          - 审判类型: 无 (is_confirmed={is_confirmed_val}, in_cooldown={g_is_in_cooldown_on_probe_date})")
        print(f"          - 当日独立确认分: {confirmation_score_today:.4f}")
        print("        --- [最终审判 (忒弥斯天平)] ---")
        final_gaia_score = max(defense_score_today, confirmation_score_today)
        print(f"          - 裁决: max(独立防守分, 独立确认分) = max({defense_score_today:.4f}, {confirmation_score_today:.4f}) = {final_gaia_score:.4f}")
        print("        --- [冷却重置解剖 (哈迪斯面纱)] ---")
        reset_cond1 = g_upper_shadow.get(probe_date, 0) > g_lower_shadow.get(probe_date, 0)
        reset_cond2 = df.get('volume_D').get(probe_date, 0) > df.get(cooldown_vol_ma_col).get(probe_date, np.inf)
        print(f"          - 条件1 (上影优势): upper_shadow > lower_shadow -> {reset_cond1}")
        print(f"          - 条件2 (成交放量): volume > {cooldown_vol_ma_col} -> {reset_cond2}")
        print(f"          - 综合判定 (is_cooldown_reset): {g_is_cooldown_reset_signal.get(probe_date, False)}")
        print("      --------------------------")
        p_fib_support = get_param_value(p_synthesis.get('fibonacci_support_params'), {})
        historical_low_support_score = _calculate_historical_low_support(df, p_fib_support)
        print(f"    - [组件3] 历史低点支撑分: {historical_low_support_score.get(probe_date, 0.0):.4f}")
        structural_support_score = np.maximum(gaia_bedrock_support_score, historical_low_support_score)
        final_bottom_context_score = np.maximum(conventional_bottom_score, structural_support_score)
        print(f"    - [融合步骤1] 结构支撑分 (盖亚 vs 历史低点): {structural_support_score.get(probe_date, 0.0):.4f}")
        print(f"    - [最终裁决] 底部上下文总分 (常规 vs 结构): {final_bottom_context_score.get(probe_date, 0.0):.4f}")
        print("\n  --- [天平右侧] 顶部上下文解剖 ---")
        atomic['strategy_instance_ref'] = self.strategy
        _, top_context = calculate_context_scores(df, atomic)
        del atomic['strategy_instance_ref']
        ma55 = df.get('MA_55_D', df['close_D'])
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        wave_channel_height = (rolling_high_55d - ma55).replace(0, 1e-9)
        stretch_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
        ma_periods = [5, 13, 21, 55]
        short_ma_cols = [f'MA_{p}_D' for p in ma_periods[:-1]]
        long_ma_cols = [f'MA_{p}_D' for p in ma_periods[1:]]
        if all(col in df for col in short_ma_cols + long_ma_cols):
            short_mas = df[short_ma_cols].values
            long_mas = df[long_ma_cols].values
            misalignment_matrix = (short_mas < long_mas).astype(np.float32)
            misalignment_score_values = np.mean(misalignment_matrix, axis=1)
            misalignment_score = pd.Series(misalignment_score_values, index=df.index)
        else:
            misalignment_score = pd.Series(0.5, index=df.index)
        bias_params = get_param_value(p_synthesis.get('bias_overheat_params'), {})
        warning_threshold = get_param_value(bias_params.get('warning_threshold'), 0.15)
        danger_threshold = get_param_value(bias_params.get('danger_threshold'), 0.25)
        bias_abs = df.get('BIAS_21_D', pd.Series(0, index=df.index)).abs()
        denominator = danger_threshold - warning_threshold
        if denominator <= 1e-6:
            overheat_score = (bias_abs > danger_threshold).astype(float)
        else:
            overheat_score = ((bias_abs - warning_threshold) / denominator).clip(0, 1)
        overheat_score = overheat_score.fillna(0.0)
        conventional_top_score = (stretch_score * misalignment_score * overheat_score)**(1/3)
        print(f"    - [组件1] 常规顶部得分 (传统三因子融合): {conventional_top_score.get(probe_date, 0.0):.4f}")
        print(f"      - 价格拉伸分: {stretch_score.get(probe_date, 0.0):.4f}")
        print(f"      - 均线混乱分: {misalignment_score.get(probe_date, 0.0):.4f}")
        print(f"      - 乖离过热分: {overheat_score.get(probe_date, 0.0):.4f} (原始BIAS: {bias_abs.get(probe_date, 0.0):.2%})")
        uranus_params = get_param_value(p_synthesis.get('uranus_ceiling_params'), {})
        from .utils import _calculate_uranus_ceiling_resistance, _calculate_historical_high_resistance
        uranus_ceiling_resistance_score_series = _calculate_uranus_ceiling_resistance(df, uranus_params)
        uranus_ceiling_resistance_score = uranus_ceiling_resistance_score_series.get(probe_date, 0.0)
        self._deploy_uranus_ceiling_probe(probe_date)
        p_fib_resistance = get_param_value(p_synthesis.get('fibonacci_resistance_params'), {})
        print("      --- [历史高点显微镜] 深入解剖 ---")
        historical_high_resistance_score_series = _calculate_historical_high_resistance(df, p_fib_resistance, uranus_params)
        final_historical_high_score = historical_high_resistance_score_series.get(probe_date, 0.0)
        if get_param_value(p_fib_resistance.get('enabled'), False):
            fib_periods = get_param_value(p_fib_resistance.get('periods'), [34, 55, 89, 144, 233])
            level_scores = get_param_value(p_fib_resistance.get('level_scores'), {})
            for period in fib_periods:
                period_str = str(period)
                if period_str not in level_scores: continue
                rolling_high_series = df['high_D'].rolling(window=period, min_periods=max(1, int(period*0.8))).max().shift(1)
                historical_high_val = rolling_high_series.get(probe_date)
                if pd.notna(historical_high_val):
                    historical_high_date = rolling_high_series.loc[:probe_date].idxmax()
                    print(f"\n        - {period}日周期: 找到前高 {historical_high_val:.2f} (日期: {historical_high_date.strftime('%Y-%m-%d')})")
                else:
                    print(f"\n        - {period}日周期: 未找到有效前高。")
                    continue
                temp_resistance_series = pd.Series(np.nan, index=df.index)
                temp_resistance_series.at[probe_date] = historical_high_val
                rejection_quality = self._deploy_apollo_arrow_probe(probe_date, uranus_params, temp_resistance_series)
                strategic_importance = level_scores[period_str]
                final_period_score = rejection_quality * strategic_importance
                print(f"        --- [融合裁决] ---")
                print(f"          - 战术质量分 (来自阿波罗之箭): {rejection_quality:.4f}")
                print(f"          - 战略重要性分 (来自配置): {strategic_importance:.2f}")
                print(f"          - ⚖️ 周期风险分 = 质量 * 重要性 = {rejection_quality:.4f} * {strategic_importance:.2f} = {final_period_score:.4f}")
        structural_resistance_score = np.maximum(uranus_ceiling_resistance_score, final_historical_high_score)
        print(f"\n    - [组件2] 结构性阻力得分: {structural_resistance_score:.4f}")
        print(f"      - 乌拉诺斯穹顶(均线)阻力分: {uranus_ceiling_resistance_score:.4f}")
        print(f"      - 历史高点(价格)阻力分: {final_historical_high_score:.4f}")
        final_top_context_score = np.maximum(conventional_top_score.get(probe_date, 0.0), structural_resistance_score)
        print(f"    - [最终裁决] 顶部上下文总分 (常规 vs 结构): {final_top_context_score:.4f}")
        print("\n--- “忒弥斯天平”称量完毕 ---")

    def _get_dominant_offense_type_for_probe(self, total_offense_score: float, active_offense: list) -> str:
        """
        【V1.1 · 记忆烙印协议版】为“宙斯之雷”探针专门提供的、用于模拟“最强进攻信号类型”判断的辅助方法。
        - 核心升级: 不再依赖脆弱的中文名反查，而是直接使用 `internal_name` 进行精准查找。
        """
        if total_offense_score <= 0 or not active_offense:
            return 'unknown'
        # 直接获取主导信号的内部名称
        dominant_signal_internal_name = active_offense[0].get('internal_name')
        if not dominant_signal_internal_name:
            return 'unknown'
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        # 使用内部名称进行精准、可靠的查找
        meta = score_map.get(dominant_signal_internal_name, {})
        return meta.get('type', 'unknown')

    def _deploy_uranus_ceiling_probe(self, probe_date: pd.Timestamp):
        """
        【V1.3 · 神盾协议探针版】“乌拉诺斯穹顶”法医探针
        - 核心升级: 调用通用的 _deploy_apollo_arrow_probe 探针来解剖拒绝质量。
        """
        print("\n      --- [乌拉诺斯显微镜] 深入解剖 ---")
        df = self.strategy.df_indicators
        strategy_instance_ref = self.strategy
        p_synthesis = get_params_block(strategy_instance_ref, 'ultimate_signal_synthesis_params', {})
        uranus_params = get_param_value(p_synthesis.get('uranus_ceiling_params'), {})
        if not get_param_value(uranus_params.get('enabled'), False):
            print("        - 乌拉诺斯穹顶系统在配置中被禁用。")
            return 0.0
        
        # [代码重构] 大部分参数获取移至阿波罗之箭探针，这里只保留确认压制所需的参数
        resistance_levels = get_param_value(uranus_params.get('resistance_levels'), [55, 89, 144, 233, 377])
        confirmation_window = get_param_value(uranus_params.get('confirmation_window'), 3)
        rejection_lookback_window = get_param_value(uranus_params.get('rejection_lookback_window'), 5)
        confirmation_cooldown_period = get_param_value(uranus_params.get('confirmation_cooldown_period'), 10)
        confirmation_score = get_param_value(uranus_params.get('confirmation_score'), 0.8)
        rejection_quality_bonus_factor = get_param_value(uranus_params.get('rejection_quality_bonus_factor'), 0.25)
        cooldown_reset_volume_ma_period = get_param_value(uranus_params.get('cooldown_reset_volume_ma_period'), 55)
        
        close_col, open_col, low_col, high_col, vol_col = 'close_D', 'open_D', 'low_D', 'high_D', 'volume_D'
        cooldown_vol_ma_col = f'VOL_MA_{cooldown_reset_volume_ma_period}_D'
        
        # 1. 寻找代理天花板
        ma_cols = [f'MA_{p}_D' for p in resistance_levels if f'MA_{p}_D' in df.columns]
        ma_df = df[ma_cols]
        ma_df_above_price = ma_df.where(ma_df.ge(df[close_col], axis=0))
        acting_ceiling = ma_df_above_price.min(axis=1).ffill()
        
        print(f"        - acting_ceiling (代理天花板): {acting_ceiling.get(probe_date, np.nan):.4f}")
        
        # 2. 调用通用的阿波罗之箭探针来获取拒绝质量分
        rejection_score_today = self._deploy_apollo_arrow_probe(probe_date, uranus_params, acting_ceiling)
        
        # 3. [代码重构] 确认压制评估逻辑保持不变，但需要重新计算 rejection_quality_score 序列
        from .utils import _calculate_rejection_quality_score
        rejection_quality_score = _calculate_rejection_quality_score(df, uranus_params, acting_ceiling)

        print("        --- [确认压制评估 (哈迪斯之锁)] ---")
        max_recent_rejection_quality = rejection_quality_score.rolling(window=rejection_lookback_window, min_periods=1).max()
        
        influence_zone_pct = get_param_value(uranus_params.get('influence_zone_pct'), 0.03)
        is_in_influence_zone = pd.Series(False, index=df.index)
        valid_indices = acting_ceiling.dropna().index
        if not valid_indices.empty:
            lower_bound = acting_ceiling[valid_indices] * (1 - influence_zone_pct)
            is_in_influence_zone.loc[valid_indices] = df.loc[valid_indices, close_col].between(lower_bound, acting_ceiling[valid_indices])

        is_failing_to_break = (df[close_col] < acting_ceiling) & is_in_influence_zone
        is_confirmed_rejection = is_failing_to_break.rolling(window=confirmation_window, min_periods=confirmation_window).sum() >= confirmation_window
        
        upper_shadow = df[high_col] - np.maximum(df[open_col], df[close_col])
        lower_shadow = np.minimum(df[open_col], df[close_col]) - df[low_col]
        is_cooldown_reset_signal = (lower_shadow > upper_shadow) & (df[vol_col] > df[cooldown_vol_ma_col])
        
        confirmation_score_series = pd.Series(0.0, index=df.index)
        last_confirmation_date = pd.NaT
        is_in_cooldown_on_probe_date = False
        for idx in df.index:
            if idx > probe_date: break
            if pd.notna(last_confirmation_date) and (idx - last_confirmation_date).days < confirmation_cooldown_period:
                if idx == probe_date: is_in_cooldown_on_probe_date = True
                if is_cooldown_reset_signal.get(idx, False): last_confirmation_date = pd.NaT
                continue
            if is_confirmed_rejection.get(idx, False):
                recent_quality = max_recent_rejection_quality.get(idx, 0.0)
                if recent_quality > 0:
                    rejection_score = confirmation_score + recent_quality * rejection_quality_bonus_factor
                    confirmation_score_series.loc[idx] = min(rejection_score, 1.0)
                else:
                    confirmation_score_series.loc[idx] = confirmation_score
                last_confirmation_date = idx
        
        is_confirmed_val = is_confirmed_rejection.get(probe_date, False)
        recent_quality_val = max_recent_rejection_quality.get(probe_date, 0.0)
        confirmation_score_today = confirmation_score_series.get(probe_date, 0.0)
        
        print(f"          - is_confirmed_rejection (是否满足压制天数): {is_confirmed_val}")
        print(f"          - is_in_cooldown (是否处于确认冷却期): {is_in_cooldown_on_probe_date}")
        print(f"          - 近期最高拒绝质量分 (lookback={rejection_lookback_window}d): {recent_quality_val:.4f}")
        if not is_in_cooldown_on_probe_date and is_confirmed_val:
            if recent_quality_val > 0:
                print(f"          - 审判类型: 强化压制 (基础分 {confirmation_score:.2f} + 质量奖励 {recent_quality_val:.2f} * {rejection_quality_bonus_factor:.2f})")
            else:
                print(f"          - 审判类型: 常规压制 (固定分 {confirmation_score:.2f})")
        else:
            print(f"          - 审判类型: 无 (is_confirmed={is_confirmed_val}, in_cooldown={is_in_cooldown_on_probe_date})")
        print(f"          - 当日独立确认分: {confirmation_score_today:.4f}")
        
        print("        --- [最终审判 (塔纳托斯之镰)] ---")
        final_score = max(rejection_score_today, confirmation_score_today)
        print(f"          - 裁决: max(独立拒绝分, 独立确认分) = max({rejection_score_today:.4f}, {confirmation_score_today:.4f}) = {final_score:.4f}")
        return final_score

    def _deploy_hades_gaze_probe(self, probe_date: pd.Timestamp, domain: str, signal_type: str):
        """
        【V1.1 · 赫淮斯托斯重铸版】“哈迪斯凝视”终极风险探针
        - 核心修复: 彻底修正了计算顺序的颠倒错误。探针现在严格遵循“先在周期内相乘，再跨周期融合”的
                      正确逻辑，确保重算结果与主引擎完全一致。
        - 核心职责: 钻透式解剖终极风险信号，揭示其从支柱健康度到最终分数的完整计算链路。
        - 调用示例: self._deploy_hades_gaze_probe(probe_date, 'CHIP', 'BEARISH_RESONANCE')
        """
        domain_upper = domain.upper()
        signal_name = f'SCORE_{domain_upper}_{signal_type}'
        print(f"\n--- [探针] 正在启用: 💀【哈迪斯凝视】💀 -> 解剖信号【{signal_name}】 ---")
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        def get_val(name, date, default=np.nan):
            series = atomic.get(name)
            if series is None: return default
            return series.get(date, default)
        # 链路层 1: 获取最终信号值
        final_score_raw = get_val(signal_name, probe_date, 0.0)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        signal_meta = score_map.get(signal_name, {})
        base_score = signal_meta.get('penalty_weight', signal_meta.get('score', 0))
        final_score_contribution = final_score_raw * base_score
        print(f"\n  [链路层 1] 最终风险贡献: {final_score_contribution:.0f}")
        print(f"    - [公式]: 原始信号值 * 基础分")
        print(f"    - [计算]: {final_score_raw:.4f} * {base_score} = {final_score_contribution:.2f}")
        # 链路层 2: 反推到中央合成引擎的输出
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        resonance_tf_weights = get_param_value(p_synthesis.get('resonance_tf_weights'), {})
        reversal_tf_weights = get_param_value(p_synthesis.get('reversal_tf_weights'), {})
        overall_health_cache = atomic.get(f'__{domain_upper}_overall_health', {})
        if not overall_health_cache:
            print("    - [探针错误] 无法找到领域健康度缓存。解剖终止。")
            return
        # 提取探针日的整体健康度
        s_bull_health = {p: v.get(probe_date, 0.5) for p, v in overall_health_cache.get('s_bull', {}).items()}
        s_bear_health = {p: v.get(probe_date, 0.5) for p, v in overall_health_cache.get('s_bear', {}).items()}
        d_intensity_health = {p: v.get(probe_date, 0.5) for p, v in overall_health_cache.get('d_intensity', {}).items()}
        print(f"\n  [链路层 2] 反推 -> 中央合成引擎 (transmute_health_to_ultimate_signals)")
        # 修正计算顺序：先在每个周期内相乘，再跨周期融合
        recalc_raw_score = 0.0
        if signal_type == 'BEARISH_RESONANCE':
            print(f"    - [公式]: 看跌共振 = Fuse(各周期s_bear * 各周期d_intensity)")
            period_scores = {p: s_bear_health.get(p, 0.5) * d_intensity_health.get(p, 0.5) for p in s_bear_health.keys()}
            print(f"    - [周期内计算]:")
            for p, score in period_scores.items():
                print(f"      - {p:<2}日周期: s_bear({s_bear_health.get(p, 0.5):.4f}) * d_intensity({d_intensity_health.get(p, 0.5):.4f}) = {score:.4f}")
            short_force = (period_scores.get(1, 0.5) * period_scores.get(5, 0.5))**0.5
            medium_trend = (period_scores.get(13, 0.5) * period_scores.get(21, 0.5))**0.5
            long_inertia = period_scores.get(55, 0.5)
            recalc_raw_score = (short_force**resonance_tf_weights.get('short', 0.2) * 
                                medium_trend**resonance_tf_weights.get('medium', 0.5) * 
                                long_inertia**resonance_tf_weights.get('long', 0.3))
            print(f"    - [跨周期融合计算]: Fuse(...) = {recalc_raw_score:.4f}")
        elif signal_type == 'TOP_REVERSAL':
            print(f"    - [公式]: 顶部反转 = Fuse(各周期s_bear * (1 - 各周期s_bull))")
            period_scores = {p: s_bear_health.get(p, 0.5) * (1.0 - s_bull_health.get(p, 0.5)) for p in s_bear_health.keys()}
            print(f"    - [周期内计算]:")
            for p, score in period_scores.items():
                print(f"      - {p:<2}日周期: s_bear({s_bear_health.get(p, 0.5):.4f}) * (1 - s_bull({s_bull_health.get(p, 0.5):.4f})) = {score:.4f}")
            short_force = (period_scores.get(1, 0.5) * period_scores.get(5, 0.5))**0.5
            medium_trend = (period_scores.get(13, 0.5) * period_scores.get(21, 0.5))**0.5
            long_inertia = period_scores.get(55, 0.5)
            recalc_raw_score = (short_force**reversal_tf_weights.get('short', 0.6) * 
                                medium_trend**reversal_tf_weights.get('medium', 0.3) * 
                                long_inertia**reversal_tf_weights.get('long', 0.1))
            print(f"    - [跨周期融合计算]: Fuse(...) = {recalc_raw_score:.4f}")
        else:
            print(f"    - [探针警告] 不支持的风险信号类型: {signal_type}")
            return
        print(f"    - [对比]: 实际原始值 {final_score_raw:.4f} vs 重算原始值 {recalc_raw_score:.4f}")
        # 链路层3和4保持不变，但现在其分析更有意义
        print(f"\n  [链路层 3] 钻透 -> 整体三维健康度来源 (融合后的参考值)")
        overall_bullish_health = ( ( (s_bull_health.get(1,0.5)*s_bull_health.get(5,0.5))**0.5 )**resonance_tf_weights.get('short',0.2) * ( (s_bull_health.get(13,0.5)*s_bull_health.get(21,0.5))**0.5 )**resonance_tf_weights.get('medium',0.5) * (s_bull_health.get(55,0.5))**resonance_tf_weights.get('long',0.3) )
        overall_bearish_health = ( ( (s_bear_health.get(1,0.5)*s_bear_health.get(5,0.5))**0.5 )**resonance_tf_weights.get('short',0.2) * ( (s_bear_health.get(13,0.5)*s_bear_health.get(21,0.5))**0.5 )**resonance_tf_weights.get('medium',0.5) * (s_bear_health.get(55,0.5))**resonance_tf_weights.get('long',0.3) )
        overall_dynamic_intensity = ( ( (d_intensity_health.get(1,0.5)*d_intensity_health.get(5,0.5))**0.5 )**resonance_tf_weights.get('short',0.2) * ( (d_intensity_health.get(13,0.5)*d_intensity_health.get(21,0.5))**0.5 )**resonance_tf_weights.get('medium',0.5) * (d_intensity_health.get(55,0.5))**resonance_tf_weights.get('long',0.3) )
        print(f"    - 整体看涨健康度 (s_bull): {overall_bullish_health:.4f}")
        print(f"    - 整体看跌健康度 (s_bear): {overall_bearish_health:.4f}  <-- 风险的主要来源之一")
        print(f"    - 整体动态强度 (d_intensity): {overall_dynamic_intensity:.4f}  <-- 风险的主要来源之一")
        print(f"\n  [链路层 4] 终极解剖 -> 各支柱健康度贡献 (以5日周期为例)")
        pillar_configs = {
            'CHIP': ['quantitative', 'advanced', 'internal', 'holder', 'fault'],
            'FUND_FLOW': ['consensus', 'conviction', 'conflict', 'sentiment'],
            'DYN': ['volatility', 'efficiency', 'momentum', 'inertia'],
            'STRUCTURE': ['ma', 'mechanics', 'mtf', 'pattern'],
            'BEHAVIOR': ['price', 'volume', 'kline'],
            'FOUNDATION': ['ema', 'rsi', 'macd', 'cmf']
        }
        pillars = pillar_configs.get(domain_upper, [])
        if not pillars:
            print(f"    - [探针警告] 未找到领域 {domain_upper} 的支柱配置。")
            return
        print(f"    {'Pillar':<15} | {'s_bull':<10} | {'s_bear':<10} | {'d_intensity':<10}")
        print(f"    {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10}")
        for pillar_name in pillars:
            pillar_health = atomic.get(f'_PILLAR_HEALTH_{domain_upper}_{pillar_name}')
            if not pillar_health:
                print(f"    {pillar_name:<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
                continue
            s_b = pillar_health.get('s_bull', {}).get(5, pd.Series(0.5)).get(probe_date, 0.5)
            s_br = pillar_health.get('s_bear', {}).get(5, pd.Series(0.5)).get(probe_date, 0.5)
            d_i = pillar_health.get('d_intensity', {}).get(5, pd.Series(0.5)).get(probe_date, 0.5)
            print(f"    {pillar_name:<15} | {s_b:<10.4f} | {s_br:<10.4f} | {d_i:<10.4f}")
        print("\n--- “哈迪斯凝视”解剖完毕 ---")

    def _deploy_apollo_arrow_probe(self, probe_date: pd.Timestamp, params: Dict, resistance_line: pd.Series) -> float:
        """
        【V1.3 · 塔纳托斯之镰探针版】通用拒绝质量评估探针 (阿波罗之箭)
        - 核心升级: 完全同步“伊卡洛斯之陨”的风险叠加逻辑，并清晰打印奖励分。
        """
        print("        --- [阿波罗之箭评估] ---")
        df = self.strategy.df_indicators
        # 从参数中获取所有必要的配置
        influence_zone_pct = get_param_value(params.get('influence_zone_pct'), 0.03)
        rejection_base_score = get_param_value(params.get('rejection_base_score'), 0.4)
        rejection_yin_line_weight = get_param_value(params.get('rejection_yin_line_weight'), 0.1)
        rejection_dominance_weight = get_param_value(params.get('rejection_dominance_weight'), 0.2)
        rejection_volume_weight = get_param_value(params.get('rejection_volume_weight'), 0.3)
        min_shadow_ratio = get_param_value(params.get('min_shadow_ratio'), 0.15) # 已按您的要求修改
        # 废除icarus_fall_base_score，引入icarus_fall_bonus
        icarus_fall_bonus = get_param_value(params.get('icarus_fall_bonus'), 0.5)
        cooldown_reset_volume_ma_period = get_param_value(params.get('cooldown_reset_volume_ma_period'), 55)
        close_col, open_col, low_col, high_col, vol_col = 'close_D', 'open_D', 'low_D', 'high_D', 'volume_D'
        ares_vol_ma_col = 'VOL_MA_5_D'
        # 检查必需列
        required_cols = [close_col, open_col, low_col, high_col, vol_col, ares_vol_ma_col, 'up_limit_D']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 1. 获取当日关键数据
        res_val = resistance_line.get(probe_date, np.nan)
        if pd.isna(res_val):
            print("          - 目标阻力位当日无有效值，评估跳过。")
            return 0.0
        # 2. 计算影响区和基础条件
        lower_bound_val = res_val * (1 - influence_zone_pct)
        is_in_influence_zone_val = lower_bound_val <= df.at[probe_date, close_col] <= res_val
        base_rejection_condition_val = (df.at[probe_date, high_col] > res_val) & is_in_influence_zone_val & (df.at[probe_date, close_col] < df.at[probe_date, high_col])
        # 3. 计算各项质量加权分
        rejection_quality_score_val = 0.0
        if base_rejection_condition_val:
            rejection_quality_score_val = rejection_base_score
        is_yin_line_val = df.at[probe_date, close_col] < df.at[probe_date, open_col]
        upper_shadow_val = df.at[probe_date, high_col] - max(df.at[probe_date, open_col], df.at[probe_date, close_col])
        lower_shadow_val = min(df.at[probe_date, open_col], df.at[probe_date, close_col]) - df.at[probe_date, low_col]
        kline_range_val = (df.at[probe_date, high_col] - df.at[probe_date, low_col])
        upper_shadow_ratio_val = upper_shadow_val / kline_range_val if kline_range_val > 0 else 0
        is_upper_shadow_significant_val = upper_shadow_ratio_val > min_shadow_ratio
        has_dominance_val = (upper_shadow_val > lower_shadow_val) & is_upper_shadow_significant_val
        yin_line_bonus = rejection_yin_line_weight if base_rejection_condition_val and is_yin_line_val else 0.0
        dominance_bonus = rejection_dominance_weight if base_rejection_condition_val and has_dominance_val else 0.0
        has_volume_spike_val = df.at[probe_date, vol_col] > df.at[probe_date, ares_vol_ma_col]
        proportional_volume_score_val = normalize_score(df[vol_col] / df[ares_vol_ma_col].replace(0, np.nan), df.index, window=cooldown_reset_volume_ma_period, ascending=True).get(probe_date, 0.0)
        volume_bonus = rejection_volume_weight * proportional_volume_score_val if base_rejection_condition_val and has_dominance_val and has_volume_spike_val else 0.0
        rejection_quality_score_val += yin_line_bonus + dominance_bonus + volume_bonus
        # 4. 应用绝对否决/奖励规则
        limit_up_price_val = df.at[probe_date, 'up_limit_D']
        is_icarus_fall_val = (df.at[probe_date, high_col] >= limit_up_price_val * 0.995) & (df.at[probe_date, close_col] < df.at[probe_date, high_col] * 0.98)
        # 将伊卡洛斯之陨的逻辑从“取最大值”改为“叠加奖励分”
        icarus_bonus_val = icarus_fall_bonus if is_icarus_fall_val else 0.0
        rejection_quality_score_val += icarus_bonus_val
        is_apollo_absorption_val = (lower_shadow_val > upper_shadow_val) & has_volume_spike_val
        if is_in_influence_zone_val and is_apollo_absorption_val:
            rejection_quality_score_val = 0.0
        final_score = np.clip(rejection_quality_score_val, 0, 1.0)
        # 5. 打印详细解剖过程
        print(f"          - 目标阻力线: {res_val:.2f}")
        print(f"          - 影响区: [{lower_bound_val:.2f}, {res_val:.2f}] -> 收盘价({df.at[probe_date, close_col]:.2f})是否在内: {'✅' if is_in_influence_zone_val else '❌'}")
        print(f"          - 基础条件 (触顶+回落): {base_rejection_condition_val} -> 基础分 {rejection_base_score if base_rejection_condition_val else 0.0:.2f}")
        print(f"          - 权重1 (空头宣告-收阴): {is_yin_line_val} -> 加分 {yin_line_bonus:.2f}")
        print(f"          - 影线长度: 上影 {upper_shadow_val:.2f} vs 下影 {lower_shadow_val:.2f}")
        print(f"          - 上影线显著性检查: (上影率 {upper_shadow_ratio_val:.2f} > 阈值 {min_shadow_ratio:.2f}) -> {is_upper_shadow_significant_val}")
        print(f"          - 权重2 (空头胜利-上影优势): (上影>下影 AND 上影显著) -> {has_dominance_val} -> 加分 {dominance_bonus:.2f}")
        print(f"          - 权重3 (主力派发-放量): {has_volume_spike_val and has_dominance_val} -> 加分 {volume_bonus:.2f}")
        print(f"          ---")
        # 修改打印说明，反映叠加逻辑
        print(f"          - 💀 塔纳托斯之镰 (涨停回落): {is_icarus_fall_val} -> 额外奖励分 {icarus_bonus_val:.2f}")
        print(f"          - ☀️ 阿波罗吸收 (多头反噬): {is_apollo_absorption_val and is_in_influence_zone_val} -> 若触发，分数强制归零")
        print(f"          - 最终裁决 (战术质量分): {final_score:.4f}")
        return final_score

    def _deploy_archangel_diagnosis_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 权限修复版】“天使长诊断探针” - 四骑士审查
        - 核心修复: 在调用 calculate_context_scores 之前，为其注入必需的 'strategy_instance_ref' 上下文引用，
                      解决因缺少权限导致计算结果为0的致命BUG。
        """
        print("\n--- [探针] 正在启用: 👼【天使长诊断探针 · 四骑士审查】👼 ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        # 步骤一：获取“第四骑士” - 结构性压力分
        # 注入计算所需的上下文引用
        atomic['strategy_instance_ref'] = self.strategy
        _, top_context_score_series = calculate_context_scores(df, atomic)
        # 计算完毕后，立即移除临时引用，保持状态纯净
        del atomic['strategy_instance_ref']
        top_context_score = top_context_score_series.get(probe_date, 0.0)
        # 步骤二：获取其他三位骑士的信号值
        upthrust_risk = atomic.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        heaven_earth_risk = atomic.get('SCORE_BOARD_HEAVEN_EARTH', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        post_peak_risk = atomic.get('COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN', pd.Series(0.0, index=df.index)).get(probe_date, 0.0)
        print("  --- [输入审查] 启示录四骑士信号值 ---")
        print(f"    - 骑士1 (上冲派发): {upthrust_risk:.4f}")
        print(f"    - 骑士2 (天地板): {heaven_earth_risk:.4f}")
        print(f"    - 骑士3 (高位回落): {post_peak_risk:.4f}")
        print(f"    - 骑士4 (结构性压力): {top_context_score:.4f}  <-- 来自“忒弥斯天平”")
        # 步骤三：复刻融合逻辑并提供证据
        risk_values = [upthrust_risk, heaven_earth_risk, post_peak_risk, top_context_score]
        archangel_score = max(risk_values)
        print("\n  --- [融合裁决] ---")
        print(f"    - 融合算法: max(骑士1, 骑士2, 骑士3, 骑士4)")
        print(f"    - 计算过程: max({upthrust_risk:.4f}, {heaven_earth_risk:.4f}, {post_peak_risk:.4f}, {top_context_score:.4f}) = {archangel_score:.4f}")
        print(f"    - 最终结论 (SCORE_ARCHANGEL_TOP_REVERSAL 的真实值): {archangel_score:.4f}")
        print("--- “天使长诊断探针”运行完毕 ---")

    def _deploy_athena_wisdom_probe(self, probe_date: pd.Timestamp):
        """
        【V3.0 · 普罗米修斯之火同步版】“雅典娜智慧”探针
        - 核心革命: 探针的重算逻辑已与主引擎的“普罗米修斯之火”协议完全同步。
        - 新核心逻辑: 探针内部完美复刻“加权几何平均 + 关系元分析”的两阶段认知过程。
        - 收益: 彻底解决了探针与主引擎逻辑脱节导致的巨大验证偏差，恢复了探针的诊断能力。
        """
        print("\n--- [探针] 正在启用: 🦉【雅典娜智慧 · 终极底部确认解剖 V3.0】🦉 ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        def get_val(name, date, default=0.0):
            series = atomic.get(name)
            if series is None:
                print(f"      - [警告] 探针无法在 atomic_states 中找到信号: {name}")
                return default
            return series.get(date, default)
        final_score = get_val('COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION', probe_date)
        print(f"\n  [链路层 1] 最终确认成品: COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION = {final_score:.4f}")
        print(f"    - [核心公式]: 终极确认分 = 原始终极底部确认分 * 底部上下文分数")
        print("\n  [链路层 2] 解剖 -> 原始终极底部确认分 (ultimate_bottom_raw)")
        print(f"    - [核心公式]: 原始分 = 认知融合底部反转分 * 形态底部反转分")
        fusion_bottom_val = get_val('COGNITIVE_FUSION_BOTTOM_REVERSAL', probe_date)
        print(f"\n    --- [组件 A] 认知融合底部反转分 (COGNITIVE_FUSION_BOTTOM_REVERSAL): {fusion_bottom_val:.4f} ---")
        # 部署与主引擎完全同步的“普罗米修斯之火”重算逻辑
        print(f"      - [核心公式]: MetaAnalysis(GeometricMean(基础, 结构, 行为))")
        print("\n        --- [组件A显微镜 · 普罗米修斯之火重算] ---")
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        fusion_weights_conf = get_param_value(p_cognitive.get('cognitive_fusion_weights'), {})
        foundation_bottom = get_unified_score(atomic, df.index, 'FOUNDATION_BOTTOM_REVERSAL')
        structure_bottom = get_unified_score(atomic, df.index, 'STRUCTURE_BOTTOM_REVERSAL')
        behavior_bottom = get_unified_score(atomic, df.index, 'BEHAVIOR_BOTTOM_REVERSAL')
        print(f"        - 输入1: 基础层反转分 = {foundation_bottom.get(probe_date, 0.0):.4f}")
        print(f"        - 输入2: 结构层反转分 = {structure_bottom.get(probe_date, 0.0):.4f}")
        print(f"        - 输入3: 行为层反转分 = {behavior_bottom.get(probe_date, 0.0):.4f}")
        print("\n        --- [阶段一: 奥林匹斯众神殿 · 共识快照] ---")
        scores_to_fuse = [foundation_bottom.values, structure_bottom.values, behavior_bottom.values]
        weights_to_fuse = [
            fusion_weights_conf.get('foundation', 0.33),
            fusion_weights_conf.get('structure', 0.33),
            fusion_weights_conf.get('behavior', 0.34)
        ]
        weights_array = np.array(weights_to_fuse)
        weights_array /= weights_array.sum()
        stacked_scores = np.stack(scores_to_fuse, axis=0)
        safe_scores = np.maximum(stacked_scores, 1e-9)
        log_signals = np.log(safe_scores)
        weighted_log_sum = np.sum(log_signals * weights_array[:, np.newaxis], axis=0)
        consensus_snapshot_score = pd.Series(np.exp(weighted_log_sum), index=df.index, dtype=np.float32)
        print(f"          - [探针重算] 共识快照分 (GeometricMean) @ {probe_date.date()}: {consensus_snapshot_score.get(probe_date, 0.0):.4f}")
        print("\n        --- [阶段二: 普罗米修斯之火 · 动态锻造] ---")
        p_meta = get_param_value(p_cognitive.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        meta_window = 5
        norm_window = 55
        state_score_val = consensus_snapshot_score.clip(0, 1).get(probe_date, 0.0)
        relationship_trend = consensus_snapshot_score.diff(meta_window).fillna(0)
        velocity_score_series = normalize_to_bipolar(series=relationship_trend, target_index=df.index, window=norm_window, sensitivity=1.0)
        velocity_score_val = velocity_score_series.get(probe_date, 0.0)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score_series = normalize_to_bipolar(series=relationship_accel, target_index=df.index, window=norm_window, sensitivity=1.0)
        acceleration_score_val = acceleration_score_series.get(probe_date, 0.0)
        print(f"          - 状态分 (State): {state_score_val:.4f}")
        print(f"          - 速度分 (Velocity): {velocity_score_val:.4f}")
        print(f"          - 加速度分 (Acceleration): {acceleration_score_val:.4f}")
        fusion_bottom_recalc = (state_score_val * w_state + velocity_score_val * w_velocity + acceleration_score_val * w_acceleration).clip(0, 1)
        print(f"          - [探针重算] 最终融合分 = ({state_score_val:.2f}*{w_state} + {velocity_score_val:.2f}*{w_velocity} + {acceleration_score_val:.2f}*{w_acceleration}) = {fusion_bottom_recalc:.4f}")
        print(f"          - [对比]: 实际值 {fusion_bottom_val:.4f} vs 重算值 {fusion_bottom_recalc:.4f}")

        pattern_bottom_val = get_val('SCORE_PATTERN_BOTTOM_REVERSAL', probe_date)
        print(f"\n    --- [组件 B] 形态底部反转分 (SCORE_PATTERN_BOTTOM_REVERSAL): {pattern_bottom_val:.4f} ---")
        print(f"      - [核心公式]: max(RSI反转, 平台突破, MACD金叉, 动能衰竭)")
        print("\n        --- [组件B显微镜] ---")
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        macd_hist = df.get('MACDh_13_34_8_D', pd.Series(0, index=df.index))
        was_oversold = (rsi.rolling(window=5, min_periods=1).min() < 35)
        is_recovering = (df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)) > 0)
        score_rsi_reversal = (was_oversold & is_recovering).astype(float).get(probe_date, 0.0)
        is_breaking_consolidation = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        score_consolidation_breakout = (is_breaking_consolidation * 0.8).get(probe_date, 0.0)
        is_macd_bull_cross = ((macd_hist > 0) & (macd_hist.shift(1) <= 0)).astype(float)
        score_macd_bullish_cross = is_macd_bull_cross.get(probe_date, 0.0)
        rsi_slope_abs = df.get('SLOPE_1_RSI_13_D', pd.Series(0, index=df.index)).abs()
        macd_hist_slope_abs = df.get('SLOPE_1_MACDh_13_34_8_D', pd.Series(0, index=df.index)).abs()
        rsi_exhaustion_score = normalize_score(rsi_slope_abs, df.index, window=60, ascending=False)
        macd_exhaustion_score = normalize_score(macd_hist_slope_abs, df.index, window=60, ascending=False)
        score_momentum_exhaustion = ((rsi_exhaustion_score * macd_exhaustion_score)**0.5).get(probe_date, 0.0)
        print(f"        - 模式1: RSI反转分: {score_rsi_reversal:.4f}")
        print(f"        - 模式2: 平台突破分: {score_consolidation_breakout:.4f}")
        print(f"        - 模式3: MACD金叉分: {score_macd_bullish_cross:.4f}")
        print(f"        - 模式4: 动能衰竭分: {score_momentum_exhaustion:.4f}")
        pattern_bottom_recalc = max(score_rsi_reversal, score_consolidation_breakout, score_macd_bullish_cross, score_momentum_exhaustion)
        print(f"        - [探针重算] 形态分 = max(...) = {pattern_bottom_recalc:.4f}")
        print("\n    --- [调节器] 底部上下文分数 (bottom_context_score) ---")
        atomic['strategy_instance_ref'] = self.strategy
        bottom_context_score_series, _ = calculate_context_scores(df, atomic)
        del atomic['strategy_instance_ref']
        bottom_context_score = bottom_context_score_series.get(probe_date, 0.0)
        print(f"      - [探针获取] 底部上下文分数: {bottom_context_score:.4f} (详情请见“忒弥斯天平”探针)")
        print("\n  [最终验证]")
        ultimate_bottom_raw_recalc = fusion_bottom_recalc * pattern_bottom_recalc
        print(f"    - [探针重算] 原始终极底部确认分 = {fusion_bottom_recalc:.4f} * {pattern_bottom_recalc:.4f} = {ultimate_bottom_raw_recalc:.4f}")
        final_score_recalc = ultimate_bottom_raw_recalc * bottom_context_score
        print(f"    - [探针重算] 终极确认分 = {ultimate_bottom_raw_recalc:.4f} * {bottom_context_score:.4f} = {final_score_recalc:.4f}")
        print(f"    - [对比]: 实际值 {final_score:.4f} vs 重算值 {final_score_recalc:.4f}")
        print("--- “雅典娜智慧”探针解剖完毕 ---")

    def _deploy_hermes_caduceus_probe(self, probe_date: pd.Timestamp):
        """
        【V1.2 · 终极解剖版】“商神杖探针” - 权力转移风险解剖
        - 核心升级: 在打印斜率的同时，增加打印当天的指标原始值，提供完整的诊断证据链。
        - 收益: 彻底阐明了风险信号的源头数据，使诊断报告的透明度和可信度达到极致。
        """
        print("\n--- [探针] 正在启用: ☤【商神杖探针 V1.2 · 权力转移风险解剖】☤ ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel.micro_behavior_engine
        def get_val(series, date, default=0.0):
            if series is None: return default
            return series.get(date, default)
        final_score = get_val(atomic.get('COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'), probe_date)
        print(f"\n  [链路层 1] 最终信号值: COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL = {final_score:.4f}")
        print("    - [解读]: 这是经过动态锻造和上下文抑制后的最终风险分。")
        suppression_context = get_val(atomic.get('SCORE_CONTEXT_RECENT_REVERSAL'), probe_date)
        suppression_factor = 1.0 - suppression_context
        dynamic_score = final_score / suppression_factor if suppression_factor > 0 else 0.0
        print(f"\n  [链路层 2] 解剖 -> 上下文抑制力场")
        print(f"    - [公式]: 最终分 = 动态锻造分 * (1 - 近期反转上下文)")
        print(f"    - 近期反转上下文 (SCORE_CONTEXT_RECENT_REVERSAL): {suppression_context:.4f}")
        print(f"    - 抑制因子 (Suppression Factor): 1.0 - {suppression_context:.4f} = {suppression_factor:.4f}")
        print(f"    - [反推] 动态锻造分 = {final_score:.4f} / {suppression_factor:.4f} = {dynamic_score:.4f}")
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        print(f"\n  [链路层 3] 解剖 -> 动态锻造 (关系元分析)")
        print(f"    - [公式]: (状态分 * {w_state}) + (速度分 * {w_velocity}) + (加速度分 * {w_acceleration})")
        print(f"    - [解读]: 捕捉风险的“变化趋势”，即使当前风险值不高，但只要增长迅猛，也会得到高分。")
        norm_window = 120
        ma_health_score = engine._calculate_ma_health(df, p_conf, 55)
        granularity_momentum_down = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=False)
        dominance_momentum_down = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=False)
        _, granularity_holo_down = calculate_holographic_dynamics(df, 'avg_order_value', norm_window)
        _, dominance_holo_down = calculate_holographic_dynamics(df, 'trade_concentration_index', norm_window)
        power_shift_to_retail_risk_raw = (granularity_momentum_down * granularity_holo_down * dominance_momentum_down * dominance_holo_down)
        snapshot_power_shift_risk = power_shift_to_retail_risk_raw * (1 - ma_health_score)
        print(f"\n  [链路层 4] 解剖 -> 关系快照分 (Snapshot Score)")
        print(f"    - [公式]: 快照分 = 原始风险分 * (1 - 均线健康度)")
        print(f"    - [解读]: 权力向散户转移的风险，在均线结构恶化时 (健康度低) 最为危险。")
        print(f"\n  [链路层 5] 解剖 -> 原始风险分 & 均线健康度 @ {probe_date.date()}")
        print(f"    - [公式]: 原始风险分 = GranularityMomentum * GranularityHolo * DominanceMomentum * DominanceHolo")
        print(f"    ---")
        print(f"    - 均线健康度 (ma_health_score): {get_val(ma_health_score, probe_date):.4f}")
        print(f"    - 原始风险分:")
        gmd_val = get_val(granularity_momentum_down, probe_date)
        ghd_val = get_val(granularity_holo_down, probe_date)
        dmd_val = get_val(dominance_momentum_down, probe_date)
        dhd_val = get_val(dominance_holo_down, probe_date)
        # 获取原始指标值
        raw_avg_order_val = get_val(df.get('avg_order_value_D'), probe_date)
        raw_trade_conc_val = get_val(df.get('trade_concentration_index_D'), probe_date)
        # 修改打印行以包含原始值
        print(f"      - Granularity Momentum (Down): {gmd_val:.4f} (原始斜率: {get_val(df.get('SLOPE_5_avg_order_value_D'), probe_date):.2f}, 当日数值: {raw_avg_order_val:,.2f}) <-- 关键证据: 订单散户化")
        print(f"      - Granularity Holo (Down)    : {ghd_val:.4f}")
        print(f"      - Dominance Momentum (Down)  : {dmd_val:.4f} (原始斜率: {get_val(df.get('SLOPE_5_trade_concentration_index_D'), probe_date):.2f}, 当日数值: {raw_trade_conc_val:.4f}) <-- 关键证据: 交易分散化")
        # 新增/修改结束
        print(f"      - Dominance Holo (Down)      : {dhd_val:.4f}")
        print(f"    - [重算] 原始风险分 = {gmd_val:.4f} * {ghd_val:.4f} * {dmd_val:.4f} * {dhd_val:.4f} = {get_val(power_shift_to_retail_risk_raw, probe_date):.4f}")
        print(f"    - [重算] 关系快照分 = {get_val(power_shift_to_retail_risk_raw, probe_date):.4f} * (1 - {get_val(ma_health_score, probe_date):.4f}) = {get_val(snapshot_power_shift_risk, probe_date):.4f}")
        state_score = snapshot_power_shift_risk.clip(0, 1)
        relationship_trend = snapshot_power_shift_risk.diff(5).fillna(0)
        velocity_score = normalize_to_bipolar(series=relationship_trend, target_index=df.index, window=55, sensitivity=1.0)
        relationship_accel = relationship_trend.diff(5).fillna(0)
        acceleration_score = normalize_to_bipolar(series=relationship_accel, target_index=df.index, window=55, sensitivity=1.0)
        final_dynamic_risk_recalc = (get_val(state_score, probe_date) * w_state + get_val(velocity_score, probe_date) * w_velocity + get_val(acceleration_score, probe_date) * w_acceleration).clip(0, 1)
        print(f"\n  [链路层 6] 动态锻造重算")
        print(f"    - 状态分 (State)      : {get_val(state_score, probe_date):.4f}  <-- 当前风险水平")
        print(f"    - 速度分 (Velocity)   : {get_val(velocity_score, probe_date):.4f}  <-- 风险增长速度")
        print(f"    - 加速度分 (Acceleration): {get_val(acceleration_score, probe_date):.4f}  <-- 风险增长加速度")
        print(f"    - [重算] 动态锻造分 = {final_dynamic_risk_recalc:.4f} <-- 趋势放大风险!")
        final_score_recalc = final_dynamic_risk_recalc * suppression_factor
        print(f"\n  [最终验证]")
        print(f"    - [探针重算] 最终风险分 = {final_dynamic_risk_recalc:.4f} (动态锻造分) * {suppression_factor:.4f} (抑制因子) = {final_score_recalc:.4f}")
        print(f"    - [对比]: 实际值 {final_score:.4f} vs 重算值 {final_score_recalc:.4f}")
        print("--- “商神杖探针”解剖完毕 ---")

    def _deploy_hermes_verdict_probe(self, probe_date: pd.Timestamp):
        """
        【V1.0 · 新增】“赫尔墨斯裁决探针” - 微观行为对质
        - 核心职责: 并行解剖“伪装吸筹(看涨)”和“权力转移(看跌)”两个相互矛盾的微观信号。
        - 收益: 揭示系统如何在两个相似但意图相反的行为模式之间进行博弈和裁决，展示其认知深度。
        """
        print("\n--- [探针] 正在启用: ⚖️【赫尔墨斯裁决探针 · 微观行为对质】⚖️ ---")
        df = self.strategy.df_indicators
        engine = self.cognitive_intel.micro_behavior_engine
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        norm_window = 120
        def get_val(series, date, default=0.0):
            if series is None: return default
            return series.get(date, default)
        def run_dissection(signal_type: str):
            if signal_type == 'bullish':
                print("\n  --- 解剖【看涨】伪装散户吸筹 (SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION) ---")
                p = get_params_block(self.strategy, 'deceptive_flow_params', {})
                retail_inflow_score = get_unified_score(self.strategy.atomic_states, df.index, 'FF_BEARISH_RESONANCE')
                chip_concentration_score = normalize_score(df.get('SLOPE_5_concentration_90pct_D'), df.index, norm_window, ascending=False)
                price_suppression_score = normalize_score(df.get('SLOPE_5_close_D').abs(), df.index, norm_window, ascending=False)
                vpa_inefficiency_score = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False)
                raw_score = (retail_inflow_score * chip_concentration_score * price_suppression_score * vpa_inefficiency_score)
                ma_health_score = engine._calculate_ma_health(df, p_conf, 55)
                snapshot_score = raw_score * (1 - ma_health_score)
                print(f"    - [原始分]: (散户流出 * 筹码集中 * 价格压制 * VPA低效) = {get_val(raw_score, probe_date):.4f}")
                print(f"    - [上下文]: (1 - 均线健康度 {get_val(ma_health_score, probe_date):.4f}) = {1-get_val(ma_health_score, probe_date):.4f}")
                print(f"    - [快照分]: {get_val(raw_score, probe_date):.4f} * {1-get_val(ma_health_score, probe_date):.4f} = {get_val(snapshot_score, probe_date):.4f}")
            elif signal_type == 'bearish':
                print("\n  --- 解剖【看跌】权力转移风险 (COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL) ---")
                granularity_momentum_down = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=False)
                dominance_momentum_down = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=False)
                _, granularity_holo_down = calculate_holographic_dynamics(df, 'avg_order_value', norm_window)
                _, dominance_holo_down = calculate_holographic_dynamics(df, 'trade_concentration_index', norm_window)
                raw_score = (granularity_momentum_down * granularity_holo_down * dominance_momentum_down * dominance_holo_down)
                ma_health_score = engine._calculate_ma_health(df, p_conf, 55)
                snapshot_score = raw_score * (1 - ma_health_score)
                print(f"    - [原始分]: (订单散户化 * 交易分散化) = {get_val(raw_score, probe_date):.4f}")
                print(f"    - [上下文]: (1 - 均线健康度 {get_val(ma_health_score, probe_date):.4f}) = {1-get_val(ma_health_score, probe_date):.4f}")
                print(f"    - [快照分]: {get_val(raw_score, probe_date):.4f} * {1-get_val(ma_health_score, probe_date):.4f} = {get_val(snapshot_score, probe_date):.4f}")
            else:
                return
            final_score = engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
            state_score = get_val(snapshot_score.clip(0, 1), probe_date)
            relationship_trend = snapshot_score.diff(5).fillna(0)
            velocity_score = get_val(normalize_to_bipolar(series=relationship_trend, target_index=df.index, window=55), probe_date)
            relationship_accel = relationship_trend.diff(5).fillna(0)
            acceleration_score = get_val(normalize_to_bipolar(series=relationship_accel, target_index=df.index, window=55), probe_date)
            print(f"    - [动态锻造]:")
            print(f"      - 状态分 (State)      : {state_score:.4f}")
            print(f"      - 速度分 (Velocity)   : {velocity_score:.4f}")
            print(f"      - 加速度分 (Acceleration): {acceleration_score:.4f}")
            print(f"    - [最终动态分]: {get_val(final_score, probe_date):.4f}")
        run_dissection('bullish')
        run_dissection('bearish')
        print("\n--- “赫尔墨斯裁决探针”运行完毕 ---")

    def _deploy_ares_spear_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 宙斯之雷协议版】“阿瑞斯之矛探针” - 主力信念瓦解解剖
        - 核心修正: 部署“宙斯之雷协议”，修正了原始风险分的计算逻辑。
          - 错误逻辑: (-score).clip(0)
          - 正确逻辑: 1 - score
        - 收益: 确保探针能够正确复现“主力信念瓦解”风险的产生过程，使其恢复神力。
        """
        print("\n--- [探针] 正在启用: ⚔️【阿瑞斯之矛探针 V1.1 · 宙斯之雷协议版】⚔️ ---")
        df = self.strategy.df_indicators
        engine = self.cognitive_intel.micro_behavior_engine
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        norm_window = 120
        def get_val(series, date, default=0.0):
            if series is None: return default
            return series.get(date, default)
        print("\n  --- 解剖【看跌】主力信念瓦解风险 ---")
        granularity_momentum_up = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=True)
        dominance_momentum_up = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=True)
        _, granularity_holo_up = calculate_holographic_dynamics(df, 'avg_order_value', norm_window)
        _, dominance_holo_up = calculate_holographic_dynamics(df, 'trade_concentration_index', norm_window)
        # 修正原始风险分的计算公式
        risk_from_granularity = (1 - granularity_momentum_up).clip(0)
        risk_from_dominance = (1 - dominance_momentum_up).clip(0)
        raw_score = (risk_from_granularity * granularity_holo_up * risk_from_dominance * dominance_holo_up)
        
        ma_health_score = engine._calculate_ma_health(df, p_conf, 55)
        snapshot_score = raw_score * (1 - ma_health_score)
        print(f"    - [原始分]: (1 - 大单动量) * (1 - 控盘动量) = {get_val(raw_score, probe_date):.4f}")
        print(f"    - [上下文]: (1 - 均线健康度 {get_val(ma_health_score, probe_date):.4f}) = {1-get_val(ma_health_score, probe_date):.4f}")
        print(f"    - [快照分]: {get_val(raw_score, probe_date):.4f} * {1-get_val(ma_health_score, probe_date):.4f} = {get_val(snapshot_score, probe_date):.4f}")
        final_score = engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
        state_score = get_val(snapshot_score.clip(0, 1), probe_date)
        relationship_trend = snapshot_score.diff(5).fillna(0)
        velocity_score = get_val(normalize_to_bipolar(series=relationship_trend, target_index=df.index, window=55), probe_date)
        relationship_accel = relationship_trend.diff(5).fillna(0)
        acceleration_score = get_val(normalize_to_bipolar(series=relationship_accel, target_index=df.index, window=55), probe_date)
        print(f"    - [动态锻造]:")
        print(f"      - 状态分 (State)      : {state_score:.4f}")
        print(f"      - 速度分 (Velocity)   : {velocity_score:.4f}")
        print(f"      - 加速度分 (Acceleration): {acceleration_score:.4f}")
        print(f"    - [最终动态分]: {get_val(final_score, probe_date):.4f}")
        print("\n--- “阿瑞斯之矛探针”运行完毕 ---")

    def _deploy_hephaestus_chip_forge_probe(self, probe_date: pd.Timestamp):
        """
        【V2.5 · 冥王之眼版】“赫淮斯托斯-公理熔炉”探针
        - 核心升级: 全面同步主引擎的“冥王之眼”协议。
                      1. 内置了“全息背离”计算逻辑。
                      2. 在公理重算中引入了“背离杠杆”。
                      3. 同步了共振信号的融合方式(加权算术平均)。
                      4. 更新了“反转信号”的计算方式，直接使用背离引擎。
        """
        # 全面同步“冥王之眼”协议
        print("\n--- [探针] 正在启用: 🔥【赫淮斯托斯 · 公理熔炉探针 V2.5】🔥 ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        chip_intel = self.chip_intel
        periods = [5, 13, 21, 55]
        norm_window = 120
        # 内部辅助函数，用于模拟“冥王之眼”协议
        def _recalc_holographic_divergence(series: pd.Series, short_p: int, long_p: int, window: int) -> pd.Series:
            slope_short = series.diff(short_p).fillna(0)
            slope_long = series.diff(long_p).fillna(0)
            velocity_divergence = slope_short - slope_long
            velocity_divergence_score = normalize_to_bipolar(velocity_divergence, series.index, window)
            accel_short = slope_short.diff(short_p).fillna(0)
            accel_long = slope_long.diff(long_p).fillna(0)
            acceleration_divergence = accel_short - accel_long
            acceleration_divergence_score = normalize_to_bipolar(acceleration_divergence, series.index, window)
            final_divergence_score = (velocity_divergence_score * 0.6 + acceleration_divergence_score * 0.4).clip(-1, 1)
            return final_divergence_score.astype(np.float32)
        def _recalc_meta_analysis(snapshot_score: pd.Series, meta_window: int, holographic_divergence_score: pd.Series):
            p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
            p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
            w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
            w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
            w_holographic = get_param_value(p_meta.get('holographic_weight'), 0.5)
            state_score = snapshot_score.clip(0, 1)
            relationship_trend = snapshot_score.diff(meta_window).fillna(0)
            velocity_score = normalize_to_bipolar(relationship_trend, df.index, 55, 1.0)
            relationship_accel = relationship_trend.diff(meta_window).fillna(0)
            acceleration_score = normalize_to_bipolar(relationship_accel, df.index, 55, 1.0)
            dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
            holographic_leverage = 1 + (holographic_divergence_score * w_holographic)
            final_score = (state_score * dynamic_leverage * holographic_leverage).clip(0, 1)
            return final_score
        def get_val(series, date, default=np.nan):
            if series is None or not isinstance(series, (pd.Series, dict)): return default
            if isinstance(series, dict): return series.get(date, default)
            return series.get(date, default)
        # 链路层 1: 最终锻造成品 (终极信号)
        print("\n  [链路层 1] 最终锻造成品 (终极信号)")
        bull_resonance = get_val(atomic.get('SCORE_CHIP_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_resonance = get_val(atomic.get('SCORE_CHIP_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_reversal = get_val(atomic.get('SCORE_CHIP_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_reversal = get_val(atomic.get('SCORE_CHIP_TOP_REVERSAL'), probe_date, 0.0)
        print(f"    - 看涨共振 (BULLISH_RESONANCE): {bull_resonance:.4f}")
        print(f"    - 看跌共振 (BEARISH_RESONANCE): {bear_resonance:.4f}")
        print(f"    - 底部反转 (BOTTOM_REVERSAL)  : {bottom_reversal:.4f}")
        print(f"    - 顶部反转 (TOP_REVERSAL)    : {top_reversal:.4f}")
        concentration_scores = atomic.get('SCORE_CHIP_MTF_CONCENTRATION', {})
        accumulation_scores = atomic.get('SCORE_CHIP_MTF_ACCUMULATION', {})
        power_transfer_scores = atomic.get('SCORE_CHIP_MTF_POWER_TRANSFER', {})
        # 链路层 2: 核心公理诊断 (三维动态分析)
        print("\n  [链路层 2] 核心公理诊断 (三维动态分析)")
        # --- 公理一：聚散动态 ---
        print("\n    --- 公理一: 聚散动态 (Concentration) ---")
        conc_90 = normalize_score(df.get('concentration_90pct_D'), df.index, norm_window, ascending=False)
        conc_70 = normalize_score(df.get('concentration_70pct_D'), df.index, norm_window, ascending=False)
        stability = normalize_score(df.get('peak_stability_D'), df.index, norm_window, ascending=True)
        concentration_snapshot = (conc_90 * conc_70 * stability)**(1/3)
        print(f"      - [快照分] 当日静态集中度: {get_val(concentration_snapshot, probe_date):.4f}")
        for p in periods:
            recalc_divergence = _recalc_holographic_divergence(concentration_snapshot, 1, p, norm_window) # 重算背离分
            recalc_dyn_conc = _recalc_meta_analysis(concentration_snapshot, p, recalc_divergence) # 传入背离分
            print(f"      - [周期 {p}d] 动态分: {get_val(concentration_scores.get(p), probe_date):.4f} (重算: {get_val(recalc_dyn_conc, probe_date):.4f})")
        # --- 公理二：主力吸派 ---
        print("\n    --- 公理二: 主力吸派 (Main Force Action) ---")
        for p in periods:
            acc_ev = (normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, norm_window, ascending=True) * normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=False) * normalize_score(df.get(f'SLOPE_{p}_trade_concentration_index_D'), df.index, norm_window, ascending=True))**(1/3)
            dist_ev = (normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D'), df.index, norm_window, ascending=False) * normalize_score(df.get(f'SLOPE_{p}_turnover_from_winners_ratio_D'), df.index, norm_window, ascending=True) * normalize_score(df.get(f'SLOPE_{p}_trade_concentration_index_D'), df.index, norm_window, ascending=False))**(1/3)
            action_snapshot = (acc_ev - dist_ev).astype(np.float32)
            recalc_divergence = _recalc_holographic_divergence(action_snapshot, 1, p, norm_window) # 重算背离分
            recalc_dyn_action = _recalc_meta_analysis(action_snapshot, p, recalc_divergence) # 传入背离分
            print(f"      - [周期 {p}d] 快照分: {get_val(action_snapshot, probe_date):.4f} -> 动态分: {get_val(accumulation_scores.get(p), probe_date):.4f} (重算: {get_val(recalc_dyn_action, probe_date):.4f})")
        # --- 公理三：权力转移 ---
        print("\n    --- 公理三: 权力转移 (Power Transfer) ---")
        for p in periods:
            to_main = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True) * normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True))**0.5
            to_retail = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False) * normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False))**0.5
            transfer_snapshot = (to_main - to_retail).astype(np.float32)
            recalc_divergence = _recalc_holographic_divergence(transfer_snapshot, 1, p, norm_window) # 重算背离分
            recalc_dyn_transfer = _recalc_meta_analysis(transfer_snapshot, p, recalc_divergence) # 传入背离分
            print(f"      - [周期 {p}d] 快照分: {get_val(transfer_snapshot, probe_date):.4f} -> 动态分: {get_val(power_transfer_scores.get(p), probe_date):.4f} (重算: {get_val(recalc_dyn_transfer, probe_date):.4f})")
        # 链路层 3: 终极信号合成 (全息共振熔炉)
        print("\n  [链路层 3] 终极信号合成 (全息共振熔炉)")
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        total_weight = sum(tf_weights.values())
        # --- 看涨共振 ---
        print("\n    --- 看涨共振 (Bullish Resonance) ---")
        bullish_scores_by_period = {}
        print("      - [周期内融合] 公式: (聚散分 + 吸派分 + 转移分) / 3")
        for p in periods:
            conc_val = get_val(concentration_scores.get(p), probe_date, 0.0)
            acc_val = get_val(accumulation_scores.get(p), probe_date, 0.0)
            trans_val = get_val(power_transfer_scores.get(p), probe_date, 0.0)
            period_score = (conc_val + acc_val + trans_val) / 3.0
            bullish_scores_by_period[p] = period_score
            print(f"        - 周期 {p}d: ({conc_val:.2f} + {acc_val:.2f} + {trans_val:.2f}) / 3 = {period_score:.4f}")
        print("      - [跨周期共振] 公式: 加权算术平均") # 更新公式描述
        recalc_bullish_resonance = 0.0 # 初始化为0
        if total_weight > 0:
            for p in periods:
                recalc_bullish_resonance += bullish_scores_by_period[p] * (tf_weights[p] / total_weight) # 使用加权算术平均
        print(f"      - [最终锻造] 实际值: {bull_resonance:.4f} vs 重算值: {recalc_bullish_resonance:.4f}")
        # --- 看跌共振 ---
        print("\n    --- 看跌共振 (Bearish Resonance) ---")
        bearish_scores_by_period = {}
        print("      - [周期内融合] 公式: ((1-聚散分) + (1-吸派分) + (1-转移分)) / 3")
        for p in periods:
            conc_val = get_val(concentration_scores.get(p), probe_date, 0.0)
            acc_val = get_val(accumulation_scores.get(p), probe_date, 0.0)
            trans_val = get_val(power_transfer_scores.get(p), probe_date, 0.0)
            period_score = ((1 - conc_val) + (1 - acc_val) + (1 - trans_val)) / 3.0
            bearish_scores_by_period[p] = period_score
            print(f"        - 周期 {p}d: ((1-{conc_val:.2f}) + (1-{acc_val:.2f}) + (1-{trans_val:.2f})) / 3 = {period_score:.4f}")
        print("      - [跨周期共振] 公式: 加权算术平均") # 更新公式描述
        recalc_bearish_resonance = 0.0 # 初始化为0
        if total_weight > 0:
            for p in periods:
                recalc_bearish_resonance += bearish_scores_by_period[p] * (tf_weights[p] / total_weight) # 使用加权算术平均
        print(f"      - [最终锻造] 实际值: {bear_resonance:.4f} vs 重算值: {recalc_bearish_resonance:.4f}")
        # 链路层 4: 反转信号锻造
        print("\n  [链路层 4] 反转信号锻造 (全息背离引擎应用)")
        # --- 底部反转 ---
        bull_res_series = atomic.get('SCORE_CHIP_BULLISH_RESONANCE')
        recalc_bottom_reversal_divergence = _recalc_holographic_divergence(bull_res_series, 5, 21, 55) # 使用背离引擎重算
        recalc_bottom_reversal = recalc_bottom_reversal_divergence.clip(0, 1) # 只取看涨背离部分
        print(f"    - 底部反转 = Divergence(看涨共振, 5, 21).clip(0,1) -> 实际值: {bottom_reversal:.4f} vs 重算值: {get_val(recalc_bottom_reversal, probe_date):.4f}") # 更新描述
        # --- 顶部反转 ---
        bear_res_series = atomic.get('SCORE_CHIP_BEARISH_RESONANCE')
        recalc_top_reversal_divergence = _recalc_holographic_divergence(bear_res_series, 5, 21, 55) # 使用背离引擎重算
        recalc_top_reversal = recalc_top_reversal_divergence.clip(0, 1) # 只取看涨背离部分
        print(f"    - 顶部反转 = Divergence(看跌共振, 5, 21).clip(0,1) -> 实际值: {top_reversal:.4f} vs 重算值: {get_val(recalc_top_reversal, probe_date):.4f}") # 更新描述
        print("\n--- “赫淮斯托斯-公理熔炉”探针运行完毕 ---")

    def _deploy_ares_tribunal_probe(self, probe_date: pd.Timestamp):
        """
        【V1.2 · 原始力量同步版】“阿瑞斯审判庭”探针
        - 核心修复: 完全同步主引擎V5.0版的“原始力量”逻辑，确保探针与主引擎的计算口径绝对一致。
        """
        print("\n--- [探针] 正在启用: ⚖️【阿瑞斯审判庭 · 真伪识别探针 V1.2】⚖️ ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        norm_window = 55
        p = 5
        def get_val(series, date, default=np.nan):
            if series is None or not isinstance(series, (pd.Series, dict)): return default
            if isinstance(series, dict): return series.get(date, default)
            return series.get(date, default)
        # --- 步骤1: 量化“近期派发强度”证据 ---
        print("\n  [链路层 1] 量化“近期派发强度”证据")
        to_main = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True) *
                   normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True))**0.5
        to_retail = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False) *
                     normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False))**0.5
        short_term_transfer_snapshot = (to_main - to_retail).astype(np.float32)
        recent_distribution_strength = (short_term_transfer_snapshot.rolling(3).mean().clip(-1, 0) * -1).astype(np.float32)
        print(f"    - 近期派发强度分: {get_val(recent_distribution_strength, probe_date):.4f}")
        # --- 步骤2: 量化“当日反转强度”与“动态质量”证据 (全新逻辑) ---
        print("\n  [链路层 2] 量化“反转强度(原始力量)”与“动态质量”证据")
        chip_reversal_raw = get_val(atomic.get('SCORE_CHIP_BOTTOM_REVERSAL'), probe_date, 0.0)
        behavior_reversal_raw = get_val(atomic.get('SCORE_BEHAVIOR_BOTTOM_REVERSAL'), probe_date, 0.0)
        dyn_reversal_raw = get_val(atomic.get('SCORE_DYN_BOTTOM_REVERSAL'), probe_date, 0.0)
        reversal_strength = np.maximum.reduce([chip_reversal_raw, behavior_reversal_raw, dyn_reversal_raw])
        print(f"    - 当日反转强度 (原始力量) = max(筹码:{chip_reversal_raw:.2f}, 行为:{behavior_reversal_raw:.2f}, 力学:{dyn_reversal_raw:.2f}) -> {reversal_strength:.4f}")
        dyn_bullish_resonance = get_val(atomic.get('SCORE_DYN_BULLISH_RESONANCE'), probe_date, 0.0)
        behavior_bullish_resonance = get_val(atomic.get('SCORE_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        reversal_dynamic_quality = (dyn_bullish_resonance * behavior_bullish_resonance)**0.5
        print(f"    - 反转动态质量 (力学: {dyn_bullish_resonance:.2f} * 行为: {behavior_bullish_resonance:.2f}) -> {reversal_dynamic_quality:.4f}")
        # --- 步骤3: 交叉验证“战术性打压” ---
        print("\n  [链路层 3] 交叉验证“战术性打压”")
        trend_quality_context = get_val(atomic.get('COGNITIVE_SCORE_TREND_QUALITY'), probe_date, 0.0)
        panic_absorption_score = get_val(atomic.get('SCORE_MICRO_PANIC_ABSORPTION'), probe_date, 0.0)
        winner_conviction_score = (get_val(atomic.get('PROCESS_META_WINNER_CONVICTION'), probe_date, 0.0) * 0.5 + 0.5)
        structural_support_score = get_val(atomic.get('SCORE_FOUNDATION_BOTTOM_CONFIRMED'), probe_date, 0.0)
        print(f"    - [看涨证据链] 趋势质量: {trend_quality_context:.2f}, 恐慌吸收: {panic_absorption_score:.2f}, 赢家信念: {winner_conviction_score:.2f}, 结构支撑: {structural_support_score:.2f}")
        absorption_evidence_chain = (trend_quality_context * panic_absorption_score * winner_conviction_score * (1 + structural_support_score * 0.5))
        recalc_tactical_suppression = (get_val(recent_distribution_strength, probe_date) * reversal_strength * reversal_dynamic_quality * absorption_evidence_chain).clip(0, 1)
        actual_tactical_suppression = get_val(atomic.get('COGNITIVE_SCORE_TACTICAL_SUPPRESSION'), probe_date, 0.0)
        print(f"    - [最终锻造] 战术性打压分 -> 实际值: {actual_tactical_suppression:.4f} vs 重算值: {recalc_tactical_suppression:.4f}")
        # --- 步骤4: 交叉验证“真实撤退” ---
        print("\n  [链路层 4] 交叉验证“真实撤退”")
        trend_decay_context = 1.0 - trend_quality_context
        no_absorption_score = 1.0 - panic_absorption_score
        winner_capitulation_score = (get_val(atomic.get('PROCESS_META_WINNER_CONVICTION'), probe_date, 0.0) * -0.5 + 0.5)
        bull_trap_evidence = 1.0 - reversal_dynamic_quality
        print(f"    - [看跌证据链] 趋势恶化: {trend_decay_context:.2f}, 无人吸收: {no_absorption_score:.2f}, 赢家动摇: {winner_capitulation_score:.2f}, 牛市陷阱: {bull_trap_evidence:.2f}")
        retreat_evidence_chain = (trend_decay_context * no_absorption_score * winner_capitulation_score * bull_trap_evidence)
        recalc_true_retreat = (get_val(recent_distribution_strength, probe_date) * retreat_evidence_chain).clip(0, 1)
        actual_true_retreat = get_val(atomic.get('COGNITIVE_SCORE_TRUE_RETREAT_RISK'), probe_date, 0.0)
        print(f"    - [最终锻造] 真实撤退风险 -> 实际值: {actual_true_retreat:.4f} vs 重算值: {recalc_true_retreat:.4f}")
        print("\n--- “阿瑞斯审判庭”探针运行完毕 ---")









