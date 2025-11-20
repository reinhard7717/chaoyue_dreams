import pandas as pd
import numpy as np
import json
from strategies.trend_following import utils
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class ChipProbes:
    """
    【探针模块】筹码情报专属探针
    """
    def __init__(self, intel_layer):
        self.intelligence_layer = intel_layer
        self.strategy = intel_layer.strategy
        self.chip_intel = intel_layer.chip_intel
    def _deploy_hephaestus_forge_probe(self, probe_date: pd.Timestamp):
        """
        【V1.5 · 焦点转移版】“赫菲斯托斯熔炉”探针
        - 核心升级: 将解剖焦点从“公理一”转移至“公理四：筹码峰健康度”。
        """
        print("\n" + "="*35 + f" [筹码探针] 正在点燃 🔥【赫菲斯托斯熔炉 · 筹码引擎解剖 V1.5】🔥 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        chip_intel = self.chip_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        print("\n  [链路层 1] 最终输出 (Final Output)")
        bull_res_actual = get_val(atomic.get('SCORE_CHIP_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_actual = get_val(atomic.get('SCORE_CHIP_BEARISH_RESONANCE'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print("\n  [链路层 2] 四大公理最终得分 (Final Axiom Scores)")
        periods = [1, 5, 13, 21, 55]
        concentration_scores = chip_intel._diagnose_concentration_dynamics(df, periods)
        accumulation_scores = chip_intel._diagnose_main_force_action(df, periods)
        power_transfer_scores = chip_intel._diagnose_power_transfer(df, periods)
        peak_integrity_scores = chip_intel._diagnose_peak_integrity_dynamics(df, periods)
        axiom_scores_by_period = {}
        for p in periods:
            axiom_scores_by_period[p] = {
                'concentration': get_val(concentration_scores.get(p), probe_date, 0.0),
                'accumulation': get_val(accumulation_scores.get(p), probe_date, 0.0),
                'power_transfer': get_val(power_transfer_scores.get(p), probe_date, 0.0),
                'peak_integrity': get_val(peak_integrity_scores.get(p), probe_date, 0.0),
            }
            print(f"    - [周期 {p:2d}] 公理得分: 聚散({axiom_scores_by_period[p]['concentration']:.2f}), 吸派({axiom_scores_by_period[p]['accumulation']:.2f}), 转移({axiom_scores_by_period[p]['power_transfer']:.2f}), 峰健康({axiom_scores_by_period[p]['peak_integrity']:.2f})")
        print("\n--- “赫菲斯托斯熔炉”探针解剖完毕 ---")
    def _deploy_chip_resonance_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 健壮性修复版】筹码共振探针
        - 核心修复: 修复了对 tf_fusion_weights 键进行排序时，未过滤 'description' 键导致的 ValueError。
        """
        import json
        print("\n" + "="*35 + f" [筹码探针] 正在启用 🏛️【筹码共振探针 V1.1】🏛️ " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.chip_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights_raw = p_conf.get('tf_fusion_weights', {})
        periods_str = [k for k in tf_weights_raw.keys() if str(k).isdigit()]
        periods = sorted([int(p) for p in periods_str])
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {})
        tf_weights = {int(k): v for k, v in tf_weights_raw.items() if str(k).isdigit()}
        numeric_tf_weights = tf_weights
        total_tf_weight = sum(numeric_tf_weights.values())
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_bull_res = get_val(atomic.get('SCORE_CHIP_BULLISH_RESONANCE'), probe_date, 0.0)
        actual_bear_res = get_val(atomic.get('SCORE_CHIP_BEARISH_RESONANCE'), probe_date, 0.0)
        print(f"    - 【看涨共振分】: {actual_bull_res:.4f}")
        print(f"    - 【看跌共振分】: {actual_bear_res:.4f}")
        print("\n  [链路层 2] 多周期健康分 (MTF Health Scores)")
        bipolar_health_by_period = {}
        bullish_scores_by_period = {}
        bearish_scores_by_period = {}
        concentration_scores = engine._diagnose_concentration_dynamics(df, periods)
        accumulation_scores = engine._diagnose_main_force_action(df, periods)
        power_transfer_scores = engine._diagnose_power_transfer(df, periods)
        peak_integrity_scores = engine._diagnose_peak_integrity_dynamics(df, periods)
        for p in periods:
            conc_score = get_val(concentration_scores.get(p), probe_date, 0.0)
            acc_score = get_val(accumulation_scores.get(p), probe_date, 0.0)
            pow_score = get_val(power_transfer_scores.get(p), probe_date, 0.0)
            peak_score = get_val(peak_integrity_scores.get(p), probe_date, 0.0)
            health = (conc_score * axiom_weights.get('concentration', 0) + acc_score * axiom_weights.get('accumulation', 0) + pow_score * axiom_weights.get('power_transfer', 0) + peak_score * axiom_weights.get('peak_integrity', 0))
            bipolar_health_by_period[p] = np.clip(health, -1, 1)
            bullish_scores_by_period[p] = max(0, bipolar_health_by_period[p])
            bearish_scores_by_period[p] = max(0, -bipolar_health_by_period[p])
            print(f"    - [周期 {p:2d}] 双极性健康分: {bipolar_health_by_period[p]:.4f} -> 看涨: {bullish_scores_by_period[p]:.4f}, 看跌: {bearish_scores_by_period[p]:.4f}")
        print("\n--- “筹码共振探针”解剖完毕 ---")
    def _deploy_bottom_accumulation_lockdown_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0】底部吸筹锁仓探针
        - 核心职责: 深度解剖“底部吸筹锁仓”信号的“设置-触发-确认”三段式逻辑，
                      定位信号未触发或误触发的根本原因。
        """
        print("\n" + "="*25 + f" [筹码探针] 正在启用 ⛓️【底部吸筹锁仓探针 V1.0】⛓️ " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.chip_intel
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        # --- 重演 diagnose_bottom_accumulation_lockdown 的逻辑 ---
        p_conf = get_params_block(self.strategy, 'chip_lockdown_params', {})
        lookback_window = get_param_value(p_conf.get('lookback_window'), 5)
        accumulation_threshold = get_param_value(p_conf.get('accumulation_threshold'), 0.3)
        norm_window = 55
        # --- 链路层 1: 最终系统输出 ---
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_score = get_val(atomic.get('SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN'), probe_date, 0.0)
        print(f"    - 【最终信号分】: {actual_score:.4f}")
        # --- 链路层 2: 设置阶段 (Setup Phase) ---
        print("\n  [链路层 2] 设置阶段: 识别“底部吸筹区” (Setup Phase)")
        gaia_params = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {}).get('gaia_bedrock_params', {})
        fib_params = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {}).get('fibonacci_support_params', {})
        gaia_support = utils._calculate_gaia_bedrock_support(df, gaia_params, atomic)
        historical_low_support = utils._calculate_historical_low_support(df, fib_params)
        authoritative_bottom_support = np.maximum(gaia_support, historical_low_support) > 0.1
        chip_accumulation_score = atomic.get('SCORE_CHIP_TRUE_ACCUMULATION', pd.Series(0.0, index=df.index))
        sustained_accumulation = chip_accumulation_score.rolling(window=3).mean() > accumulation_threshold
        is_in_bottom_accumulation_zone = authoritative_bottom_support & sustained_accumulation
        print("    --- [回溯检查] 点火日前 '底部吸筹区' 状态 ---")
        for i in range(lookback_window, 0, -1):
            prev_date = probe_date - pd.Timedelta(days=i)
            if prev_date in df.index:
                auth_support_val = get_val(authoritative_bottom_support, prev_date, False)
                sust_acc_val = get_val(sustained_accumulation, prev_date, False)
                zone_val = get_val(is_in_bottom_accumulation_zone, prev_date, False)
                print(f"    - [T-{i}] 日期: {prev_date.strftime('%Y-%m-%d')}: 权威底部支撑: {auth_support_val}, 持续吸筹: {sust_acc_val} -> 吸筹区: {zone_val}")
        # --- 链路层 3: 触发阶段 (Trigger Phase) ---
        print("\n  [链路层 3] 触发阶段: 识别“点火日” (Trigger Phase)")
        is_ignition_day = (df['pct_change_D'] > 0.01) & (df['volume_D'] > df.get('VOL_MA_21_D', 0))
        ignition_val = get_val(is_ignition_day, probe_date, False)
        pct_change_val = get_val(df['pct_change_D'], probe_date)
        vol_val = get_val(df['volume_D'], probe_date)
        vol_ma21_val = get_val(df.get('VOL_MA_21_D'), probe_date)
        print(f"    - 【点火日判断】: {ignition_val} (涨幅: {pct_change_val:.2%}, 成交量: {vol_val:.0f} > MA21量: {vol_ma21_val:.0f})")
        # --- 链路层 4: 确认阶段 (Confirmation Phase) ---
        print("\n  [链路层 4] 确认阶段: 回溯验证与当天确认 (Confirmation Phase)")
        has_recent_setup = is_in_bottom_accumulation_zone.rolling(window=lookback_window).max().shift(1).fillna(0).astype(bool)
        has_recent_setup_val = get_val(has_recent_setup, probe_date, False)
        print(f"    - [回溯验证]: {has_recent_setup_val} (回看 {lookback_window} 天内是否存在吸筹区)")
        low_profit_taking_urgency = 1.0 - normalize_score(df.get('profit_taking_urgency_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        no_main_force_distribution = 1.0 - normalize_score(df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        winner_conviction_raw = atomic.get('PROCESS_META_WINNER_CONVICTION', pd.Series(0.0, index=df.index))
        winner_conviction_score = (winner_conviction_raw.clip(-1, 1) * 0.5 + 0.5)
        lockdown_confirmation_score = (low_profit_taking_urgency * no_main_force_distribution * winner_conviction_score)**(1/3)
        lptu_val = get_val(low_profit_taking_urgency, probe_date)
        nmfd_val = get_val(no_main_force_distribution, probe_date)
        wcs_val = get_val(winner_conviction_score, probe_date)
        lockdown_conf_val = get_val(lockdown_confirmation_score, probe_date)
        print(f"    - [当天锁仓确认]: 得分 {lockdown_conf_val:.4f}")
        print(f"      - (低)获利了结紧迫度: {lptu_val:.4f}")
        print(f"      - (无)主力拉高派发: {nmfd_val:.4f}")
        print(f"      - 赢家信念: {wcs_val:.4f}")
        # --- 链路层 5: 最终裁决 ---
        print("\n  [链路层 5] 最终裁决 (Final Adjudication)")
        recalc_score = (ignition_val & has_recent_setup_val) * lockdown_conf_val
        print(f"    - 【探针重算-最终分】: ({ignition_val} AND {has_recent_setup_val}) * {lockdown_conf_val:.4f} = {recalc_score:.4f}")
        match = np.isclose(actual_score, recalc_score)
        print(f"    - [对比]: 系统最终值 {actual_score:.4f} vs. 探针正确值 {recalc_score:.4f} -> {'✅ 一致' if match else '❌ 不一致'}")
        print("\n--- “底部吸筹锁仓探针”解剖完毕 ---")
    def _deploy_lockdown_scramble_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V2.0 · 原始数据穿透版】锁仓抢筹探针
        - 核心升级: 1. 修正前提验证逻辑，确保读取增强前的原始“底部锁仓”信号。
                      2. 新增“原始证据值”链路层，穿透归一化过程，直达最原始的输入数据。
        """
        print("\n" + "="*25 + f" [筹码探针] 正在启用 🏃【锁仓抢筹探针 V2.0】🏃 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'SCORE_CHIP_LOCKDOWN_SCRAMBLE'
        norm_window = 55
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终信号分】: {actual_score:.4f}")
        print("\n  [链路层 2] 前提验证 (Prerequisite Validation)")
        # 核心修正：读取由微观行为引擎保存的“增强前”的原始锁仓信号值，确保与引擎计算基准一致
        # 提供一个回退，以防利润兑现模块未运行时探针崩溃
        lockdown_trigger_series = atomic.get(
            'SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN_PRE_ENHANCEMENT',
            atomic.get('SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN', pd.Series(0.0, index=df.index))
        )
        lockdown_trigger_val = get_val(lockdown_trigger_series, probe_date)
        print(f"    - [前提: 底部锁仓信号 (增强前)] -> 得分: {lockdown_trigger_val:.4f}")
        print("\n  [链路层 3] 原始证据值 (Raw Evidence Values)")
        raw_cost_advantage = get_val(df.get('main_buy_cost_advantage_D'), probe_date)
        raw_net_flow = get_val(df.get('main_force_net_flow_consensus_D'), probe_date)
        raw_conviction = get_val(df.get('main_force_conviction_ratio_D'), probe_date)
        raw_costly_buy = get_val(df.get('main_force_intraday_profit_D'), probe_date)
        print(f"    - [原始证据 I: 主力成本优势] -> 原始值: {raw_cost_advantage:.4f}")
        print(f"    - [原始证据 II: 主力净流入] -> 原始值: {raw_net_flow:.4f}")
        print(f"    - [原始证据 III: 主力信念] -> 原始值: {raw_conviction:.4f}")
        print(f"    - [原始证据 IV: 亏钱也要买(日内盈亏)] -> 原始值: {raw_costly_buy:.4f}")
        print("\n  [链路层 4] 抢筹证据归一化 (Scramble Evidence Normalization)")
        cost_advantage_series = normalize_score(df.get('main_buy_cost_advantage_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        net_flow_series = normalize_score(df.get('main_force_net_flow_consensus_D', pd.Series(0.0, index=df.index)).clip(lower=0), df.index, norm_window, ascending=True)
        conviction_series = normalize_score(df.get('main_force_conviction_ratio_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        costly_buy_series = normalize_score(df.get('main_force_intraday_profit_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, norm_window, ascending=True)
        cost_advantage_val = get_val(cost_advantage_series, probe_date)
        net_flow_val = get_val(net_flow_series, probe_date)
        conviction_val = get_val(conviction_series, probe_date)
        costly_buy_val = get_val(costly_buy_series, probe_date)
        print(f"    - [证据 I: 主力成本优势] -> 归一化得分: {cost_advantage_val:.4f}")
        print(f"    - [证据 II: 主力净流入] -> 归一化得分: {net_flow_val:.4f}")
        print(f"    - [证据 III: 主力信念] -> 归一化得分: {conviction_val:.4f}")
        print(f"    - [证据 IV: 亏钱也要买] -> 归一化得分: {costly_buy_val:.4f}")
        print("\n  [链路层 5] 证据融合 (Evidence Fusion)")
        recalc_evidence_score = (cost_advantage_val * net_flow_val * conviction_val * costly_buy_val)**(1/4)
        print(f"    - [融合公式]: (成本优势 * 净流入 * 信念 * 亏钱买入) ** (1/4)")
        print(f"    - 【探针重算-证据总分】: {recalc_evidence_score:.4f}")
        print("\n  [链路层 6] 最终裁决 (Final Adjudication)")
        recalc_final_score = lockdown_trigger_val * recalc_evidence_score
        print(f"    - [裁决公式]: 前提分 * 证据总分")
        print(f"    - 【探针重算-最终分】: {lockdown_trigger_val:.4f} * {recalc_evidence_score:.4f} = {recalc_final_score:.4f}")
        match = np.isclose(actual_score, recalc_final_score)
        print(f"    - [对比]: 系统最终值 {actual_score:.4f} vs. 探针正确值 {recalc_final_score:.4f} -> {'✅ 一致' if match else '❌ 不一致'}")
        print("\n--- “锁仓抢筹探针”解剖完毕 ---")















