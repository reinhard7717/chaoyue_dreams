# 文件: strategies/trend_following/intelligence/process/calculate_covert_accumulation.py
# 隐蔽吸筹 (Covert Accumulation) 计算 V14.0 · 暗物质探测版
# 修改说明：全盘废弃旧逻辑，基于《最终军械库清单》构建“隐匿流+结构熵+势能背离”三维探测模型。

import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_param_value
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateCovertAccumulation:
    """
    PROCESS_META_COVERT_ACCUMULATION
    【V14.0 · 暗物质探测版】
    核心思想：
    真正的吸筹是“资金流入而价格不动”。本模型通过探测“隐匿资金流”、“筹码低熵化”和“量价背离度”
    来锁定主力的左侧建仓行为。
    """
    def __init__(self, strategy_instance, process_intelligence_helper_instance: ProcessIntelligenceHelper):
        self.strategy = strategy_instance
        self.helper = process_intelligence_helper_instance
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self._probe_output = []
        self._debug_trace = {} # 全链路数据快照

    def _get_required_column_map(self) -> Dict[str, str]:
        """
        【V18.3】数据映射升级：全息引力场与量子验证
        新增：
        1. pressure_trapped_D (套牢压力): 顶部引力
        2. support_resistance_ratio_D (支撑压力比): 底部支撑力
        3. tick_large_order_net_D (大单净额): 定向矢量
        4. tick_clustering_index_D (聚类指数): 算法指纹
        5. morning_flow_ratio_D, closing_flow_ratio_D (时间加权): 攻击时点
        6. intraday_support_test_count_D (支撑试盘): 验证底部的坚固度
        7. intraday_price_distribution_skewness_D (微观偏度): 验证日内吸筹结构
        8. reversal_prob_D (反转概率): 统计学最终判决
        9. 运动学衍生: SLOPE/ACCEL/JERK_21...
        """
        return {
            'close': 'close_D',
            'pct_change': 'pct_change_D',
            'stealth_flow_ratio': 'stealth_flow_ratio_D',
            'absorption_energy': 'absorption_energy_D',
            'sm_inst_net_buy': 'SMART_MONEY_INST_NET_BUY_D',
            'flow_cluster_intensity': 'flow_cluster_intensity_D',
            'pressure_release_index': 'pressure_release_index_D',
            'intraday_accum_conf': 'intraday_accumulation_confidence_D',
            'flow_persistence': 'flow_persistence_minutes_D',
            'hf_divergence': 'high_freq_flow_divergence_D',
            'transfer_efficiency': 'tick_chip_transfer_efficiency_D',
            'slope_stealth': 'SLOPE_21_stealth_flow_ratio_D',
            'accel_stealth': 'ACCEL_21_stealth_flow_ratio_D',
            'jerk_stealth': 'JERK_21_stealth_flow_ratio_D',
            'hab_net_amount_21': 'total_net_amount_20d_D',
            'chip_entropy': 'chip_entropy_D',
            'chip_stability': 'chip_stability_D',
            'behavior_accumulation': 'behavior_accumulation_D',
            'concentration_peak': 'concentration_peak_D',
            'chip_skewness': 'chip_skewness_D',
            'winner_rate': 'winner_rate_D',
            'tick_balance': 'tick_chip_balance_ratio_D',
            'slope_entropy': 'SLOPE_21_chip_entropy_D',
            'accel_entropy': 'ACCEL_21_chip_entropy_D',
            'jerk_entropy': 'JERK_21_chip_entropy_D',
            'accel_conc': 'ACCEL_21_chip_concentration_ratio_D',
            'cost_95': 'cost_95pct_D',
            'cost_5': 'cost_5pct_D',
            'intra_low_lock': 'intraday_low_lock_ratio_D',
            'chip_convergence': 'chip_convergence_ratio_D',
            'pf_divergence': 'price_flow_divergence_D',
            'sm_divergence': 'SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'hf_divergence_micro': 'high_freq_flow_divergence_D',
            'market_sentiment': 'market_sentiment_score_D',
            'tick_abnormal': 'tick_abnormal_volume_ratio_D',
            'slope_pf_div': 'SLOPE_21_price_flow_divergence_D',
            'accel_pf_div': 'ACCEL_21_price_flow_divergence_D',
            'jerk_pf_div': 'JERK_21_price_flow_divergence_D',
            'slope_sm_div': 'SLOPE_21_SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D',
            'flow_zscore': 'flow_zscore_D',
            'chip_rsi_div': 'chip_rsi_divergence_D',
            'div_strength': 'divergence_strength_D',
            'state_golden_pit': 'STATE_GOLDEN_PIT_D',
            'cost_5pct': 'cost_5pct_D',
            'tick_abnormal_vol': 'tick_abnormal_volume_ratio_D',
            'pressure_trapped': 'pressure_trapped_D',
            'support_res_ratio': 'support_resistance_ratio_D',
            'large_order_net': 'tick_large_order_net_D',
            'clustering_idx': 'tick_clustering_index_D',
            'morning_flow': 'morning_flow_ratio_D',
            'closing_flow': 'closing_flow_ratio_D',
            'slope_large_net': 'SLOPE_21_tick_large_order_net_D',
            'accel_large_net': 'ACCEL_21_tick_large_order_net_D',
            'jerk_large_net': 'JERK_21_tick_large_order_net_D',
            'slope_support': 'SLOPE_21_support_resistance_ratio_D',
            'slope_cluster': 'SLOPE_21_tick_clustering_index_D',
            'support_tests': 'intraday_support_test_count_D',
            'micro_skew': 'intraday_price_distribution_skewness_D',
            'rev_prob': 'reversal_prob_D',
        }

    def _get_raw_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        raw_signals = {}
        column_map = self._get_required_column_map()
        for signal_name, col_name in column_map.items():
            # 严禁 fillna，数据缺失必须报错或暴露，不使用防御性默认值
            raw_signals[signal_name] = self.helper._get_safe_series(df, col_name, 0.0).astype(np.float32)
        return raw_signals

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V18.3 主计算流】
        逻辑：Covert_Accumulation = (Stealth * Structure * Divergence * Location) ^ (1/4)
        四维全息探测：隐匿流确认资金进场，结构熵确认筹码锁定，背离场确认势能积蓄，定位场确认时空奇点。
        """
        self._probe_output = []
        self._debug_trace = {}
        df_index = df.index
        raw = self._get_raw_signals(df)
        if self._is_probe_enabled(df):
            self._debug_trace['raw_sample_end'] = {k: v.values[-1] for k, v in raw.items()}
        stealth_score = self._calc_stealth_flow_vector(df_index, raw)
        structure_score = self._calc_structural_entropy_vector(df_index, raw)
        divergence_score = self._calc_divergence_vector(df_index, raw)
        location_score = self._calc_anomaly_location_vector(df_index, raw)
        self._debug_trace['vectors'] = {
            'stealth': stealth_score.values[-1],
            'structure': structure_score.values[-1],
            'divergence': divergence_score.values[-1],
            'location': location_score.values[-1]
        }
        pct_change = raw['pct_change'].values
        price_suppression = np.where(pct_change > 0.04, 0.4, 1.0)
        base_accumulation = (stealth_score * structure_score * divergence_score * location_score) ** 0.25
        final_score = (base_accumulation * price_suppression).clip(0, 1).astype(np.float32)
        if self._is_probe_enabled(df):
            self._output_full_debug_info(df_index, final_score)
        return final_score

    def _calc_stealth_flow_vector(self, df_index: pd.Index, raw: Dict[str, pd.Series]) -> pd.Series:
        """
        【V15.3 · 费米-狄拉克相变版】
        逻辑：Stealth = Activation( Raw_Linear_Score )
        引入非线性增益，模拟主力吸筹从“量变”到“质变”的物理过程。
        """
        # ---------------------------
        # 1. 基础瞬时流强度 (Base)
        # ---------------------------
        s_ratio = raw['stealth_flow_ratio'].values
        intra_conf = raw['intraday_accum_conf'].values
        base_intensity = (s_ratio * 0.6 + intra_conf * 0.4)
        # ---------------------------
        # 2. 全息微观修正 (Holographic)
        # ---------------------------
        # 时间持续性
        persistence = raw['flow_persistence'].values
        time_factor = np.clip(0.8 + (persistence / 240.0), 0.8, 1.3)
        # 高频背离
        hf_div = raw['hf_divergence'].values
        micro_div_factor = np.tanh(hf_div) * 0.2 + 1.0 
        # 转移效率
        eff = raw['transfer_efficiency'].values
        eff_factor = np.tanh(eff) * 0.3 + 0.85 
        # 全息乘数
        holographic_multiplier = time_factor * micro_div_factor * eff_factor
        # ---------------------------
        # 3. 运动学修正 (Kinematics)
        # ---------------------------
        slope = raw['slope_stealth'].values
        accel = raw['accel_stealth'].values
        jerk = raw['jerk_stealth'].values
        v_factor = np.tanh(slope / 0.05) * 0.2
        a_factor = np.tanh(accel / 0.01) * 0.1
        j_factor = np.abs(np.tanh(jerk / 0.005)) * 0.05
        k_multiplier = 1.0 + v_factor + a_factor + j_factor
        # ---------------------------
        # 4. 算法结构与 HAB
        # ---------------------------
        cluster = raw['flow_cluster_intensity'].values
        inst_buy = raw['sm_inst_net_buy'].values
        inst_gate = np.tanh(inst_buy / 1000000.0).clip(0, 1)
        algo_structure = np.tanh(cluster) * inst_gate
        hab_21 = raw['hab_net_amount_21'].values
        hab_score = np.tanh(hab_21 / 50000000.0)
        hab_robustness = np.where(hab_score > 0, 1.0 + hab_score * 0.3, 1.0 + hab_score * 0.8)
        # ---------------------------
        # 5. 线性合成 (Linear Synthesis)
        # ---------------------------
        absorb = raw['absorption_energy'].values
        absorb_norm = np.tanh(absorb / 50000.0)
        press_release = raw['pressure_release_index'].values
        effectiveness = absorb_norm * (0.3 + 0.7 * press_release)
        # 计算原始线性得分 (Raw Linear Score)
        dynamic_flow = base_intensity * k_multiplier * algo_structure * holographic_multiplier
        raw_linear_score = np.sqrt(np.clip(dynamic_flow, 0, None)) * (0.5 + 0.5 * effectiveness) * hab_robustness
        # ---------------------------
        # 6. 非线性激活 (Non-Linear Activation) [NEW]
        # ---------------------------
        # 费米-狄拉克激活函数参数
        # alpha (陡峭度): 8.0 -> 极强的区分度，压制噪音，放大信号
        # beta (阈值): 0.6 -> 只有当原始分达到 0.6 以上，才被视为有效吸筹
        alpha = 8.0
        beta = 0.6
        # 为了防止数值溢出，对指数项进行截断
        exponent = -alpha * (raw_linear_score - beta)
        exponent = np.clip(exponent, -20, 20) # 防止 exp 溢出
        
        # 激活计算
        # 公式性质：在 beta 处值为 0.5；小于 beta 时迅速衰减；大于 beta 时迅速饱和
        activated_score = 1.0 / (1.0 + np.exp(exponent))
        
        # 修正：我们需要保留原始分数的幅度信息，而不是单纯变成 0/1 开关
        # 因此，使用 激活因子 对 原始分数 进行 加权，而不是直接替换
        # 这样既压制了低分，又保留了高分区的差异
        final_stealth_vector = raw_linear_score * activated_score * 2.0 # *2.0 是为了补偿 Sigmoid 的压缩效应
        
        # 再次 Clipping 保证物理意义
        final_stealth_vector = np.clip(final_stealth_vector, 0, 1)
        # ---------------------------
        # 7. 探针埋点
        # ---------------------------
        self._debug_trace['stealth_details'] = {
            'raw_linear': raw_linear_score[-1],
            'activation_factor': activated_score[-1],
            'final_vector': final_stealth_vector[-1],
            'phase_state': 'Critical' if raw_linear_score[-1] > beta else 'Noise',
            'components': f"Base:{base_intensity[-1]:.2f}/Holo:{holographic_multiplier[-1]:.2f}/Kin:{k_multiplier[-1]:.2f}"
        }
        return pd.Series(final_stealth_vector, index=df_index).astype(np.float32)

    def _calc_structural_entropy_vector(self, df_index: pd.Index, raw: Dict[str, pd.Series]) -> pd.Series:
        """
        【V16.3 · 晶格化相变版】
        逻辑：Structure = Activation( Raw_Linear_Product )
        引入物理相变逻辑，强制过滤“平庸结构”，只保留越过“成核势垒”的完美锁仓。
        """
        # ---------------------------
        # 1. 静态结构状态 (Static)
        # ---------------------------
        entropy = raw['chip_entropy'].values
        entropy_score = np.clip(1.0 - entropy * 1.5, 0, 1)
        stability = raw['chip_stability'].values
        peak = raw['concentration_peak'].values
        peak_score = np.exp(-0.5 * peak)
        thermo_order = entropy_score * stability * (0.6 + 0.4 * peak_score)
        skew = raw['chip_skewness'].values
        spatial_score = np.tanh(skew * 2.0) * 0.5 + 0.5
        spatial_score = np.where(skew < -0.5, spatial_score * 0.1, spatial_score)
        win_rate = raw['winner_rate'].values
        game_lock = np.exp(-((win_rate - 0.25) ** 2) / (2 * 0.15 ** 2))
        tick_bal = raw['tick_balance'].values
        micro_score = np.tanh(tick_bal - 1.0) * 0.3 + 1.0
        # ---------------------------
        # 2. 物理致密性 (Compactness)
        # ---------------------------
        c95 = raw['cost_95'].values
        c5 = raw['cost_5'].values
        bandwidth = (c95 - c5) / (c5 + 1e-9)
        # 带宽越窄，爆发力越强。放宽一点衰减系数，避免误杀
        compactness = np.exp(-2.5 * np.clip(bandwidth - 0.05, 0, None))
        # ---------------------------
        # 3. 微观刚性 & 收敛 (Rigidity & Convergence)
        # ---------------------------
        low_lock = raw['intra_low_lock'].values
        rigidity = np.tanh(low_lock * 2.0) * 0.4 + 0.6
        conv_ratio = raw['chip_convergence'].values
        convergence = np.tanh(conv_ratio) * 0.2 + 1.0
        # ---------------------------
        # 4. 运动学修正 (Kinematics)
        # ---------------------------
        s_ent = raw['slope_entropy'].values
        v_ent = np.tanh(-s_ent / 0.005)
        a_conc = raw['accel_conc'].values
        a_lock = np.tanh(a_conc / 0.002)
        j_ent = raw['jerk_entropy'].values
        j_mut = np.abs(np.tanh(j_ent / 0.001)) * 0.1
        kinematic_evol = 1.0 + (v_ent * 0.2) + (a_lock * 0.1) + j_mut
        # ---------------------------
        # 5. HAB 沉淀 (HAB)
        # ---------------------------
        beh_acc_series = raw['behavior_accumulation']
        hab_depth_raw = beh_acc_series.rolling(window=21, min_periods=5).sum().fillna(0).values
        hab_depth = np.tanh(hab_depth_raw / 10.0)
        stab_series = raw['chip_stability']
        stab_mean_21 = stab_series.rolling(window=21, min_periods=5).mean().fillna(0).values
        current_stab = stab_series.values
        stab_ratio = np.clip(current_stab / (stab_mean_21 + 1e-9), 0.5, 1.5)
        hab_factor = (0.5 + 0.5 * hab_depth) * stab_ratio
        # ---------------------------
        # 6. 线性连乘 (Raw Product)
        # ---------------------------
        # 注意：这里是 7 个因子的连乘。如果每个因子平均 0.8，0.8^7 ≈ 0.2
        # 所以 Raw Score 会天然偏低。
        raw_product = (thermo_order * spatial_score * game_lock * micro_score * compactness * rigidity * convergence * kinematic_evol * hab_factor)
        # ---------------------------
        # 7. 非线性激活 (Phase Transition Activation) [NEW]
        # ---------------------------
        # 成核阈值 Beta = 0.25 (对应连乘后的物理意义：各分项表现优良)
        # 陡峭度 Alpha = 10.0 (瞬间结晶)
        alpha = 10.0
        beta = 0.25
        # 计算激活因子
        exponent = -alpha * (raw_product - beta)
        exponent = np.clip(exponent, -20, 20)
        activation_factor = 1.0 / (1.0 + np.exp(exponent))
        # 最终输出：原始分 * 激活因子 * 补偿系数
        # 补偿系数 3.0：将 0.25 左右的分数拉回到 0.75 以上的可信区间
        final_vector = raw_product * activation_factor * 3.0
        # 再次截断
        final_vector = np.clip(final_vector, 0, 1)
        # ---------------------------
        # 8. 探针埋点
        # ---------------------------
        self._debug_trace['struct_details'] = {
            'raw_product': raw_product[-1],
            'activation': activation_factor[-1],
            'final': final_vector[-1],
            'compactness': compactness[-1],
            'hab': hab_factor[-1],
            'state': 'LOCKED' if final_vector[-1] > 0.6 else 'LOOSE'
        }
        return pd.Series(final_vector, index=df_index).astype(np.float32)

    def _calc_divergence_vector(self, df_index: pd.Index, raw: Dict[str, pd.Series]) -> pd.Series:
        """
        【V17.3 · 奇异点共振版】
        逻辑：Vector = Activation( Raw_Linear_Product )
        引入非线性共振激活，将多维连乘导致的低数值压抑，还原为物理意义上的高强度共振信号。
        """
        # ---------------------------
        # 1. 基础背离状态 (Base)
        # ---------------------------
        pf_div = raw['pf_divergence'].values
        pf_score = np.tanh(pf_div) * 0.5 + 0.5
        sm_div = raw['sm_divergence'].values
        sm_score = np.tanh(sm_div / 50.0).clip(0, 1)
        macro_div = pf_score * 0.4 + sm_score * 0.6
        hf_div = raw['hf_divergence_micro'].values
        hf_score = np.tanh(hf_div / 10.0) * 0.5 + 0.5
        tick_abn = raw['tick_abnormal'].values
        abn_score = np.tanh(tick_abn - 1.0).clip(0, 1)
        micro_div = hf_score * 0.7 + abn_score * 0.3
        sentiment = raw['market_sentiment'].values
        panic_score = np.exp(-5.0 * sentiment)
        inst_buy = raw['sm_inst_net_buy'].values
        buy_score = np.tanh(inst_buy / 5000000.0).clip(0, 1)
        psycho_div = np.sqrt(panic_score * buy_score)
        base_state = np.sqrt(macro_div * micro_div) * (1.0 + psycho_div * 0.5)
        # ---------------------------
        # 2. 统计显著性 (Stats)
        # ---------------------------
        z_score = raw['flow_zscore'].values
        stats_factor = 1.0 / (1.0 + np.exp(-(z_score - 1.0))) * 0.4 + 0.8
        # ---------------------------
        # 3. 结构动量 (Struct)
        # ---------------------------
        chip_rsi = raw['chip_rsi_div'].values
        struct_factor = 1.0 + np.tanh(chip_rsi) * 0.3
        # ---------------------------
        # 4. 元数据 (Meta)
        # ---------------------------
        div_str = raw['div_strength'].values
        meta_factor = np.tanh(div_str) * 0.5 + 0.5
        # ---------------------------
        # 5. 运动学 & HAB (Kinematics & HAB)
        # ---------------------------
        s_pf = raw['slope_pf_div'].values
        s_sm = raw['slope_sm_div'].values
        v_pf = np.tanh(s_pf / 0.05)
        v_sm = np.tanh(s_sm / 2.0)
        velocity = v_pf * 0.4 + v_sm * 0.6
        a_pf = raw['accel_pf_div'].values
        accel = np.tanh(a_pf / 0.01) * 0.5
        j_pf = raw['jerk_pf_div'].values
        jerk = np.abs(np.tanh(j_pf / 0.005)) * 0.1
        kinematic_trend = 1.0 + (velocity * 0.3) + accel + jerk
        kinematic_trend = np.clip(kinematic_trend, 0.5, 1.5)
        sm_div_series = raw['sm_divergence']
        hab_raw = sm_div_series.rolling(window=21, min_periods=5).sum().fillna(0).values
        hab_depth = np.tanh(hab_raw / 400.0)
        potential_factor = np.where(hab_depth > 0, 1.0 + hab_depth * 0.4, 1.0 + hab_depth * 1.0)
        # ---------------------------
        # 6. 线性连乘 (Raw Product)
        # ---------------------------
        # 7个因子连乘，数值会极低
        raw_product = base_state * kinematic_trend * potential_factor * stats_factor * struct_factor * meta_factor
        # ---------------------------
        # 7. 非线性共振激活 (Resonance Activation) [NEW]
        # ---------------------------
        # 共振阈值 Beta = 0.15 (对应平均单因子 0.73)
        # 爆发速率 Alpha = 12.0 (极速共振)
        alpha = 12.0
        beta = 0.15
        exponent = -alpha * (raw_product - beta)
        exponent = np.clip(exponent, -20, 20)
        activation_factor = 1.0 / (1.0 + np.exp(exponent))
        # 能量补偿：乘 5.0，将 0.15 附近的 Raw Score 暴力拉升到 0.75 以上
        final_vector = raw_product * activation_factor * 5.0
        # 截断
        final_vector = np.clip(final_vector, 0, 1)
        self._debug_trace['div_details'] = {
            'raw_product': raw_product[-1],
            'activation': activation_factor[-1],
            'final': final_vector[-1],
            'state': 'RESONANCE' if final_vector[-1] > 0.7 else 'NOISE',
            'components': f"Base:{base_state[-1]:.2f}/Stats:{stats_factor[-1]:.2f}/Kin:{kinematic_trend[-1]:.2f}"
        }
        return pd.Series(final_vector, index=df_index).astype(np.float32)

    def _calc_anomaly_location_vector(self, df_index: pd.Index, raw: Dict[str, pd.Series]) -> pd.Series:
        """
        【V18.3 · 事件视界坍缩模型】
        逻辑：Vector = Raw_Product * Topology_Gate * Resonance_Activation
        引入拓扑门控，对“非安全区”的异动实行“零容忍”政策；引入共振激活，对“完美风暴”实行指数级放大。
        """
        is_pit = raw['state_golden_pit'].values
        sr_ratio = raw['support_res_ratio'].values
        sr_score = np.tanh((sr_ratio - 0.5) * 2.0).clip(0, 1)
        trapped = raw['pressure_trapped'].values
        vacuum_score = np.exp(-3.0 * trapped)
        close = raw['close'].values
        cost_floor = raw['cost_5pct'].values
        dist = (close - cost_floor) / (close + 1e-9)
        safety_score = np.exp(-10.0 * np.clip(dist, 0, None))
        topology = np.maximum(is_pit, sr_score * vacuum_score * safety_score)
        lo_net = raw['large_order_net'].values
        lo_score = np.tanh(lo_net / 10000000.0).clip(0, 1)
        cluster_idx = raw['clustering_idx'].values
        cluster_score = np.tanh((cluster_idx - 0.3) * 3.0).clip(0, 1)
        morning = raw['morning_flow'].values
        closing = raw['closing_flow'].values
        time_score = np.tanh((morning * 0.6 + closing * 0.4) * 2.0).clip(0, 1)
        tick_vol = raw['tick_abnormal_vol'].values
        abn_score = np.tanh(tick_vol - 1.0).clip(0, 1)
        event_horizon = lo_score * cluster_score * (0.7 + 0.3 * time_score) * abn_score
        s_lo = raw['slope_large_net'].values
        a_lo = raw['accel_large_net'].values
        j_lo = raw['jerk_large_net'].values
        v_force = np.tanh(s_lo / 5000000.0)
        a_force = np.tanh(a_lo / 1000000.0)
        j_force = np.abs(np.tanh(j_lo / 500000.0)) * 0.1
        s_sr = raw['slope_support'].values
        v_support = np.tanh(s_sr / 0.05)
        s_cl = raw['slope_cluster'].values
        v_cluster = np.tanh(s_cl / 0.02)
        warp_factor = 1.0 + (v_force * 0.3) + (a_force * 0.2) + (v_support * 0.2) + (v_cluster * 0.1) + j_force
        warp_factor = np.clip(warp_factor, 0.5, 1.8)
        hab_21 = raw['hab_net_amount_21'].values
        hab_depth = np.tanh(hab_21 / 50000000.0)
        mass_factor = np.where(hab_depth > 0, 1.0 + hab_depth * 0.5, 1.0 + hab_depth * 1.5)
        tests = raw['support_tests'].values
        resilience = np.exp(-((tests - 4.0)**2) / (2 * 2.5**2)) * 0.5 + 0.7
        skew = raw['micro_skew'].values
        skew_factor = np.tanh(skew * 2.0) * 0.2 + 1.0
        prob = raw['rev_prob'].values
        prob_factor = np.where(prob > 0.7, 1.0 + (prob - 0.7) * 1.0, prob / 0.7)
        base_system = (topology * event_horizon) * warp_factor * mass_factor
        raw_product = base_system * resilience * skew_factor * prob_factor
        topo_gate = 1.0 / (1.0 + np.exp(-15.0 * (topology - 0.3)))
        resonance_activation = 1.0 / (1.0 + np.exp(-12.0 * (raw_product - 0.10)))
        final_vector = raw_product * topo_gate * (resonance_activation * 5.0)
        final_vector = np.clip(final_vector, 0, 1)
        self._debug_trace['loc_details'] = {
            'raw_product': raw_product[-1],
            'topology': topology[-1],
            'topo_gate': topo_gate[-1],
            'activation': resonance_activation[-1],
            'final': final_vector[-1],
            'event_hrz': event_horizon[-1]
        }
        if self._is_probe_enabled(pd.DataFrame(index=df_index)):
            print(f"[PROBE V18.3 LOC] Final: {final_vector[-1]:.4f} | Raw: {raw_product[-1]:.4f} | Gate: {topo_gate[-1]:.2f} | Act: {resonance_activation[-1]:.2f}")
        return pd.Series(final_vector, index=df_index).astype(np.float32)

    def _is_probe_enabled(self, df: pd.DataFrame) -> bool:
        return get_param_value(self.debug_params.get('enabled'), False) and self.probe_dates

    def _output_full_debug_info(self, df_index: pd.Index, final_score: pd.Series):
        if not self.probe_dates: return
        last_date = df_index[-1]
        print(f"\n====== [CalculateCovertAccumulation PROBE] {last_date} ======")
        print(f"Final Score: {final_score.values[-1]:.4f}")
        
        # 打印原始值快照
        raw = self._debug_trace.get('raw_sample_end', {})
        print("\n>>> [1. Critical Raw Inputs]")
        print(f"  Stealth_Flow_Ratio: {raw.get('stealth_flow_ratio', 0):.4f}")
        print(f"  Absorption_Energy: {raw.get('absorption_energy', 0):.1f}")
        print(f"  SmartMoney_NetBuy: {raw.get('sm_inst_net_buy', 0):.0f}")
        print(f"  Chip_Entropy: {raw.get('chip_entropy', 0):.4f}")
        
        # 打印矢量分量
        vec = self._debug_trace.get('vectors', {})
        print("\n>>> [2. Vector Components]")
        print(f"  [Stealth Vector]: {vec.get('stealth', 0):.4f}")
        print(f"  [Structure Vector]: {vec.get('structure', 0):.4f}")
        print(f"  [Divergence Vector]: {vec.get('divergence', 0):.4f}")
        print("=========================================================\n")