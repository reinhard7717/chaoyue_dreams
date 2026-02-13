# 文件: strategies/trend_following/intelligence/process/calculate_main_force_rally_intent.py
# 版本: V15.0 · 全息推力-阻力张量版
# 说明: 引入“推力-阻力”物理模型，深度集成VPA效率与控盘坚实度，废弃线性加权，采用张量乘积合成。全链路暴露中间变量。
import pandas as pd
import numpy as np
import numba
from typing import Dict, List, Any
from strategies.trend_following.utils import get_params_block, get_param_value

class CalculateMainForceRallyIntent:
    """
    PROCESS_META_MAIN_FORCE_RALLY_INTENT
    【V15.0 · 全息推力-阻力张量版】
    基于 A 股“资金-结构-效率”三维物理场。
    核心方程：RallyIntent = (Kinetics * Control) / (1 + Resistance)
    1. 引入 VPA_EFFICIENCY_D 衡量拉升效率（阻力系数）。
    2. 引入 control_solidity_index_D 衡量控盘状态。
    3. 引入 flow_consistency_D 剔除突击一日游资金。
    """
    def __init__(self, strategy_instance, process_intelligence_helper_instance):
        self.strategy = strategy_instance
        self.helper = process_intelligence_helper_instance
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        self._probe_cache = []

    def _get_required_column_map(self) -> Dict[str, str]:
        """
        【V32.0】数据映射升级：引入HAB历史累积缓冲与斐波那契动力学
        新增：
        1. flow_21d, flow_55d: 斐波那契周期历史资金累积 (HAB存量意识)
        2. sm_slope_13, sm_accel_13, sm_jerk_13: 主力资金的13日动力学衍生
        3. energy_conc: 用于动力学的能量阻尼(消除零基陷阱)
        """
        return {
            'close': 'close_D',
            'open': 'open_D',
            'high': 'high_D',
            'low': 'low_D',
            'cost_avg': 'cost_50pct_D',
            'sm_net_buy': 'SMART_MONEY_HM_NET_BUY_D',
            'hab_inventory': 'total_net_amount_21d_D',
            'sm_slope': 'SLOPE_5_SMART_MONEY_HM_NET_BUY_D',
            'sm_accel': 'ACCEL_5_SMART_MONEY_HM_NET_BUY_D',
            'sm_jerk': 'JERK_5_SMART_MONEY_HM_NET_BUY_D',
            'sm_synergy': 'SMART_MONEY_SYNERGY_BUY_D',
            'opening_strength': 'opening_buy_strength_D',
            'flow_directionality': 'main_force_flow_directionality_D',
            'pushing_score': 'pushing_score_D',
            'abnormal_vol': 'tick_abnormal_volume_ratio_D',
            'breakout_conf': 'breakout_confidence_D',
            'intraday_support': 'INTRADAY_SUPPORT_INTENT_D',
            'market_sentiment': 'market_sentiment_score_D',
            'mf_conviction': 'main_force_conviction_index_D',
            'closing_strength': 'closing_strength_index_D',
            'winner_rate': 'winner_rate_D',
            'control_solidity': 'control_solidity_index_D',
            'chip_entropy': 'chip_entropy_D',
            'chip_stability': 'chip_stability_D',
            'peak_conc': 'peak_concentration_D',
            'accumulation_score': 'accumulation_signal_score_D',
            'trend_alignment': 'trend_alignment_index_D',
            'hab_structure': 'long_term_chip_ratio_D',
            'conc_slope': 'SLOPE_5_peak_concentration_D',
            'winner_accel': 'ACCEL_5_winner_rate_D',
            'platform_quality': 'consolidation_quality_score_D',
            'foundation_strength': 'support_strength_D',
            'vpa_efficiency': 'VPA_EFFICIENCY_D',
            'profit_pressure': 'profit_pressure_D',
            'turnover': 'turnover_rate_D',
            'trapped_pressure': 'pressure_trapped_D',
            'dist_score': 'distribution_score_D',
            'intraday_dist': 'intraday_distribution_confidence_D',
            'instability': 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'pressure_release': 'pressure_release_index_D',
            'shakeout_score': 'shakeout_score_D',
            'chip_divergence': 'chip_divergence_ratio_D',
            'dist_slope': 'SLOPE_5_distribution_score_D',
            'dist_accel': 'ACCEL_5_distribution_score_D',
            'dist_jerk': 'JERK_5_distribution_score_D',
            'gap_momentum': 'GAP_MOMENTUM_STRENGTH_D',
            'emotional_extreme': 'STATE_EMOTIONAL_EXTREME_D',
            'energy_conc': 'energy_concentration_D',
            'reversal_prob': 'reversal_prob_D',
            'is_leader': 'STATE_MARKET_LEADER_D',
            'theme_hotness': 'THEME_HOTNESS_SCORE_D',
            'lock_ratio': 'high_position_lock_ratio_90_D',
            'coordinated_attack': 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'flow_21d': 'total_net_amount_21d_D',
            'flow_55d': 'total_net_amount_55d_D',
            'sm_slope_13': 'SLOPE_13_SMART_MONEY_HM_NET_BUY_D',
            'sm_accel_13': 'ACCEL_13_SMART_MONEY_HM_NET_BUY_D',
            'sm_jerk_13': 'JERK_13_SMART_MONEY_HM_NET_BUY_D'
        }

    def _load_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V15.0】加载数据，NaN填充为0.0以保持张量计算的连续性
        """
        data = {}
        col_map = self._get_required_column_map()
        for key, col_name in col_map.items():
            series = self.helper._get_safe_series(df, col_name, 0.0)
            data[key] = series.astype(np.float32)
        return data

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V32.0】全息张量计算执行器：全链路引入动力学与HAB历史缓冲层
        """
        self._probe_cache = []
        raw = self._load_data(df)
        idx = df.index
        count = len(idx)
        if count < 5:
            print(f"[PROBE-FATAL] 数据行数不足5行，当前行数: {count}，直接阻断。")
            return pd.Series(0.0, index=idx)
        print(f"[PROBE-INFO] 开始执行全息推力计算(含HAB与Kinematics)，处理条目数: {count}")
        thrust = self._calc_thrust_component(
            raw['sm_net_buy'].values, raw['hab_inventory'].values,
            raw['sm_slope'].values, raw['sm_accel'].values, raw['sm_jerk'].values,
            raw['sm_synergy'].values, raw['opening_strength'].values,
            raw['flow_directionality'].values, raw['pushing_score'].values,
            raw['abnormal_vol'].values, raw['breakout_conf'].values,
            raw['intraday_support'].values, raw['market_sentiment'].values,
            raw['mf_conviction'].values, raw['closing_strength'].values
        )
        structure = self._calc_structure_component(
            raw['winner_rate'].values, raw['control_solidity'].values,
            raw['chip_entropy'].values, raw['chip_stability'].values,
            raw['peak_conc'].values, raw['accumulation_score'].values,
            raw['cost_avg'].values, raw['close'].values,
            raw['trend_alignment'].values, raw['hab_structure'].values,
            raw['conc_slope'].values, raw['winner_accel'].values,
            raw['platform_quality'].values, raw['foundation_strength'].values
        )
        drag = self._calc_drag_component(
            raw['vpa_efficiency'].values, raw['profit_pressure'].values,
            raw['turnover'].values, raw['trapped_pressure'].values,
            raw['dist_score'].values, raw['intraday_dist'].values,
            raw['instability'].values, raw['hab_inventory'].values,
            raw['pressure_release'].values, raw['shakeout_score'].values,
            raw['chip_divergence'].values,
            raw['dist_slope'].values, raw['dist_accel'].values, raw['dist_jerk'].values
        )
        raw_intent = self._calc_tensor_synthesis(
            thrust, structure, drag,
            raw['gap_momentum'].values, raw['emotional_extreme'].values,
            raw['energy_conc'].values, raw['reversal_prob'].values,
            raw['is_leader'].values, raw['theme_hotness'].values,
            raw['lock_ratio'].values, raw['coordinated_attack'].values,
            raw['flow_21d'].values, raw['flow_55d'].values,
            raw['sm_slope_13'].values, raw['sm_accel_13'].values, raw['sm_jerk_13'].values,
            raw['sm_net_buy'].values, idx
        )
        med = np.median(raw_intent)
        mad = np.median(np.abs(raw_intent - med)) + 1e-9
        print(f"[PROBE-STAT] Raw Intent | Median: {med:.4f} | MAD: {mad:.4f}")
        z_scores = (raw_intent - med) / (mad * 3.0)
        final_scores = 1.0 / (1.0 + np.exp(-z_scores))
        if self._is_probe_enabled():
            self._generate_probe_report(idx, raw, thrust, structure, drag, raw_intent, final_scores)
        return pd.Series(final_scores, index=idx, dtype=np.float32)

    def _calc_thrust_component(self, sm_net_buy: np.ndarray, hab_inventory: np.ndarray, sm_slope: np.ndarray, sm_accel: np.ndarray, sm_jerk: np.ndarray, sm_synergy: np.ndarray, opening_strength: np.ndarray, flow_directionality: np.ndarray, pushing_score: np.ndarray, abnormal_vol: np.ndarray, breakout_conf: np.ndarray, intraday_support: np.ndarray, market_sentiment: np.ndarray, mf_conviction: np.ndarray, closing_strength: np.ndarray) -> np.ndarray:
        """
        【V21.0】维度A：非线性临界激发推力模型 (Non-Linear Critical Excitation Thrust)
        核心逻辑：引入相变机制。当宏观动力学与微观爆发力发生“临界共振”时，推力呈指数级放大。
        方程：FinalThrust = LinearThrust * (1 + Sigmoid(Resonance - Threshold))
        """
        # --- 1. 线性基础推力 (Linear Base) ---
        # [HAB & Flow]
        safe_flow = np.where(np.abs(sm_net_buy) < 1.0, 1.0, sm_net_buy)
        hab_bonus = np.where((hab_inventory > 0) & (sm_net_buy > 0), 1.0 + np.tanh(hab_inventory / 100000.0) * 0.5, 1.0)
        hab_damper = np.where((hab_inventory > 0) & (sm_net_buy < 0), 0.5, 1.0)
        base_kinetic = (sm_net_buy * hab_damper * hab_bonus) + (sm_synergy * 1.5)
        # [Kinematics]
        k_slope = np.tanh(sm_slope * 0.5)
        k_accel = np.tanh(sm_accel * 0.3)
        k_jerk  = np.tanh(sm_jerk * 0.2)
        kinematic_factor = 1.0 + ((k_slope * 0.5 + k_accel * 0.3 + k_jerk * 0.2) * 0.5)
        # [Micro-Burst]
        engine_load = 1.0 + np.maximum(0.0, (pushing_score - 50.0) / 50.0)
        ignition_boost = 1.0 + np.tanh(np.maximum(0, abnormal_vol - 1.0))
        # [Modifiers]
        initial_impulse = 1.0 + np.clip(opening_strength / 100.0, 0.0, 1.0)
        purity_factor = np.clip((1.0 + flow_directionality) / 2.0, 0.1, 1.0)
        conviction_factor = 1.0 + np.maximum(0.0, (mf_conviction - 50.0) / 50.0)
        closing_factor = 1.0 + (closing_strength / 100.0) * 0.5
        breakout_factor = 1.0 + np.clip(breakout_conf / 100.0, 0.0, 1.0)
        micro_boost = 1.0 + np.maximum(0.0, intraday_support * 2.0)
        sentiment_bonus = np.where((base_kinetic > 0) & (market_sentiment < 0.4), 1.0 + (0.4 - market_sentiment) * 1.5, 1.0)
        
        # 计算线性推力 (作为基准)
        linear_thrust = base_kinetic * kinematic_factor * (engine_load * ignition_boost) * initial_impulse * purity_factor * conviction_factor * closing_factor * breakout_factor * micro_boost * sentiment_bonus

        # --- 2. 非线性临界激发 (Non-Linear Critical Excitation) ---
        # 构建“临界共振指数” (Critical Resonance Index, CRI)
        # 逻辑：当 速度(Slope)、推升(Pushing)、点火(Ignition)、纯度(Purity) 同时具备时，CRI 极高。
        # 我们使用归一化后的因子相乘来寻找共振点。
        # items: 
        #   k_slope ( -1 ~ 1 ) -> map to 0 ~ 1: (x+1)/2
        #   pushing ( 0 ~ 100 ) -> map to 0 ~ 1: x/100
        #   purity  ( -1 ~ 1 ) -> map to 0 ~ 1: (x+1)/2
        norm_slope = (k_slope + 1.0) * 0.5
        norm_pushing = np.clip(pushing_score / 100.0, 0.0, 1.0)
        norm_purity = (flow_directionality + 1.0) * 0.5
        norm_ignition = np.tanh(abnormal_vol) # 0 ~ 1
        
        # CRI = 几何平均或乘积。这里使用加权乘积强调“短板效应” (只要有一个不行，整体就不行)
        cri = norm_slope * norm_pushing * norm_purity * norm_ignition
        
        # 激发函数：Sigmoid 变体
        # 当 CRI > 0.15 (经验阈值，意味着各项指标均值 > 0.6) 时，开始非线性放大
        # 放大倍数最大限制为 2.0 倍 (即总推力翻倍)
        # tanh((x - threshold) * sharp)
        # threshold=0.15, sharpness=5.0
        excitation_gain = 1.0 + np.maximum(0.0, np.tanh((cri - 0.15) * 5.0)) * 1.0 
        
        return linear_thrust * excitation_gain

    def _calc_structure_component(self, winner_rate: np.ndarray, control_solidity: np.ndarray, chip_entropy: np.ndarray, chip_stability: np.ndarray, peak_conc: np.ndarray, accumulation_score: np.ndarray, cost_avg: np.ndarray, close: np.ndarray, trend_alignment: np.ndarray, hab_structure: np.ndarray, conc_slope: np.ndarray, winner_accel: np.ndarray, platform_quality: np.ndarray, foundation_strength: np.ndarray) -> np.ndarray:
        """
        【V25.0】维度B：金刚石结构共振模型 (Diamond Structure Resonance)
        核心逻辑：
        1. 引入“金刚石相变”机制。当 有序度、稳定性、平台质量、HAB底仓 四维共振时，结构发生质变。
        2. 方程：FinalStructure = LinearStructure * (1 + Sigmoid(SRI - Threshold))
        """
        # --- 1. Linear Base Structure (线性基础结构 - V24.0逻辑) ---
        # [Static]
        cost_gap = (close - cost_avg) / (cost_avg + 1e-9)
        cost_rbf = np.exp(-10.0 * (cost_gap - 0.05)**2)
        safe_entropy = np.clip(chip_entropy, 0.0, 1.0)
        safe_stability = np.clip(chip_stability, 0.0, 1.0)
        orderliness = (1.0 - safe_entropy) * (0.5 + 0.5 * safe_stability)
        peak_efficiency = peak_conc * winner_rate
        control_factor = 1.0 + np.clip(control_solidity, -0.5, 0.5) * 0.4
        acc_factor = 1.0 + (accumulation_score / 100.0)
        trend_factor = 1.0 + trend_alignment * 0.5
        static_score = orderliness * peak_efficiency * cost_rbf * control_factor * acc_factor * trend_factor

        # [Inertia & Kinematics]
        inertia_bonus = 1.0 + np.maximum(0.0, (hab_structure - 0.6) * 1.25)
        k_conc_slope = np.tanh(conc_slope * 2.0)
        k_winner_accel = np.tanh(winner_accel * 1.0)
        evolution_factor = 1.0 + (k_conc_slope * 0.2 + k_winner_accel * 0.1)

        # [Platform & Foundation]
        norm_platform = np.clip(platform_quality / 100.0, 0.0, 1.0)
        platform_factor = 1.0 + norm_platform * 0.5
        norm_foundation = np.clip(foundation_strength / 100.0, 0.0, 1.0)
        foundation_factor = 1.0 + norm_foundation * 0.3
        
        linear_structure = static_score * inertia_bonus * evolution_factor * platform_factor * foundation_factor

        # --- 2. Diamond Resonance Excitation (金刚石共振激发) ---
        # 构建 SRI (Structural Resonance Index)
        # 选取四大支柱：
        # 1. Norm_Entropy_Inv (有序度): 越低越好 -> 1 - entropy
        # 2. Norm_Stability (锁仓度): 越高越好
        # 3. Norm_Platform (整固度): 越高越好
        # 4. Norm_HAB (底仓深度): 越高越好
        norm_entropy_inv = 1.0 - safe_entropy
        norm_stability = safe_stability
        # norm_platform 已计算
        norm_hab = np.clip(hab_structure, 0.0, 1.0)
        
        # SRI 计算：采用乘积逻辑，强调无短板
        sri = norm_entropy_inv * norm_stability * norm_platform * norm_hab
        
        # 激发函数
        # Threshold = 0.25 (意味着四项均值约为 0.7，即 0.7^4 ≈ 0.24)
        # 只有当 SRI > 0.25 时，开始显著放大，最大放大 2.0 倍
        excitation_gain = 1.0 + np.maximum(0.0, np.tanh((sri - 0.25) * 3.0)) * 1.0
        
        return linear_structure * excitation_gain

    def _calc_drag_component(self, vpa_efficiency: np.ndarray, profit_pressure: np.ndarray, turnover_rate: np.ndarray, trapped_pressure: np.ndarray, dist_score: np.ndarray, intraday_dist: np.ndarray, instability: np.ndarray, hab_inventory: np.ndarray, pressure_release: np.ndarray, shakeout_score: np.ndarray, chip_divergence: np.ndarray, dist_slope: np.ndarray, dist_accel: np.ndarray, dist_jerk: np.ndarray) -> np.ndarray:
        """
        【V29.0】维度C：非线性临界阻力模型 (Non-Linear Critical Drag)
        核心逻辑：引入“阻力相变”机制。当被动负载与主动派发在不稳定环境中对齐时，阻力将产生非线性爆炸。
        方程：FinalDrag = LinearDrag * (1 + NonLinearExcitation) + HiddenDivergence
        """
        # --- 1. Linear Base Drag (线性基础阻力) ---
        passive_load = (np.maximum(0.0, profit_pressure) * 1.5) + (np.maximum(0.0, trapped_pressure) * 2.0)
        norm_dist = np.clip(dist_score / 100.0, 0.0, 1.0)
        norm_intra = np.clip(intraday_dist / 100.0, 0.0, 1.0)
        hab_resistance_base = np.tanh(np.maximum(0.0, -hab_inventory / 100000.0))
        # HAB 缓冲 (存量意识)
        hab_relief = 1.0 / (1.0 + np.maximum(0.0, np.tanh(hab_inventory / 500000.0)) * 1.5)
        active_barrier_score = (norm_dist + norm_intra + hab_resistance_base) * hab_relief
        active_barrier = np.expm1(active_barrier_score) * 2.0
        # 动力学因子
        k_d_slope = np.tanh(dist_slope * 0.4)
        k_d_accel = np.tanh(dist_accel * 0.2)
        k_d_jerk  = np.tanh(dist_jerk * 0.1)
        kinematic_drag_factor = 1.0 + (k_d_slope * 0.4 + k_d_accel * 0.2 + k_d_jerk * 0.1)
        # 粘滞与泄压
        viscosity = 1.0 + (np.clip(instability / 100.0, 0.0, 1.0) * 0.5 + (1.0 - np.clip(vpa_efficiency, 0.0, 1.0)) * 0.5)
        relief_valve = 1.0 + (np.clip(pressure_release / 100.0, 0.0, 1.0) * 1.0 + np.clip(shakeout_score / 100.0, 0.0, 1.0) * 0.5)
        turnover_drag = np.maximum(0.0, turnover_rate - 0.03) * 5.0
        # 计算初步线性阻力
        linear_drag = ((passive_load + active_barrier) * kinematic_drag_factor + turnover_drag) * viscosity / relief_valve
        # --- 2. Non-Linear Critical Excitation (非线性临界激发) [New] ---
        # 构建“阻力共振指数 (DRI)”
        # 选取三个核心负面因子：派发力度、环境不稳、低效率
        norm_instability = np.clip(instability / 100.0, 0.0, 1.0)
        norm_friction = 1.0 - np.clip(vpa_efficiency, 0.0, 1.0)
        # 只有三者在高位对齐时，DRI 才会显著上升 (乘法效应)
        dri = active_barrier_score * norm_instability * norm_friction
        # 非线性增益：当 DRI > 0.2 时，阻力进入“崩溃激发区”
        # 使用 tanh 模拟饱和非线性
        non_linear_gain = 1.0 + np.maximum(0.0, np.tanh((dri - 0.2) * 4.0)) * 2.0
        # --- 3. Synthesis (合成) ---
        hidden_drag = np.maximum(0.0, chip_divergence) * 2.0
        return linear_drag * non_linear_gain + hidden_drag

    def _calc_tensor_synthesis(self, thrust: np.ndarray, structure: np.ndarray, drag: np.ndarray, gap_momentum: np.ndarray, emotional_extreme: np.ndarray, energy_conc: np.ndarray, reversal_prob: np.ndarray, is_leader: np.ndarray, theme_hotness: np.ndarray, lock_ratio: np.ndarray, coordinated_attack: np.ndarray, flow_21d: np.ndarray, flow_55d: np.ndarray, sm_slope_13: np.ndarray, sm_accel_13: np.ndarray, sm_jerk_13: np.ndarray, sm_net_buy: np.ndarray, idx: pd.Index) -> np.ndarray:
        """
        【V33.0】张量合成 - 引入全息共振指数(HRI)与非线性指数增益
        1. 消除零基陷阱与引入动力学跃迁、HAB存量意识。
        2. [新增] 提取全息共振指数(HRI)，捕捉多维物理场极值对齐的瞬间。
        3. [新增] 施加指数级非线性增益(Exponential Resonance Gain)，撕裂妖股与普通票的得分差距。
        """
        energy_damping = np.tanh(np.abs(sm_net_buy) / 10000000.0) * np.clip(energy_conc / 100.0, 0.0, 1.0)
        k_slope = np.tanh(sm_slope_13) * 0.3
        k_accel = np.tanh(sm_accel_13) * 0.3
        k_jerk  = np.tanh(sm_jerk_13) * 0.4
        kinematic_burst = 1.0 + np.maximum(0.0, (k_slope + k_accel + k_jerk) * energy_damping)
        combined_inventory = (flow_21d * 0.6) + (flow_55d * 0.4)
        hab_buffer = np.clip(1.0 - (1.0 / (1.0 + np.exp(combined_inventory / 50000000.0))), 0.0, 0.9)
        norm_theme = np.clip(theme_hotness / 100.0, 0.0, 1.0)
        leader_premium = 1.0 + (is_leader * 0.5) + (norm_theme * 0.2)
        base_tensor = thrust * structure * (1.0 + gap_momentum) * leader_premium * kinematic_burst
        alpha_threshold = 1.5
        raw_effective_drag = drag * (1.0 - hab_buffer)
        squeeze_transition = 1.0 / (1.0 + np.exp(-2.0 * (base_tensor - alpha_threshold * raw_effective_drag)))
        squeeze_bonus = squeeze_transition * raw_effective_drag * emotional_extreme * (energy_conc/100.0) * (1.0 + coordinated_attack) * kinematic_burst
        safe_lock_ratio = np.clip(lock_ratio / 100.0, 0.0, 0.95)
        final_drag = (raw_effective_drag * raw_effective_drag) * (1.0 - squeeze_transition) * (1.0 - reversal_prob) * (1.0 - safe_lock_ratio)
        raw_intent = (base_tensor / (1.0 + final_drag)) + squeeze_bonus
        hri = (base_tensor * (1.0 + squeeze_bonus)) / (1.0 + final_drag)
        hri_threshold = 3.0
        hri_excess = np.clip(hri - hri_threshold, 0.0, 2.5)
        resonance_gain = 1.0 + np.expm1(hri_excess * 1.5)
        final_intent = raw_intent * resonance_gain
        if self._is_probe_enabled():
            print(f"\n[PROBE-SYNTHESIS-V33.0] 非线性指数增益与HRI全息审计...")
            for i in range(len(base_tensor)):
                if hri_excess[i] > 0.0 or np.isnan(final_intent[i]):
                    ts = idx[i].strftime('%Y-%m-%d')
                    print(f"[{ts}] --- V33.0 非线性增益相变探针 ---")
                    print(f"  [RAW INTENT] BaseTensor: {base_tensor[i]:.4f} | SqueezeBonus: {squeeze_bonus[i]:.4f} | FinalDrag: {final_drag[i]:.4f}")
                    print(f"  [RESONANCE HRI] HRI Value: {hri[i]:.4f} (Threshold: {hri_threshold})")
                    print(f"  [EXP GAIN] HRI Excess: {hri_excess[i]:.4f} | Resonance Multiplier: x{resonance_gain[i]:.4f}")
                    print(f"  [OUT] Raw Intent: {raw_intent[i]:.4f} -> Final Amplified Intent: {final_intent[i]:.4f}\n")
        return final_intent

    def _is_probe_enabled(self) -> bool:
        return get_param_value(self.debug_params.get('enabled'), False) and \
               get_param_value(self.debug_params.get('should_probe'), False)

    def _generate_probe_report(self, idx, raw, thrust, structure, drag, raw_intent, final):
        """
        【V33.0】探针终极升级：全息共振、HAB存量与动力学突变透视
        不再只是输出结果，而是通过“反向推演”展示每一个关键物理量对最终意图的贡献。
        """
        if not self.probe_dates: return
        target_dates = pd.to_datetime(self.probe_dates).tz_localize(None).normalize()
        current_dates = idx.tz_localize(None).normalize()
        locs = np.where(current_dates.isin(target_dates))[0]
        if len(locs) == 0: locs = [-1]
        for i in locs:
            ts = idx[i]
            # --- 关键节点重新计算用于回显 ---
            # 动力学与能量阻尼
            net_buy = raw['sm_net_buy'].values[i]
            energy_damping = np.tanh(np.abs(net_buy) / 10000000.0) * np.clip(raw['energy_conc'].values[i] / 100.0, 0.0, 1.0)
            k_burst = 1.0 + max(0.0, (np.tanh(raw['sm_slope_13'].values[i]) * 0.3 + np.tanh(raw['sm_acc_13'].values[i]) * 0.3 + np.tanh(raw['sm_jerk_13'].values[i]) * 0.4) * energy_damping)
            # HAB 免疫力
            comb_inv = (raw['flow_21d'].values[i] * 0.6) + (raw['flow_55d'].values[i] * 0.4)
            hab_imm = np.clip(1.0 - (1.0 / (1.0 + np.exp(comb_inv / 50000000.0))), 0.0, 0.9)
            # 阻力相变
            eff_drag = drag[i] * (1.0 - hab_imm)
            # 共振增益
            hri = (thrust[i] * structure[i] * (1.0 + raw['gap_momentum'].values[i]) * (1.0 + (raw['is_leader'].values[i]*0.5)) * k_burst) / (1.0 + eff_drag)
            res_gain = 1.0 + np.expm1(np.clip(hri - 3.0, 0.0, 2.5) * 1.5)
            report = [
                f"\n=== [PROBE V33.0] CalculateMainForceRallyIntent Holographic Resonance Audit @ {ts.strftime('%Y-%m-%d')} ===",
                f"【A. Kinematics (动力学)】 Burst: x{k_burst:.4f} | Damping: {energy_damping:.4f} | Jerk: {raw['sm_jerk_13'].values[i]:.2f}",
                f"【B. HAB (存量意识)】 21d/55d Inv: {raw['flow_21d'].values[i]:.0f}/{raw['flow_55d'].values[i]:.0f} | Immunity: {hab_imm*100:.1f}%",
                f"【C. Ecosystem (生态)】 Leader: {raw['is_leader'].values[i]} | LockRatio: {raw['lock_ratio'].values[i]:.2f}% | Attack: {raw['coordinated_attack'].values[i]}",
                f"【D. Resonance (共振)】 HRI: {hri:.4f} (Threshold: 3.0) -> Resonance Multiplier: x{res_gain:.4f}",
                f"【E. Synthesis (合成)】 Thrust: {thrust[i]:.4f} | Structure: {structure[i]:.4f} | EffectiveDrag: {eff_drag:.4f}",
                f"【F. Result (最终)】 Raw Intent: {raw_intent[i]:.4f} | Final Normalized Score: {final[i]:.4f}",
                f"===============================================================\n"
            ]
            self._probe_cache.extend(report)
            for line in report: print(line)
















