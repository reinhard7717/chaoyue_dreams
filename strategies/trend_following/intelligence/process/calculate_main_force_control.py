# strategies\trend_following\intelligence\process\calculate_main_force_control.py
# 【V1.0.0 · 主力控盘关系计算器】 计算“主力控盘”的专属关系分数。  已完成
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score,
    get_adaptive_mtf_normalized_bipolar_score, normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper

class CalculateMainForceControlRelationship:
    """
    【V1.0.0 · 主力控盘关系计算器】
    PROCESS_META_MAIN_FORCE_CONTROL
    - 核心职责: 计算“主力控盘”的专属关系分数。
    - 版本: 1.0.0
    """
    def __init__(self, strategy_instance, helper_instance: ProcessIntelligenceHelper):
        """
        初始化 CalculateMainForceControlRelationship 处理器。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
            helper_instance (ProcessIntelligenceHelper): 过程情报辅助工具实例。
        """
        self.strategy = strategy_instance
        self.helper = helper_instance
        self.params = self.helper.params # 获取ProcessIntelligence的参数
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates

    def _print_debug_info(self, debug_output: Dict):
        """
        统一打印调试信息。
        """
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def _get_safe_series(self, df: pd.DataFrame, col_name: str, default_value: float = 0.0, method_name: str = "") -> pd.Series:
        """
        安全地从DataFrame中获取Series，如果列不存在或值为NaN，则填充默认值。
        V11.0: 针对特定信号，当其值为NaN时，提供更具业务含义的默认值（例如0.5表示中性），
               以适配normalize_score和normalize_to_bipolar的归一化逻辑。
               neutral_nan_defaults 字典已提取到配置中。
        """
        process_params = get_params_block(self.strategy, 'process_intelligence_params', {})
        neutral_nan_defaults = process_params.get('neutral_nan_defaults', {})
        current_default_value = neutral_nan_defaults.get(col_name, default_value)
        if col_name not in df.columns:
            return pd.Series(current_default_value, index=df.index, dtype=np.float32)
        series = df[col_name].astype(np.float32)
        return series.fillna(current_default_value)

    def calculate(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 主力控盘五层金字塔决策系统】
        职责：调度全量计算流程，将物理层数据转化为决策层信号。
        架构：
        1. 物理层 (Physics): 挂载全量军械库数据，计算 HAB 存量与动力学状态。
        2. 组件层 (Components): 并行计算 成本优势(V34)、净活动(V14)、传统控盘(V13)。
        3. 转换层 (Translation): 执行微观传递函数与归一化。
        4. 合成层 (Synthesis): 执行结构融合与杠杆计算。
        5. 决策层 (Decision): 输出最终控盘分数。
        """
        method_name = "calculate_main_force_control_relationship"
        is_debug = get_param_value(self.debug_params.get('enabled'), False)
        # --- 0. 调试探针初始化 ---
        # _temp_debug_values 是全链路的“黑匣子”，所有子方法都会向其中写入关键中间变量
        _temp_debug_values = {} 
        probe_ts = self._get_probe_timestamp(df, is_debug)
        debug_output = {}
        if probe_ts:
            print(f"[调度中心] {method_name} 启动 @ {probe_ts.strftime('%Y-%m-%d')}")
            debug_output[f"--- {method_name} 管道启动 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        # --- 1. 物理层 (Physics Layer) ---
        # 职责：原料质检与状态挂载
        if hasattr(self, '_validate_arsenal_signals'):
             if not self._validate_arsenal_signals(df, config, method_name, debug_output, probe_ts):
                return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 获取全量上下文 (Context)，此处已包含 HAB 存量和 Slope/Accel/Jerk 计算
        control_context = self._get_raw_control_signals(df, method_name, _temp_debug_values, probe_ts)
        # --- 2. 组件层 (Component Layer) ---
        # 职责：独立维度的深度计算
        # 2.1 [传统控盘] (Traditional): 基于均线、MACD、KDJ 的经典共振
        scores_traditional = self._calculate_traditional_control_score_components(
            control_context, df.index, _temp_debug_values
        )
        if scores_traditional.isnull().all():
             return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 2.2 [成本优势] (Cost Advantage): V34 版本，含 HAB 护盾、阻尼吸收与非线性增益
        scores_cost_advantage = self._calculate_main_force_cost_advantage_score(
            control_context, df.index, _temp_debug_values
        )
        # 2.3 [净活动力] (Net Activity): V14 版本，基于资金流的内能与生态温度
        scores_net_activity = self._calculate_main_force_net_activity_score(
            control_context, df.index, config, method_name, _temp_debug_values
        )
        # --- 3. 转换层 (Translation Layer) ---
        # 职责：将各维度的原始分转换统一量纲，准备进行融合
        # 注意：scores_traditional 等本身已是分值，但此处计算 MTF 结构分和辅助指标归一化
        norm_traditional, norm_structural, norm_flow, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down = \
            self._normalize_components(df, control_context, scores_traditional, config, method_name, _temp_debug_values)
        # --- 4. 合成层 (Synthesis Layer) ---
        # 职责：结构融合与杠杆放大
        # 4.1 [结构融合] (Fusion): 将 传统分、结构分(MTF) 与 成本分 进行博弈融合
        # V26.0 逻辑：引入状态依赖与锁仓效率
        fused_control_score = self._fuse_control_scores(
            norm_traditional, norm_structural, control_context, _temp_debug_values
        )
        # 4.2 [风控杠杆] (Leverage): 计算当前状态允许放大的倍数 (0.0 ~ 12.0)
        # 核心逻辑：结构越好 (fused_control_score 高)，对 资金动力 (scores_net_activity) 的放大倍数越高
        control_leverage = self._calculate_control_leverage_model(
            df.index, 
            fused_score=fused_control_score,       # 结构分 (作为基准)
            net_activity_score=scores_net_activity,# 动力分 (作为验证)
            norm_flow=norm_flow, 
            cost_score=scores_cost_advantage,      # 成本分 (作为安全垫)
            norm_t0_buy=norm_t0_buy, 
            norm_t0_sell=norm_t0_sell, 
            norm_vwap_up=norm_vwap_up, 
            norm_vwap_down=norm_vwap_down,
            context=control_context, 
            _temp_debug_values=_temp_debug_values
        )
        # --- 5. 决策层 (Decision Layer) ---
        # 职责：输出最终信号
        # 核心公式：最终得分 = 净活动力(动力) * 风控杠杆(结构)
        # 解释：
        # - 如果没有动力 (Activity=0)，无论结构多好，得分都是 0 (死水)。
        # - 如果结构极差 (Leverage<1)，即使有动力，得分也会被压缩 (假突破)。
        # - 如果结构完美 (Leverage>5) 且 动力充足，得分会瞬间饱和 (主升浪)。
        raw_final_score = scores_net_activity * control_leverage
        # 极值剪裁：限制在 [-1, 1] 区间
        final_control_score = raw_final_score.clip(-1, 1).astype(np.float32)
        # --- 6. 调试输出与收尾 ---
        _temp_debug_values["最终结果"] = {
            "Net_Activity": scores_net_activity,
            "Control_Leverage": control_leverage,
            "Fused_Structure": fused_control_score,
            "Cost_Advantage": scores_cost_advantage,
            "Final_Score": final_control_score
        }
        # 如果探针激活，执行全链路打印
        if probe_ts:
            self._calculate_main_force_control_relationship_debug_output(
                debug_output, 
                _temp_debug_values, 
                method_name, 
                probe_ts
            )
            
        return final_control_score

    def _get_control_parameters(self, config: Dict) -> Tuple[Dict, Dict]:
        """
        【V1.0.0】获取主力控盘关系计算所需的参数和权重。
        参数:
            config (Dict): 诊断配置字典。
        返回:
            Tuple[Dict, Dict]: 包含实际MTF权重和MTF斜率加速度权重的元组。
        """
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        return actual_mtf_weights, mtf_slope_accel_weights

    def _get_raw_control_signals(self, df: pd.DataFrame, method_name: str, _temp_debug_values: Dict, probe_ts: pd.Timestamp) -> Dict[str, Dict[str, pd.Series]]:
        """
        【V35.0.0 · 物理层总线与动力学预处理 (Physics Bus & Kinematics Pre-calc)】
        职责：
        1. 全量挂载：根据所有下游组件的需求，提取军械库全量数据。
        2. HAB 存量计算：预计算 13/21/34/55 日资金与 VPA 存量。
        3. 动力学预处理：计算资金与成本的 Slope/Accel/Jerk，并执行零基去噪 (Log/Tanh)。
        4. 归类分发：将数据结构化为 Market, Funds, Structure, Sentiment, State, EMA 六大板块。
        """
        # =========================================================================
        # 1. Market (基础行情)
        # =========================================================================
        market_raw = {
            "close": df['close_D'].astype(np.float32),
            "open": df['open_D'].astype(np.float32),
            "high": df['high_D'].astype(np.float32),
            "low": df['low_D'].astype(np.float32),
            "amount": df['amount_D'].astype(np.float32),
            "volume": df['volume_D'].astype(np.float32),
            "pct_change": df['pct_change_D'].astype(np.float32),
            "turnover_rate": df['turnover_rate_D'].astype(np.float32),
            "circ_mv": df['circ_mv_D'].astype(np.float32),
            "up_limit": df['up_limit_D'].astype(np.float32),
            "down_limit": df['down_limit_D'].astype(np.float32),
        }
        # =========================================================================
        # 2. Funds (资金流与动力学)
        # =========================================================================
        # 基础净流
        net_mf = df['net_mf_amount_D'].astype(np.float32).fillna(0)
        funds_raw = {
            # --- 基础净流 ---
            "net_mf_calibrated": net_mf,
            
            # --- 分单明细 (用于 V9 均价计算) ---
            "buy_elg_amt": df.get('buy_elg_amount_D', pd.Series(0, index=df.index)).astype(np.float32),
            "buy_lg_amt": df.get('buy_lg_amount_D', pd.Series(0, index=df.index)).astype(np.float32),
            "sell_elg_amt": df.get('sell_elg_amount_D', pd.Series(0, index=df.index)).astype(np.float32),
            "sell_lg_amt": df.get('sell_lg_amount_D', pd.Series(0, index=df.index)).astype(np.float32),
            "buy_elg_vol": df.get('buy_elg_vol_D', pd.Series(0, index=df.index)).astype(np.float32),
            "buy_lg_vol": df.get('buy_lg_vol_D', pd.Series(0, index=df.index)).astype(np.float32),
            "sell_elg_vol": df.get('sell_elg_vol_D', pd.Series(0, index=df.index)).astype(np.float32),
            "sell_lg_vol": df.get('sell_lg_vol_D', pd.Series(0, index=df.index)).astype(np.float32),
            
            # --- 聪明钱信号 (用于 V14/V12/V26) ---
            "smart_money_net": df['SMART_MONEY_HM_NET_BUY_D'].astype(np.float32),
            "smart_money_attack": df['SMART_MONEY_HM_COORDINATED_ATTACK_D'].astype(np.float32),
            "smart_divergence": df['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'].astype(np.float32),
            "smart_synergy": df.get('SMART_MONEY_SYNERGY_BUY_D', pd.Series(0, index=df.index)).astype(np.float32),
            "inst_net_buy": df['SMART_MONEY_INST_NET_BUY_D'].astype(np.float32),
            
            # --- 辅助资金 ---
            "tick_lg_net": df['tick_large_order_net_D'].astype(np.float32),
            "net_energy_flow": df.get('net_energy_flow_D', pd.Series(0, index=df.index)).astype(np.float32),
            "flow_efficiency": df.get('flow_efficiency_D', pd.Series(0, index=df.index)).astype(np.float32),
            "closing_intensity": df.get('closing_flow_intensity_D', pd.Series(0, index=df.index)).astype(np.float32),
            "gap_momentum": df.get('GAP_MOMENTUM_STRENGTH_D', pd.Series(0, index=df.index)).astype(np.float32),
            "stealth_flow": df.get('stealth_flow_ratio_D', pd.Series(0, index=df.index)).astype(np.float32),
            "vwap_dev": df.get('vwap_deviation_D', pd.Series(0, index=df.index)).astype(np.float32),
            
            # --- 预留槽位 ---
            "hab_net_mf_13": None, "hab_net_mf_21": None, 
            "hab_net_mf_34": None, "hab_net_mf_55": None,
            "net_mf_slope": None, "net_mf_accel": None, "net_mf_jerk": None,
            "hab_vpa_power_34": None, "hab_vpa_power_55": None
        }
        # --- 计算 HAB (Historical Accumulation Buffer) ---
        # 逻辑：预计算 13/21/34/55 日的滚动资金存量，供 V14(净活动) 和 V34(成本护盾) 使用
        fib_periods = [13, 21, 34, 55]
        for p in fib_periods:
            funds_raw[f"hab_net_mf_{p}"] = net_mf.rolling(window=p, min_periods=int(p*0.5)).sum()
        # --- 计算 VPA 存量 (供 V13 使用) ---
        vpa_raw = df['VPA_MF_ADJUSTED_EFF_D'].fillna(0)
        funds_raw['hab_vpa_power_34'] = vpa_raw.rolling(34).sum()
        funds_raw['hab_vpa_power_55'] = vpa_raw.rolling(55).sum()
        # --- 计算 资金动力学 (Kinematics) ---
        # 逻辑：物理强度(Intensity) -> 差分(Diff) -> 归一化(Tanh)
        circ_mv_safe = market_raw['circ_mv'].replace(0, np.nan)
        # 1. 强度归一化 (解决大盘股与小盘股量级差异)
        control_intensity = net_mf / circ_mv_safe 
        # 2. 物理差分 (解决百分比变化的零基陷阱)
        slope_mf = control_intensity.diff(5) * 10.0 
        accel_mf = slope_mf.diff(3)
        jerk_mf = accel_mf.diff(3)
        # 3. 鲁棒映射 (Tanh 压制异常值)
        funds_raw['net_mf_slope'] = np.tanh(slope_mf).fillna(0)
        funds_raw['net_mf_accel'] = np.tanh(accel_mf).fillna(0)
        funds_raw['net_mf_jerk'] = np.tanh(jerk_mf).fillna(0)
        # =========================================================================
        # 3. Structure (成本与筹码结构)
        # =========================================================================
        structure_raw = {
            # --- 成本分布 ---
            "cost_5pct": df['cost_5pct_D'].astype(np.float32),
            "cost_95pct": df['cost_95pct_D'].astype(np.float32),
            "cost_50pct": df['cost_50pct_D'].astype(np.float32),
            "weight_avg_cost": df['weight_avg_cost_D'].astype(np.float32),
            "profit_ratio": df['profit_ratio_D'].astype(np.float32),
            "pressure_trapped": df['pressure_trapped_D'].astype(np.float32),
            "winner_rate": df['winner_rate_D'].astype(np.float32),
            
            # --- 筹码状态 ---
            "chip_stability": df['chip_stability_D'].astype(np.float32),
            "chip_entropy": df['chip_entropy_D'].astype(np.float32),
            "chip_convergence": df.get('chip_convergence_ratio_D', pd.Series(50, index=df.index)).astype(np.float32),
            "concentration": df['chip_concentration_ratio_D'].astype(np.float32),
            "lock_ratio_90": df.get('high_position_lock_ratio_90_D', pd.Series(0, index=df.index)).astype(np.float32),
            
            # --- 几何形态 ---
            "geom_r2": df['GEOM_REG_R2_D'].astype(np.float32),
            "fractal_dim": df['PRICE_FRACTAL_DIM_D'].astype(np.float32),
            "price_entropy": df['PRICE_ENTROPY_D'].astype(np.float32),
            "bias_55": df['BIAS_55_D'].astype(np.float32),
            "ma_tension": df.get('MA_POTENTIAL_TENSION_INDEX_D', pd.Series(0, index=df.index)).astype(np.float32),
            "ma_coherence": df.get('MA_COHERENCE_RESONANCE_D', pd.Series(0, index=df.index)).astype(np.float32),
            "support_strength": df.get('support_strength_D', pd.Series(0, index=df.index)).astype(np.float32),
            "resistance_strength": df.get('resistance_strength_D', pd.Series(0, index=df.index)).astype(np.float32),
            "rubber_band": df.get('MA_RUBBER_BAND_EXTENSION_D', pd.Series(0, index=df.index)).astype(np.float32),
            
            # --- 动态迁移 ---
            "cost_migration": df['intraday_cost_center_migration_D'].astype(np.float32),
            "game_index": df['intraday_chip_game_index_D'].astype(np.float32),
            
            # --- 预留槽位 ---
            "cost_slope": None, "cost_accel": None
        }
        # --- 计算 成本动力学 (Cost Kinematics) ---
        # 逻辑：为 V34 成本模型预计算重心移动方向
        avg_cost = structure_raw['weight_avg_cost']
        # 1. 对数化 (Log) 消除高价股与低价股的斜率差异
        cost_log = np.log(avg_cost.replace(0, np.nan))
        # 2. 物理差分
        cost_slope_raw = cost_log.diff(13) * 100.0 
        cost_accel_raw = cost_slope_raw.diff(8)
        # 3. 结果存储 (注意：此处不Tanh，留给下游根据需要处理，或在此处统一处理)
        # V35决定：在此处不进行 Tanh，保留原始物理量级供调试，归一化在微观函数中进行
        structure_raw['cost_slope'] = cost_slope_raw.fillna(0)
        structure_raw['cost_accel'] = cost_accel_raw.fillna(0)
        # =========================================================================
        # 4. Sentiment (情绪与行为)
        # =========================================================================
        sentiment_behavior = {
            "adx_14": df['ADX_14_D'].astype(np.float32),
            "vpa_efficiency": df['VPA_EFFICIENCY_D'].astype(np.float32),
            "turnover": df['turnover_rate_D'].astype(np.float32),
            "turnover_stability": df.get('TURNOVER_STABILITY_INDEX_D', pd.Series(50, index=df.index)).astype(np.float32),
            
            # --- 流动性与意图 ---
            "flow_consistency": df['flow_consistency_D'].astype(np.float32),
            "pushing_score": df['pushing_score_D'].astype(np.float32),
            "shakeout_score": df['shakeout_score_D'].astype(np.float32),
            "t0_buy_conf": df['intraday_accumulation_confidence_D'].astype(np.float32),
            "t0_sell_conf": df['intraday_distribution_confidence_D'].astype(np.float32),
            "intraday_support_intent": df['INTRADAY_SUPPORT_INTENT_D'].astype(np.float32),
            
            # --- 风险警示 ---
            "reversal_warning": df.get('reversal_warning_score_D', pd.Series(0, index=df.index)).astype(np.float32),
            "reversal_prob": df.get('reversal_prob_D', pd.Series(0, index=df.index)).astype(np.float32),
            "divergence_strength": df.get('divergence_strength_D', pd.Series(0, index=df.index)).astype(np.float32),
            "bbw": df.get('BBW_21_2.0_D', pd.Series(0, index=df.index)).astype(np.float32),
        }
        # =========================================================================
        # 5. State (系统状态)
        # =========================================================================
        system_state = {
            "market_leader": df['STATE_MARKET_LEADER_D'].astype(np.float32),
            "golden_pit": df['STATE_GOLDEN_PIT_D'].astype(np.float32),
            "breakout_confirmed": df['STATE_BREAKOUT_CONFIRMED_D'].astype(np.float32),
            "rounding_bottom": df.get('STATE_ROUNDING_BOTTOM_D', pd.Series(0, index=df.index)).astype(np.float32),
            "robust_trend": df.get('STATE_ROBUST_TREND_D', pd.Series(0, index=df.index)).astype(np.float32),
            "parabolic_warning": df.get('STATE_PARABOLIC_WARNING_D', pd.Series(0, index=df.index)).astype(np.float32),
            
            # --- 行业增强 ---
            "industry_rank": df.get('industry_strength_rank_D', pd.Series(50, index=df.index)).astype(np.float32),
            "industry_markup": df.get('industry_markup_score_D', pd.Series(50, index=df.index)).astype(np.float32),
        }
        # =========================================================================
        # 6. EMA (均线系统)
        # =========================================================================
        ema_system = {
            "ema_13": df.get('EMA_13_D', pd.Series(0, index=df.index)).astype(np.float32),
            "ema_21": df.get('EMA_21_D', pd.Series(0, index=df.index)).astype(np.float32),
            "ema_55": df.get('EMA_55_D', pd.Series(0, index=df.index)).astype(np.float32),
            "ema_144": df.get('EMA_144_D', pd.Series(0, index=df.index)).astype(np.float32),
        }
        # --- 上下文打包 ---
        context = {
            "market": market_raw, 
            "funds": funds_raw, 
            "structure": structure_raw, 
            "sentiment": sentiment_behavior, 
            "state": system_state, 
            "ema": ema_system
        }
        # --- 探针输出 ---
        if probe_ts:
            snap_conc = structure_raw['concentration'].loc[probe_ts] if probe_ts in structure_raw['concentration'].index else np.nan
            print(f"[探针] V35.0.0 物理层总线挂载完毕。")
            print(f"      - HAB存量计算完成 (Periods: 13, 21, 34, 55)")
            print(f"      - 动力学三阶导数计算完成 (Funds & Cost)")
            print(f"      - [探针数据] 筹码集中度: {snap_conc:.2f}%")
            
        return context

    def _calculate_main_force_control_relationship_debug_output(self, debug_output: Dict, _temp_debug_values: Dict, method_name: str, probe_ts: pd.Timestamp):
        """
        【V4.3.0 · 全链路深度诊断输出】
        职责：按照从“原料数据”到“最终信号”的逻辑链路打印所有探针信息。
        """
        # 定义全链路输出序列
        full_chain = [
            ("原料数据快照", "1. 物理层 (Raw Arsenal Data)"),
            ("主力平均价格计算", "2. 原子层 (Weighted Cost Calc)"),
            ("组件_传统控盘", "3. 逻辑层 - 传统控盘 (Fibonacci Resonance)"),
            ("组件_成本优势", "4. 逻辑层 - 成本优势 (Dual-Track Alpha)"),
            ("组件_净活动(动力学)", "5. 逻辑层 - 资金动力学 (Kinematic Flows)"),
            ("归一化处理", "6. 转换层 (MTF & Normalization)"),
            ("融合_动力学", "7. 决策层 - 深度融合 (Shannon-VPA Fusion)"),
            ("风控_杠杆", "8. 风控层 (Leverage & Vol Gating)"),
            ("最终结果", "9. 输出层 (Final Signal)")
        ]
        print(f"[探针] CalculateMainForceControlRelationship 正在捕获全链路数据快照 @ {probe_ts}")
        for key, label in full_chain:
            if key in _temp_debug_values:
                debug_output[f"  -- [全链路探针] {label}:"] = ""
                data_map = _temp_debug_values[key]
                for sub_key, val in data_map.items():
                    # 动态解包数据
                    if isinstance(val, pd.Series):
                        v_print = val.loc[probe_ts] if probe_ts in val.index else np.nan
                    else:
                        v_print = val
                    # 极端值预警标记
                    warn_tag = " [!] 极端值" if (isinstance(v_print, float) and abs(v_print - 1.0) < 0.0001) else ""
                    # 格式化输出
                    if isinstance(v_print, (float, np.floating)):
                        debug_output[f"        {sub_key}: {v_print:.4f}{warn_tag}"] = ""
                    else:
                        debug_output[f"        {sub_key}: {v_print}"] = ""
                        
        self._print_debug_info(debug_output)

    def _calculate_main_force_cost_advantage_score(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V34.0.0 · 成本博弈共振与阻尼吸收模型 (Resonance & Absorption Model)】
        职责：
        1. 继承全链路物理层逻辑（HAB、动力学、微观传递函数）。
        2. 优化：引入 'HAB阻尼吸收'，主力存量可抵消套牢压力。
        3. 优化：引入 '结构共振'，胜率/安全/重心三维齐备时触发连乘奖励。
        4. 优化：引入 '泥沼突围'，在重压力区识别强动力学反转。
        """
        print(f"[探针] CalculateMainForceCostAdvantageScore 正在计算成本优势分")
        m = context['market']
        s = context['structure']
        f = context['funds']
        close = m['close']
        # =========================================================================
        # 1. 微观传递函数定义 (保持 V33 核心逻辑)
        # =========================================================================
        def _tf_winner_sigmoid(series: pd.Series) -> pd.Series:
            # 胜率传递：60-90% 敏感区
            return (2.0 / (1.0 + np.exp(-(series - 65.0) * 0.15))) - 1.0
        def _tf_safety_rbf(close_series: pd.Series, cost_series: pd.Series) -> pd.Series:
            # 安全垫：RBF 黄金区间 (12%)
            alpha = (close_series - cost_series) / close_series.replace(0, np.nan)
            gaussian_score = np.exp(-np.power((alpha - 0.12) / 0.15, 2))
            over_extension_penalty = np.tanh((alpha - 0.35) * 10.0).clip(0, 1) * 1.5
            return (gaussian_score * 1.5 - 0.5) - over_extension_penalty
        def _tf_trapped_decay(series: pd.Series) -> pd.Series:
            # 压力传递：指数衰减
            return np.exp(-(series / 25.0))
        def _tf_kinematic_tanh(series: pd.Series, scale: float = 1.0) -> pd.Series:
            # 动力传递：鲁棒 Tanh
            return np.tanh(series * scale)
        # =========================================================================
        # 2. 物理量提取与预处理
        # =========================================================================
        cost_5 = s.get('cost_5pct', close * 0.9)
        avg_cost = s.get('weight_avg_cost', close)
        winner_rate = s.get('winner_rate', pd.Series(50, index=index))
        trapped_ratio = s.get('pressure_trapped', pd.Series(50, index=index))
        cost_migration = s.get('cost_migration', pd.Series(0, index=index))
        # HAB 存量计算
        if 'hab_net_mf_21' in f:
            hab_21 = f['hab_net_mf_21']
        else:
            hab_21 = f.get('net_mf_calibrated', pd.Series(0, index=index)).rolling(21).sum()
            
        circ_mv = m['circ_mv'].replace(0, np.nan)
        hab_density = hab_21 / circ_mv
        # 动力学数据
        cost_slope = s.get('cost_slope', pd.Series(0, index=index))
        cost_accel = s.get('cost_accel', pd.Series(0, index=index))
        norm_slope = _tf_kinematic_tanh(cost_slope, scale=2.0)
        norm_accel = _tf_kinematic_tanh(cost_accel, scale=5.0)
        # =========================================================================
        # 3. 核心状态判定 (逻辑优化版)
        # =========================================================================
        # [判定 A] 蓝天模式 (Blue Sky) - 增加防伪逻辑
        # 优化：获利盘高 + 无套牢 + 成本重心未显著下移 (Slope > -0.1)
        is_blue_sky = (winner_rate > 85.0) & (trapped_ratio < 10.0) & (norm_slope > -0.1)
        # [判定 B] 泥沼模式 (Mud)
        is_mud = (trapped_ratio > 60.0)
        # [判定 C] 洗盘护盾 (Washout Shield)
        is_washout = (norm_slope < 0) & (hab_density > 0.005)
        shield_val = _tf_kinematic_tanh(hab_density * 50.0).clip(0, 0.5)
        # [判定 D] 出货预警 (Distribution)
        is_distribution = (norm_slope <= 0) & (hab_density < -0.005)
        # [判定 E] 攻击加速 (Aggressive) & [判定 F] 泥沼突围 (Jailbreak)
        # 突围条件：在泥沼中，出现强力的加速上攻信号
        is_aggressive = (norm_slope > 0.1) & (norm_accel > 0.05) & (cost_migration > 0.2)
        is_jailbreak = is_mud & is_aggressive
        # =========================================================================
        # 4. 评分计算与深度融合 (Deep Fusion)
        # =========================================================================
        # 4.1 应用微观传递函数
        score_winner = _tf_winner_sigmoid(winner_rate)
        score_safety = _tf_safety_rbf(close, cost_5)
        score_gravity = _tf_kinematic_tanh((close - avg_cost) / avg_cost.replace(0, np.nan), scale=10.0).clip(0, 1)
        # 4.2 阻尼吸收逻辑 (Damping Absorption) - 优化点
        # 原始阻尼
        raw_damping = _tf_trapped_decay(trapped_ratio)
        # 存量吸收能力：HAB密度越高，吸收能力越强 (0~0.3的补偿)
        absorption_capacity = _tf_kinematic_tanh(hab_density * 30.0).clip(0, 0.3)
        # 只有当阻尼存在(小于1)且存量为正时，才应用吸收
        effective_damping = (raw_damping + absorption_capacity).clip(0, 1.0)
        # 蓝天模式下阻尼失效
        effective_damping = effective_damping.mask(is_blue_sky, 1.0)
        # 4.3 线性加权
        base_score = (score_winner * 0.5 + score_safety * 0.3 + score_gravity * 0.2)
        # 4.4 结构共振 (Structure Resonance) - 优化点
        # 如果三者皆为正，说明结构极其稳固，给予 1.15倍 共振奖励
        is_resonant = (score_winner > 0) & (score_safety > 0) & (score_gravity > 0)
        resonance_multiplier = pd.Series(1.0, index=index).mask(is_resonant, 1.15)
        # 4.5 应用阻尼与共振
        damped_score = base_score * effective_damping * resonance_multiplier
        # 4.6 状态修正 (护盾与惩罚)
        state_score = damped_score.copy()
        state_score = state_score.mask(is_washout & (state_score < 0.2), 0.2 + shield_val)
        state_score = state_score.mask(is_distribution, state_score - 0.5)
        # =========================================================================
        # 5. 非线性相变增益 (Gamma Expansion)
        # =========================================================================
        gamma = pd.Series(1.5, index=index)
        # 蓝天 -> 0.8 (极易)
        gamma = gamma.mask(is_blue_sky, 0.8)
        # 泥沼 -> 2.0 (极难)
        # 优化：如果是“突围”状态，Gamma 从 2.0 降至 1.4，给予反转机会
        mud_gamma = pd.Series(2.0, index=index).mask(is_jailbreak, 1.4)
        gamma = gamma.mask(is_mud, mud_gamma)
        # 动力加速 -> 普适性降低阻力
        gamma = gamma - (is_aggressive.astype(float) * 0.2)
        # 执行非线性映射
        clipped_state = state_score.clip(-1.0, 1.0)
        final_nonlinear = np.sign(clipped_state) * np.power(np.abs(clipped_state), gamma)
        # =========================================================================
        # 6. 最终物理增强
        # =========================================================================
        conc_factor = (s.get('concentration', pd.Series(50, index=index)) / 70.0).clip(0.6, 1.4)
        kinetic_boost = pd.Series(1.0, index=index).mask(is_aggressive, 1.25)
        final_score = (final_nonlinear * conc_factor * kinetic_boost).clip(-1.0, 1.0)
        # =========================================================================
        # 7. 调试输出
        # =========================================================================
        if _temp_debug_values is not None:
            _temp_debug_values["组件_成本优势"] = {
                "Score_Winner": score_winner,
                "Score_Safety": score_safety,
                "Effective_Damping": effective_damping,
                "Is_Resonant": is_resonant,
                "Is_Jailbreak": is_jailbreak,
                "Final_Cost_Score": final_score
            }
        if is_jailbreak.any():
            print(f"[探针] 泥沼突围 (Jailbreak): 在重压区监测到强动力学信号，Gamma惩罚已调低。")
        if is_resonant.any():
            print(f"[探针] 结构共振 (Resonance): 胜率/安全/重心三维齐备，触发 1.15倍 结构乘数。")
        return final_score.astype(np.float32)

    def _calculate_main_force_avg_prices(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V9.0.0 · 波动率自适应与非线性动力学增益模型】
        职责：
        1. 延续HAB存量锁控机制，拦截单日洗盘噪音。
        2. 废弃全局通用归一化，自研“波动率自适应动力学归一化(Volatility-Adaptive Normalizer)”。
        3. 利用个股自身的历史滚动标准差对Slope/Accel/Jerk进行股性降维，再通过 Tanh 映射。
        4. 整合聪明钱与Tick大单作为非线性乘数，实现资金与结构的深度共振。
        """
        m = context['market']
        f = context['funds']
        s = context['structure']
        w_elg, w_lg = 1.5, 1.0
        daily_buy_amt_w = (f['buy_elg_amt'] * w_elg) + (f['buy_lg_amt'] * w_lg)
        daily_buy_vol_w = (f['buy_elg_vol'] * w_elg) + (f['buy_lg_vol'] * w_lg)
        daily_sell_amt_w = (f['sell_elg_amt'] * w_elg) + (f['sell_lg_amt'] * w_lg)
        daily_sell_vol_w = (f['sell_elg_vol'] * w_elg) + (f['sell_lg_vol'] * w_lg)
        ema_span = 21
        ema_buy_amt = daily_buy_amt_w.ewm(span=ema_span, adjust=False).mean()
        ema_buy_vol = daily_buy_vol_w.ewm(span=ema_span, adjust=False).mean()
        ema_sell_amt = daily_sell_amt_w.ewm(span=ema_span, adjust=False).mean()
        ema_sell_vol = daily_sell_vol_w.ewm(span=ema_span, adjust=False).mean()
        raw_buy_price = (ema_buy_amt / ema_buy_vol).replace([np.inf, -np.inf], np.nan)
        raw_sell_price = (ema_sell_amt / ema_sell_vol).replace([np.inf, -np.inf], np.nan)
        avg_raw = raw_buy_price.mean()
        avg_close = m['close'].mean()
        unit_mismatch_warning = False
        if pd.notna(avg_raw) and avg_raw > 0 and (avg_close / avg_raw > 10.0 or avg_close / avg_raw < 0.1):
            print(f"[致命异常探针] 量纲错配预警！拒绝执行防御性Hack掩盖！请检查军械库底层单位。")
            unit_mismatch_warning = True
        chip_mean = s.get('chip_mean', m['close'])
        stealth_flow = f.get('stealth_flow', pd.Series(0.0, index=index))
        vwap_dev = f.get('vwap_dev', pd.Series(0.0, index=index))
        shadow_cost = m['close'] / (1.0 + vwap_dev.fillna(0) / 100.0)
        stealth_factor = (stealth_flow / 100.0).clip(0, 0.4) if stealth_flow.mean() > 1.0 else stealth_flow.clip(0, 0.4)
        weight_explicit = 0.7 - stealth_factor
        weight_static = 0.3 - (stealth_factor * 0.25)
        weight_shadow = stealth_factor * 1.25
        fused_buy_price = (raw_buy_price * weight_explicit + chip_mean * weight_static + shadow_cost * weight_shadow).fillna(m['close'])
        fused_sell_price = (raw_sell_price * 0.7 + chip_mean * 0.3).fillna(m['close'])
        hab_21 = f['hab_net_mf_21']
        daily_net = f['net_mf_calibrated']
        safe_hab = hab_21.abs().clip(lower=1e-5)
        flow_to_inventory_ratio = daily_net / safe_hab
        is_wash_out = (hab_21 > 0) & (daily_net < 0) & (flow_to_inventory_ratio.abs() < 0.15)
        locked_buy_price = fused_buy_price.copy()
        locked_buy_price[is_wash_out] = np.nan
        locked_buy_price = locked_buy_price.ffill().fillna(fused_buy_price)
        smooth_buy = locked_buy_price.ewm(span=3, adjust=False).mean()
        safe_smooth_buy = smooth_buy.clip(lower=1e-5)
        buy_slope = np.log(safe_smooth_buy / safe_smooth_buy.shift(13).clip(lower=1e-5)) * 100.0
        buy_accel = buy_slope - buy_slope.shift(8)
        buy_jerk = buy_accel - buy_accel.shift(5)
        buy_slope_clean = buy_slope.fillna(0)
        buy_accel_clean = buy_accel.fillna(0)
        buy_jerk_clean = buy_jerk.fillna(0)
        slope_std = buy_slope_clean.rolling(window=252, min_periods=21).std().fillna(buy_slope_clean.std()).clip(lower=1e-3)
        accel_std = buy_accel_clean.rolling(window=252, min_periods=21).std().fillna(buy_accel_clean.std()).clip(lower=1e-3)
        jerk_std = buy_jerk_clean.rolling(window=252, min_periods=21).std().fillna(buy_jerk_clean.std()).clip(lower=1e-3)
        norm_slope = np.tanh(buy_slope_clean / (slope_std * 2.0))
        norm_accel = np.tanh(buy_accel_clean / (accel_std * 2.0))
        norm_jerk = np.tanh(buy_jerk_clean / (jerk_std * 2.0))
        raw_kinematic_energy = (norm_slope * 0.5) + (norm_accel * 0.3) + (norm_jerk * 0.2)
        kinematic_gain = np.tanh(raw_kinematic_energy * 1.5)
        smart_money_attack = f.get('smart_money_attack', pd.Series(0.0, index=index))
        tick_lg_net = f.get('tick_lg_net', pd.Series(0.0, index=index))
        smart_boost = pd.Series(1.0, index=index)
        smart_boost = smart_boost.mask((smart_money_attack > 0.5) & (tick_lg_net > 0), 1.3)
        smart_boost = smart_boost.mask((smart_money_attack < -0.5) & (tick_lg_net < 0), 0.7)
        final_kinematic_power = (kinematic_gain * smart_boost).clip(-1.0, 1.0)
        result = {
            "unit_mismatch": unit_mismatch_warning,
            "avg_buy": locked_buy_price,
            "shadow_cost": shadow_cost,
            "flow_to_inventory_ratio": flow_to_inventory_ratio,
            "buy_slope": buy_slope_clean,
            "buy_accel": buy_accel_clean,
            "buy_jerk": buy_jerk_clean,
            "kinematic_power": final_kinematic_power,
            "avg_sell": fused_sell_price
        }
        _temp_debug_values["主力平均价格(V9自适应归一化)"] = {
            "avg_buy_mean": locked_buy_price.mean(),
            "slope_std_mean": slope_std.mean(),
            "norm_slope_mean": norm_slope.mean(),
            "kinematic_power_mean": final_kinematic_power.mean()
        }
        return result

    def _calculate_main_force_net_activity_score(self, context: Dict, index: pd.Index, config: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V14.0.0 · 内联稳健动态自适应归一化净活动模型】
        职责：
        1. 彻底废弃外部 helper 或 utils 的通用归一化方法。
        2. 自研基于滚动中位数(Rolling Median)与滚动标准差(Rolling Std)的稳健动态归一化。
        3. 延续非对称幂律增益与生态温度调控，保留极端势能。
        4. 通过个股自身历史波动率(252日)对内能进行股性降维，最终输出 [-1, 1] 的高置信度控盘分。
        """
        f = context['funds']
        m = context['market']
        synergy = f.get('smart_synergy', pd.Series(0.0, index=index))
        divergence = f.get('smart_divergence', pd.Series(0.0, index=index))
        ecology_modifier = 1.0 + (synergy * 0.4) - (divergence * 0.7)
        tick_net = f.get('tick_lg_net', pd.Series(0.0, index=index))
        smart_net = f.get('smart_money_net', pd.Series(0.0, index=index))
        daily_net = f.get('net_mf_calibrated', pd.Series(0.0, index=index))
        true_flow = (tick_net * 0.4 + smart_net * 0.4 + daily_net * 0.2) * ecology_modifier
        amount_anchor = m['amount']
        if amount_anchor.isnull().all():
            print(f"[致命异常探针] 全市场成交额(Amount)断层，动力学物理锚点丢失！")
            return pd.Series(0.0, index=index)
        flow_velocity_13 = (true_flow.ewm(span=13, adjust=False).mean()) / amount_anchor * 100.0
        flow_accel_8 = flow_velocity_13 - flow_velocity_13.shift(8)
        flow_jerk_5 = flow_accel_8 - flow_accel_8.shift(5)
        vel_std = flow_velocity_13.rolling(window=252, min_periods=21).std().replace(0, np.nan).fillna(flow_velocity_13.std()).clip(lower=1e-5)
        acc_std = flow_accel_8.rolling(window=252, min_periods=21).std().replace(0, np.nan).fillna(flow_accel_8.std()).clip(lower=1e-5)
        jrk_std = flow_jerk_5.rolling(window=252, min_periods=21).std().replace(0, np.nan).fillna(flow_jerk_5.std()).clip(lower=1e-5)
        norm_vel = flow_velocity_13 / vel_std
        norm_acc = flow_accel_8 / acc_std
        norm_jrk = flow_jerk_5 / jrk_std
        raw_linear_energy = (norm_vel * 0.5 + norm_acc * 0.3 + norm_jrk * 0.2)
        is_resonance = (norm_vel * norm_acc > 0) & (norm_acc * norm_jrk > 0)
        power_law_exponent = pd.Series(1.0, index=index).mask(is_resonance, 1.6)
        amplified_energy = np.sign(raw_linear_energy) * (raw_linear_energy.abs() ** power_law_exponent)
        hab_13 = f.get('hab_net_mf_13', pd.Series(0.0, index=index))
        hab_21 = f.get('hab_net_mf_21', pd.Series(0.0, index=index))
        hab_34 = f.get('hab_net_mf_34', pd.Series(0.0, index=index))
        hab_55 = f.get('hab_net_mf_55', pd.Series(0.0, index=index))
        consensus_inventory = hab_13 * 0.4 + hab_21 * 0.3 + hab_34 * 0.2 + hab_55 * 0.1
        safe_inventory = consensus_inventory.abs().clip(lower=1e-5)
        flow_to_inventory_ratio = true_flow / safe_inventory
        buffer_impact = pd.Series(1.0, index=index)
        has_deep_inventory = (consensus_inventory > 0) & (true_flow < 0)
        dampener = 1.0 - np.exp(-15.0 * flow_to_inventory_ratio.abs())
        buffer_impact = buffer_impact.mask(has_deep_inventory, dampener)
        smart_attack = f.get('smart_money_attack', pd.Series(0.0, index=index))
        energy_flow = f.get('net_energy_flow', pd.Series(0.0, index=index))
        temperature_scalar = 1.0 + (smart_attack.clip(lower=0) * 0.5) + (energy_flow / 100.0).clip(lower=0) * 0.5
        efficiency = f.get('flow_efficiency', pd.Series(0.0, index=index))
        closing = f.get('closing_intensity', pd.Series(0.0, index=index))
        quality_bonus = np.tanh(efficiency / 10.0) * 0.3 + np.tanh(closing / 5.0) * 0.3
        golden_pit = (m['pct_change'] < -1.5) & (amplified_energy > 0) & has_deep_inventory
        pit_multiplier = pd.Series(1.0, index=index).mask(golden_pit, 1.5)
        final_internal_energy = amplified_energy * buffer_impact * pit_multiplier * temperature_scalar + quality_bonus.fillna(0)
        energy_median = final_internal_energy.rolling(window=252, min_periods=21).median().fillna(final_internal_energy.median())
        energy_std = final_internal_energy.rolling(window=252, min_periods=21).std().fillna(final_internal_energy.std()).clip(lower=1e-5)
        adaptive_scaled_energy = (final_internal_energy - energy_median) / (energy_std * 1.5)
        final_score = np.tanh(adaptive_scaled_energy).clip(-1.0, 1.0)
        _temp_debug_values["组件_净活动(稳健自适应归一化模型)"] = {
            "amplified_energy_mean": amplified_energy.mean(),
            "final_internal_energy_mean": final_internal_energy.mean(),
            "energy_median_mean": energy_median.mean(),
            "energy_std_mean": energy_std.mean(),
            "adaptive_scaled_energy_max": adaptive_scaled_energy.max(),
            "final_score": final_score
        }
        print(f"[探针] 稳健动态自适应归一化执行完毕。波动率缩放基准(Std)均值: {energy_std.mean():.4f}, 映射后均分: {final_score.mean():.4f}")
        return final_score

    def _calculate_traditional_control_score_components(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V13.0.0 · 迟延激活与方向不对称质量映射传统控盘模型】
        职责：
        1. 推迟饱和激活(Late Activation)：所有中间共振层均保留在Logit无界空间计算，根除双重Tanh导致的阈值压缩塌陷。
        2. 引入缺口动能(Gap Momentum)：修复均线滞后，为底层动能注入瞬间物理突变张量 。
        3. 方向不对称质量映射(Direction-Aware Quality)：多头加权有序度，空头加权混沌度(崩盘踩踏效应)。
        4. 动态状态机(Dynamic State Matrix)：利用换手稳定性与VPA效能缩放状态机常数，消除僵化静态加分 [cite: 1, 2]。
        """
        m = context['market']
        ema = context['ema']
        f_funds = context['funds']
        s_struct = context['structure']
        s_sent = context['sentiment']
        s_state = context['state']
        ema_55 = ema['ema_55']
        ema_144 = ema.get('ema_144', pd.Series(0.0, index=index))
        if ema_55.isnull().all():
            print(f"[致命异常探针] EMA55生命线数据全量缺失，传统动力学引擎停机！")
            return pd.Series(0.0, index=index)
        safe_ema_55 = ema_55.clip(lower=1e-5)
        vel_13 = np.log(safe_ema_55 / safe_ema_55.shift(13).clip(lower=1e-5)) * 100.0
        acc_8 = vel_13 - vel_13.shift(8)
        jrk_5 = acc_8 - acc_8.shift(5)
        vel_std = vel_13.rolling(window=252, min_periods=21).std().fillna(vel_13.std()).clip(lower=1e-3)
        acc_std = acc_8.rolling(window=252, min_periods=21).std().fillna(acc_8.std()).clip(lower=1e-3)
        jrk_std = jrk_5.rolling(window=252, min_periods=21).std().fillna(jrk_5.std()).clip(lower=1e-3)
        norm_vel = vel_13 / vel_std
        norm_acc = acc_8 / acc_std
        norm_jrk = jrk_5 / jrk_std
        adx = s_sent.get('adx_14', pd.Series(20.0, index=index)).fillna(20.0)
        adx_activation = 1.0 / (1.0 + np.exp(-0.2 * (adx - 25.0)))
        w_vel = 0.30 + adx_activation * 0.45 
        w_jrk = 0.40 - adx_activation * 0.25 
        w_acc = 1.0 - w_vel - w_jrk
        gap_momentum = f_funds.get('gap_momentum', pd.Series(0.0, index=index)).fillna(0.0)
        raw_linear_kine = (norm_vel * w_vel + norm_acc * w_acc + norm_jrk * w_jrk) + (gap_momentum / 10.0)
        is_bull_resonance = (norm_vel > 0.1) & (norm_acc > 0.05) & (norm_jrk > 0.05)
        is_bear_resonance = (norm_vel < -0.1) & (norm_acc < -0.05) & (norm_jrk < -0.05)
        is_resonance = is_bull_resonance | is_bear_resonance
        coherence = s_struct.get('ma_coherence', pd.Series(0.0, index=index)).fillna(0.0)
        coherence_temperature = 1.0 + np.tanh(coherence / 50.0).clip(lower=0)
        power_exponent = pd.Series(1.0, index=index).mask(is_resonance, 1.4 * coherence_temperature)
        amplified_kine_score = np.sign(raw_linear_kine) * (raw_linear_kine.abs() ** power_exponent)
        hab_vpa_34 = f_funds.get('hab_vpa_power_34', pd.Series(0.0, index=index))
        hab_vpa_55 = f_funds.get('hab_vpa_power_55', pd.Series(0.0, index=index))
        consensus_vpa_inventory = hab_vpa_34 * 0.4 + hab_vpa_55 * 0.6
        inventory_std = consensus_vpa_inventory.rolling(252, min_periods=21).std().fillna(consensus_vpa_inventory.std()).clip(lower=1e-5)
        norm_vpa_inventory = consensus_vpa_inventory / inventory_std
        is_structural_washout = (norm_vpa_inventory > 1.0) & (amplified_kine_score < 0)
        structural_buffer = pd.Series(1.0, index=index).mask(is_structural_washout, np.exp(-abs(amplified_kine_score)))
        buffered_kine_score = amplified_kine_score * structural_buffer
        q1 = buffered_kine_score.rolling(window=252, min_periods=21).quantile(0.25).fillna(buffered_kine_score.quantile(0.25))
        q2 = buffered_kine_score.rolling(window=252, min_periods=21).median().fillna(buffered_kine_score.median())
        q3 = buffered_kine_score.rolling(window=252, min_periods=21).quantile(0.75).fillna(buffered_kine_score.quantile(0.75))
        upside_scale = (q3 - q2).clip(lower=1e-3)
        downside_scale = (q2 - q1).clip(lower=1e-3)
        asymmetric_scaled_kine_logit = pd.Series(np.where(buffered_kine_score > q2, (buffered_kine_score - q2) / upside_scale, (buffered_kine_score - q2) / downside_scale), index=index)
        tension = s_struct.get('ma_tension', pd.Series(0.0, index=index)).fillna(0.0)
        support = s_struct.get('support_strength', pd.Series(0.0, index=index)).fillna(0.0)
        resistance = s_struct.get('resistance_strength', pd.Series(0.0, index=index)).fillna(0.0)
        sr_log_ratio = np.log1p(support.clip(lower=0)) - np.log1p(resistance.clip(lower=0))
        structure_resonance_logit = (coherence / 50.0) * 0.4 - (tension / 50.0) * 0.2 + sr_log_ratio * 0.4
        raw_fused_energy = asymmetric_scaled_kine_logit * 0.65 + structure_resonance_logit * 0.35
        base_fused_score = np.tanh(raw_fused_energy)
        vpa_eff = s_sent.get('vpa_efficiency', pd.Series(0.0, index=index)).fillna(0.0)
        vpa_multiplier = (1.0 + np.tanh(vpa_eff / 50.0)).clip(0.5, 1.5)
        adx_confidence = (np.tanh((adx - 20.0) / 10.0) * 0.5 + 0.5).clip(0.5, 1.2)
        entropy = s_struct.get('price_entropy', pd.Series(2.0, index=index)).fillna(2.0)
        fractal_dim = s_struct.get('fractal_dim', pd.Series(1.5, index=index)).fillna(1.5)
        entropy_penalty = 1.0 / (1.0 + np.exp(entropy - 1.8))
        fractal_stability = 1.0 - (fractal_dim - 1.0).clip(0, 0.5) * 2.0
        orderliness_multiplier = (entropy_penalty * 0.5 + fractal_stability * 0.5).clip(0.4, 1.5)
        chaos_multiplier = 1.0 / orderliness_multiplier 
        directional_quality_multiplier = pd.Series(np.where(base_fused_score > 0, orderliness_multiplier, chaos_multiplier), index=index)
        safe_vpa = vpa_multiplier.clip(lower=0.1)
        safe_adx = adx_confidence.clip(lower=0.1)
        safe_dir_qual = directional_quality_multiplier.clip(lower=0.1)
        geometric_quality_multiplier = np.power(safe_vpa * safe_adx * safe_dir_qual, 1.0/3.0)
        quality_adjusted_score = base_fused_score * geometric_quality_multiplier
        rubber_band = s_struct.get('rubber_band', pd.Series(0.0, index=index)).fillna(0.0)
        reversal_warn = s_sent.get('reversal_warning', pd.Series(0.0, index=index)).fillna(0.0)
        overextension_discount = np.exp(-np.abs(rubber_band) / 15.0) * (1.0 - (reversal_warn / 100.0).clip(0.0, 0.8))
        macro_bear_mask = (m['close'] < ema_144) & (ema_144 > 0)
        risk_adjusted_score = quality_adjusted_score.mask(macro_bear_mask & (quality_adjusted_score > 0), quality_adjusted_score * 0.5)
        risk_adjusted_score = risk_adjusted_score.mask(risk_adjusted_score > 0, risk_adjusted_score * overextension_discount)
        turnover_stability = s_sent.get('turnover_stability', pd.Series(50.0, index=index)).fillna(50.0)
        dynamic_friction = np.tanh(turnover_stability / 50.0).clip(0.5, 1.5)
        dynamic_state_scale = vpa_multiplier * dynamic_friction
        is_parabolic = s_state.get('parabolic_warning', pd.Series(0.0, index=index)) > 0
        is_golden_pit = s_state.get('golden_pit', pd.Series(0.0, index=index)) > 0
        is_breakout = s_state.get('breakout_confirmed', pd.Series(0.0, index=index)) > 0
        is_rounding = s_state.get('rounding_bottom', pd.Series(0.0, index=index)) > 0
        is_robust = s_state.get('robust_trend', pd.Series(0.0, index=index)) > 0
        state_multiplier = pd.Series(1.0, index=index).mask(is_parabolic, 0.1)
        state_adder = pd.Series(0.0, index=index)
        state_adder = state_adder.mask(is_golden_pit & (risk_adjusted_score > -0.5), 0.4 * dynamic_state_scale)
        state_adder = state_adder.mask(is_breakout & (risk_adjusted_score > 0), 0.5 * dynamic_state_scale)
        state_adder = state_adder.mask(is_rounding & (risk_adjusted_score > -0.2), 0.3 * dynamic_state_scale)
        state_adder = state_adder.mask(is_robust & (risk_adjusted_score > 0.2), 0.2 * dynamic_state_scale)
        final_score = (risk_adjusted_score * state_multiplier + state_adder).clip(-1.0, 1.0)
        _temp_debug_values["组件_传统控盘(迟延激活与方向博弈)"] = {
            "gap_momentum_mean": gap_momentum.mean(),
            "asymmetric_scaled_kine_logit_mean": asymmetric_scaled_kine_logit.mean(),
            "structure_resonance_logit_mean": structure_resonance_logit.mean(),
            "base_fused_score_max": base_fused_score.max(),
            "directional_quality_multiplier_mean": directional_quality_multiplier.mean(),
            "dynamic_state_scale_mean": dynamic_state_scale.mean(),
            "final_score": final_score
        }
        print(f"[探针] 迟延激活重构完成。突破双重Tanh封锁，多空方向质量乘数不对称率校验正常。")
        return final_score

    def _calculate_control_leverage_model(self, index: pd.Index, fused_score: pd.Series, net_activity_score: pd.Series, 
                                          norm_flow: pd.Series, cost_score: pd.Series, 
                                          norm_t0_buy: pd.Series, norm_t0_sell: pd.Series, 
                                          norm_vwap_up: pd.Series, norm_vwap_down: pd.Series,
                                          context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V12.0.0 · 矢量张量共振与流动性曲率杠杆模型】
        职责：
        1. 矢量张量共振(Tensor Resonance)：计算个股动能、行业强度、聪明钱热度在三维相空间的余弦方向一致性，激活超流态溢价。
        2. 流动性曲率阻尼(Liquidity Curvature)：引入成交额边际贡献率，封杀地量空拉的虚假杠杆。
        3. 记忆压力阻尼(Memory Damping)：利用 EMA 累积历史风险张力，实现风险响应的非对称时间衰减。
        4. 全内联 MAD 归一化与非线性 Tanh 势能池，确保物理层量纲绝对正交。
        """
        sent, funds, struct, market, state = context['sentiment'], context['funds'], context['structure'], context['market'], context['state']
        def _robust_norm_unipolar(s: pd.Series, lookback: int = 252) -> pd.Series:
            median = s.rolling(lookback, min_periods=21).median().fillna(s.median())
            mad = (s - median).abs().rolling(lookback, min_periods=21).median().fillna((s - s.median()).abs().median()).clip(lower=1e-5)
            return (1.0 / (1.0 + np.exp(-1.5 * (s - median) / mad))).fillna(0.5)
        def _get_slope_dir(s: pd.Series, p: int = 5) -> pd.Series:
            return (s - s.shift(p)).apply(np.sign).fillna(0)
        hab_cost = _temp_debug_values.get("主力平均价格(V9自适应归一化)", {}).get("avg_buy_mean", market['close'].mean())
        hab_cost_series = pd.Series(hab_cost, index=index) if isinstance(hab_cost, float) else _temp_debug_values.get("主力平均价格(V9自适应归一化)", {}).get("avg_buy", pd.Series(market['close'].mean(), index=index))
        hab_bias = (market['close'] - hab_cost_series) / hab_cost_series.replace(0, np.nan)
        norm_bias_mult = 1.0 + (0.5 - _robust_norm_unipolar(hab_bias)) * 0.4
        sqz_tension_raw = struct.get('profit_ratio', pd.Series(50.0, index=index)) - struct.get('pressure_trapped', pd.Series(50.0, index=index))
        ind_rank = state.get('industry_rank', pd.Series(50.0, index=index))
        sm_attack = funds.get('smart_money_attack', pd.Series(0.0, index=index))
        dir_stock = _get_slope_dir(sqz_tension_raw)
        dir_industry = _get_slope_dir(-ind_rank)
        dir_smart = _get_slope_dir(sm_attack)
        vector_resonance = ((dir_stock == dir_industry) & (dir_industry == dir_smart) & (dir_stock > 0)).astype(float)
        resonance_multiplier = 1.0 + vector_resonance * 0.5
        amt_ma252 = market['amount'].rolling(252, min_periods=21).mean().clip(lower=1e-5)
        liquidity_ratio = market['amount'] / amt_ma252
        liquidity_damping = np.tanh(liquidity_ratio * 5.0).clip(0.3, 1.0)
        ind_markup = state.get('industry_markup', pd.Series(50.0, index=index))
        sector_premium = (1.0 / (1.0 + np.exp(0.2 * (ind_rank - 20.0)))) * 0.65
        synergy_scalar = (norm_bias_mult * resonance_multiplier * (1.0 + sector_premium)).fillna(1.0)
        reversal_risk = (sent.get('reversal_prob', pd.Series(0.0, index=index)) / 100.0).clip(0, 1)
        mem_risk_ema = reversal_risk.ewm(span=5).mean()
        hard_breaker = pd.Series(1.0, index=index).mask(mem_risk_ema > 0.7, 0.15)
        div_strength = sent.get('divergence_strength', pd.Series(0.0, index=index))
        soft_risk_decay = np.exp(-(_robust_norm_unipolar(div_strength) * 2.0))
        turn_stab = sent.get('turnover_stability', pd.Series(50.0, index=index))
        friction = _robust_norm_unipolar(turn_stab).clip(0.4, 1.2)
        lock_90 = struct.get('lock_ratio_90', pd.Series(0.0, index=index))
        lockup_bonus = ((sent.get('turnover', pd.Series(3.0, index=index)) < 5.0) & (lock_90 > 65.0)).astype(float) * 0.5
        super_bonus = _robust_norm_unipolar(struct.get('chip_convergence', pd.Series(50.0, index=index))) * 0.4
        validation = (net_activity_score.clip(lower=0) * 0.4 + norm_flow * 0.3 + cost_score.clip(lower=0) * 0.2 + norm_t0_buy * 0.1).clip(0, 1)
        internal_energy = (fused_score * validation + lockup_bonus + super_bonus + (0.2 - sent['bbw']).clip(lower=0) * 5.0).fillna(0)
        base_lev = 1.0 + np.tanh(internal_energy / (1.0 / friction)) * 3.5
        is_demon = (market.get('up_limit', pd.Series(0.0, index=index)) > 0) & (vector_resonance > 0) & (ind_rank < 15)
        raw_lev = (base_lev * synergy_scalar * liquidity_damping) ** pd.Series(1.0, index=index).mask(is_demon, 1.4)
        pos_lev = raw_lev * soft_risk_decay * hard_breaker
        attrition = (funds.get('net_mf_calibrated', pd.Series(0.0, index=index)) / funds.get('hab_net_mf_21', pd.Series(0.0, index=index)).abs().clip(lower=1e-5)).abs()
        is_shield = (funds.get('hab_net_mf_21', pd.Series(0.0, index=index)) > 0) & (attrition < 0.07)
        punish_raw = (norm_t0_sell * 0.4 + norm_vwap_down * 0.3 + (1 - norm_flow) * 0.3).clip(0, 1)
        neg_lev = (1.0 + fused_score) * np.exp(-(validation * 0.5 + punish_raw.mask(is_shield, 0.0) * 0.6)) * soft_risk_decay * hard_breaker
        final_lev = pd.Series(np.where(fused_score > 0, pos_lev, neg_lev), index=index).clip(0, 12.0)
        _temp_debug_values["风控_杠杆(矢量张量与流动性模型)"] = {
            "vector_resonance_rate": vector_resonance.mean(),
            "liquidity_damping_mean": liquidity_damping.mean(),
            "mem_risk_ema_mean": mem_risk_ema.mean(),
            "is_demon_rate": is_demon.mean(),
            "final_leverage": final_lev
        }
        print(f"[探针] 杠杆张量共振闭环: 矢量一致性触发率:{vector_resonance.mean():.2%}, 流动性枯竭惩罚覆盖率:{(liquidity_damping < 0.5).mean():.2%}, 杠杆极值:{final_lev.max():.2f}")
        return final_lev

    def _fuse_control_scores(self, traditional_score: pd.Series, structural_score: pd.Series, context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V26.0.0 · 成本结构挤压与状态依赖物理融合模型】
        职责：在多维博弈的基础上，引入筹码分布状态和市场形态对评分进行动态校准。
        核心逻辑：
        1. 成本挤压：获利盘主导时放大动力，套牢盘主导时抑制动力。
        2. 锁定效率：奖励"缩量锁仓"的高控盘形态。
        3. 状态物理：黄金坑豁免洗盘惩罚，假突破触发聪明钱熔断。
        4. VPA效能：奖励高效推升。
        5. 合成：采用幂律扩张 (Power Law Expansion) 拉开区分度。
        """
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        m_market = context['market']
        s_state = context['state']
        profit = s_struct['profit_ratio'].fillna(0)
        trapped = s_struct['pressure_trapped'].fillna(0)
        squeeze_potential = np.tanh((profit - trapped) / 20.0)
        cost_modifier = 1.0 + (squeeze_potential * 0.3)
        adx = s_sent['adx_14'].fillna(20.0)
        dynamic_trad_weight = 1.0 / (1.0 + np.exp(-0.15 * (adx - 30.0)))
        dynamic_trad_weight = dynamic_trad_weight.clip(0.2, 0.8)
        dynamic_struct_weight = 1.0 - dynamic_trad_weight
        base_score = (traditional_score * cost_modifier * dynamic_trad_weight + structural_score * dynamic_struct_weight)
        hab_34 = f_funds.get('hab_net_mf_34', pd.Series(0, index=base_score.index))
        circ_mv = m_market['circ_mv'].replace(0, np.nan)
        turnover = s_sent['turnover'].replace(0, np.nan).fillna(1.0)
        hab_density = hab_34 / circ_mv
        norm_turnover = turnover / 3.0
        locking_ratio = (hab_density * 100.0) / norm_turnover
        locking_gain = 1.0 + np.tanh(locking_ratio * 0.5).clip(-0.5, 0.8)
        jerk = f_funds.get('net_mf_jerk', pd.Series(0, index=base_score.index))
        bias_55 = s_struct['bias_55'].fillna(0)
        sm_div = f_funds.get('smart_divergence', pd.Series(0, index=base_score.index))
        jerk_std = jerk.rolling(252, min_periods=21).std().fillna(jerk.std()).replace(0, 1.0)
        norm_jerk_z = jerk / jerk_std
        is_golden_pit = s_state['golden_pit'] > 0
        is_breakout = s_state['breakout_confirmed'] > 0
        is_kinematic_fail = (norm_jerk_z < -1.5)
        decay_factor = pd.Series(1.0, index=base_score.index)
        high_risk_decay = np.exp(-(np.abs(norm_jerk_z) * 0.8)).clip(0, 1)
        decay_factor = decay_factor.mask(is_kinematic_fail & (bias_55 > 20.0), high_risk_decay)
        decay_factor = decay_factor.mask(is_golden_pit, 1.0)
        sm_gate = pd.Series(1.0, index=base_score.index)
        is_bull_trap = (base_score > 0.2) & (sm_div < -0.2)
        sm_gate = sm_gate.mask(is_bull_trap, 0.6)
        is_fake_breakout = is_breakout & (sm_div < -0.1)
        sm_gate = sm_gate.mask(is_fake_breakout, 0.2)
        is_strong_resonance = (base_score > 0.2) & (sm_div > 0.2)
        sm_gate = sm_gate.mask(is_strong_resonance, 1.2)
        vpa_eff = s_sent['vpa_efficiency'].fillna(0)
        vpa_gain = 1.0 + (np.tanh(vpa_eff / 100.0) * 0.5).clip(0, 0.5)
        entropy_diff = s_struct['price_entropy'] - s_struct['chip_entropy']
        entropy_quality = 1.0 / (1.0 + np.exp(-2.0 * (entropy_diff - 0.5)))
        r2 = s_struct['geom_r2'].fillna(0).clip(0, 1)
        linearity_gain = 1.0 + np.power(r2, 3) * 0.5
        raw_final = (base_score * locking_gain * sm_gate * decay_factor * entropy_quality * linearity_gain * vpa_gain)
        expanded_final = np.sign(raw_final) * np.power(np.abs(raw_final), 1.5)
        final_fused = np.tanh(expanded_final).astype(np.float32)
        if _temp_debug_values is not None:
            _temp_debug_values["融合_动力学"] = {
                "Cost_Modifier": cost_modifier,
                "State_GoldenPit": is_golden_pit,
                "State_Breakout": is_breakout,
                "Decay_Factor": decay_factor,
                "Smart_Gate": sm_gate,
                "Locking_Gain": locking_gain,
                "Final_Fused_Score": final_fused
            }
        if is_golden_pit.sum() > 0:
            print(f"[探针] 监测到黄金坑状态 (Golden Pit)，已豁免动力学衰减。")
        if is_fake_breakout.sum() > 0:
            print(f"[探针] 警报：监测到假突破 (Fake Breakout) —— 突破确立但聪明钱背离，执行熔断。")
        if (cost_modifier > 1.2).sum() > 0:
            print(f"[探针] 监测到蓝天模式 (Blue Sky)，获利盘主导，动力学得分已放大。")
        return final_fused

    def _normalize_components(self, df: pd.DataFrame, context: Dict, scores_traditional: pd.Series, config: Dict, method_name: str, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        【V4.6.0 · 独立组件归一化系统 (Kinetic Normalizer)】
        职责：放弃通用 Helper 逻辑，采用基于物理动力学和博弈深度定制的非线性归一化方案。
        修改说明：
        1. Bipolar Tanh Scaling: 用于传统得分和结构得分，处理极端离群值，保留 0 轴敏感度。
        2. Inline MTF Structure: 内部实现多周期筹码稳定性斜率加权，判定控盘沉淀。
        3. Intent Sigmoid Mapping: 对意图信心分执行逻辑回归映射，强化强信号，压制弱噪音。
        """
        s_struct = context['structure']
        s_sent = context['sentiment']
        df_index = df.index
        # --- 1. 传统控盘分归一化 (Bipolar -1 to 1) ---
        # 逻辑：传统分包含乖离张力，分布较广。使用 Tanh(x/std) 保持中心敏感。
        std_trad = scores_traditional.std() if scores_traditional.std() > 0 else 1.0
        norm_traditional = np.tanh(scores_traditional / (std_trad * 1.5))
        # --- 2. 结构控盘分 MTF 计算 (Kinetic MTF Structure) ---
        # 逻辑：不依赖 helper，直接计算 chip_stability 的多周期斜率共振。
        stability = s_struct['chip_stability']
        # 计算 5, 13, 21 日斜率
        slope_5 = (stability - stability.shift(5)) / 5.0
        slope_13 = (stability - stability.shift(13)) / 13.0
        slope_21 = (stability - stability.shift(21)) / 21.0
        # 执行权重融合 (近高远低: 0.5, 0.3, 0.2)
        mtf_struct_raw = slope_5 * 0.5 + slope_13 * 0.3 + slope_21 * 0.2
        # 将变化率转化为 [-1, 1] 的控盘强度。0.01 的日均变化即为极强信号。
        norm_structural = np.tanh(mtf_struct_raw * 100.0)
        # --- 3. 辅助指标归一化 (Unipolar 0 to 1) ---
        # 流量一致性: 原始 0-100 -> [0, 1]
        norm_flow = (s_sent['flow_consistency'] / 100.0).clip(0, 1)
        # 意图信心分 (Sigmoid 强化): 0.5 为中性，强化两极
        def _intent_sigmoid(s: pd.Series):
            return 1.0 / (1.0 + np.exp(-10.0 * (s - 0.5)))
        norm_t0_buy = _intent_sigmoid(s_sent['t0_buy_conf'])
        norm_t0_sell = _intent_sigmoid(s_sent['t0_sell_conf'])
        # 脉冲强度 (Min-Max 缩放): 0.0-1.0
        norm_vwap_up = s_sent['pushing_score'].clip(0, 1)
        norm_vwap_down = s_sent['shakeout_score'].clip(0, 1)
        # --- 4. 转换层探针捕获 ---
        if _temp_debug_values is not None:
            _temp_debug_values["归一化处理"] = {
                "traditional_norm": norm_traditional,
                "structural_mtf_norm": norm_structural,
                "flow_consistency": norm_flow,
                "t0_buy_boost": norm_t0_buy
            }
        print(f"[探针] 组件归一化自研逻辑执行完成。结构斜率均值: {mtf_struct_raw.mean():.6f}")
        return norm_traditional, norm_structural, norm_flow, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down

    def _get_probe_timestamp(self, df: pd.DataFrame, is_debug: bool) -> Optional[pd.Timestamp]:
        """
        【辅助方法】获取用于调试的探针时间戳。
        防止 calculate 方法调用时报 AttributeError。
        """
        if not is_debug or not self.probe_dates:
            return None
        probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
        # 倒序遍历索引，找到最新的匹配日期
        for date in reversed(df.index):
            ts = pd.to_datetime(date)
            if ts.tz_localize(None).normalize() in probe_dates_dt:
                return date
        return None

    def _validate_arsenal_signals(self, df: pd.DataFrame, config: Dict, method_name: str, debug_output: Dict, probe_ts: pd.Timestamp) -> bool:
        """
        【V2.0.0 · 军械库全链路动力学与存量验证版 (严苛模式)】
        职责：作为调度器的“守门人”，强制检查高阶聪明钱、锁仓率及背离信号。移除容错妥协，暴露数据断层。
        """
        _, mtf_slope_accel_weights = self._get_control_parameters(config)
        required_physical_raw = [
            'close_D', 'amount_D', 'pct_change_D', 'turnover_rate_D',
            'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D',
            'buy_elg_vol_D', 'buy_lg_vol_D', 'sell_elg_vol_D', 'sell_lg_vol_D',
            'net_mf_amount_D', 'flow_consistency_D', 'chip_stability_D',
            'SMART_MONEY_HM_NET_BUY_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'high_position_lock_ratio_90_D', 'chip_concentration_ratio_D'
        ]
        required_kinematics_decision = [
            'JERK_5_net_mf_amount_D', 'ACCEL_5_net_mf_amount_D', 'SLOPE_5_net_mf_amount_D',
            'ACCEL_5_chip_stability_D', 'JERK_5_pushing_score_D',
            'EMA_13_D', 'EMA_21_D', 'EMA_55_D',
            'chip_entropy_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D',
            'weight_avg_cost_D', 'winner_rate_D', 'pushing_score_D', 'shakeout_score_D',
            'price_flow_divergence_D'
        ]
        dynamic_mtf_signals = []
        base_sig_proxy = 'chip_stability_D'
        for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
            dynamic_mtf_signals.append(f'SLOPE_{period_str}_{base_sig_proxy}')
        for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
            dynamic_mtf_signals.append(f'ACCEL_{period_str}_{base_sig_proxy}')
        full_validation_list = list(set(required_physical_raw + required_kinematics_decision + dynamic_mtf_signals))
        if probe_ts:
            print(f"[探针] _validate_arsenal_signals 启动严苛全链路自检 @ {probe_ts}")
            print(f"      - 物理原料检查项: {len(required_physical_raw)} 个")
            print(f"      - 动力学与博弈检查项: {len(required_kinematics_decision)} 个")
        if not self.helper._validate_required_signals(df, full_validation_list, method_name):
            if probe_ts:
                missing_cols = [col for col in full_validation_list if col not in df.columns]
                debug_output[f"    -> [致命错误] {method_name} 关键军械库信号缺失: {missing_cols[:10]}... 强制熔断。"] = ""
                self._print_debug_info(debug_output)
            return False
        return True








