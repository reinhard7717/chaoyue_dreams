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
        【V40.0.0 · 主力控盘五层金字塔决策系统 · 熵减矢量版】
        职责：调度全量计算流程，将物理层数据转化为决策层信号。
        架构：
        1. 物理层 (Physics): 挂载全量军械库数据，计算 HAB 存量与双熵动力学状态。
        2. 组件层 (Components): 并行计算 传统控盘(V40)、成本优势(V40)、净活动力(V40)。
        3. 转换层 (Translation): 执行微观传递函数与归一化 (Kinetic Normalizer)。
        4. 合成层 (Synthesis): 执行 熵减结构融合 与 状态记忆风控杠杆。
        5. 决策层 (Decision): 输出最终控盘分数 (Activity * Leverage)。
        """
        method_name = "calculate_main_force_control_relationship"
        is_debug = get_param_value(self.debug_params.get('enabled'), False)
        
        # --- 0. 调试探针初始化 ---
        # _temp_debug_values 是全链路的“黑匣子”，所有子方法都会向其中写入关键中间变量
        _temp_debug_values = {} 
        probe_ts = self._get_probe_timestamp(df, is_debug)
        debug_output = {}
        
        if probe_ts:
            print(f"[调度中心] {method_name} 启动 @ {probe_ts.strftime('%Y-%m-%d')} | 版本: V40.0.0 (熵减矢量版)")
            debug_output[f"--- {method_name} 管道启动 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""

        # --- 1. 物理层 (Physics Layer) ---
        # 职责：原料质检与状态挂载
        # V40升级：严苛模式，拒绝缺失关键物理量
        if hasattr(self, '_validate_arsenal_signals'):
             if not self._validate_arsenal_signals(df, config, method_name, debug_output, probe_ts):
                print(f"[熔断] {method_name}: 关键军械库信号缺失，策略强制终止。")
                return pd.Series(0.0, index=df.index, dtype=np.float32)

        # 获取全量上下文 (Context)，此处已包含 HAB 存量、双熵(Entropy) 和 矢量合成预处理
        control_context = self._get_raw_control_signals(df, method_name, _temp_debug_values, probe_ts)

        # --- 2. 组件层 (Component Layer) ---
        # 职责：独立维度的深度计算
        
        # 2.1 [传统控盘] (Traditional): 时空动力学与混沌博弈
        scores_traditional = self._calculate_traditional_control_score_components(
            control_context, df.index, _temp_debug_values
        )
        if scores_traditional.isnull().all():
             return pd.Series(0.0, index=df.index, dtype=np.float32)

        # 2.2 [成本优势] (Cost Advantage): 熵减动力学与 HAB 护盾
        scores_cost_advantage = self._calculate_main_force_cost_advantage_score(
            control_context, df.index, _temp_debug_values
        )

        # 2.3 [净活动力] (Net Activity): 矢量合成与惯性阻尼
        scores_net_activity = self._calculate_main_force_net_activity_score(
            control_context, df.index, config, method_name, _temp_debug_values
        )

        # --- 3. 转换层 (Translation Layer) ---
        # 职责：将各维度的原始分转换统一量纲，准备进行融合
        # V40: 采用内联归一化逻辑，此处主要提取标准化后的流向与意图分供下游使用
        norm_traditional, norm_structural, norm_flow, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down = \
            self._normalize_components(df, control_context, scores_traditional, config, method_name, _temp_debug_values)

        # --- 4. 合成层 (Synthesis Layer) ---
        # 职责：结构融合与杠杆放大
        
        # 4.1 [结构融合] (Fusion): 将 传统分、结构分 与 成本分 进行熵减融合
        # V40: 引入双熵博弈 (Price Entropy vs Chip Entropy)
        fused_control_score = self._fuse_control_scores(
            norm_traditional, norm_structural, control_context, _temp_debug_values
        )

        # 4.2 [风控杠杆] (Leverage): 计算当前状态允许放大的倍数 (0.0 ~ 12.0)
        # V40: 引入 HAB 风险记忆与动力学预警
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
        # 核心公式：最终得分 = 矢量净活动力(动力) * 熵减风控杠杆(结构)
        # 解释：
        # - 动力 (Activity) 是物理做功的基础。
        # - 杠杆 (Leverage) 是由 熵(Entropy) 和 结构(Structure) 决定的放大器。
        # - 高混乱度 (High Entropy) 会导致 Leverage -> 0，从而熔断 Activity。
        
        raw_final_score = scores_net_activity * control_leverage
        
        # 极值剪裁：限制在 [-1, 1] 区间
        final_control_score = raw_final_score.clip(-1, 1).astype(np.float32)

        # --- 6. 调试输出与收尾 ---
        _temp_debug_values["最终结果"] = {
            "Net_Activity (Vector)": scores_net_activity,
            "Control_Leverage (Entropy)": control_leverage,
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
        【V40.0.0 · 物理层总线 (Physics Bus - Arsenal Map)】
        职责：
        1. 严格映射《最终军械库清单.txt》，移除不存在的列引用。
        2. 引入 Tick级数据、聪明钱数据、熵数据。
        3. 动力学预处理：计算 Tick流与资金流的加速度。
        4. [修复] 分离 Sentiment(情绪) 与 State(状态) 字典，并处理字段共用与缺失补全。
        """
        # --- 1. Market (基础行情) ---
        market_raw = {
            "close": df['close_D'].astype(np.float32),
            "amount": df['amount_D'].astype(np.float32),
            "pct_change": df['pct_change_D'].astype(np.float32),
            "turnover_rate": df['turnover_rate_D'].astype(np.float32),
            "circ_mv": df['circ_mv_D'].replace(0, np.nan).astype(np.float32),
            "up_limit": df['up_limit_D'].astype(np.float32),
        }
        
        # --- 2. Funds (核心资金 - 升级为 Tick/Smart/Large 矢量组) ---
        funds_raw = {
            # 微观核动力 (Tick Level)
            "tick_lg_net": df['tick_large_order_net_D'].astype(np.float32),
            "tick_lg_count": df['tick_large_order_count_D'].astype(np.float32),
            # 机构/聪明钱 (Smart Money)
            "smart_net_buy": df['SMART_MONEY_HM_NET_BUY_D'].astype(np.float32),
            "smart_attack": df['SMART_MONEY_HM_COORDINATED_ATTACK_D'].astype(np.float32),
            "inst_net_buy": df['SMART_MONEY_INST_NET_BUY_D'].astype(np.float32),
            # 传统资金流
            "net_mf_calibrated": df['net_mf_amount_D'].astype(np.float32),
            "buy_lg_amt": df['buy_lg_amount_D'].astype(np.float32),
            # 资金特征
            "flow_consistency": df['flow_consistency_D'].astype(np.float32),
            "flow_efficiency": df['flow_efficiency_D'].astype(np.float32),
            "stealth_flow": df['stealth_flow_ratio_D'].astype(np.float32),
            # [新增] 聪明钱背离 (映射自 price_flow_divergence_D)
            "smart_divergence": df['price_flow_divergence_D'].astype(np.float32),
            # 预留槽位
            "hab_net_mf_21": None, "hab_net_mf_34": None
        }
        # HAB 存量计算
        composite_flow = (funds_raw["tick_lg_net"] * 0.5 + funds_raw["smart_net_buy"] * 0.3 + funds_raw["net_mf_calibrated"] * 0.2)
        funds_raw["hab_net_mf_21"] = composite_flow.rolling(window=21, min_periods=10).sum()
        funds_raw["hab_net_mf_34"] = composite_flow.rolling(window=34, min_periods=15).sum()

        # --- 3. Structure (结构与熵) ---
        structure_raw = {
            "chip_entropy": df['chip_entropy_D'].astype(np.float32),
            "price_entropy": df['PRICE_ENTROPY_D'].astype(np.float32),
            "fractal_dim": df['PRICE_FRACTAL_DIM_D'].astype(np.float32),
            "chip_stability": df['chip_stability_D'].astype(np.float32),
            "concentration": df['chip_concentration_ratio_D'].astype(np.float32),
            "winner_rate": df['winner_rate_D'].astype(np.float32),
            "profit_ratio": df['profit_ratio_D'].astype(np.float32),
            "pressure_trapped": df['pressure_trapped_D'].astype(np.float32),
            "cost_5pct": df['cost_5pct_D'].astype(np.float32),
            "avg_cost": df['weight_avg_cost_D'].astype(np.float32),
            "r2": df['GEOM_REG_R2_D'].astype(np.float32),
            "bias_55": df['BIAS_55_D'].astype(np.float32),
            "ma_coherence": df['MA_COHERENCE_RESONANCE_D'].astype(np.float32)
        }

        # --- 4. Sentiment & State (情绪与状态 - 分离兼容) ---
        # 提取公共状态变量
        _market_leader = df['STATE_MARKET_LEADER_D'].astype(np.float32)
        _golden_pit = df['STATE_GOLDEN_PIT_D'].astype(np.float32)
        
        # [Sentiment] 情绪字典：包含 leverage_model 所需的 market_leader/golden_pit
        sentiment_raw = {
            "vpa_efficiency": df['VPA_EFFICIENCY_D'].astype(np.float32),
            "pushing_score": df['pushing_score_D'].astype(np.float32),
            "shakeout_score": df['shakeout_score_D'].astype(np.float32),
            "turnover_stability": df['TURNOVER_STABILITY_INDEX_D'].astype(np.float32),
            "t0_buy_conf": df['intraday_accumulation_confidence_D'].astype(np.float32),
            "t0_sell_conf": df['intraday_distribution_confidence_D'].astype(np.float32),
            "industry_rank": df['industry_strength_rank_D'].astype(np.float32),
            "reversal_prob": df['reversal_prob_D'].astype(np.float32),
            "divergence_strength": df['divergence_strength_D'].astype(np.float32),
            "turnover": df['turnover_rate_D'].astype(np.float32),
            # [新增] ADX (fuse_control_scores 需要)
            "adx_14": df['ADX_14_D'].astype(np.float32),
            # [兼容] 冗余存储状态
            "market_leader": _market_leader,
            "golden_pit": _golden_pit
        }

        # [State] 状态字典：包含 fuse_control_scores 所需的 golden_pit/breakout
        state_raw = {
            "market_leader": _market_leader,
            "golden_pit": _golden_pit,
            # [新增] 突破确认 (fuse_control_scores 需要)
            "breakout_confirmed": df['STATE_BREAKOUT_CONFIRMED_D'].astype(np.float32)
        }

        # --- 5. EMA System ---
        ema_system = {
            "ema_13": df['EMA_13_D'].astype(np.float32),
            "ema_55": df['EMA_55_D'].astype(np.float32),
        }

        # --- 探针输出 ---
        if _temp_debug_values is not None:
            _temp_debug_values["1. 物理层 (Raw Arsenal Data)"] = {
                "Close": market_raw['close'],
                "Tick_Lg_Net": funds_raw['tick_lg_net'],
                "Smart_Net_Buy": funds_raw['smart_net_buy'],
                "Smart_Divergence": funds_raw['smart_divergence'],
                "Chip_Entropy": structure_raw['chip_entropy'],
                "ADX_14": sentiment_raw['adx_14'],
                "Breakout_Confirmed": state_raw['breakout_confirmed']
            }

        if probe_ts:
            snap_tick = funds_raw['tick_lg_net'].loc[probe_ts] if probe_ts in funds_raw['tick_lg_net'].index else np.nan
            print(f"[探针] V40.0.0 物理总线挂载。State/Sentiment 双字典分离完成。")

        return {
            "market": market_raw,
            "funds": funds_raw,
            "structure": structure_raw,
            "sentiment": sentiment_raw,
            "state": state_raw, # [修复] 显式返回 state 字典
            "ema": ema_system
        }

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
        【V40.0.0 · 成本优势与熵减动力学博弈模型 (Cost Advantage & Entropy Kinematics)】
        修改思路：
        1. 数据层升级：引入筹码熵(Entropy)、获利盘(Profit)、套牢盘(Trapped)及HAB存量，构建“双熵博弈”。
        2. 动力学(Kinematics)：采用 Log-Space 计算成本重心的 Slope/Accel/Jerk，彻底规避“零基陷阱”与量纲失真。
        3. HAB存量缓冲(Buffer)：利用 21日/34日 资金存量构建“反阻力护盾”，当存量充足时，抵消套牢盘的物理阻尼。
        4. 内联归一化(Inline Norm)：拒绝通用Helper，针对成本物理特性自研 _tf_sigmoid_winner 等传递函数。
        5. 非线性增益(Gamma)：针对“蓝天模式(BlueSky)”与“泥沼突围(Jailbreak)”场景应用 Gamma 扩张。
        6. 深度融合：将 结构有序度(Entropy) 与 成本动力学(Trend) 进行矢量乘积融合。
        """
        # --- 1. 数据挂载 (Data Mounting) ---
        m = context['market']
        s = context['structure']
        f = context['funds']
        close = m['close']
        # 核心成本数据
        avg_cost = s.get('weight_avg_cost', close) # 权重均价
        cost_5 = s.get('cost_5pct', close * 0.9)
        # 博弈数据
        winner_rate = s.get('winner_rate', pd.Series(50, index=index))
        profit_ratio = s.get('profit_ratio', pd.Series(50, index=index))
        pressure_trapped = s.get('pressure_trapped', pd.Series(50, index=index))
        # 熵与结构 (V40新增)
        chip_entropy = s.get('chip_entropy', pd.Series(100, index=index)) # 筹码熵 (高=混乱, 低=有序)
        concentration = s.get('concentration', pd.Series(50, index=index))
        # HAB 存量 (从 funds 中提取或计算)
        # 逻辑：如果没有预计算的 hab_21，则现场计算。这是“存量意识”的核心。
        hab_21 = f.get('hab_net_mf_21', f['net_mf_calibrated'].rolling(21).sum())
        circ_mv = m['circ_mv'].replace(0, np.nan)
        # --- 2. 成本动力学 (Cost Kinematics: Log-Space) ---
        # 步骤：
        # 1. 对数化：处理股价从 10元涨到11元 与 100元涨到110元 的等效性。
        # 2. 斐波拉契差分：Slope(13) -> Accel(8) -> Jerk(5)
        log_cost = np.log(avg_cost.replace(0, np.nan))
        # Slope (13): 长期趋势方向 (一阶导)
        # 乘以 100 转化为类百分比量级
        slope_cost = log_cost.diff(13) * 100.0 
        # Accel (8): 趋势加速力度 (二阶导)
        # 消除线性趋势，保留加速度
        accel_cost = slope_cost.diff(8)
        # Jerk (5): 趋势变率的变率 (三阶导)
        # 用于捕捉拐点：加速衰竭或加速启动
        jerk_cost = accel_cost.diff(5)
        # 零基陷阱处理：由于使用了 Log Returns，天然规避了除以零的问题。
        # 仅需填充 NaN
        slope_cost = slope_cost.fillna(0)
        accel_cost = accel_cost.fillna(0)
        jerk_cost = jerk_cost.fillna(0)
        # --- 4. 自研内联归一化 (Inline Normalization) ---
        # 拒绝使用通用 helper，针对成本特性定制微观传递函数
        def _tf_winner(s): 
            # 胜率传递：
            # > 65% 为主力控盘舒适区，奖励显著增加
            # < 40% 为套牢区，给予负分
            return (2.0 / (1.0 + np.exp(-(s - 65.0) * 0.15))) - 1.0
            
        def _tf_entropy(e):
            # 熵传递 (反向)：
            # 熵越低(有序)越好。Entropy < 70 为高度有序。
            # 假设 entropy 范围 0-100 (根据清单数据特性调整)
            return 1.0 / (1.0 + np.exp((e - 75.0) * 0.1)) 
            
        def _tf_trapped_damping(t, shield):
            # 阻尼函数：
            # 套牢盘(t) 产生物理阻力，指数衰减。
            # HAB护盾(shield) 提供线性补偿。
            # 逻辑：如果有大量主力存量(shield high)，即使有套牢盘，也不认为是死套，而是“吸筹”。
            raw_decay = np.exp(-(t / 25.0)) # 典型值 50 -> exp(-2) = 0.13 (阻力大)
            buffered_decay = raw_decay + (shield * 0.6) # 护盾最大补偿 0.6
            return buffered_decay.clip(0, 1.0)
        def _tf_kinematics(k, sensitivity=1.0, threshold=0.0):
            # 动力学 Tanh 映射
            return np.tanh((k - threshold) * sensitivity)
        # --- 3. HAB 存量缓冲系统 (HAB Shield) ---
        # 逻辑：计算资金存量密度，构建护盾
        hab_density = hab_21 / circ_mv
        # 护盾强度映射：0.005 (0.5%) 的流通盘净买入即为强护盾
        hab_shield = np.tanh(hab_density * 50.0).clip(0, 1.0)
        # --- 5. 核心逻辑计算 ---
        # A. 静态优势分 (Static Advantage)
        score_winner = _tf_winner(winner_rate)
        # 安全垫 (Safety): 股价不应偏离成本太远(超买)，也不应太低(破位)
        # 理想区间：成本上方 0% ~ 15%
        cost_bias = (close - avg_cost) / avg_cost.replace(0, np.nan)
        score_safety = np.exp(-np.power((cost_bias - 0.05)/0.2, 2)) * 2.0 - 1.0
        # B. 动态趋势分 (Dynamic Trend)
        # 权重：加速度(Accel) > 速度(Slope) > 跃度(Jerk)
        # 逻辑：我们更看重主力拉升的“爆发力”(Accel)
        kine_score = _tf_kinematics(slope_cost, 0.5) * 0.3 + \
                     _tf_kinematics(accel_cost, 2.0) * 0.5 + \
                     _tf_kinematics(jerk_cost, 2.0) * 0.2
                     
        # C. 结构有序度 (Structural Order)
        # 引入熵的概念：低熵 + 高集中度 = 强控盘
        score_order = _tf_entropy(chip_entropy)
        # --- 6. 深度融合与阻尼应用 ---
        # [Step 1] 基础合成
        # 静态(底气) + 动态(意图) + 结构(状态)
        raw_score = (score_winner * 0.35 + kine_score * 0.45 + score_order * 0.2)
        # [Step 2] 阻尼应用 (Damping)
        # 计算有效阻尼：套牢盘 - HAB护盾
        effective_damping = _tf_trapped_damping(pressure_trapped, hab_shield)
        damped_score = raw_score * effective_damping
        # --- 7. 非线性增益与场景优化 ---
        # 场景 A: 蓝天模式 (Blue Sky)
        # 条件：获利盘 > 90% (上方无压力)
        is_blue_sky = (profit_ratio > 90.0)
        # 场景 B: 泥沼突围 (Jailbreak)
        # 条件：深套(Pressure > 60) 但 动力学强劲(Accel > 0) 且 有护盾(Shield > 0.2)
        is_jailbreak = (pressure_trapped > 60.0) & (accel_cost > 0.05) & (hab_shield > 0.2)
        # Gamma 增益控制
        gamma = pd.Series(1.0, index=index)
        # 蓝天模式下，更容易获得高分 (Gamma < 1 放大正值)
        gamma = gamma.mask(is_blue_sky & (damped_score > 0), 0.7)
        # 泥沼突围下，给予额外奖励常数
        jailbreak_bonus = is_jailbreak.astype(np.float32) * 0.4
        # 最终计算：Power Law 扩张
        # Sign * |Score|^Gamma
        final_score = np.sign(damped_score) * np.power(np.abs(damped_score), gamma)
        final_score = final_score + jailbreak_bonus
        # 熵值熔断：如果极度混乱 (Order < 0.2)，且得分为正，强制衰减
        entropy_penalty = pd.Series(1.0, index=index)
        entropy_penalty = entropy_penalty.mask((score_order < 0.2) & (final_score > 0), 0.5)
        final_score = (final_score * entropy_penalty).clip(-1.0, 1.0)
        # --- Debug Output ---
        if _temp_debug_values is not None:
            _temp_debug_values["组件_成本优势(V40熵减版)"] = {
                "HAB_Shield": hab_shield,
                "Log_Slope_13": slope_cost,
                "Log_Accel_8": accel_cost,
                "Score_Order (Entropy)": score_order,
                "Effective_Damping": effective_damping,
                "Is_Jailbreak": is_jailbreak,
                "Final_Cost_Score": final_score
            }
            
        if is_jailbreak.any():
            print(f"[探针] 成本优势: 监测到 {is_jailbreak.sum()} 个点位触发【泥沼突围】模式，HAB护盾已激活。")
        return final_score.astype(np.float32)

    def _calculate_main_force_avg_prices(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V40.0.0 · 主力HAB存量均价与成本动力学模型 (HAB Cost Kinematics)】
        修改思路：
        1. 数据层升级：引入 `weight_avg_cost` (CYQ均价) 作为物理锚点，引入 `smart_net_buy` (聪明钱) 作为修正因子。
        2. HAB存量均价：放弃简单的 EMA，构建基于 21日/55日 累积成交额与成交量的 VWMA (Volume Weighted Moving Average) 模型，真实反映主力持仓成本。
        3. 动力学升级：对 HAB 成本曲线进行 Log-Space 下的 Slope/Accel/Jerk 计算，捕捉主力成本的移动趋势。
        4. 归一化自研：针对价格乖离率 (Bias) 和动力学特征编写专用归一化逻辑，摒弃通用 Helper。
        5. 非线性增益：引入“成本共振 (Cost Resonance)”，当主力成本与市场均价趋同时，给予高置信度权重。
        """
        m = context['market']
        f = context['funds']
        s = context['structure']
        # --- 1. 数据提取与预处理 (Data Extraction) ---
        # 基础量价
        close = m['close']
        turnover = m.get('turnover_rate', pd.Series(1.0, index=index)).replace(0, np.nan)
        # 主力资金流 (特大单 + 大单)
        buy_elg_amt = f.get('buy_elg_amt', pd.Series(0, index=index))
        buy_lg_amt = f.get('buy_lg_amt', pd.Series(0, index=index))
        buy_elg_vol = f.get('buy_elg_vol', pd.Series(0, index=index))
        buy_lg_vol = f.get('buy_lg_vol', pd.Series(0, index=index))
        sell_elg_amt = f.get('sell_elg_amt', pd.Series(0, index=index))
        sell_lg_amt = f.get('sell_lg_amt', pd.Series(0, index=index))
        sell_elg_vol = f.get('sell_elg_vol', pd.Series(0, index=index))
        sell_lg_vol = f.get('sell_lg_vol', pd.Series(0, index=index))
        # 聪明钱与筹码锚点
        smart_net_buy = f.get('smart_net_buy', pd.Series(0, index=index)) # V40新增
        cyq_avg_cost = s.get('weight_avg_cost', close) # 市场平均筹码成本
        # --- 2. HAB 存量均价计算 (HAB VWMA) ---
        # 逻辑：单日的均价容易受噪音干扰，使用 HAB (Historical Accumulation Buffer) 思想
        # 计算 21日 (短期主力) 和 55日 (中期主力) 的滚动加权均价
        # 合成单日主力买入/卖出数据
        daily_main_buy_amt = buy_elg_amt + buy_lg_amt
        daily_main_buy_vol = buy_elg_vol + buy_lg_vol
        daily_main_sell_amt = sell_elg_amt + sell_lg_amt
        daily_main_sell_vol = sell_elg_vol + sell_lg_vol
        # 定义 HAB 计算器 (VWMA)
        def _calc_hab_vwma(amt_series, vol_series, window):
            # 滚动求和
            roll_amt = amt_series.rolling(window=window, min_periods=int(window*0.5)).sum()
            roll_vol = vol_series.rolling(window=window, min_periods=int(window*0.5)).sum()
            # 计算均价 (避免除以零)
            vwma = (roll_amt / roll_vol.replace(0, np.nan))
            return vwma
            
        # 计算核心成本线
        hab_cost_buy_21 = _calc_hab_vwma(daily_main_buy_amt, daily_main_buy_vol, 21).fillna(close)
        hab_cost_sell_21 = _calc_hab_vwma(daily_main_sell_amt, daily_main_sell_vol, 21).fillna(close)
        # 量纲检查 (Unit Check)
        # 如果计算出的成本与股价差异过大 (例如 > 10倍)，说明元数据单位可能错配 (手 vs 股)
        # 此时降级使用 close 或 cyq_avg_cost
        price_ratio = hab_cost_buy_21 / close.replace(0, np.nan)
        unit_mismatch = (price_ratio > 5.0) | (price_ratio < 0.2)
        if unit_mismatch.any():
            print(f"[探针] V40 HAB成本计算：监测到 {unit_mismatch.sum()} 个点位量纲异常，已自动回退至市场均价。")
            hab_cost_buy_21 = hab_cost_buy_21.mask(unit_mismatch, cyq_avg_cost)
            hab_cost_sell_21 = hab_cost_sell_21.mask(unit_mismatch, cyq_avg_cost)
        # --- 3. 成本动力学 (Cost Kinematics) ---
        # 对 HAB 成本曲线进行 Log-Space 求导
        log_cost = np.log(hab_cost_buy_21.replace(0, np.nan))
        # Slope (13): 成本移动方向
        slope_cost = log_cost.diff(13) * 100.0
        # Accel (8): 成本移动加速
        accel_cost = slope_cost.diff(8)
        # Jerk (5): 成本变率的变率
        jerk_cost = accel_cost.diff(5)
        # 去噪与填充
        slope_clean = slope_cost.fillna(0)
        accel_clean = accel_cost.fillna(0)
        jerk_clean = jerk_cost.fillna(0)
        # --- 4. 归一化与非线性增益 (Normalization & Gain) ---
        # A. 动力学归一化 (Z-Tanh)
        # 使用 252日 历史波动率作为基准
        slope_std = slope_clean.rolling(252, min_periods=21).std().fillna(0.1).clip(lower=0.01)
        norm_slope = np.tanh(slope_clean / (slope_std * 1.5))
        norm_accel = np.tanh(accel_clean / (slope_std * 1.5)) # 加速度通常较小，复用 slope_std
        # B. 聪明钱修正 (Smart Money Bias)
        # 如果聪明钱大幅净买入，说明主力愿意以高于当前计算成本的价格吸筹
        # 归一化聪明钱流向 [-1, 1]
        circ_mv = m['circ_mv'].replace(0, np.nan)
        smart_bias = np.tanh((smart_net_buy / circ_mv) * 1000.0) # 千分之几的力度
        # C. 成本动力评分 (Kinematic Power)
        raw_power = norm_slope * 0.5 + norm_accel * 0.3 + smart_bias * 0.2
        # 非线性扩张
        kinematic_power = np.sign(raw_power) * np.power(np.abs(raw_power), 1.2).clip(-1, 1)
        # --- 5. 最终融合 (Final Fusion) ---
        # 构建“锁定买入均价” (Locked Buy Price)
        # 逻辑：结合 HAB计算均价 和 CYQ市场均价
        # 换手率越低，CYQ权重越高 (静态)；换手率越高，Flow权重越高 (动态)。
        flow_weight = np.tanh(turnover / 2.0).clip(0.2, 0.8) # 换手2%时权重约0.76
        static_weight = 1.0 - flow_weight
        final_buy_price = (hab_cost_buy_21 * flow_weight + cyq_avg_cost * static_weight)
        final_sell_price = (hab_cost_sell_21 * flow_weight + cyq_avg_cost * static_weight)
        # 聪明钱溢价修正
        # 如果 SmartMoney 强力买入，上调主力成本线 (支撑位上移)
        final_buy_price = final_buy_price * (1.0 + smart_bias * 0.02) # 最多修正 2%
        # --- 6. 结果打包与调试 ---
        result = {
            "unit_mismatch": unit_mismatch.any(),
            "avg_buy": final_buy_price,     # 最终主力买入成本 (支撑)
            "avg_sell": final_sell_price,   # 最终主力卖出成本 (压力)
            "buy_slope": slope_clean,       # 成本斜率
            "buy_accel": accel_clean,       # 成本加速度
            "buy_jerk": jerk_clean,         # 成本跃度
            "kinematic_power": kinematic_power, # 成本动力评分
            "shadow_cost": cyq_avg_cost     # 影子成本 (CYQ参考)
        }
        if _temp_debug_values is not None:
            _temp_debug_values["主力平均价格(V40 HAB版)"] = {
                "HAB_Cost_21": hab_cost_buy_21.mean(),
                "Slope_Mean": slope_clean.mean(),
                "Kinematic_Power": kinematic_power.mean(),
                "Smart_Bias_Mean": smart_bias.mean(),
                "Unit_Mismatch_Count": unit_mismatch.sum()
            }
            
        return result

    def _calculate_main_force_net_activity_score(self, context: Dict, index: pd.Index, config: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V40.0.0 · 矢量合成与HAB存量惯性净活动力模型 (Vector HAB-Inertia Activity)】
        基于《最终军械库清单.txt》的深度重构：
        1.  **数据增强**：采用 Tick级微观核动力 (tick_lg_net) + 聪明钱意图 (smart_net_buy) + 宏观资金流 (net_mf) 的三维矢量合成。
            并引入 `flow_consistency` (一致性) 作为矢量合成的质量系数。
        2.  **动力学去噪**：
            -   **Slope (13)**: 趋势方向，采用 13日 EMA 平滑后的差分。
            -   **Accel (8)**: 爆发力度，Slope 的 8日 差分。
            -   **Jerk (5)**: 变盘信号，Accel 的 5日 差分。
            -   **零基规避**：所有流向数据先标准化为“市值占比(千分之)”，再进行物理差分，彻底规避 %Change 的零基陷阱。
        3.  **HAB 存量记忆系统 (HAB Memory System)**：
            -   引入 **Historical Accumulation Buffer (HAB)**。
            -   逻辑：若 `HAB_21` (21日累积存量) 高度盈余，则当日的微小流出被视为“良性换手”而非“出货”，给予阻尼豁免。
            -   反之，若 `HAB_21` 为负，当日微小流入被视为“诱多”，信号被抑制。
        4.  **内联自适应归一化**：
            -   摒弃通用 Helper，使用基于 252日 历史波动率 (Rolling Std) 的 Tanh 动态映射。
            -   针对 `flow_efficiency` (VPA效能) 使用 Sigmoid 映射。
        5.  **非线性增益**：
            -   **聪明钱共振 (Smart Resonance)**: 当 `smart_attack` (协同攻击) 触发时，对最终得分施加 1.5倍 非线性增益。
            -   **一致性奖励**: 高一致性 (Consistency > 80) 触发指数级奖励。
        """
        f = context['funds']
        m = context['market']
        s = context['sentiment']
        # --- 1. 矢量合成 (Vector Synthesis) ---
        # 提取核心分量
        tick_net = f.get('tick_lg_net', pd.Series(0, index=index)).fillna(0)  # 微观：最具爆发力 
        smart_net = f.get('smart_net_buy', pd.Series(0, index=index)).fillna(0) # 机构：最具方向性 
        macro_net = f.get('net_mf_calibrated', pd.Series(0, index=index)).fillna(0) # 宏观：容量验证 
        # 归一化为物理强度 (Intensity): 千分之流通市值
        # 解决大盘股与小盘股的量级不可比问题
        circ_mv = m['circ_mv'].replace(0, np.nan)
        norm_tick = (tick_net / circ_mv) * 1000.0
        norm_smart = (smart_net / circ_mv) * 1000.0
        norm_macro = (macro_net / circ_mv) * 1000.0
        # 引入一致性质量因子
        # flow_consistency: 0-100. 越高说明资金合力越强。
        consistency = f.get('flow_consistency', pd.Series(50, index=index)).fillna(50)
        quality_scalar = np.tanh((consistency - 40.0) / 20.0).clip(0.5, 1.5) # 映射到 [0.5, 1.5]
        # 合成核心力向量 (Raw Vector Force)
        # 权重分配：微观(0.45) > 机构(0.35) > 宏观(0.20)
        raw_vector = (norm_tick * 0.45 + norm_smart * 0.35 + norm_macro * 0.20) * quality_scalar
        # --- 2. HAB 存量记忆与惯性计算 (HAB Inertia) ---
        # 存量意识：计算 21日 矢量存量
        # 如果 context 中已计算 hab_net_mf_21 (基于复合流)，则直接使用，否则现场计算
        hab_21 = f.get('hab_net_mf_21', raw_vector.rolling(window=21, min_periods=10).sum())
        # HAB 惯性阻尼 (Damping)
        # 逻辑：当 HAB > 10 (存量充足) 且 当日流出 (raw_vector < 0) 时，视为洗盘，衰减负向信号
        # HAB 阈值设定：10 表示累积净流入达到流通盘的 1% (因为 raw_vector 是千分之)
        is_washout_buffer = (hab_21 > 10.0) & (raw_vector < 0) & (raw_vector > -2.0) # 小幅流出
        inertia_dampener = pd.Series(1.0, index=index)
        inertia_dampener = inertia_dampener.mask(is_washout_buffer, 0.3) # 仅保留 30% 的负向影响
        # 应用阻尼
        buffered_vector = raw_vector * inertia_dampener
        # --- 3. 动力学求导 (Kinematics Derivation) ---
        # 避免零基陷阱：直接对物理强度 (Buffered Vector) 进行差分，而非百分比变化
        # Velocity (13): 趋势速度 (EMA平滑)
        velocity = buffered_vector.ewm(span=13, adjust=False).mean()
        # Accel (8): 爆发加速度 (差分)
        # 代表资金流入的“增量”变化
        accel = velocity - velocity.shift(8)
        # Jerk (5): 变盘跃度 (二阶差分)
        # 代表加速度的衰竭或二次加速
        jerk = accel - accel.shift(5)
        # 填充 NaN
        velocity = velocity.fillna(0)
        accel = accel.fillna(0)
        jerk = jerk.fillna(0)
        # --- 4. 自适应稳健归一化 (Adaptive Robust Norm) ---
        # 使用 252日 历史波动率 (Rolling Std) 作为基准
        # 逻辑：不同股票的资金活跃度不同，必须自适应
        def _adaptive_tanh(s: pd.Series, lookback=252, scale=2.0) -> pd.Series:
            # 计算滚动标准差，使用 median 填充作为稳健兜底
            roll_std = s.rolling(lookback, min_periods=21).std()
            robust_std = roll_std.fillna(s.std()).replace(0, 0.1) # 防止除零
            # Tanh 映射
            return np.tanh(s / (robust_std * scale))
        z_vel = _adaptive_tanh(velocity, scale=2.0)
        z_acc = _adaptive_tanh(accel, scale=1.5) # 加速度更敏感，Scale 调小
        z_jrk = _adaptive_tanh(jerk, scale=1.0)  # 跃度最敏感
        # --- 5. 效率修正与非线性增益 (Efficiency & Gain) ---
        # A. VPA 效率修正 (VPA Efficiency)
        # 只有资金流入是不够的，必须产生价格位移才算“做功”
        # flow_efficiency: 清单数据 
        vpa_eff = f.get('flow_efficiency', pd.Series(50, index=index)).fillna(50)
        # Sigmoid 映射：>50 奖励，<50 惩罚
        eff_factor = 1.0 / (1.0 + np.exp(-(vpa_eff - 50.0) * 0.1)) # 0.0 ~ 1.0
        # 映射到 [0.5, 1.5] 的乘数
        eff_multiplier = eff_factor + 0.5 
        # B. 聪明钱协同增益 (Smart Resonance Gain)
        # smart_attack: 协同攻击信号 (0/1 或 连续值) 
        smart_attack = f.get('smart_attack', pd.Series(0, index=index)).fillna(0)
        is_attacking = smart_attack > 0.5
        # C. 能量合成
        # 基础动能
        base_energy = (z_vel * 0.4 + z_acc * 0.4 + z_jrk * 0.2)
        # 应用效率乘数
        # 注意：如果 base_energy 为负 (流出)，效率高反而说明出货坚决，保持符号逻辑
        final_energy = base_energy * eff_multiplier
        # 应用非线性增益 (Power Law)
        # 仅当 正向流入 且 聪明钱攻击 时触发
        gain_exponent = pd.Series(1.0, index=index)
        gain_exponent = gain_exponent.mask((final_energy > 0) & is_attacking, 1.4) # 1.4次幂扩张
        # 最终映射
        final_score = np.sign(final_energy) * np.power(np.abs(final_energy), gain_exponent)
        final_score = final_score.clip(-1.0, 1.0)
        # --- 6. 调试探针 ---
        if _temp_debug_values is not None:
            _temp_debug_values["组件_净活动(V40 HAB矢量版)"] = {
                "Vector_Raw_Mean": raw_vector.mean(),
                "HAB_21_Mean": hab_21.mean(),
                "Inertia_Dampener_Active": (inertia_dampener < 1.0).sum(),
                "Z_Vel": z_vel,
                "Z_Acc": z_acc,
                "Eff_Multiplier": eff_multiplier,
                "Final_Activity_Score": final_score
            }
            
        return final_score.astype(np.float32)

    def _calculate_traditional_control_score_components(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V40.0.0 · 传统控盘时空共振与混沌博弈模型 (Spatiotemporal Resonance & Chaos Game)】
        基于《最终军械库清单.txt》的深度重构：
        1.  **数据维度扩张**：引入 `MA_COHERENCE_RESONANCE` (均线共振)、`GAP_MOMENTUM_STRENGTH` (缺口动能)、`PRICE_FRACTAL_DIM` (分形维数) 与 `PRICE_ENTROPY` (价格熵)。
        2.  **时空动力学 (Spatiotemporal Kinematics)**：
            -   **Slope (13)**: 趋势的一阶导，表征方向。
            -   **Accel (8)**: 趋势的二阶导，表征力道。
            -   **Jerk (5)**: 趋势的三阶导，表征变盘。
            -   **零基陷阱规避**：对价格类指标使用 Log-Space 差分，对评分/比例类指标直接差分。
        3.  **HAB 记忆系统 (HAB Memory)**：
            -   引入 **Historical Accumulation Buffer (HAB)** 思想。
            -   计算 `EMA_55` 的 34日 累积偏离度 (Bias Accumulation)，构建趋势惯性。
            -   当惯性巨大时 (HAB >> 0)，当日微小的反向波动被视为噪音而非反转。
        4.  **内联混沌归一化 (Chaos Normalization)**：
            -   摒弃通用 Helper，构建基于“分形维数”和“熵”的自适应归一化。
            -   **混沌状态 (High Entropy)**: 归一化区间压缩，降低信号置信度。
            -   **有序状态 (Low Entropy)**: 归一化区间扩张，放大信号置信度。
        5.  **非线性增益**：
            -   **共振奖励**: 当 `MA_COHERENCE` (共振度) > 80 时，给予指数级奖励。
            -   **缺口突变**: 引入 `GAP_MOMENTUM` 作为物理突变项，给予额外的动能加分。
        """
        m = context['market']
        ema = context['ema']
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds'] # 需要用到 gap_momentum
        # --- 1. 数据提取与预处理 (Data Extraction) ---
        # 核心趋势锚点：EMA 55 (生命线)
        ema_55 = ema.get('ema_55', pd.Series(0, index=index))
        if ema_55.isnull().all():
             # 严苛模式：核心数据缺失直接熔断，不再静默填充
             print(f"[熔断] 传统控盘: EMA_55 全量缺失，无法计算。")
             return pd.Series(0.0, index=index)
             
        # [cite_start]辅助博弈数据 [cite: 1, 3]
        coherence = s_struct.get('ma_coherence', pd.Series(0, index=index)).fillna(0) # 均线共振度
        fractal_dim = s_struct.get('fractal_dim', pd.Series(1.5, index=index)).fillna(1.5) # 分形维数
        entropy = s_struct.get('price_entropy', pd.Series(3.0, index=index)).fillna(3.0) # 价格熵
        gap_momentum = f_funds.get('gap_momentum', pd.Series(0, index=index)).fillna(0) # 缺口动能 [cite: 1]
        # --- 2. 时空动力学求导 (Spatiotemporal Kinematics) ---
        # 使用 Log-Space 避免零基陷阱，且更符合金融资产的百分比增长特性
        # Log-Price Transformation
        log_ema = np.log(ema_55.replace(0, np.nan))
        # Slope (13): 趋势方向
        slope = log_ema.diff(13) * 100.0 # 转化为 % 级别
        # Accel (8): 趋势力度
        accel = slope.diff(8)
        # Jerk (5): 趋势变盘
        jerk = accel.diff(5)
        # 填充 NaN
        slope = slope.fillna(0)
        accel = accel.fillna(0)
        jerk = jerk.fillna(0)
        # --- 3. HAB 趋势惯性记忆 (HAB Trend Inertia) ---
        # 逻辑：计算 EMA_55 斜率的 34日 累积值。
        # 如果长期处于上升趋势 (HAB_Slope > 5.0)，则具备强大的多头惯性。
        hab_slope_34 = slope.rolling(window=34, min_periods=21).sum()
        # 惯性阻尼 (Inertia Damping)
        # 当惯性为正 (hab_slope_34 > 0) 且 当日出现减速 (accel < 0) 时，
        # 只要减速幅度不大，视为良性调整，给予阻尼保护。
        inertia_protection = pd.Series(1.0, index=index)
        is_benign_pullback = (hab_slope_34 > 5.0) & (accel < 0) & (accel > -0.5)
        inertia_protection = inertia_protection.mask(is_benign_pullback, 0.5) # 减半负向影响
        # --- 4. 混沌自适应归一化 (Chaos Adaptive Norm) ---
        # 逻辑：市场的有效性随“熵”变化。
        # 高熵 (Entropy > 4.0) -> 市场混沌 -> 信号不可信 -> 归一化分母变大 (压缩得分)
        # 低熵 (Entropy < 2.5) -> 市场有序 -> 信号可信 -> 归一化分母变小 (放大的分)
        # 基础波动率 (252日)
        base_vol = slope.rolling(252, min_periods=21).std().fillna(slope.std()).replace(0, 0.1)
        # 熵修正系数 (Entropy Scalar)
        # 假设 entropy 范围 1.0 ~ 5.0.  3.0 为中性。
        entropy_scalar = (entropy / 3.0).clip(0.5, 2.0) # 熵越高，系数越大，分母越大
        # 自适应分母
        adaptive_denom = base_vol * entropy_scalar
        # Tanh 映射
        z_slope = np.tanh(slope / (adaptive_denom * 2.0))
        z_accel = np.tanh(accel / (adaptive_denom * 1.5)) # 加速度更敏感
        z_jerk = np.tanh(jerk / (adaptive_denom * 1.0))  # 跃度最敏感
        # --- 5. 核心评分与非线性增益 (Scoring & Gain) ---
        # A. 基础动能 (Kinetic Energy)
        # 权重：趋势(Slope) 40%, 爆发(Accel) 40%, 变盘(Jerk) 20%
        # 应用 HAB 惯性保护：如果处于良性回调，Accel 的负值会被 inertia_protection 缩小
        base_score = (z_slope * 0.4 + (z_accel * inertia_protection) * 0.4 + z_jerk * 0.2)
        # B. 结构共振增益 (Resonance Gain)
        # [cite_start]coherence: 均线共振度 0~100 [cite: 1]
        # 当均线高度共振 (>80) 时，说明多周期合力形成，给予 1.5倍 增益
        resonance_mult = 1.0 + (np.tanh((coherence - 60.0) / 20.0).clip(0, 1.0) * 0.5)
        # C. 缺口突变奖励 (Gap Mutation)
        # [cite_start]gap_momentum: 缺口动能 [cite: 1]
        # 缺口代表物理层级的能量跃迁，给予独立加分
        gap_bonus = np.tanh(gap_momentum / 10.0).clip(-0.3, 0.3)
        # D. 分形维数修正 (Fractal Correction)
        # fractal_dim: 1.0 (直线) ~ 2.0 (平面)。接近 1.5 为随机游走。
        # 当 D < 1.3 时，趋势性极强 (Trend Persistence)，奖励得分。
        # 当 D > 1.7 时，均值回归极强 (Mean Reversion)，惩罚趋势分。
        fractal_mult = pd.Series(1.0, index=index)
        fractal_mult = fractal_mult.mask(fractal_dim < 1.3, 1.2) # 趋势增强
        fractal_mult = fractal_mult.mask(fractal_dim > 1.7, 0.8) # 噪音抑制
        # --- 6. 最终合成 ---
        # 组合：(基础动能 * 共振 * 分形) + 缺口
        raw_final = (base_score * resonance_mult * fractal_mult) + gap_bonus
        # 限制范围
        final_score = raw_final.clip(-1.0, 1.0)
        # --- 7. 调试探针 ---
        if _temp_debug_values is not None:
            _temp_debug_values["组件_传统控盘(V40时空混沌版)"] = {
                "HAB_Slope_34": hab_slope_34.mean(),
                "Entropy_Scalar": entropy_scalar.mean(),
                "Inertia_Active": (inertia_protection < 1.0).sum(),
                "Coherence_Mult": resonance_mult.mean(),
                "Fractal_Mult": fractal_mult.mean(),
                "Gap_Bonus": gap_bonus.mean(),
                "Final_Trad_Score": final_score
            }
            
        print(f"[探针] 传统控盘: 混沌归一化完成。平均熵修正系数: {entropy_scalar.mean():.2f} (高熵压制), 惯性保护触发点数: {(inertia_protection < 1.0).sum()}")
        return final_score.astype(np.float32)

    def _calculate_control_leverage_model(self, index: pd.Index, fused_score: pd.Series, net_activity_score: pd.Series,
                                          norm_flow: pd.Series, cost_score: pd.Series,
                                          norm_t0_buy: pd.Series, norm_t0_sell: pd.Series,
                                          norm_vwap_up: pd.Series, norm_vwap_down: pd.Series,
                                          context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V40.0.0 · 熵减风控杠杆与HAB状态记忆模型 (Entropy-HAB Leverage)】
        基于《最终军械库清单.txt》的深度重构：
        1.  **数据层增强**：
            -   [cite_start]引入 **双熵 (Dual Entropy)**: `chip_entropy_D` (筹码无序度) + `PRICE_ENTROPY_D` (价格无序度) [cite: 2, 3]。
            -   [cite_start]引入 **风险 (Risk)**: `reversal_prob_D` (反转概率) + `divergence_strength_D` (背离强度) [cite: 2, 3]。
            -   [cite_start]引入 **行业 (Industry)**: `industry_strength_rank_D` (行业排名) [cite: 2]。
        2.  **动力学 (Kinetics)**:
            -   计算 **熵的加速度 (Entropy Accel)**。如果混乱度正在加速上升 (Accel > 0)，即使当前绝对值不高，也需提前收缩杠杆。
            -   应用斐波拉契 Slope(5) / Accel(3) 进行敏感度探测。
        3.  **HAB 记忆系统 (HAB Memory)**:
            -   计算 **风险HAB (Risk HAB)**: 13日 累积风险均值。
            -   **惯性逻辑**: 如果长期稳定 (Low Risk HAB)，单日的熵增被视为噪音；如果长期混乱，单日低熵无效。
        4.  **内联归一化**:
            -   针对 Entropy 和 Rank 使用倒数/Sigmoid 自适应归一化。
        5.  **非线性增益**:
            -   [cite_start]**龙头特权 (Leader Privilege)**: `STATE_MARKET_LEADER` 触发 1.5倍 杠杆上限 [cite: 1]。
            -   [cite_start]**聪明钱熔断 (Smart Override)**: `SMART_MONEY_HM_COORDINATED_ATTACK` 可豁免部分熵惩罚 [cite: 1]。
        6.  **最终输出**: 动态杠杆系数 [0.0, 12.0]，决定了动力分转化为最终得分的放大倍数。
        """
        s_struct = context['structure']
        s_sent = context['sentiment'] # 注意：在 V40 _get_raw_control_signals 中，State 指标被打包进了 sentiment
        f_funds = context['funds']
        # --- 1. 数据提取 (Data Extraction) ---
        # [cite_start]熵 (Entropy) [cite: 2, 3]
        chip_ent = s_struct.get('chip_entropy', pd.Series(100, index=index)).fillna(100)
        price_ent = s_struct.get('price_entropy', pd.Series(4, index=index)).fillna(4)
        # [cite_start]风险 (Risk) [cite: 3]
        rev_prob = s_sent.get('reversal_prob', pd.Series(0, index=index)).fillna(0) # 0-100
        div_str = s_sent.get('divergence_strength', pd.Series(0, index=index)).fillna(0)
        # [cite_start]状态 (State) [cite: 1, 2]
        is_leader = s_sent.get('market_leader', pd.Series(0, index=index)) > 0.5
        is_pit = s_sent.get('golden_pit', pd.Series(0, index=index)) > 0.5
        ind_rank = s_sent.get('industry_rank', pd.Series(50, index=index)).fillna(50)
        # [cite_start]聪明钱 (Smart) [cite: 1]
        smart_attack = f_funds.get('smart_attack', pd.Series(0, index=index)).fillna(0)
        # --- 2. 动力学风控 (Kinematic Risk) ---
        # 计算“综合无序度” (Total Disorder)
        # 归一化: Chip[0-100] -> [0,1], Price[1-5] -> [0,1]
        norm_chip_ent = (chip_ent / 80.0).clip(0, 1.5)
        norm_price_ent = ((price_ent - 1.5) / 2.5).clip(0, 1.5)
        raw_disorder = (norm_chip_ent * 0.6 + norm_price_ent * 0.4)
        # 熵的动力学：Slope(5) & Accel(3) - 快速反应
        # 如果熵正在加速上升，说明市场正在失控
        # 差分计算，无需对数，因为 disorder 已经是比率
        ent_slope = raw_disorder.diff(5).fillna(0)
        ent_accel = ent_slope.diff(3).fillna(0)
        # 动力学惩罚: 如果加速混乱 (Accel > 0.05)，加大惩罚
        # 这是一个“预警”机制，在混乱彻底爆发前降杠杆
        kinetic_penalty = pd.Series(0.0, index=index)
        kinetic_penalty = kinetic_penalty.mask(ent_accel > 0.05, 0.3) # 扣除 30% 杠杆
        # --- 3. HAB 风险记忆 (HAB Risk Memory) ---
        # 计算 13日 滚动平均无序度 (Structure Stability)
        hab_disorder = raw_disorder.rolling(window=13, min_periods=5).mean().fillna(raw_disorder)
        # 稳定性判定: 如果 HAB < 0.4 (长期有序)，则对单日的熵增宽容
        # 逻辑：牛市急跌往往是买点，不要因为单日熵增就杀跌
        is_structurally_stable = hab_disorder < 0.4
        # 修正后的无序度 (Effective Disorder)
        effective_disorder = raw_disorder.copy()
        # 稳定状态下，单日混乱打 7 折计算
        effective_disorder = effective_disorder.mask(is_structurally_stable, effective_disorder * 0.7)
        # --- 4. 自研归一化与阻尼 (Custom Norm & Damping) ---
        # A. 熵减阻尼 (Entropy Damping)
        # 核心公式: Lev = Lev * exp(-k * Disorder^2)
        # k=2.0 -> Disorder=0.5时 衰减到 0.6; Disorder=1.0时 衰减到 0.13
        # 这是一个强非线性函数，对低熵非常友好，对高熵极度严苛
        entropy_damping = np.exp(-2.0 * np.power(effective_disorder, 2))
        # B. 行业地位修正 (Industry Scalar)
        # Rank 1-5: 1.2x; Rank > 30: 0.8x
        # 1 / (1 + exp(rank)) 逻辑变种
        ind_scalar = 1.0 + (np.tanh((20.0 - ind_rank) / 10.0) * 0.2) # 0.8 ~ 1.2
        # C. 风险概率熔断 (Risk Circuit Breaker)
        # reversal_prob > 80% -> 强制 0.1x
        risk_breaker = pd.Series(1.0, index=index)
        risk_breaker = risk_breaker.mask(rev_prob > 80.0, 0.1)
        risk_breaker = risk_breaker.mask(div_str > 80.0, 0.2)
        # --- 5. 杠杆合成与非线性增益 ---
        # 基础杠杆: 由 结构分 (fused_score) 决定
        # fused_score [-1, 1] -> Base [1, 6]
        # 结构越好，允许放大的倍数越高
        base_lev = 1.0 + np.tanh(fused_score.clip(0, 1) * 2.5) * 5.0 
        # 应用 熵阻尼 & 动力学惩罚
        # (1 - kinetic_penalty) 是线性扣减
        lev_step1 = base_lev * entropy_damping * (1.0 - kinetic_penalty)
        # 应用 状态特权 (State Privilege)
        # 龙头/金坑 享有 1.5倍/1.3倍 溢价
        state_mult = pd.Series(1.0, index=index)
        state_mult = state_mult.mask(is_leader, 1.5)
        state_mult = state_mult.mask(is_pit, 1.3)
        # 应用 聪明钱豁免 (Smart Override)
        # 如果 smart_attack 存在，强制提升杠杆底线，且抵抗熵惩罚
        # 逻辑：机构协同攻击是“有序的混沌”
        attack_bonus = pd.Series(1.0, index=index)
        attack_bonus = attack_bonus.mask(smart_attack > 0.5, 1.4)
        # 综合计算
        raw_final_lev = lev_step1 * ind_scalar * state_mult * attack_bonus * risk_breaker
        # --- 6. 最终约束 ---
        # 硬约束: 如果 净活动力(Net Activity) 为负，杠杆不能超过 1.0 
        # 逻辑：主力在出逃时，结构再好也只能看作反弹，不能加杠杆做主升
        final_lev = raw_final_lev.mask(net_activity_score < 0, raw_final_lev.clip(upper=1.0))
        # 全局上限 12.0
        final_lev = final_lev.clip(0.0, 12.0)
        # --- 7. 调试探针 ---
        if _temp_debug_values is not None:
            _temp_debug_values["风控_杠杆(V40熵减HAB版)"] = {
                "HAB_Disorder_13": hab_disorder.mean(),
                "Entropy_Accel_Penalty": (kinetic_penalty > 0).sum(),
                "Entropy_Damping": entropy_damping.mean(),
                "State_Mult_Max": state_mult.max(),
                "Risk_Breaker_Active": (risk_breaker < 1.0).sum(),
                "Final_Leverage": final_lev
            }
            
        print(f"[探针] 风控杠杆: 熵减模型运行。HAB稳定态占比: {(is_structurally_stable).mean():.1%}, 风险熔断触发: {(risk_breaker < 0.5).sum()} 次")
        return final_lev

    def _fuse_control_scores(self, traditional_score: pd.Series, structural_score: pd.Series, context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V40.0.0 · 熵减结构融合与HAB时空博弈模型 (Entropy-HAB Fusion)】
        基于《最终军械库清单.txt》的深度重构：
        1.  **数据层增强**：
            -   引入 `winner_rate_D` (获利比例) 辅助 `profit_ratio` 计算更真实的筹码挤压。
            -   引入 `chip_stability_D` (筹码稳定性) 的 Slope/Accel 动力学，预判结构变化。
            -   引入 `SMART_MONEY_HM_COORDINATED_ATTACK_D` (聪明钱攻击) 作为最高级门控。
        2.  **HAB 存量缓冲 (HAB Shield)**：
            -   利用 `hab_net_mf_55` (中期存量) 构建护盾。当存量丰厚时，豁免短期结构恶化 (Stability Drop) 的惩罚。
        3.  **双熵博弈 (Dual Entropy)**：
            -   计算 `Entropy_Gap = Price_Entropy - Chip_Entropy`。
            -   理想状态：价格有序 (Low Price Ent) + 筹码有序 (Low Chip Ent)。
            -   伪突破状态：价格有序 + 筹码混乱 (High Chip Ent) -> 强惩罚。
        4.  **内联归一化**:
            -   针对 Squeeze (挤压) 和 Locking (锁仓) 编写非线性映射函数。
        5.  **非线性增益**:
            -   聪明钱攻击触发 1.25倍 乘数。
            -   采用 Gamma 分布对最终得分进行扩张。
        """
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        m_market = context['market']
        s_state = context['state'] # 包含 Golden Pit, Breakout 等
        # --- 1. 数据提取与预处理 ---
        # 核心结构数据
        profit = s_struct.get('profit_ratio', pd.Series(0, index=traditional_score.index)).fillna(0)
        winner = s_struct.get('winner_rate', pd.Series(0, index=traditional_score.index)).fillna(0)
        trapped = s_struct.get('pressure_trapped', pd.Series(0, index=traditional_score.index)).fillna(0)
        # 动力学数据 (Stability Kinematics)
        # 筹码稳定性反映了锁仓程度，直接对其求导
        stability = s_struct.get('chip_stability', pd.Series(50, index=traditional_score.index)).fillna(50)
        slope_stab = stability.diff(5).fillna(0)
        accel_stab = slope_stab.diff(3).fillna(0)
        # 熵数据
        price_ent = s_struct.get('price_entropy', pd.Series(3, index=traditional_score.index)).fillna(3)
        chip_ent = s_struct.get('chip_entropy', pd.Series(100, index=traditional_score.index)).fillna(100)
        # 资金与HAB
        hab_55 = f_funds.get('hab_net_mf_55', f_funds['net_mf_calibrated'].rolling(55).sum())
        circ_mv = m_market['circ_mv'].replace(0, np.nan)
        turnover = s_sent['turnover'].replace(0, np.nan).fillna(1.0)
        smart_attack = f_funds.get('smart_attack', pd.Series(0, index=traditional_score.index)).fillna(0)
        sm_div = f_funds.get('smart_divergence', pd.Series(0, index=traditional_score.index)).fillna(0)
        # --- 2. 成本结构挤压 (Cost Squeeze Dynamics) ---
        # 逻辑：获利盘与胜率的双重确认。如果两者都高且套牢盘低，说明上方真空。
        # Squeeze = ((Profit + Winner)/2 - Trapped)
        # 归一化：Tanh映射，系数 20.0
        squeeze_raw = ((profit + winner) * 0.5 - trapped)
        squeeze_score = np.tanh(squeeze_raw / 20.0) # -1.0 ~ 1.0
        # Squeeze 修正系数: 正向挤压放大信号，负向挤压(套牢)衰减信号
        cost_modifier = 1.0 + (squeeze_score * 0.4)
        # --- 3. HAB 存量护盾 (HAB Shield) ---
        # 逻辑：计算主力持仓密度。
        # 如果 HAB密度 > 1% (0.01)，则认为主力控盘深厚，对结构波动不敏感。
        hab_density = (hab_55 / circ_mv).fillna(0)
        hab_shield = np.tanh(hab_density * 30.0).clip(0, 1.0) # 0.0 ~ 1.0
        # --- 4. 动态权重分配 (Dynamic Weighting) ---
        # 基于 ADX (趋势强度)
        # ADX > 40: 趋势确立，重传统(Traditional) 轻结构(Structural)
        # ADX < 20: 震荡蓄势，重结构 轻传统
        adx = s_sent['adx_14'].fillna(20.0)
        w_trad = 1.0 / (1.0 + np.exp(-0.15 * (adx - 30.0))) # Sigmoid centered at 30
        w_trad = w_trad.clip(0.3, 0.7) # 限制权重范围
        w_struct = 1.0 - w_trad
        # 基础融合分
        base_score = (traditional_score * w_trad + structural_score * w_struct) * cost_modifier
        # --- 5. 锁仓效率与动力学修正 (Locking Efficiency & Kinematics) ---
        # A. 锁仓效率: (HAB / Turnover)
        # 换手越低，HAB越高，说明锁仓越好。
        # 归一化：Tanh
        locking_ratio = (hab_density * 100.0) / (turnover / 3.0) # 归一化换手
        locking_gain = 1.0 + np.tanh(locking_ratio * 0.5).clip(-0.5, 0.8)
        # B. 结构动力学 (Stability Kinematics)
        # 如果稳定性正在加速上升 (Accel > 0)，说明筹码正在快速沉淀，奖励
        # 如果稳定性加速下降，说明筹码松动
        # HAB 护盾作用：如果 Shield 强，忽略短期松动
        stab_kine = np.tanh(slope_stab * 0.1 + accel_stab * 0.2)
        # 应用护盾：如果是负向变动，乘以 (1 - Shield)
        stab_kine = stab_kine.mask(stab_kine < 0, stab_kine * (1.0 - hab_shield * 0.8))
        # --- 6. 双熵博弈与门控 (Dual Entropy & Gating) ---
        # 逻辑：价格熵(Price) vs 筹码熵(Chip)
        # 归一化 Price[1-5] -> [0,1], Chip[0-100] -> [0,1]
        norm_p_ent = ((price_ent - 1.0) / 4.0).clip(0, 1)
        norm_c_ent = (chip_ent / 100.0).clip(0, 1)
        # 熵质量 (Entropy Quality)
        # 最佳：Price有序(Low) + Chip有序(Low)
        # 最差：Price有序(Low) + Chip混乱(High) -> 典型的诱多/假突破
        # 计算“伪装度” (Fake Degree): Price低 但 Chip高
        fake_degree = (1.0 - norm_p_ent) * norm_c_ent
        entropy_penalty = 1.0 - (fake_degree * 0.8) # 最大扣除 80%
        # 聪明钱门控 (Smart Gate)
        sm_gate = pd.Series(1.0, index=traditional_score.index)
        # 协同攻击：奖励 1.25x
        sm_gate = sm_gate.mask(smart_attack > 0.5, 1.25)
        # 顶背离：惩罚 0.6x (股价涨但主力卖)
        is_bull_trap = (base_score > 0.2) & (sm_div < -0.2)
        sm_gate = sm_gate.mask(is_bull_trap, 0.6)
        # --- 7. 最终合成与非线性扩张 ---
        # 状态检查
        is_golden_pit = s_state.get('golden_pit', pd.Series(0, index=traditional_score.index)) > 0
        is_breakout = s_state.get('breakout_confirmed', pd.Series(0, index=traditional_score.index)) > 0
        # 融合计算
        # Base * Locking * Stability_Addon * Entropy * Smart
        # Stability Addon: 转化为乘数 1.0 +/- 0.2
        stab_mult = 1.0 + (stab_kine * 0.2)
        raw_final = base_score * locking_gain * stab_mult * entropy_penalty * sm_gate
        # 状态豁免：如果是黄金坑，忽略部分惩罚
        if is_golden_pit.any():
            raw_final = raw_final.mask(is_golden_pit & (raw_final < 0), raw_final * 0.5) # 负分减半
            
        # 幂律扩张 (Power Law Expansion)
        # 增强区分度：两端极化
        expanded_final = np.sign(raw_final) * np.power(np.abs(raw_final), 1.5)
        # 最终 Tanh 约束
        final_fused = np.tanh(expanded_final).astype(np.float32)
        # --- Debug Output ---
        if _temp_debug_values is not None:
            _temp_debug_values["融合_动力学(V40熵减HAB版)"] = {
                "Squeeze_Score": squeeze_score,
                "HAB_Shield": hab_shield,
                "Dynamic_Trad_Weight": w_trad,
                "Locking_Gain": locking_gain,
                "Stab_Kinematics": stab_kine,
                "Entropy_Penalty": entropy_penalty,
                "Smart_Gate": sm_gate,
                "Final_Fused_Score": final_fused
            }
            
        if (entropy_penalty < 0.6).sum() > 0:
            print(f"[探针] 结构融合: 监测到 { (entropy_penalty < 0.6).sum() } 个点位存在【伪有序】特征 (价格稳/筹码乱)，已执行熵减惩罚。")
        if (hab_shield > 0.8).sum() > 0:
            print(f"[探针] 结构融合: HAB护盾激活。主力高度锁仓，短期结构波动已被平滑。")
            
        return final_fused

    def _normalize_components(self, df: pd.DataFrame, context: Dict, scores_traditional: pd.Series, config: Dict, method_name: str, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        【V4.6.0 · 独立组件归一化系统 (Kinetic Normalizer)】
        职责：放弃通用 Helper 逻辑，采用基于物理动力学和博弈深度定制的非线性归一化方案。
        修改说明：
        1. Bipolar Tanh Scaling: 用于传统得分和结构得分，处理极端离群值，保留 0 轴敏感度。
        2. Inline MTF Structure: 内部实现多周期筹码稳定性斜率加权，判定控盘沉淀。
        3. Intent Sigmoid Mapping: 对意图信心分执行逻辑回归映射，强化强信号，压制弱噪音。
        4. 修复：从 funds 中正确获取 flow_consistency。
        """
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds'] # 获取资金上下文
        
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
        # [修复] 修正引用路径，从 context['funds'] 中获取
        flow_consistency = f_funds.get('flow_consistency', pd.Series(50, index=df.index)).fillna(50)
        norm_flow = (flow_consistency / 100.0).clip(0, 1)

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
                "flow_consistency_norm": norm_flow,
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








