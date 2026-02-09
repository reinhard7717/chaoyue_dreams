# strategies\trend_following\intelligence\process\calculate_main_force_control.py
# 【V1.0.0 · 主力控盘关系计算器】
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
        【V4.2.0 · 主力控盘关系计算器 - 最终形态】
        职责：调度全量计算流程。
        优化：
        1. 保留 _get_probe_timestamp 用于控制调试开关。
        2. 合并输出逻辑至 _calculate_main_force_control_relationship_debug_output。
        """
        method_name = "calculate_main_force_control_relationship"
        is_debug = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        # 1. 获取探针时间 (必须独立，用于控制后续的数据收集)
        probe_ts = self._get_probe_timestamp(df, is_debug)
        debug_output = {}
        _temp_debug_values = {} 
        if probe_ts:
            debug_output[f"--- {method_name} 管道启动 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        # 2. 验证必要信号
        if hasattr(self, '_validate_arsenal_signals'):
             if not self._validate_arsenal_signals(df, config, method_name, debug_output, probe_ts):
                return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 3. 数据准备
        control_context = self._get_raw_control_signals(df, method_name, _temp_debug_values, probe_ts)
        # 4. 组件计算
        scores_traditional = self._calculate_traditional_control_score_components(control_context, df.index, _temp_debug_values)
        if scores_traditional.isnull().all():
             return pd.Series(0.0, index=df.index, dtype=np.float32)
        scores_cost_advantage = self._calculate_main_force_cost_advantage_score(control_context, df.index, _temp_debug_values)
        scores_net_activity = self._calculate_main_force_net_activity_score(control_context, df.index, config, method_name, _temp_debug_values)
        # 5. 模型融合
        norm_traditional, norm_structural, norm_flow, norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down = \
            self._normalize_components(df, control_context, scores_traditional, config, method_name, _temp_debug_values)
        fused_control_score = self._fuse_control_scores(norm_traditional, norm_structural, control_context, _temp_debug_values)
        # 6. 风控杠杆
        control_leverage = self._calculate_control_leverage_model(
            df.index, fused_control_score, scores_net_activity, 
            norm_flow, scores_cost_advantage, 
            norm_t0_buy, norm_t0_sell, norm_vwap_up, norm_vwap_down,
            control_context, _temp_debug_values
        )
        # 7. 最终输出
        final_control_score = (scores_net_activity * control_leverage).clip(-1, 1).astype(np.float32)
        # 将最终结果放入 debug 容器，供统一输出
        _temp_debug_values["最终结果"] = {"Final_Score": final_control_score}
        # 8. 调试输出 (调用合并后的标准输出方法)
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
        【V2.1.0 · 全维动力学信号提取】
        新增提取 Slope(斜率), Accel(加速度), Jerk(加加速度) 指标。
        锁定 Lookback = 5 (斐波那契周线窗口)，捕捉主力短线意图的突变。
        动力学核心对象：
        1. net_mf_amount_D (资金流): 捕捉点火与衰竭。
        2. chip_stability_D (筹码结构): 捕捉锁仓加速。
        3. pushing_score_D (推升意图): 捕捉攻击信号突变。
        """
        # 1. 基础数据提取 (保持原有逻辑)
        market = {
            "close": self._get_safe_series(df, 'close_D', method_name=method_name),
            "amount": self._get_safe_series(df, 'amount_D', method_name=method_name).replace(0, np.nan),
            "pct_change": self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name),
        }
        structure = {
            "ema_13": self._get_safe_series(df, 'EMA_13_D', method_name=method_name),
            "ema_21": self._get_safe_series(df, 'EMA_21_D', method_name=method_name),
            "ema_55": self._get_safe_series(df, 'EMA_55_D', method_name=method_name),
            "chip_stability": self._get_safe_series(df, 'chip_stability_D', 0.5, method_name=method_name),
            "chip_entropy": self._get_safe_series(df, 'chip_entropy_D', 1.0, method_name=method_name),
            "cost_50pct": self._get_safe_series(df, 'cost_50pct_D', method_name=method_name).replace(0, np.nan),
            "winner_rate": self._get_safe_series(df, 'winner_rate_D', 50.0, method_name=method_name),
        }
        buy_lg = self._get_safe_series(df, 'buy_lg_amount_D', 0.0, method_name=method_name)
        buy_elg = self._get_safe_series(df, 'buy_elg_amount_D', 0.0, method_name=method_name)
        sell_lg = self._get_safe_series(df, 'sell_lg_amount_D', 0.0, method_name=method_name)
        sell_elg = self._get_safe_series(df, 'sell_elg_amount_D', 0.0, method_name=method_name)
        funds = {
            "buy_lg_amt": buy_lg,
            "buy_elg_amt": buy_elg,
            "sell_lg_amt": sell_lg,
            "sell_elg_amt": sell_elg,
            "buy_lg_vol": self._get_safe_series(df, 'buy_lg_vol_D', 0.0, method_name=method_name),
            "buy_elg_vol": self._get_safe_series(df, 'buy_elg_vol_D', 0.0, method_name=method_name),
            "sell_lg_vol": self._get_safe_series(df, 'sell_lg_vol_D', 0.0, method_name=method_name),
            "sell_elg_vol": self._get_safe_series(df, 'sell_elg_vol_D', 0.0, method_name=method_name),
            "net_mf_calibrated": self._get_safe_series(df, 'net_mf_amount_D', 0.0, method_name=method_name),
            "total_buy_amt": buy_lg + buy_elg,
            "total_sell_amt": sell_lg + sell_elg,
        }
        sentiment = {
            "flow_consistency": self._get_safe_series(df, 'flow_consistency_D', 0.5, method_name=method_name),
            "t0_buy_conf": self._get_safe_series(df, 'intraday_accumulation_confidence_D', 0.0, method_name=method_name),
            "t0_sell_conf": self._get_safe_series(df, 'intraday_distribution_confidence_D', 0.0, method_name=method_name),
            "pushing_score": self._get_safe_series(df, 'pushing_score_D', 0.0, method_name=method_name),
            "shakeout_score": self._get_safe_series(df, 'shakeout_score_D', 0.0, method_name=method_name),
            "bbw": self._get_safe_series(df, 'BBW_21_2.0_D', 0.1, method_name=method_name),
            "turnover": self._get_safe_series(df, 'turnover_rate_D', 1.0, method_name=method_name),
            "vpa_efficiency": self._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.0, method_name=method_name),
        }
        # 2. 动力学衍生数据 (Kinematics Extraction)
        # 针对核心指标提取 5日 (Lookback=5) 的 derivatives
        # 注意：如果数据层没有提供现成的列，get_safe_series 会返回默认值(0.0)，不会报错
        # 资金流动力学
        funds["net_mf_slope"] = self._get_safe_series(df, 'SLOPE_5_net_mf_amount_D', 0.0, method_name=method_name)
        funds["net_mf_accel"] = self._get_safe_series(df, 'ACCEL_5_net_mf_amount_D', 0.0, method_name=method_name)
        funds["net_mf_jerk"]  = self._get_safe_series(df, 'JERK_5_net_mf_amount_D', 0.0, method_name=method_name)
        # 筹码结构动力学
        structure["stability_accel"] = self._get_safe_series(df, 'ACCEL_5_chip_stability_D', 0.0, method_name=method_name)
        # 行为意图动力学
        sentiment["pushing_jerk"] = self._get_safe_series(df, 'JERK_5_pushing_score_D', 0.0, method_name=method_name)
        context = {"market": market, "structure": structure, "funds": funds, "sentiment": sentiment}
        if probe_ts:
            debug_vals = {}
            # 仅记录部分核心动力学数据防止日志爆炸
            debug_vals["funds.net_mf_jerk"] = funds["net_mf_jerk"].loc[probe_ts] if probe_ts in funds["net_mf_jerk"].index else np.nan
            debug_vals["funds.net_mf_accel"] = funds["net_mf_accel"].loc[probe_ts] if probe_ts in funds["net_mf_accel"].index else np.nan
            debug_vals["structure.stability_accel"] = structure["stability_accel"].loc[probe_ts] if probe_ts in structure["stability_accel"].index else np.nan
            _temp_debug_values["原始信号快照(动力学)"] = debug_vals
        return context

    def _calculate_main_force_control_relationship_debug_output(self, debug_output: Dict, _temp_debug_values: Dict, method_name: str, probe_ts: pd.Timestamp):
        """
        【V4.2.0 · 统一调试输出】
        合并了原有的 _print_pipeline_debug 逻辑。
        使用结构化遍历，根据 _temp_debug_values 中的键值自动生成报告。
        """
        # 定义输出顺序，确保日志逻辑清晰
        sections = [
            ("原始信号快照", "原始信号值"),
            ("原始信号快照(动力学)", "动力学信号"),
            ("组件_传统控盘", "传统控盘组件"),
            ("组件_成本优势", "成本优势组件"),
            ("组件_净活动", "净活动组件"),
            ("组件_净活动(动力学)", "净活动动力学"),
            ("归一化处理", "归一化中间态"),
            ("融合_中间态", "融合中间态"),
            ("融合_动力学", "结构动力学融合"),
            ("风控_杠杆", "风控杠杆模型"),
            ("最终结果", "最终输出")
        ]
        for key, label in sections:
            if key in _temp_debug_values:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- {label} ---"] = ""
                data_map = _temp_debug_values[key]
                for sub_key, val in data_map.items():
                    # 统一处理 Series (取单点) 或 Scalar
                    if isinstance(val, pd.Series):
                        v_print = val.loc[probe_ts] if probe_ts in val.index else np.nan
                    else:
                        v_print = val
                    # 格式化输出
                    if isinstance(v_print, (float, np.floating)):
                        debug_output[f"        {sub_key}: {v_print:.4f}"] = ""
                    else:
                        debug_output[f"        {sub_key}: {v_print}"] = ""
        self._print_debug_info(debug_output)

    def _calculate_main_force_avg_prices(self, df_index: pd.Index, close_price: pd.Series, 
                                         buy_lg_amt: pd.Series, buy_elg_amt: pd.Series, 
                                         sell_lg_amt: pd.Series, sell_elg_amt: pd.Series, 
                                         buy_lg_vol: pd.Series, buy_elg_vol: pd.Series, 
                                         sell_lg_vol: pd.Series, sell_elg_vol: pd.Series, 
                                         _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V2.4.0 · 主力成本动力学模型 - 零基陷阱防御版】
        核心防御机制：
        1. 预平滑 (Pre-Smoothing): 对原始均价进行 EMA 处理，防止噪音在二阶导数中爆炸。
        2. 算术差分 (Arithmetic Difference): 
           - Slope 计算使用百分比 (相对变化)。
           - Accel/Jerk 计算使用差值 (绝对变化)，严禁使用 (Slope_t - Slope_t-1)/Slope_t-1，
             彻底规避 Slope=0 导致的除零陷阱和符号翻转陷阱。
        3. 软压缩 (Soft Clipping): 使用 tanh 对高阶导数进行去极值处理。
        """
        # 定义权重：给予 Smart Money (超大单) 更高权重
        w_elg, w_lg = 1.5, 1.0
        # --- 1. 计算买入成本 (Buy Cost) ---
        buy_amt_w = (buy_elg_amt * w_elg) + (buy_lg_amt * w_lg)
        buy_vol_w = (buy_elg_vol * w_elg) + (buy_lg_vol * w_lg)
        # 基础计算：加权均价
        # 防御1: 使用 fillna(close_price) 处理无成交的情况，避免 NaN
        daily_buy = (buy_amt_w / buy_vol_w.replace(0, np.nan)).fillna(close_price)
        # 防御2: 预平滑 (Pre-Smoothing)
        # 成本线本身的波动需要被平滑，否则 Jerk 指标会全是噪音
        avg_buy = daily_buy.ewm(span=5, adjust=False).mean()
        # --- 动力学计算 (Kinematics) ---
        # Level 1: Slope (速度)
        # 逻辑：成本的 3日变动率。价格永远 > 0，使用百分比是安全的。
        # replace(0, np.nan) 是最后的保险，防止极端数据错误
        buy_slope_raw = (avg_buy - avg_buy.shift(3)) / avg_buy.shift(3).replace(0, np.nan) * 100
        # Level 2: Accel (加速度)
        # 【关键防御】使用算术差分 (Difference)，而非增长率。
        # 含义：斜率增加了多少个百分点。避免了 Slope=0 时的除零错误。
        buy_accel_raw = buy_slope_raw - buy_slope_raw.shift(1)
        # Level 3: Jerk (加加速度/变盘)
        # 【关键防御】同样使用算术差分。
        buy_jerk_raw = buy_accel_raw - buy_accel_raw.shift(1)
        # --- 数据清洗与压缩 ---
        # 使用 tanh 将动力学指标限制在合理区间，防止个别妖股的极端数据破坏整体模型
        # 系数说明：
        # Slope * 1.0: 1% 的日均变动对应 tanh(1) ~= 0.76 (合理)
        # Accel * 2.0: 加速度通常很小，放大处理
        # Jerk * 5.0: 突变信号通常极小，大幅放大以捕捉信号
        result = {
            "avg_buy": avg_buy,
            "buy_slope": buy_slope_raw.fillna(0), # 输出原始值供逻辑判断(如 >0.2%)
            "buy_accel": buy_accel_raw.fillna(0),
            "buy_jerk": buy_jerk_raw.fillna(0),
            # 归一化版本 (用于机器学习或打分融合)
            "buy_slope_norm": np.tanh(buy_slope_raw),
            "buy_accel_norm": np.tanh(buy_accel_raw * 2.0),
            "buy_jerk_norm": np.tanh(buy_jerk_raw * 5.0),
        }
        # --- 2. 计算卖出成本 (Sell Cost) ---
        # 逻辑同上
        sell_amt_w = (sell_elg_amt * w_elg) + (sell_lg_amt * w_lg)
        sell_vol_w = (sell_elg_vol * w_elg) + (sell_lg_vol * w_lg)
        daily_sell = (sell_amt_w / sell_vol_w.replace(0, np.nan)).fillna(close_price)
        avg_sell = daily_sell.ewm(span=5, adjust=False).mean()
        sell_slope_raw = (avg_sell - avg_sell.shift(3)) / avg_sell.shift(3).replace(0, np.nan) * 100
        result["avg_sell"] = avg_sell
        result["sell_slope"] = sell_slope_raw.fillna(0)
        _temp_debug_values["主力平均价格计算"] = {
            "avg_buy": avg_buy,
            "buy_slope": buy_slope_raw,
            "buy_accel": buy_accel_raw,
            "buy_jerk": buy_jerk_raw
        }
        return result

    def _calculate_main_force_cost_advantage_score(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V2.3.0 · 双轨成本Alpha模型 - 动力学增强版】
        引入成本斜率 (Cost Slope) 作为核心判定因子。
        逻辑升级：
        1. 静态优势 (Static Alpha): 市场成本 vs 主力成本 (原有逻辑)。
        2. 动态意图 (Dynamic Intent):
           - 如果主力成本在显著上移 (Slope > 0.5%)，说明主力在“抬轿”，给予 Alpha 加成。
           - 如果股价涨但主力成本走平 (Slope ~ 0)，可能是虚拉，Alpha 降权。
        """
        m, f, s = context['market'], context['funds'], context['structure']
        # 1. 调用新的价格计算方法 (返回字典)
        # 注意：这里需要传入拆分后的资金流数据
        prices = self._calculate_main_force_avg_prices(
            index, m['close'],
            f['buy_lg_amt'], f['buy_elg_amt'], f['sell_lg_amt'], f['sell_elg_amt'],
            f['buy_lg_vol'], f['buy_elg_vol'], f['sell_lg_vol'], f['sell_elg_vol'],
            _temp_debug_values
        )
        avg_buy = prices['avg_buy']
        avg_sell = prices['avg_sell']
        buy_slope = prices['buy_slope'] # 成本趋势
        buy_accel = prices['buy_accel'] # 抢筹急迫度
        # 2. 计算静态优势 (Static Score)
        # 战略Alpha: 主力成本比市场成本低多少
        strategic_alpha = (s['cost_50pct'] - avg_buy) / s['cost_50pct'].replace(0, np.nan)
        # 安全垫: 现价高于主力买入价多少
        safety_margin = (m['close'] - avg_buy) / avg_buy.replace(0, np.nan)
        # 收割能力: 卖出均价高于市场成本多少
        harvest_prem = (avg_sell - s['cost_50pct']) / s['cost_50pct'].replace(0, np.nan)
        # 3. 动态意图修正 (Dynamic Modifier)
        # 构建“进攻因子” (Aggression Factor)
        # 逻辑：成本上移(Slope>0) 且 加速(Accel>0) = 强力进攻
        aggression_bonus = pd.Series(0.0, index=index)
        # 情景A: 推升 (Pushing) -> 成本Slope > 0.2%
        aggression_bonus = aggression_bonus.mask(buy_slope > 0.2, 0.2)
        # 情景B: 抢筹 (Scramble) -> 成本Slope > 0.2% 且 Accel > 0
        aggression_bonus = aggression_bonus.mask((buy_slope > 0.2) & (buy_accel > 0), 0.4)
        # 情景C: 虚拉/对倒 (Fake Pump) -> 股价大涨(>3%) 但成本几乎不动(|Slope| < 0.1%)
        # 这种情况下，静态优势再大也是虚的，需要惩罚
        is_fake_pump = (m['pct_change'] > 3.0) & (buy_slope.abs() < 0.1)
        aggression_bonus = aggression_bonus.mask(is_fake_pump, -0.3)
        # 4. 综合评分
        # 静态分 (0.8) + 动态修正 (0.2 + bonus)
        raw_score = (
            strategic_alpha.fillna(0) * 0.4 + 
            safety_margin.fillna(0) * 0.3 + 
            harvest_prem.fillna(0) * 0.1 + 
            aggression_bonus # 直接叠加 bonus
        )
        final_score = np.tanh(raw_score * 5.0)
        _temp_debug_values["组件_成本优势"] = {
            "avg_buy": avg_buy,
            "buy_slope": buy_slope,
            "aggression_bonus": aggression_bonus,
            "final_score": final_score
        }
        return pd.Series(final_score, index=index).fillna(0)

    def _calculate_main_force_net_activity_score(self, context: Dict, index: pd.Index, config: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V2.2.0 · 资金流动力学模型 (Funds Kinematics Model)】
        在渗透率基础上，引入 Jerk (脉冲) 和 Accel (力度) 进行非线性修正。
        核心逻辑：
        1. 基础渗透率 (Base Penetration): (Buy - Sell) / Amount。
        2. 动力学修正 (Kinematic Adjustment):
           - 点火 (Ignition): Jerk > 0 且 Accel > 0。表示主力突然发力，权重 * 1.4。
           - 衰竭 (Exhaustion): Slope > 0 但 Accel < 0。表示买力虽然为正但在减弱，权重 * 0.8。
           - 恐慌 (Panic): Jerk < 0 且 Accel < 0。表示主力加速出逃，权重 * 1.2 (负向增强)。
        """
        f, m = context['funds'], context['market']
        # 1. 基础渗透率
        net_amt = f['total_buy_amt'] - f['total_sell_amt']
        penetration = net_amt / m['amount']
        # 2. 动力学因子提取
        # 数据层提供的导数通常量级较大或不统一，建议先做方向性判断，或者使用 tanh 压缩
        # 这里我们主要利用其符号和相对强弱
        jerk = f['net_mf_jerk']
        accel = f['net_mf_accel']
        slope = f['net_mf_slope']
        # 3. 构建动力学乘数 (Kinematic Multiplier)
        k_multiplier = pd.Series(1.0, index=index)
        # 场景A: 暴力点火 (Ignition) -> Jerk正向突变，Accel正向加速
        # 这种时候往往是行情的起点，给予高溢价
        mask_ignition = (jerk > 0) & (accel > 0)
        k_multiplier = k_multiplier.mask(mask_ignition, 1.4)
        # 场景B: 买力衰竭 (Exhaustion) -> 资金净流入(Slope>0) 但 加速度为负(Accel<0)
        # 说明主力虽然在买，但手软了，需要降低得分置信度
        mask_exhaustion = (slope > 0) & (accel < 0)
        k_multiplier = k_multiplier.mask(mask_exhaustion, 0.8)
        # 场景C: 加速出逃 (Panic Bailout) -> 资金净流出(Slope<0) 且 加速度为负(Accel<0)
        # 负向加速，说明砸盘力度在加大，让负分更负
        mask_panic = (slope < 0) & (accel < 0)
        # 注意：渗透率为负时，乘以 1.2 会变得更小(更负)，符合逻辑
        k_multiplier = k_multiplier.mask(mask_panic, 1.2)
        # 4. 逆势博弈修正 (保留原逻辑)
        # 黄金坑 & 诱多
        game_multiplier = pd.Series(1.0, index=index)
        golden_pit = (m['pct_change'] < -1.5) & (penetration > 0)
        bull_trap = (m['pct_change'] > 2.5) & (penetration < 0)
        game_multiplier = game_multiplier.mask(golden_pit, 1.3).mask(bull_trap, 1.3)
        # 5. 综合计算
        # Final = tanh(Penetration * Game_Mult * Kinematic_Mult * 5.0)
        raw_score = penetration * game_multiplier * k_multiplier
        final_score = np.tanh(raw_score * 5.0)
        # 6. MTF 融合 (保留)
        _, mtf_weights = self._get_control_parameters(config)
        base_series = pd.Series(final_score, index=index).fillna(0)
        mtf_score = self.helper._get_mtf_score_from_series_slope_accel(
            base_series, mtf_weights, index, method_name, bipolar=True
        )
        _temp_debug_values["组件_净活动(动力学)"] = {
            "penetration": penetration,
            "k_multiplier": k_multiplier, 
            "final_score": final_score
        }
        return mtf_score

    def _calculate_traditional_control_score_components(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V2.1.0 · EMA斐波那契共振模型 - 接口适配版】
        从 context['structure'] 中提取 EMA13/21/55 进行计算。
        """
        s = context['structure']
        ema_13, ema_21, ema_55 = s['ema_13'], s['ema_21'], s['ema_55']
        if ema_13.isnull().all() or ema_55.isnull().all():
            return pd.Series(np.nan, index=index, dtype=np.float32)
        # 1. 趋势矢量 (Trend Vector)
        slope_13 = (ema_13 - ema_13.shift(1)) / ema_13.shift(1).replace(0, np.nan) * 100
        slope_21 = (ema_21 - ema_21.shift(1)) / ema_21.shift(1).replace(0, np.nan) * 100
        slope_55 = (ema_55 - ema_55.shift(1)) / ema_55.shift(1).replace(0, np.nan) * 100
        composite_trend = slope_55 * 0.5 + slope_13 * 0.3 + slope_21 * 0.2
        # 2. 发散张力 (Expansion Tension)
        tension_score = np.tanh((ema_13 - ema_55) / ema_55.replace(0, np.nan) * 10.0) # 系数调整为10以适配百分比
        # 3. 形态共振 (Resonance)
        resonance = pd.Series(1.0, index=index)
        bullish = (ema_13 > ema_21) & (ema_21 > ema_55)
        bearish = (ema_13 < ema_21) & (ema_21 < ema_55)
        resonance = resonance.mask(bullish, 1.2).mask(bearish, 0.8)
        # 4. 汇总
        score = (composite_trend + tension_score) * resonance
        _temp_debug_values["组件_传统控盘"] = {"score": score, "resonance": resonance}
        return score

    def _fuse_control_scores(self, traditional_score: pd.Series, structural_score: pd.Series, context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V3.2.0 · 结构动力学融合 (Structural Dynamics Fusion)】
        在香农熵模型基础上，引入“锁仓加速”验证。
        核心逻辑：
        1. 熵效能 (Entropy Efficiency): 有序度 * VPA效率。
        2. 锁仓加速 (Lock-up Acceleration):
           - 如果筹码稳定性 (Stability) 的加速度 (Accel) > 0，说明筹码沉淀正在加速。
           - 这是一个极强的“主升浪前兆”信号。
        """
        s_struct = context['structure']
        s_sent = context['sentiment']
        # 1. 熵效能因子 (保留原逻辑)
        orderedness = 1.0 / (1.0 + np.exp(s_struct['chip_entropy'] - 1.5)) * 2.0
        vpa_norm = s_sent['vpa_efficiency'].clip(0, 1)
        entropy_factor = (orderedness * 0.6 + vpa_norm * 0.4).clip(0, 1.5)
        # 2. 结构动力学修正 (Structural Dynamics Adjustment)
        # stability_accel > 0 意味着“越来越稳定”
        stab_accel = s_struct['stability_accel']
        # 构建结构乘数
        struct_multiplier = pd.Series(1.0, index=traditional_score.index)
        # 锁仓加速奖励：稳定性在增加(Slope>0) 且 加速增加(Accel>0)
        # 这种状态下，任何技术形态的突破都极可能是真的
        is_locking_up = (s_struct['chip_stability'] > 0.6) & (stab_accel > 0)
        struct_multiplier = struct_multiplier.mask(is_locking_up, 1.2)
        # 3. 资金二阶动量校验 (保留原逻辑，改用 context 中的 jerk/accel)
        # 如果资金流 Jerk < 0 (主力突然撤力)，即使趋势还在，也要小心
        mf_jerk = context['funds']['net_mf_jerk']
        momentum_penalty = pd.Series(1.0, index=traditional_score.index)
        momentum_penalty = momentum_penalty.mask(mf_jerk < 0, 0.9) # 脉冲撤退惩罚 10%
        # 4. 融合
        base_score = traditional_score * 0.3 + structural_score * 0.7
        # Final = Base * Entropy * Struct_Dyn * Momentum_Penalty
        fused_score = base_score * entropy_factor * struct_multiplier * momentum_penalty
        fused_score = fused_score.clip(-1, 1)
        _temp_debug_values["融合_动力学"] = {
            "entropy_factor": entropy_factor, 
            "struct_multiplier": struct_multiplier,
            "fused_score": fused_score
        }
        return fused_score

    def _calculate_control_leverage_model(self, index: pd.Index, fused_score: pd.Series, net_activity_score: pd.Series, 
                                          norm_flow: pd.Series, cost_score: pd.Series, 
                                          norm_t0_buy: pd.Series, norm_t0_sell: pd.Series, 
                                          norm_vwap_up: pd.Series, norm_vwap_down: pd.Series,
                                          context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V2.1.0 · 锁仓共振与波动率门控杠杆 - 接口适配版】
        依赖 context['sentiment'] 中的 bbw 和 turnover。
        """
        sent = context['sentiment']
        # 1. 验证因子
        validation = (net_activity_score.clip(lower=0) * 0.4 + norm_flow * 0.3 + cost_score.clip(lower=0) * 0.2 + norm_t0_buy * 0.1).clip(0, 1)
        # 2. 锁仓奖励 (低换手 + 控盘)
        lockup_bonus = pd.Series(0.0, index=index)
        lockup_bonus = lockup_bonus.mask((fused_score > 0) & (sent['turnover'] < 3.0), 0.5)
        # 3. 波动率门控 (低BBW)
        vol_bonus = (0.2 - sent['bbw']).clip(lower=0) * 5.0
        # 4. 过热惩罚 (高换手)
        overheat = pd.Series(0.0, index=index)
        overheat = overheat.mask(sent['turnover'] > 15.0, (sent['turnover'] - 15.0) * 0.05).clip(upper=1.0)
        # 5. 计算杠杆
        leverage = pd.Series(1.0, index=index, dtype=np.float32)
        # 正向增强
        pos_lev = 1.0 + fused_score * validation + lockup_bonus + vol_bonus - overheat
        leverage = leverage.mask(fused_score > 0, pos_lev)
        # 负向惩罚
        punish = (norm_t0_sell * 0.4 + norm_vwap_down * 0.3 + (1 - norm_flow) * 0.3).clip(0, 1)
        neg_lev = (1.0 + fused_score) * (1.0 - validation * 0.5 - punish * 0.5)
        leverage = leverage.mask(fused_score <= 0, neg_lev)
        final_lev = leverage.clip(0, 2.5)
        _temp_debug_values["风控_杠杆"] = {"leverage": final_lev, "lockup": lockup_bonus, "vol": vol_bonus}
        return final_lev

    def _normalize_components(self, df: pd.DataFrame, context: Dict, scores_traditional: pd.Series, config: Dict, method_name: str, _temp_debug_values: Dict) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        【V4.2.0 · 归一化组件 (Context-Aware)】
        重构：
        1. 废弃原 _normalize_and_mtf_control_components 方法。
        2. 直接基于 context 字典提取数据进行归一化和 MTF 计算。
        3. 逻辑内收，消除中间层，代码更紧凑。
        """
        # 解包数据
        s_struct = context['structure']
        s_sent = context['sentiment']
        # 获取参数
        _, mtf_weights = self._get_control_parameters(config)
        df_index = df.index
        # 1. 传统控盘分归一化 (Bipolar: -1~1)
        norm_traditional = self.helper._normalize_series(scores_traditional, df_index, bipolar=True)
        # 2. 结构控盘分 MTF计算 (基于筹码稳定性)
        # 注意：这里直接使用 context 中的数据列名或 Series
        # 由于 helper._get_mtf_slope_accel_score 需要从 df 中读取衍生列(slope/accel)，
        # 我们依然传入 col_name='chip_stability_D'，前提是 df 中必须有相关列。
        # 如果 context 中的 series 已经是处理过的（如填补了 NaN），
        # 理论上 helper 应该支持直接传入 Series，但根据 helper 签名它需要 df 和 col_name。
        # 为了稳健，这里我们依然指向 df 中的原始列名。
        norm_structural = self.helper._get_mtf_slope_accel_score(
            df, 'chip_stability_D', mtf_weights, df_index, method_name, bipolar=True
        )
        # 3. 辅助指标归一化 (Unipolar: 0~1)
        norm_flow = self.helper._normalize_series(s_sent['flow_consistency'], df_index, bipolar=False)
        norm_t0_buy = self.helper._normalize_series(s_sent['t0_buy_conf'], df_index, bipolar=False)
        norm_t0_sell = self.helper._normalize_series(s_sent['t0_sell_conf'], df_index, bipolar=False)
        norm_vwap_up = self.helper._normalize_series(s_sent['pushing_score'], df_index, bipolar=False)
        norm_vwap_down = self.helper._normalize_series(s_sent['shakeout_score'], df_index, bipolar=False)

        _temp_debug_values["归一化处理"] = {
            "traditional": norm_traditional,
            "structural_mtf": norm_structural,
            "flow_consistency": norm_flow
        }
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
        【辅助方法】验证军械库清单中所需的物理信号是否存在。
        """
        _, mtf_slope_accel_weights = self._get_control_parameters(config)
        # 核心物理信号列表 (基于 V4.1.0 需求)
        required_signals = [
            'close_D', 'amount_D', 'pct_change_D',
            'net_mf_amount_D', 'chip_stability_D', 'flow_consistency_D',
            'buy_lg_amount_D', 'buy_elg_amount_D', 'sell_lg_amount_D', 'sell_elg_amount_D',
            'EMA_13_D', 'EMA_21_D', 'EMA_55_D',
            'BBW_21_2.0_D', 'turnover_rate_D',
            'chip_entropy_D', 'VPA_EFFICIENCY_D',
            'cost_50pct_D', 'winner_rate_D'
        ]
        # 动态添加动力学衍生信号 (Slope/Accel)
        # 注意：数据层可能根据配置生成这些列，如果缺失通常由 _get_safe_series 处理默认值，
        # 但这里的校验是为了确保核心逻辑不跑偏。
        base_sig_proxy = 'chip_stability_D'
        for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
            required_signals.append(f'SLOPE_{period_str}_{base_sig_proxy}')
        for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
            required_signals.append(f'ACCEL_{period_str}_{base_sig_proxy}')
        # 执行校验
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，计算中断。"] = ""
                self._print_debug_info(debug_output)
            return False
        return True










