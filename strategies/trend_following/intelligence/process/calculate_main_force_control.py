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
        【V4.2.0 · 主力控盘关系计算器 - 最终形态】
        职责：调度全量计算流程。
        优化：
        1. 保留 _get_probe_timestamp 用于控制调试开关。
        2. 合并输出逻辑至 _calculate_main_force_control_relationship_debug_output。
        """
        method_name = "calculate_main_force_control_relationship"
        is_debug = get_param_value(self.debug_params.get('enabled'), False)
        print(f"[调试] {method_name} 正在运行，调试模式: {is_debug}, 探针日期: {self.probe_dates}")
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
        # if probe_ts:
        #     self._calculate_main_force_control_relationship_debug_output(
        #         debug_output, 
        #         _temp_debug_values, 
        #         method_name, 
        #         probe_ts
        #     )
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
        【V4.4.0 · HAB 历史累积记忆缓冲系统集成版】
        职责：提取全量物理信号，并计算历史累积数据 (Rolling Accumulation)，构建 HAB 底座。
        新增：
        1. 13/21/34 日的主力资金累积净额 (hab_net_mf_*)。
        2. 13/21/34 日的主力买入总额与总量 (hab_buy_amt_*, hab_buy_vol_*)，用于计算周期 VWAP。
        """
        # 1. 基础量价映射
        market_raw = {
            "close": self._get_safe_series(df, 'close_D', method_name=method_name),
            "amount": self._get_safe_series(df, 'amount_D', method_name=method_name).replace(0, np.nan),
            "pct_change": self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        }
        # 2. 斐波那契均线
        ema_signals = {
            "ema_13": self._get_safe_series(df, 'EMA_13_D', method_name=method_name),
            "ema_21": self._get_safe_series(df, 'EMA_21_D', method_name=method_name),
            "ema_55": self._get_safe_series(df, 'EMA_55_D', method_name=method_name)
        }
        # 3. 资金分层映射 (基础日频)
        funds_raw = {
            "buy_elg_amt": self._get_safe_series(df, 'buy_elg_amount_D', 0.0, method_name=method_name),
            "buy_lg_amt": self._get_safe_series(df, 'buy_lg_amount_D', 0.0, method_name=method_name),
            "sell_elg_amt": self._get_safe_series(df, 'sell_elg_amount_D', 0.0, method_name=method_name),
            "sell_lg_amt": self._get_safe_series(df, 'sell_lg_amount_D', 0.0, method_name=method_name),
            "buy_elg_vol": self._get_safe_series(df, 'buy_elg_vol_D', 0.0, method_name=method_name),
            "buy_lg_vol": self._get_safe_series(df, 'buy_lg_vol_D', 0.0, method_name=method_name),
            "sell_elg_vol": self._get_safe_series(df, 'sell_elg_vol_D', 0.0, method_name=method_name),
            "sell_lg_vol": self._get_safe_series(df, 'sell_lg_vol_D', 0.0, method_name=method_name),
            "net_mf_calibrated": self._get_safe_series(df, 'net_mf_amount_D', 0.0, method_name=method_name),
            "net_mf_jerk": self._get_safe_series(df, 'JERK_5_net_mf_amount_D', 0.0, method_name=method_name),
            "net_mf_accel": self._get_safe_series(df, 'ACCEL_5_net_mf_amount_D', 0.0, method_name=method_name),
            "net_mf_slope": self._get_safe_series(df, 'SLOPE_5_net_mf_amount_D', 0.0, method_name=method_name)
        }
        # 聚合日频总量 (用于 HAB 计算)
        # 加权逻辑：超大单(ELG)权重 1.5，大单(LG)权重 1.0 (与 avg_prices 保持一致)
        w_elg, w_lg = 1.5, 1.0
        daily_buy_amt_weighted = funds_raw["buy_elg_amt"] * w_elg + funds_raw["buy_lg_amt"] * w_lg
        daily_buy_vol_weighted = funds_raw["buy_elg_vol"] * w_elg + funds_raw["buy_lg_vol"] * w_lg
        daily_net_mf = funds_raw["net_mf_calibrated"]
        funds_raw["total_buy_amt"] = funds_raw["buy_elg_amt"] + funds_raw["buy_lg_amt"]
        funds_raw["total_sell_amt"] = funds_raw["sell_elg_amt"] + funds_raw["sell_lg_amt"]

        # --- HAB 系统构建 (Historical Accumulation Buffer) ---
        # 计算 13, 21, 34 日的滚动累积
        hab_periods = [13, 21, 34]
        hab_data = {}
        for p in hab_periods:
            # 资金 HAB: 累积净流入 (Inventory)
            hab_data[f"hab_net_mf_{p}"] = daily_net_mf.rolling(window=p, min_periods=1).sum()
            # 成本 HAB: 累积买入金额与量 (用于计算 Rolling VWAP)
            hab_data[f"hab_buy_amt_{p}"] = daily_buy_amt_weighted.rolling(window=p, min_periods=1).sum()
            hab_data[f"hab_buy_vol_{p}"] = daily_buy_vol_weighted.rolling(window=p, min_periods=1).sum()

        # 将 HAB 数据注入 funds
        funds_raw.update(hab_data)

        # 4. 筹码结构映射
        structure_raw = {
            "cost_50pct": self._get_safe_series(df, 'cost_50pct_D', method_name=method_name).replace(0, np.nan),
            "chip_stability": self._get_safe_series(df, 'chip_stability_D', 0.5, method_name=method_name),
            "chip_entropy": self._get_safe_series(df, 'chip_entropy_D', 1.0, method_name=method_name),
            "winner_rate": self._get_safe_series(df, 'winner_rate_D', 50.0, method_name=method_name),
            "stability_accel": self._get_safe_series(df, 'ACCEL_5_chip_stability_D', 0.0, method_name=method_name)
        }
        # 5. 情绪与行为
        sentiment_behavior = {
            "vpa_efficiency": self._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.0, method_name=method_name),
            "bbw": self._get_safe_series(df, 'BBW_21_2.0_D', 0.1, method_name=method_name),
            "flow_consistency": self._get_safe_series(df, 'flow_consistency_D', 0.5, method_name=method_name),
            "t0_buy_conf": self._get_safe_series(df, 'intraday_accumulation_confidence_D', 0.0, method_name=method_name),
            "t0_sell_conf": self._get_safe_series(df, 'intraday_distribution_confidence_D', 0.0, method_name=method_name),
            "pushing_score": self._get_safe_series(df, 'pushing_score_D', 0.0, method_name=method_name),
            "shakeout_score": self._get_safe_series(df, 'shakeout_score_D', 0.0, method_name=method_name),
            "pushing_jerk": self._get_safe_series(df, 'JERK_5_pushing_score_D', 0.0, method_name=method_name),
            "turnover": self._get_safe_series(df, 'turnover_rate_D', 1.0, method_name=method_name)
        }
        context = {"market": market_raw, "ema": ema_signals, "funds": funds_raw, "structure": structure_raw, "sentiment": sentiment_behavior}
        if probe_ts:
            print(f"[探针] HAB 系统数据注入校验:")
            print(f"      - funds.hab_net_mf_21 exists: {'hab_net_mf_21' in context['funds']}")
            print(f"      - funds.hab_buy_amt_21 exists: {'hab_buy_amt_21' in context['funds']}")
            raw_snapshot = {}
            for cat_name, cat_data in context.items():
                for k, v in cat_data.items():
                    raw_snapshot[f"{cat_name}.{k}"] = v.loc[probe_ts] if probe_ts in v.index else np.nan
            _temp_debug_values["原料数据快照"] = raw_snapshot
            
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

    def _calculate_main_force_avg_prices(self, df_index: pd.Index, close_price: pd.Series, 
                                         buy_lg_amt: pd.Series, buy_elg_amt: pd.Series, 
                                         sell_lg_amt: pd.Series, sell_elg_amt: pd.Series, 
                                         buy_lg_vol: pd.Series, buy_elg_vol: pd.Series, 
                                         sell_lg_vol: pd.Series, sell_elg_vol: pd.Series, 
                                         _temp_debug_values: Dict) -> Dict[str, pd.Series]:
        """
        【V3.0.0 · HAB-VWAP 成本动力学模型】
        引入历史累积记忆缓冲 (HAB) 来计算主力真实的持仓成本 (Inventory Cost)。
        核心升级：
        1. 存量意识 (Inventory Awareness): 
           - 废弃 EMA(5) 短线均价。
           - 启用 Rolling(21) 计算“21日主力持仓成本线” (HAB-VWAP)。
           - 逻辑：主力建仓周期通常在 13-34 天，21 天是斐波那契中枢，能有效过滤日内对倒噪音。
        2. 动力学 (Kinematics): 
           - 在 HAB 成本线基础上计算 Slope(趋势), Accel(力度), Jerk(突变)。
           - 此时的 Slope > 0 代表主力“底仓抬升”，而非仅仅是“今日买贵了”。
        """
        # 权重定义：超大单(ELG)权重 1.5 (Smart Money)，大单(LG)权重 1.0
        w_elg, w_lg = 1.5, 1.0
        # 1. 计算日频加权数据 (Daily Weighted Data)
        daily_buy_amt_w = (buy_elg_amt * w_elg) + (buy_lg_amt * w_lg)
        daily_buy_vol_w = (buy_elg_vol * w_elg) + (buy_lg_vol * w_lg)
        daily_sell_amt_w = (sell_elg_amt * w_elg) + (sell_lg_amt * w_lg)
        daily_sell_vol_w = (sell_elg_vol * w_elg) + (sell_lg_vol * w_lg)
        # 2. 构建 HAB-VWAP (21日滚动累积)
        # 逻辑：21日累积金额 / 21日累积成交量 = 21日持仓均价
        hab_window = 21
        # 买入侧 HAB
        hab_buy_amt = daily_buy_amt_w.rolling(window=hab_window, min_periods=1).sum()
        hab_buy_vol = daily_buy_vol_w.rolling(window=hab_window, min_periods=1).sum()
        # 卖出侧 HAB (压力位)
        hab_sell_amt = daily_sell_amt_w.rolling(window=hab_window, min_periods=1).sum()
        hab_sell_vol = daily_sell_vol_w.rolling(window=hab_window, min_periods=1).sum()
        # 3. 价格计算与量级修正 (Unit Correction)
        # 防止 A 股成交额(万元)与成交量(股)单位不统一导致的计算偏离
        raw_buy_price = hab_buy_amt / hab_buy_vol.replace(0, np.nan)
        unit_correction = 1.0
        # 动态探测修正因子
        avg_raw = raw_buy_price.mean()
        avg_close = close_price.mean()
        if avg_raw > 0 and avg_close / avg_raw > 50:
            unit_correction = 100.0 if avg_close / avg_raw < 500 else 10000.0
            print(f"[探针] HAB-VWAP 成本量级修正因子: {unit_correction}")
            
        hab_buy_price = (raw_buy_price * unit_correction).fillna(close_price)
        hab_sell_price = (hab_sell_amt / hab_sell_vol.replace(0, np.nan) * unit_correction).fillna(close_price)
        # 4. 基于 HAB 成本的动力学计算 (Kinematics on HAB)
        # 由于 HAB 本身已具备平滑性，直接求导即可，无需再做 EMA
        # Level 1: Slope (趋势) - 使用百分比
        buy_slope = (hab_buy_price - hab_buy_price.shift(3)) / hab_buy_price.shift(3).replace(0, np.nan) * 100
        sell_slope = (hab_sell_price - hab_sell_price.shift(3)) / hab_sell_price.shift(3).replace(0, np.nan) * 100
        # Level 2: Accel (力度) - 使用差分
        buy_accel = buy_slope - buy_slope.shift(1)
        # Level 3: Jerk (突变) - 使用差分
        buy_jerk = buy_accel - buy_accel.shift(1)
        result = {
            "avg_buy": hab_buy_price,     # HAB-VWAP
            "buy_slope": buy_slope.fillna(0),
            "buy_accel": buy_accel.fillna(0),
            "buy_jerk": buy_jerk.fillna(0),
            "avg_sell": hab_sell_price,   # HAB-VWAP
            "sell_slope": sell_slope.fillna(0)
        }
        _temp_debug_values["主力平均价格计算"] = result
        return result

    def _calculate_main_force_cost_advantage_score(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V2.5.0 · 成本优势模型 - HAB 适配版】
        职责：利用 HAB-VWAP 和动力学指标，评估主力成本优势与进攻意图。
        核心逻辑：
        1. 静态优势 (Static): HAB成本 vs 市场成本。
        2. 动态进攻 (Aggression): HAB成本 Slope/Accel。
        """
        m, f, s = context['market'], context['funds'], context['structure']
        # 1. 调用 HAB-VWAP 计算逻辑
        prices = self._calculate_main_force_avg_prices(
            index, m['close'],
            f['buy_lg_amt'], f['buy_elg_amt'], f['sell_lg_amt'], f['sell_elg_amt'],
            f['buy_lg_vol'], f['buy_elg_vol'], f['sell_lg_vol'], f['sell_elg_vol'],
            _temp_debug_values
        )
        avg_buy = prices['avg_buy']     # 21日持仓均价
        avg_sell = prices['avg_sell']   # 21日卖出均价
        buy_slope = prices['buy_slope'] # 持仓成本趋势
        buy_accel = prices['buy_accel'] # 抢筹力度
        # 2. 静态优势 (Static Alpha)
        # 战略Alpha: 主力21日成本比市场平均成本(CYQ)低多少 -> 成本护城河
        strategic_alpha = (s['cost_50pct'] - avg_buy) / s['cost_50pct'].replace(0, np.nan)
        # 安全垫: 现价 vs 主力21日成本 -> 获利空间
        safety_margin = (m['close'] - avg_buy) / avg_buy.replace(0, np.nan)
        # 收割能力: 主力21日卖出均价 vs 市场成本 -> 高位派发能力
        harvest_prem = (avg_sell - s['cost_50pct']) / s['cost_50pct'].replace(0, np.nan)
        # 3. 动态意图修正 (Dynamic Modifier)
        aggression_bonus = pd.Series(0.0, index=index)
        # 场景A: 趋势推升 (Trend Push) -> 成本线稳步上移 (Slope > 0.1%)
        # 这代表主力在不断用更高的价格拿货，是真金白银的做多
        aggression_bonus = aggression_bonus.mask(buy_slope > 0.1, 0.2)
        # 场景B: 暴力抢筹 (Aggressive Scramble) -> 成本加速上移 (Accel > 0)
        aggression_bonus = aggression_bonus.mask((buy_slope > 0.1) & (buy_accel > 0), 0.4)
        # 场景C: 虚拉诱多 (Fake Pump) -> 股价大涨(>3%) 但成本线不动(|Slope| < 0.05%)
        # 说明主力在用少量筹码对倒拉升股价，底仓没动，极度危险
        is_fake_pump = (m['pct_change'] > 3.0) & (buy_slope.abs() < 0.05)
        aggression_bonus = aggression_bonus.mask(is_fake_pump, -0.3)
        # 4. 综合评分
        # 静态分(0.8) + 动态修正(Bonus)
        raw_score = (
            strategic_alpha.fillna(0) * 0.4 + 
            safety_margin.fillna(0) * 0.3 + 
            harvest_prem.fillna(0) * 0.1 + 
            aggression_bonus
        )
        final_score = np.tanh(raw_score * 5.0)
        _temp_debug_values["组件_成本优势"] = {
            "hab_vwap_buy": avg_buy,
            "buy_slope": buy_slope,
            "aggression_bonus": aggression_bonus,
            "final_score": final_score
        }
        # 探针
        print(f"[探针] HAB-VWAP 成本模型计算完成。HAB均价: {avg_buy.mean():.4f}, 成本趋势(Slope): {buy_slope.mean():.4f}")
        return pd.Series(final_score, index=index).fillna(0)

    def _calculate_main_force_net_activity_score(self, context: Dict, index: pd.Index, config: Dict, method_name: str, _temp_debug_values: Dict) -> pd.Series:
        """
        【V3.0.0 · HAB 缓冲型净活动模型】
        引入历史累积记忆缓冲 (HAB) 来评估资金流出的真实冲击。
        核心逻辑：
        1. 存量缓冲 (Inventory Buffer): 使用 21日累积净流入 (hab_net_mf_21) 作为分母。
        2. 冲击率 (Impact Ratio): 当日净流出 / 累积净流入。
           - 若累积了100亿，流出2亿，Impact = -2%，微不足道 -> 得分不应大幅下降。
           - 若累积仅5亿，流出2亿，Impact = -40%，严重出逃 -> 得分大幅下降。
        """
        f, m = context['funds'], context['market']
        # 1. 提取 HAB 数据 (21日窗口)
        hab_net = f['hab_net_mf_21']
        daily_net = f['net_mf_calibrated']
        # 2. 计算 HAB 修正后的渗透率
        # 基础渗透率 (针对全市场成交额)
        base_penetration = daily_net / m['amount'].replace(0, np.nan)
        # 缓冲因子 (Buffer Factor)
        # 逻辑：如果 HAB 是正的(有底仓)，则流出会被缓冲；如果 HAB 是负的(空头主导)，流出会加剧恐慌
        buffer_impact = pd.Series(1.0, index=index)
        # 情况A: 有底仓 (HAB > 0) 且 今日流出 (Daily < 0)
        # 计算流出占底仓的比例
        has_inventory = (hab_net > 0) & (daily_net < 0)
        outflow_ratio = (daily_net / hab_net.replace(0, np.nan)).abs() # 例如 2亿/100亿 = 0.02
        # 如果流出比例很小 (<10%)，则大幅降低 base_penetration 的负面权重
        # 意为：“这点流出对主力底仓来说是九牛一毛”
        # 修正系数：(1 - exp(-10 * ratio)) -> ratio=0.02时系数~0.18，即负分打2折
        dampener = 1.0 - np.exp(-10.0 * outflow_ratio)
        buffer_impact = buffer_impact.mask(has_inventory, dampener)
        # 情况B: 无底仓 (HAB < 0) 且 今日流出 (Daily < 0)
        # 落井下石，无需缓冲，保持原样或放大 (此处保持 1.0)
        # 3. 应用缓冲
        buffered_score = base_penetration * buffer_impact
        # 4. 动力学修正 (保留原有逻辑，应用在缓冲后的分数上)
        jerk = f['net_mf_jerk']
        accel = f['net_mf_accel']
        slope = f['net_mf_slope']
        k_multiplier = pd.Series(1.0, index=index)
        mask_ignition = (jerk > 0) & (accel > 0) # 点火
        k_multiplier = k_multiplier.mask(mask_ignition, 1.4)
        mask_panic = (slope < 0) & (accel < 0) # 恐慌加速
        k_multiplier = k_multiplier.mask(mask_panic, 1.2)
        # 5. 逆势博弈修正
        game_multiplier = pd.Series(1.0, index=index)
        golden_pit = (m['pct_change'] < -1.5) & (buffered_score > 0)
        game_multiplier = game_multiplier.mask(golden_pit, 1.3)
        # 6. 最终计算
        final_score = np.tanh(buffered_score * game_multiplier * k_multiplier * 5.0)
        # MTF 融合
        _, mtf_weights = self._get_control_parameters(config)
        base_series = pd.Series(final_score, index=index).fillna(0)
        mtf_score = self.helper._get_mtf_score_from_series_slope_accel(
            base_series, mtf_weights, index, method_name, bipolar=True
        )
        _temp_debug_values["组件_净活动(动力学)"] = {
            "daily_net": daily_net,
            "hab_net": hab_net,
            "buffer_impact": buffer_impact, 
            "final_score": final_score
        }
        return mtf_score

    def _calculate_traditional_control_score_components(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.5.0 · 传统控盘 - 三阶动力学全息模型】
        职责：通过均线系统的三阶导数（Slope, Accel, Jerk）识别趋势的生长与衰竭。
        修改说明：
        1. 引入 EMA-Jerk: 捕捉均线弯曲曲率的突变，用于预警“假突破”后的瞬间反杀 。
        2. 结构动力学加权: 趋势分不再是静态的 0/1，而是由生命线(EMA55)的斜率与加速度共同决定 。
        3. 离散度压力测试: 利用 Tension (乖离率) 评估趋势的拉伸极限 。
        """
        # 1. 信号提取 (基于 context 映射) 
        ema = context['ema']
        ema_13, ema_21, ema_55 = ema['ema_13'], ema['ema_21'], ema['ema_55']
        close = context['market']['close']
        # 2. EMA 55 (生命线) 动力学建模 
        # Slope: 趋势方向 (Velocity)
        slope_55 = (ema_55 - ema_55.shift(1)) / ema_55.shift(1).replace(0, np.nan) * 100
        # Accel: 趋势力度 (Acceleration)
        accel_55 = slope_55 - slope_55.shift(1)
        # Jerk: 趋势拐点预警 (Jerk) - 核心新增
        jerk_55 = accel_55 - accel_55.shift(1)
        # 3. EMA 13 (攻击线) 动力学建模 (用于捕捉超短线背离) 
        slope_13 = (ema_13 - ema_13.shift(1)) / ema_13.shift(1).replace(0, np.nan) * 100
        accel_13 = slope_13 - slope_13.shift(1)
        # 4. 动力学多级评分逻辑 
        # 基础分: 生命线斜率映射 (tanh 压缩)
        base_trend_score = np.tanh(slope_55 * 5.0) 
        # 动态修正 (Kinematic Modifier)
        modifier = pd.Series(0.0, index=index)
        # 场景A: 趋势共振增强 (Jerk > 0 且 Accel > 0) -> 加速主升 
        modifier = modifier.mask((slope_55 > 0) & (accel_55 > 0) & (jerk_55 > 0), 0.25)
        # 场景B: 顶部曲率钝化 (Slope > 0 但 Jerk < 0) -> 预警“踩刹车” 
        modifier = modifier.mask((slope_55 > 0) & (jerk_55 < 0), -0.2)
        # 场景C: 空头衰竭 (Slope < 0 但 Accel > 0) -> 底部圆弧化 
        modifier = modifier.mask((slope_55 < 0) & (accel_55 > 0), 0.15)
        # 5. 形态共振与张力 (Resonance & Tension) 
        # 多头/空头排列判断
        bullish_alignment = (ema_13 > ema_21) & (ema_21 > ema_55)
        bearish_alignment = (ema_13 < ema_21) & (ema_21 < ema_55)
        alignment_score = pd.Series(0.0, index=index)
        alignment_score = alignment_score.mask(bullish_alignment, 0.4).mask(bearish_alignment, -0.4)
        # 乖离率张力 (Tension)
        tension = (close - ema_55) / ema_55.replace(0, np.nan)
        tension_score = np.tanh(tension * 3.0) * 0.2
        # 6. 综合计算 (Final Fusion) 
        final_score = (base_trend_score * 0.4 + alignment_score + tension_score + modifier).clip(-1, 1)
        # 7. 探针捕获与输出 
        _temp_debug_values["组件_传统控盘"] = {
            "slope_55": slope_55,
            "accel_55": accel_55,
            "jerk_55": jerk_55,
            "modifier": modifier,
            "final_score": final_score
        }
        print(f"[探针] EMA动力学诊断: 斜率={slope_55.mean():.4f}, 加速度={accel_55.mean():.4f}, 急动度={jerk_55.mean():.4f}")
        return final_score

    def _fuse_control_scores(self, traditional_score: pd.Series, structural_score: pd.Series, context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V4.0.0 · 香农熵-HAB动力学全息融合模型】
        职责：将各组件分数进行非线性融合，识别具备“存量支撑”且“增量加速”的真控盘。
        版本更新说明：
        1. 引入 HAB 存量意识：使用 34 日累积资金流作为融合得分的“物理质量”背景。
        2. 引入动力学校验：利用 net_mf_jerk(急动度) 识别主力是否在共振点“踩刹车”。
        3. 结构熵增强：结合 chip_entropy_D 评估筹码结构的“有序度”对融合分的乘数效应。
        """
        s_struct = context['structure']
        s_sent = context['sentiment']
        f_funds = context['funds']
        # 1. 计算【香农-VPA效能因子】(增量效能)
        # 有序度映射：熵越低(有序)，分越高。中心点 1.5 位移。
        orderedness = 1.0 / (1.0 + np.exp(s_struct['chip_entropy'] - 1.5)) * 2.0
        vpa_norm = s_sent['vpa_efficiency'].clip(0, 1)
        # 效能 = 有序度(60%) + VPA效率(40%)
        entropy_efficiency_factor = (orderedness * 0.6 + vpa_norm * 0.4).clip(0, 1.5)
        # 2. 计算【HAB 信心乘数】(存量支撑)
        # 逻辑：考察 34 日累积资金存量。若存量深厚，则得分具备更强的抗噪能力。
        hab_34 = f_funds.get('hab_net_mf_34', pd.Series(0.0, index=traditional_score.index))
        # 信心函数：使用 tanh 映射，当存量达到正向极端值时，给予 1.2 倍的融合溢价
        hab_confidence = 1.0 + np.tanh(hab_34 / 100000.0) * 0.2 # 100000为量级基准
        # 3. 计算【动力学一票否决】(瞬时修正)
        # 提取资金流的急动度 (Jerk) 和加速度 (Accel)
        mf_jerk = f_funds.get('net_mf_jerk', pd.Series(0.0, index=traditional_score.index))
        mf_accel = f_funds.get('net_mf_accel', pd.Series(0.0, index=traditional_score.index))
        # 预警逻辑：如果融合分 > 0 且 (Jerk < 0 或 Accel < 0)，说明虽然在控盘，但“力道”在衰减
        momentum_deterioration = (traditional_score > 0) & ((mf_jerk < 0) | (mf_accel < 0))
        momentum_penalty = pd.Series(1.0, index=traditional_score.index)
        momentum_penalty = momentum_penalty.mask(momentum_deterioration, 0.85) # 发现减速，融合分贴现 15%
        # 4. 最终非线性融合公式
        # $$Fused = (Trad * 0.3 + Struct * 0.7) * EntropyEff * HABConf * MomentumPenalty$$
        base_fusion = (traditional_score * 0.3 + structural_score * 0.7)
        final_fused = (base_fusion * entropy_efficiency_factor * hab_confidence * momentum_penalty).clip(-1, 1)
        # 5. 探针捕获与快照
        _temp_debug_values["融合_动力学"] = {
            "entropy_eff": entropy_efficiency_factor,
            "hab_conf": hab_confidence,
            "momentum_penalty": momentum_penalty,
            "fused_score": final_fused
        }
        print(f"[探针] 融合模型执行完毕。HAB信心系数均值: {hab_confidence.mean():.4f}, 动力学惩罚覆盖率: {momentum_deterioration.mean():.2%}")
        return final_fused

    def _calculate_control_leverage_model(self, index: pd.Index, fused_score: pd.Series, net_activity_score: pd.Series, 
                                          norm_flow: pd.Series, cost_score: pd.Series, 
                                          norm_t0_buy: pd.Series, norm_t0_sell: pd.Series, 
                                          norm_vwap_up: pd.Series, norm_vwap_down: pd.Series,
                                          context: Dict, _temp_debug_values: Dict) -> pd.Series:
        """
        【V3.0.0 · HAB-动力学增强型全息杠杆模型】
        职责：基于存量成本水位与瞬时动力学状态，动态锁定最终控盘信号的杠杆倍率。
        修改说明：
        1. 引入 HAB-Bias: 计算现价与 21日 HAB 成本线的偏离，定义“博弈安全边际”。
        2. 动力学门控 (Kinematic Gating): 利用 net_mf_jerk 识别主力撤力风险。
        3. 筹码熵约束: 利用 chip_entropy 识别控盘质量，对混乱筹码结构执行杠杆惩罚。
        """
        sent = context['sentiment']
        funds = context['funds']
        struct = context['structure']
        market = context['market']
        # 1. 计算【HAB-Bias 存量偏离因子】
        # 提取之前在 _calculate_main_force_avg_prices 计算并存入 debug 的 hab_vwap_buy
        # 若缺失则现场重算 (21日 Rolling VWAP 逻辑)
        hab_cost = _temp_debug_values.get("主力平均价格计算", {}).get("avg_buy", market['close'])
        hab_bias = (market['close'] - hab_cost) / hab_cost.replace(0, np.nan)
        # 逻辑：偏离度越负(深套)，反转潜力越大，给予杠杆补偿；偏离度过正(获利盘多)，杠杆衰减。
        bias_multiplier = 1.0 - np.tanh(hab_bias * 2.0) * 0.3
        # 2. 计算【动力学安全阀 (Kinetic Safety Valve)】
        # 识别资金流是否在“踩刹车”
        mf_jerk = funds.get('net_mf_jerk', pd.Series(0.0, index=index))
        mf_accel = funds.get('net_mf_accel', pd.Series(0.0, index=index))
        kinetic_valve = pd.Series(1.0, index=index)
        # 若控盘分 > 0 且出现负向突变 (Jerk < 0)，强制杠杆打 7 折
        kinetic_valve = kinetic_valve.mask((fused_score > 0) & (mf_jerk < 0), 0.7)
        # 3. 计算【筹码结构熵惩罚】
        # 筹码越乱 (Entropy > 2.5)，杠杆越低
        entropy_penalty = (1.0 / (1.0 + np.exp(struct['chip_entropy'] - 2.5))).clip(0.5, 1.0)
        # 4. 基础验证因子 (继承原有全息验证逻辑)
        validation = (net_activity_score.clip(lower=0) * 0.4 + norm_flow * 0.3 + cost_score.clip(lower=0) * 0.2 + norm_t0_buy * 0.1).clip(0, 1)
        # 5. 波动率与换手率因子 (锁仓共振)
        lockup_bonus = pd.Series(0.0, index=index)
        lockup_bonus = lockup_bonus.mask((fused_score > 0) & (sent['turnover'] < 3.0), 0.5)
        vol_bonus = (0.2 - sent['bbw']).clip(lower=0) * 5.0
        # 6. 综合杠杆合成
        # Leverage = (Base + Bonuses) * BiasFactor * KineticValve * EntropyPenalty
        leverage = pd.Series(1.0, index=index, dtype=np.float32)
        pos_lev = (1.0 + fused_score * validation + lockup_bonus + vol_bonus) * bias_multiplier * kinetic_valve * entropy_penalty
        leverage = leverage.mask(fused_score > 0, pos_lev)
        # 负向惩罚逻辑：保留原有高效出货识别
        punish = (norm_t0_sell * 0.4 + norm_vwap_down * 0.3 + (1 - norm_flow) * 0.3).clip(0, 1)
        neg_lev = (1.0 + fused_score) * (1.0 - validation * 0.5 - punish * 0.5)
        leverage = leverage.mask(fused_score <= 0, neg_lev)
        final_lev = leverage.clip(0, 2.5)
        # 7. 探针捕获
        _temp_debug_values["风控_杠杆"] = {
            "hab_bias": hab_bias,
            "bias_multiplier": bias_multiplier,
            "kinetic_valve": kinetic_valve,
            "entropy_penalty": entropy_penalty,
            "final_leverage": final_lev
        }
        print(f"[探针] 杠杆模型闭环。HAB偏离均值: {hab_bias.mean():.4f}, 最终平均杠杆: {final_lev.mean():.4f}")
        return final_lev

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
        【V1.3.0 · 军械库全链路动力学与存量验证版】
        职责：作为调度器的“守门人”，在计算开始前强制检查动力学信号（增量）与HAB基础信号（存量）的完整性。
        更新说明：
        1. 动力学校验：新增对 JERK、ACCEL、SLOPE 等高阶导数信号的强制检查。 
        2. HAB原料校验：确保 buy_lg_vol_D 等计算持仓成本所需的物理量能字段存在。 
        3. 决策原料校验：加入 chip_entropy_D 和 VPA_EFFICIENCY_D 等结构化决策指标。 
        """
        _, mtf_slope_accel_weights = self._get_control_parameters(config)
        # 1. 定义核心物理原料清单 (用于 HAB 存量计算)
        required_physical_raw = [
            'close_D', 'amount_D', 'pct_change_D', 'turnover_rate_D', # 基础量价 [cite: 2]
            'buy_elg_amount_D', 'buy_lg_amount_D', 'sell_elg_amount_D', 'sell_lg_amount_D', # 资金金额 [cite: 2]
            'buy_elg_vol_D', 'buy_lg_vol_D', 'sell_elg_vol_D', 'sell_lg_vol_D', # 资金量能 (HAB-VWAP核心) [cite: 2]
            'net_mf_amount_D', 'flow_consistency_D', 'chip_stability_D' # 资金/筹码性质 [cite: 2]
        ]
        # 2. 定义动力学与决策指标清单 (用于 增量与共振 计算)
        required_kinematics_decision = [
            'JERK_5_net_mf_amount_D', 'ACCEL_5_net_mf_amount_D', 'SLOPE_5_net_mf_amount_D', # 资金动力学 
            'ACCEL_5_chip_stability_D', 'JERK_5_pushing_score_D', # 结构动力学 
            'EMA_13_D', 'EMA_21_D', 'EMA_55_D', # 斐波那契均线系统 
            'chip_entropy_D', 'VPA_EFFICIENCY_D', 'BBW_21_2.0_D', # 结构熵与效能 
            'cost_50pct_D', 'winner_rate_D', 'pushing_score_D', 'shakeout_score_D' # 筹码博弈与意图 
        ]
        # 3. 动态扩展 MTF 信号校验 (基于配置)
        dynamic_mtf_signals = []
        base_sig_proxy = 'chip_stability_D'
        for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
            dynamic_mtf_signals.append(f'SLOPE_{period_str}_{base_sig_proxy}')
        for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
            dynamic_mtf_signals.append(f'ACCEL_{period_str}_{base_sig_proxy}')
        # 4. 汇总全链路验证清单
        full_validation_list = list(set(required_physical_raw + required_kinematics_decision + dynamic_mtf_signals))
        # 5. 执行验证探针
        if probe_ts:
            print(f"[探针] _validate_arsenal_signals 启动全链路自检 @ {probe_ts}")
            print(f"      - 物理原料检查项: {len(required_physical_raw)} 个")
            print(f"      - 动力学与决策检查项: {len(required_kinematics_decision)} 个")
        # 6. 调用 helper 执行底层列存在性检查
        if not self.helper._validate_required_signals(df, full_validation_list, method_name):
            if probe_ts:
                missing_cols = [col for col in full_validation_list if col not in df.columns]
                debug_output[f"    -> [过程情报警告] {method_name} 关键军械库信号缺失: {missing_cols[:5]}... 计算被迫中断。"] = ""
                self._print_debug_info(debug_output)
            return False
        return True








