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
        【V4.3.4 · 结构化上下文全量映射版】
        职责：对照《最终军械库清单》提取全量物理信号，并将其精准投喂到下游各组件所需的字典键位。
        修改：1. 将 turnover 移入 sentiment 以适配杠杆模型。2. 将 stability_accel 移入 structure 以适配融合模型。3. 补全了 funds 桶中的全量导数指标。
        """
        market_raw = {
            "close": self._get_safe_series(df, 'close_D', method_name=method_name),
            "amount": self._get_safe_series(df, 'amount_D', method_name=method_name).replace(0, np.nan),
            "pct_change": self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        }
        ema_signals = {
            "ema_13": self._get_safe_series(df, 'EMA_13_D', method_name=method_name),
            "ema_21": self._get_safe_series(df, 'EMA_21_D', method_name=method_name),
            "ema_55": self._get_safe_series(df, 'EMA_55_D', method_name=method_name)
        }
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
        funds_raw["total_buy_amt"] = funds_raw["buy_elg_amt"] + funds_raw["buy_lg_amt"]
        funds_raw["total_sell_amt"] = funds_raw["sell_elg_amt"] + funds_raw["sell_lg_amt"]
        structure_raw = {
            "cost_50pct": self._get_safe_series(df, 'cost_50pct_D', method_name=method_name).replace(0, np.nan),
            "chip_stability": self._get_safe_series(df, 'chip_stability_D', 0.5, method_name=method_name),
            "chip_entropy": self._get_safe_series(df, 'chip_entropy_D', 1.0, method_name=method_name),
            "winner_rate": self._get_safe_series(df, 'winner_rate_D', 50.0, method_name=method_name),
            "stability_accel": self._get_safe_series(df, 'ACCEL_5_chip_stability_D', 0.0, method_name=method_name)
        }
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
            print(f"[探针] _get_raw_control_signals 数据链路注入校验:")
            print(f"      - funds.net_mf_jerk exists: {'net_mf_jerk' in context['funds']}")
            print(f"      - structure.stability_accel exists: {'stability_accel' in context['structure']}")
            print(f"      - sentiment.turnover exists: {'turnover' in context['sentiment']}")
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
        【V2.4.1 · 主力成本动力学模型 - 量级修正版】
        修改说明：引入 unit_correction_factor 修正 A 股成交额与成交量的单位差异（通常为 100 或 10000）。
        """
        w_elg, w_lg = 1.5, 1.0
        # 1. 计算加权总量
        buy_amt_w = (buy_elg_amt * w_elg) + (buy_lg_amt * w_lg)
        buy_vol_w = (buy_elg_vol * w_elg) + (buy_lg_vol * w_lg)
        # 2. 价格计算与单位修正
        # 基于探针诊断：当前 0.1153 vs 11.48，需放大 100 倍。
        # 动态逻辑：如果计算出的单日均价与现价偏离超过 5 倍，自动尝试数量级对齐。
        raw_daily_buy = (buy_amt_w / buy_vol_w.replace(0, np.nan))
        unit_correction = 1.0
        # 探针逻辑：自动识别万元/股或百元/股的差异
        avg_raw = raw_daily_buy.mean()
        avg_close = close_price.mean()
        if avg_raw > 0 and avg_close / avg_raw > 50:
            unit_correction = 100.0 if avg_close / avg_raw < 500 else 10000.0
            print(f"[探针] 检测到成本量级偏离，应用修正因子: {unit_correction}")
        daily_buy = (raw_daily_buy * unit_correction).fillna(close_price)
        # 3. 预平滑与动力学 (保持原有逻辑)
        avg_buy = daily_buy.ewm(span=5, adjust=False).mean()
        buy_slope_raw = (avg_buy - avg_buy.shift(3)) / avg_buy.shift(3).replace(0, np.nan) * 100
        buy_accel_raw = buy_slope_raw - buy_slope_raw.shift(1)
        buy_jerk_raw = buy_accel_raw - buy_accel_raw.shift(1)
        # 卖出侧同步修正
        sell_amt_w = (sell_elg_amt * w_elg) + (sell_lg_amt * w_lg)
        sell_vol_w = (sell_elg_vol * w_elg) + (sell_lg_vol * w_lg)
        daily_sell = (sell_amt_w / sell_vol_w.replace(0, np.nan) * unit_correction).fillna(close_price)
        avg_sell = daily_sell.ewm(span=5, adjust=False).mean()
        result = {
            "avg_buy": avg_buy,
            "buy_slope": buy_slope_raw.fillna(0),
            "buy_accel": buy_accel_raw.fillna(0),
            "buy_jerk": buy_jerk_raw.fillna(0),
            "avg_sell": avg_sell,
            "sell_slope": (avg_sell - avg_sell.shift(3)) / avg_sell.shift(3).replace(0, np.nan) * 100
        }
        _temp_debug_values["主力平均价格计算"] = result
        return result

    def _calculate_main_force_cost_advantage_score(self, context: Dict, index: pd.Index, _temp_debug_values: Dict) -> pd.Series:
        """
        【V2.3.1 · 双轨成本Alpha模型 - 接口对齐版】
        职责：计算主力持仓成本与市场成本的博弈优势。
        修改：统一使用 context['funds'] 中的量能字段，确保与 _get_raw_control_signals 补全后的键名一致。
        """
        m, f, s = context['market'], context['funds'], context['structure']
        # 1. 内部调用主力均价计算逻辑
        prices = self._calculate_main_force_avg_prices(
            index, m['close'],
            f['buy_lg_amt'], f['buy_elg_amt'], f['sell_lg_amt'], f['sell_elg_amt'],
            f['buy_lg_vol'], f['buy_elg_vol'], f['sell_lg_vol'], f['sell_elg_vol'], # 已对齐补全后的键名
            _temp_debug_values
        )
        avg_buy = prices['avg_buy']
        avg_sell = prices['avg_sell']
        buy_slope = prices['buy_slope']
        buy_accel = prices['buy_accel']
        # 2. 静态优势 (Static Score)
        strategic_alpha = (s['cost_50pct'] - avg_buy) / s['cost_50pct'].replace(0, np.nan)
        safety_margin = (m['close'] - avg_buy) / avg_buy.replace(0, np.nan)
        harvest_prem = (avg_sell - s['cost_50pct']) / s['cost_50pct'].replace(0, np.nan)
        # 3. 动态意图修正 (Dynamic Modifier)
        aggression_bonus = pd.Series(0.0, index=index)
        aggression_bonus = aggression_bonus.mask(buy_slope > 0.2, 0.2)
        aggression_bonus = aggression_bonus.mask((buy_slope > 0.2) & (buy_accel > 0), 0.4)
        is_fake_pump = (m['pct_change'] > 3.0) & (buy_slope.abs() < 0.1)
        aggression_bonus = aggression_bonus.mask(is_fake_pump, -0.3)
        # 4. 综合评分计算
        raw_score = (strategic_alpha.fillna(0) * 0.4 + safety_margin.fillna(0) * 0.3 + harvest_prem.fillna(0) * 0.1 + aggression_bonus)
        final_score = np.tanh(raw_score * 5.0)
        # 5. 探针捕获
        _temp_debug_values["组件_成本优势"] = {
            "avg_buy": avg_buy,
            "buy_slope": buy_slope,
            "aggression_bonus": aggression_bonus,
            "final_score": final_score
        }
        print(f"[探针] 成本优势模型计算完成，主力买入成本均值: {avg_buy.mean():.4f}")
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
        【V4.3.1 · 传统控盘计算逻辑修复】
        职责：基于 EMA 排列计算传统技术面控盘分数。
        修改：修正 context 访问路径，从 context['ema'] 获取已映射的信号。
        """
        # 1. 提取已映射的信号
        ema = context['ema']
        ema_13, ema_21, ema_55 = ema['ema_13'], ema['ema_21'], ema['ema_55']
        close = context['market']['close']
        # 2. 计算斐波那契共振排列
        bullish_alignment = (ema_13 > ema_21) & (ema_21 > ema_55)
        bearish_alignment = (ema_13 < ema_21) & (ema_21 < ema_55)
        # 3. 计算离散度与偏离度
        alignment_score = pd.Series(0.0, index=index)
        alignment_score = alignment_score.mask(bullish_alignment, 0.8)
        alignment_score = alignment_score.mask(bearish_alignment, -0.8)
        # 4. 探针捕获
        _temp_debug_values["组件_传统控盘"] = {
            "ema_13_val": ema_13,
            "ema_21_val": ema_21,
            "alignment_resonance": alignment_score
        }
        print(f"[探针] 传统控盘逻辑执行完成，共振分快照已存入容器")
        return (alignment_score + (close - ema_55) / ema_55).clip(-1, 1)

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
        【V1.2.0 · 军械库全链路信号验证版】
        职责：在所有计算开始前，强制验证军械库清单中所需的物理信号是否存在。
        修改：扩充了验证清单，包含高阶动力学衍生指标及主力意图指标，确保逻辑链路不因缺数中断。
        """
        _, mtf_slope_accel_weights = self._get_control_parameters(config)
        required_signals = [
            'close_D', 'amount_D', 'pct_change_D', 'turnover_rate_D',
            'net_mf_amount_D', 'chip_stability_D', 'flow_consistency_D',
            'buy_lg_amount_D', 'buy_elg_amount_D', 'sell_lg_amount_D', 'sell_elg_amount_D',
            'buy_lg_vol_D', 'buy_elg_vol_D', 'sell_lg_vol_D', 'sell_elg_vol_D',
            'EMA_13_D', 'EMA_21_D', 'EMA_55_D',
            'BBW_21_2.0_D', 'chip_entropy_D', 'VPA_EFFICIENCY_D',
            'cost_50pct_D', 'winner_rate_D',
            'intraday_accumulation_confidence_D', 'intraday_distribution_confidence_D',
            'pushing_score_D', 'shakeout_score_D',
            'JERK_5_net_mf_amount_D', 'ACCEL_5_net_mf_amount_D', 'SLOPE_5_net_mf_amount_D',
            'ACCEL_5_chip_stability_D', 'JERK_5_pushing_score_D'
        ]
        base_sig_proxy = 'chip_stability_D'
        for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
            required_signals.append(f'SLOPE_{period_str}_{base_sig_proxy}')
        for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
            required_signals.append(f'ACCEL_{period_str}_{base_sig_proxy}')
        if not self.helper._validate_required_signals(df, required_signals, method_name):
            if probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 关键军械库信号缺失，计算终止。"] = ""
                self._print_debug_info(debug_output)
            return False
        return True









