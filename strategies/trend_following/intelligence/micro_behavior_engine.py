import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from strategies.trend_following.utils import (
    get_params_block, get_param_value, bipolar_to_exclusive_unipolar, 
    get_adaptive_mtf_normalized_score, is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score
)
class MicroBehaviorEngine:
    """
    【V2.0 · 三大公理重构版】
    - 核心升级: 废弃旧的复杂诊断模型，引入基于主力微观操盘本质的“伪装、试探、效率”三大公理。
                使引擎更聚焦、逻辑更清晰、信号更纯粹。
    """
    def __init__(self, strategy_instance):
        """
        初始化微观行为诊断引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [微观行为情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def _get_signal(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        【V1.0】信号获取哨兵方法
        - 核心职责: 安全地从DataFrame获取信号。
        - 预警机制: 如果信号不存在，打印明确的警告信息，并返回一个包含默认值的Series，以防止程序崩溃。
        """
        if signal_name not in df.columns:
            print(f"    -> [微观行为引擎警告] 依赖信号 '{signal_name}' 在数据帧中不存在，将使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[signal_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # 调整校验信息为“微观行为情报校验”
            print(f"    -> [微观行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_micro_behavior_synthesis(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 · 和谐拐点升维版】微观行为诊断引擎总指挥
        - 核心升级: 在生成顶层“战略意图”信号后，进一步调用`_diagnose_harmony_inflection`方法，
                    生成终极机会信号 `SCORE_MICRO_HARMONY_INFLECTION`，
                    实现从“诊断”到“预见”的终极升维。
        """
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 微观行为引擎在配置中被禁用，跳过分析。")
            return {}
        # 获取调试参数
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates_str: List[str] = get_param_value(debug_params.get('probe_dates'), [])
        probe_dates: List[pd.Timestamp] = []
        if is_debug_enabled and probe_dates_str:
            try:
                # 关键修改：将探针日期转换为与DataFrame索引相同的UTC时区
                probe_dates = [pd.Timestamp(d).tz_localize('UTC') for d in probe_dates_str]
            except Exception as e:
                print(f"    -> [微观行为情报警告] 调试日期解析失败或时区转换失败: {e}。禁用调试。")
                is_debug_enabled = False
        all_states = {}
        # 借用行为层的MTF权重配置
        p_behavior_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior_conf.get('mtf_normalization_params'), {})
        # 修正键名 'default_weights' 为 'default'，并使用正确的默认字典结构
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # if is_debug_enabled:
        #     print(f"\n-> [微观行为情报] 启动微观行为诊断引擎，调试模式 {'启用' if is_debug_enabled else '禁用'}。")
        #     if probe_dates:
        #         print(f"   - 探针日期: {[d.strftime('%Y-%m-%d %H:%M:%S%z') for d in probe_dates]}") # 打印带时区信息
        #     # --- 新增诊断信息 ---
        #     print(f"   - 传入DataFrame是否为空: {df.empty}")
        #     if not df.empty:
        #         print(f"   - 传入DataFrame索引类型: {df.index.dtype}")
        #         # 打印索引中元素的实际类型（以防 dtype 误导）
        #         print(f"   - 传入DataFrame索引元素类型: {df.index.map(type).unique()}")
        #         print(f"   - 传入DataFrame索引范围: {df.index.min().strftime('%Y-%m-%d %H:%M:%S%z')} to {df.index.max().strftime('%Y-%m-%d %H:%M:%S%z')}") # 打印带时区信息
        #         for p_date in probe_dates:
        #             if p_date not in df.index:
        #                 print(f"   - 警告: 探针日期 {p_date.strftime('%Y-%m-%d %H:%M:%S%z')} (类型: {type(p_date)}) 不在传入DataFrame的索引中。")
        #                 # 尝试检查字符串形式的日期是否存在，以诊断潜在的类型不匹配问题
        #                 # 注意：这里检查字符串形式可能仍然会因为时区问题而失败，但可以作为辅助诊断
        #                 if p_date.tz_convert(None).strftime('%Y-%m-%d') in df.index.strftime('%Y-%m-%d'): # 比较无时区日期字符串
        #                     print(f"     - 注意: 探针日期 '{p_date.strftime('%Y-%m-%d')}' 的无时区形式在DataFrame索引的无时区形式中存在。")
        #             else:
        #                 print(f"   - 确认: 探针日期 {p_date.strftime('%Y-%m-%d %H:%M:%S%z')} (类型: {type(p_date)}) 在传入DataFrame的索引中。")
        #     else:
        #         print(f"   - 警告: 传入DataFrame为空，无法进行详细探针。")
        #     # --- 诊断信息结束 ---
        # --- 调用“诡道三策”和“背离”公理 ---
        strategy_stealth_ops = self._diagnose_strategy_stealth_ops(df, default_weights, is_debug_enabled, probe_dates)
        strategy_shock_and_awe = self._diagnose_strategy_shock_and_awe(df, default_weights, is_debug_enabled, probe_dates)
        strategy_cost_control = self._diagnose_strategy_cost_control(df, default_weights, is_debug_enabled, probe_dates)
        axiom_divergence = self._diagnose_axiom_divergence(df, default_weights, is_debug_enabled, probe_dates)
        # --- 更新原子/战术信号状态 ---
        all_states['SCORE_MICRO_STRATEGY_STEALTH_OPS'] = strategy_stealth_ops
        all_states['SCORE_MICRO_STRATEGY_SHOCK_AND_AWE'] = strategy_shock_and_awe
        all_states['SCORE_MICRO_STRATEGY_COST_CONTROL'] = strategy_cost_control
        all_states['SCORE_MICRO_AXIOM_DIVERGENCE'] = axiom_divergence
        # --- 调用战略意图合成器 ---
        strategic_intent = self._synthesize_strategic_intent(
            stealth_ops=strategy_stealth_ops,
            shock_awe=strategy_shock_and_awe,
            cost_control=strategy_cost_control,
            divergence=axiom_divergence,
            is_debug_enabled=is_debug_enabled,
            probe_dates=probe_dates
        )
        all_states['SCORE_MICRO_STRATEGIC_INTENT'] = strategic_intent
        print(f"    -> [微观行为情报校验] 计算“战略意图(SCORE_MICRO_STRATEGIC_INTENT)” 分数：{strategic_intent.mean():.4f}")
        # --- 新增：调用和谐拐点诊断器，生成终极机会信号 ---
        harmony_inflection = self._diagnose_harmony_inflection(strategic_intent, is_debug_enabled, probe_dates) # 新增代码
        all_states['SCORE_MICRO_HARMONY_INFLECTION'] = harmony_inflection # 新增代码
        # 引入微观行为层面的看涨/看跌背离信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_MICRO_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_MICRO_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        # if is_debug_enabled:
        #     print(f"\n-> [微观行为情报] 微观行为诊断引擎完成。")
        #     for probe_date in probe_dates:
        #         if probe_date in df.index: # 再次确认，因为这里是最终输出
        #             print(f"   --- 探针日期: {probe_date.strftime('%Y-%m-%d %H:%M:%S%z')} ---")
        #             for signal_name, series in all_states.items():
        #                 if probe_date in series.index:
        #                     print(f"     - {signal_name}: {series.loc[probe_date]:.4f}")
        #         else:
        #             print(f"   - 警告: 探针日期 {probe_date.strftime('%Y-%m-%d %H:%M:%S%z')} 不在传入DataFrame的索引中，无法输出最终信号。")
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_dates: List[pd.Timestamp]) -> pd.Series:
        """
        【V2.5 · 多时间框架背离重构版】微观行为公理四：诊断“微观背离”
        - 核心重构: 修正了多时间框架背离的计算逻辑。现在，价格趋势和微观意图趋势都将通过
                    融合多个时间框架的斜率来构建，从而真正实现多时间框架的背离分析。
        - 核心修正: 鉴于探针日志中`deception_index_D`在“诱多出货日”显示正值，
                    且其在`_synthesize_strategic_intent`中作为惩罚因子使用，
                    我们调整`micro_intent_fused`中`deception_index_score`的贡献方式。
                    现在，`deception_index_score`（其正值代表诱多/拉高出货，负值代表诱空/压价吸筹）
                    将以负向权重参与融合，即正向欺骗（诱多）会降低微观意图的看涨分数，
                    负向欺骗（诱空）会提高微观意图的看涨分数，从而更准确地反映主力真实意图。
        - **【修复】修正 `ewm(span=period)` 中 `period` 类型错误的问题，将其转换为整数。**
        """
        method_name = "_diagnose_axiom_divergence"
        # 更新 required_signals，包含多时间框架的价格斜率
        required_signals = [
            'SLOPE_5_EMA_5_D', 'SLOPE_13_EMA_13_D', 'SLOPE_21_EMA_21_D', # 价格趋势多时间框架
            'order_book_imbalance_D', 'main_force_ofi_D', 'deception_index_D' # 微观意图原始数据
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)

        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        # tf_weights 用于 get_adaptive_mtf_normalized_bipolar_score 内部的加权
        # 这里我们还需要一个用于融合不同时间框架斜率的权重
        slope_fusion_weights = get_param_value(p_conf.get('slope_fusion_weights'), {'5': 0.4, '13': 0.3, '21': 0.3})

        # --- 纯微观意图重构：从原始微观数据构建微观意图 ---
        order_imbalance = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name=method_name)
        main_force_ofi = self._get_safe_series(df, 'main_force_ofi_D', 0.0, method_name=method_name)
        deception_index = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)

        # 将原始微观数据归一化为双极性分数 (这里 get_adaptive_mtf_normalized_bipolar_score 已经处理了内部MTF加权)
        order_imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_imbalance, df.index, tf_weights, debug_info=(False, None, ""))
        main_force_ofi_score = get_adaptive_mtf_normalized_bipolar_score(main_force_ofi, df.index, tf_weights, debug_info=(False, None, ""))
        deception_index_score = get_adaptive_mtf_normalized_bipolar_score(deception_index, df.index, tf_weights, debug_info=(False, None, ""))

        # 融合微观意图。根据用户反馈，deception_index_D正值代表诱多出货（看跌），负值代表压价吸筹（看涨）。
        # 因此，在构建看涨微观意图时，deception_index_score应以负向权重参与融合。
        # 即：正向的deception_index_score（诱多）会降低看涨意图，负向的deception_index_score（诱空）会提高看涨意图。
        micro_intent_fused = (order_imbalance_score * 0.4 + main_force_ofi_score * 0.4 - deception_index_score * 0.2).clip(-1, 1)

        # --- 1. 构建多时间框架的价格趋势 (MTF Price Trend) ---
        price_trend_mtf_raw = pd.Series(0.0, index=df.index, dtype=np.float32)
        total_price_slope_weight = sum(slope_fusion_weights.values())
        if total_price_slope_weight > 0:
            for period_str, weight in slope_fusion_weights.items(): # 迭代键值对
                slope_signal_name = f'SLOPE_{period_str}_EMA_{period_str}_D'
                if slope_signal_name in df.columns:
                    price_trend_mtf_raw += self._get_safe_series(df, slope_signal_name, 0.0, method_name=method_name) * weight
                else:
                    print(f"    -> [微观行为情报警告] 方法 '{method_name}' 缺少价格斜率信号 '{slope_signal_name}'。")
            price_trend_mtf_raw /= total_price_slope_weight
        else:
            price_trend_mtf_raw = self._get_safe_series(df, 'SLOPE_5_EMA_5_D', 0.0, method_name=method_name) # 至少使用一个默认值

        # --- 2. 构建多时间框架的微观意图趋势 (MTF Micro Intent Trend) ---
        # 对 micro_intent_fused 计算不同时间框架的EMA斜率，然后加权融合
        micro_intent_trend_mtf_raw = pd.Series(0.0, index=df.index, dtype=np.float32)
        total_micro_intent_slope_weight = sum(slope_fusion_weights.values()) # 沿用相同的权重结构
        if total_micro_intent_slope_weight > 0:
            for period_str, weight in slope_fusion_weights.items(): # 迭代键值对
                period = int(period_str) # 将字符串键转换为整数
                # 计算 micro_intent_fused 的 EMA 斜率
                ema_series = micro_intent_fused.ewm(span=period, adjust=False).mean()
                # 使用 diff() 来近似斜率，或者更精确地计算斜率
                # 这里简化为 diff()，如果需要更精确的斜率，可以引入单独的斜率计算函数
                micro_intent_slope = ema_series.diff().fillna(0)
                micro_intent_trend_mtf_raw += micro_intent_slope * weight
            micro_intent_trend_mtf_raw /= total_micro_intent_slope_weight
        else:
            micro_intent_trend_mtf_raw = micro_intent_fused.ewm(span=5, adjust=False).mean().diff().fillna(0) # 至少使用一个默认值

        # --- 3. 将 MTF 趋势信号进行最终归一化 ---
        price_trend_normalized = get_adaptive_mtf_normalized_bipolar_score(price_trend_mtf_raw, df.index, tf_weights, debug_info=(False, None, ""))
        micro_intent_trend_normalized = get_adaptive_mtf_normalized_bipolar_score(micro_intent_trend_mtf_raw, df.index, tf_weights, debug_info=(False, None, ""))

        # --- 4. 计算背离分数 ---
        divergence_score = (micro_intent_trend_normalized - price_trend_normalized).clip(-1, 1)

        if is_debug_enabled:
            print(f"    -> [微观行为情报探针] 方法 '{method_name}' 调试启动。")
            for probe_date in probe_dates:
                if probe_date in df.index:
                    print(f"    -> [微观行为情报探针] 方法 '{method_name}' @ {probe_date.strftime('%Y-%m-%d')}:")
                    # 原始微观意图数据
                    if probe_date in order_imbalance.index:
                        print(f"       - 原始订单簿不平衡 (order_book_imbalance_D): {order_imbalance.loc[probe_date]:.4f}")
                    if probe_date in main_force_ofi.index:
                        print(f"       - 原始主力订单流 (main_force_ofi_D): {main_force_ofi.loc[probe_date]:.4f}")
                    if probe_date in deception_index.index:
                        print(f"       - 原始欺骗指数 (deception_index_D): {deception_index.loc[probe_date]:.4f}")
                    # 归一化微观意图组件
                    if probe_date in order_imbalance_score.index:
                        print(f"       - 归一化订单簿不平衡 (order_imbalance_score): {order_imbalance_score.loc[probe_date]:.4f}")
                    if probe_date in main_force_ofi_score.index:
                        print(f"       - 归一化主力订单流 (main_force_ofi_score): {main_force_ofi_score.loc[probe_date]:.4f}")
                    if probe_date in deception_index_score.index:
                        print(f"       - 归一化欺骗指数 (deception_index_score): {deception_index_score.loc[probe_date]:.4f}")
                    # 融合微观意图
                    if probe_date in micro_intent_fused.index:
                        print(f"       - 融合微观意图 (micro_intent_fused): {micro_intent_fused.loc[probe_date]:.4f}")
                    # MTF 价格趋势
                    if probe_date in price_trend_mtf_raw.index:
                        print(f"       - 原始MTF价格趋势 (price_trend_mtf_raw): {price_trend_mtf_raw.loc[probe_date]:.4f}")
                    if probe_date in price_trend_normalized.index:
                        print(f"       - 归一化MTF价格趋势 (price_trend_normalized): {price_trend_normalized.loc[probe_date]:.4f}")
                    # MTF 微观意图趋势
                    if probe_date in micro_intent_trend_mtf_raw.index:
                        print(f"       - 原始MTF微观意图趋势 (micro_intent_trend_mtf_raw): {micro_intent_trend_mtf_raw.loc[probe_date]:.4f}")
                    if probe_date in micro_intent_trend_normalized.index:
                        print(f"       - 归一化MTF微观意图趋势 (micro_intent_trend_normalized): {micro_intent_trend_normalized.loc[probe_date]:.4f}")
                    # 最终背离分数
                    if probe_date in divergence_score.index:
                        print(f"       - 最终微观背离分数 (SCORE_MICRO_AXIOM_DIVERGENCE): {divergence_score.loc[probe_date]:.4f}")
                else:
                    print(f"    -> [微观行为情报探针] 方法 '{method_name}' - 探针日期 {probe_date.strftime('%Y-%m-%d')} 不在当前DataFrame索引中。")
        return divergence_score.astype(np.float32)

    def _diagnose_strategy_stealth_ops(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_dates: List[pd.Timestamp]) -> pd.Series:
        """
        【V2.2 · 欺骗惩罚增强版】微观诡道一策：诊断“隐秘行动”
        - 核心升级: 引入`deception_index_D`的正向部分作为惩罚因子。当存在“拉高出货”的欺骗意图时，
                    隐秘吸筹的得分将被显著降低，以过滤掉虚假的、表演性质的吸筹行为。
        """
        method_name = "_diagnose_strategy_stealth_ops"
        required_signals = ['large_order_pressure_D', 'hidden_accumulation_intensity_D', 'wash_trade_intensity_D', 'deception_index_D']
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        # --- 获取战术证据 ---
        pressure_raw = self._get_safe_series(df, 'large_order_pressure_D', 0.0, method_name=method_name)
        accumulation_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name)
        # --- 获取纯度证据 ---
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        # --- 获取欺骗证据 ---
        deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        # --- 归一化证据 ---
        pressure_score = get_adaptive_mtf_normalized_score(pressure_raw, df.index, ascending=True, tf_weights=tf_weights, debug_info=(False, None, ""))
        accumulation_score = get_adaptive_mtf_normalized_score(accumulation_raw, df.index, ascending=True, tf_weights=tf_weights, debug_info=(False, None, ""))
        # 归一化纯度调节器 (对倒强度越高，得分越低，因此ascending=False)
        wash_trade_score = get_adaptive_mtf_normalized_score(wash_trade_raw, df.index, ascending=False, tf_weights=tf_weights, debug_info=(False, None, ""))
        purity_modulator = wash_trade_score
        # 归一化欺骗惩罚因子：只取deception_index_D的正向部分（拉高出货）作为惩罚
        # 使用一个固定的norm_window，例如55，或者从配置中获取
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        deception_penalty_score = normalize_score(deception_raw.clip(lower=0), df.index, norm_window, ascending=True, debug_info=(False, None, ""))
        # 欺骗惩罚调节器：1 - deception_penalty_score，欺骗越高，调节器越低
        deception_modulator = (1 - deception_penalty_score).clip(0, 1)
        # --- 战术合成 ---
        base_score = (pressure_score * accumulation_score).pow(0.5).fillna(0.0)
        # 最终得分 = 基础得分 * 纯度调节器 * 欺骗调节器
        stealth_ops_score = (base_score * purity_modulator * deception_modulator).fillna(0.0)
        # if is_debug_enabled:
        #     print(f"    -> [微观行为情报探针] 方法 '{method_name}' 调试启动。")
        #     for probe_date in probe_dates:
        #         if probe_date in df.index:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' @ {probe_date.strftime('%Y-%m-%d')}:")
        #             if probe_date in pressure_raw.index:
        #                 print(f"       - 原始大单压力 (large_order_pressure_D): {pressure_raw.loc[probe_date]:.4f}")
        #             if probe_date in accumulation_raw.index:
        #                 print(f"       - 原始隐蔽吸筹强度 (hidden_accumulation_intensity_D): {accumulation_raw.loc[probe_date]:.4f}")
        #             if probe_date in wash_trade_raw.index:
        #                 print(f"       - 原始对倒强度 (wash_trade_intensity_D): {wash_trade_raw.loc[probe_date]:.4f}")
        #             if probe_date in deception_raw.index:
        #                 print(f"       - 原始欺骗指数 (deception_index_D): {deception_raw.loc[probe_date]:.4f}")
        #             if probe_date in pressure_score.index:
        #                 print(f"       - 归一化大单压力 (pressure_score): {pressure_score.loc[probe_date]:.4f}")
        #             if probe_date in accumulation_score.index:
        #                 print(f"       - 归一化隐蔽吸筹强度 (accumulation_score): {accumulation_score.loc[probe_date]:.4f}")
        #             if probe_date in wash_trade_score.index:
        #                 print(f"       - 归一化对倒强度 (wash_trade_score): {wash_trade_score.loc[probe_date]:.4f}")
        #             if probe_date in purity_modulator.index:
        #                 print(f"       - 纯度调节器 (purity_modulator): {purity_modulator.loc[probe_date]:.4f}")
        #             if probe_date in deception_penalty_score.index:
        #                 print(f"       - 欺骗惩罚分数 (deception_penalty_score): {deception_penalty_score.loc[probe_date]:.4f}")
        #             if probe_date in deception_modulator.index:
        #                 print(f"       - 欺骗调节器 (deception_modulator): {deception_modulator.loc[probe_date]:.4f}")
        #             if probe_date in base_score.index:
        #                 print(f"       - 基础得分 (base_score): {base_score.loc[probe_date]:.4f}")
        #             if probe_date in stealth_ops_score.index:
        #                 print(f"       - 最终隐秘行动分数 (SCORE_MICRO_STRATEGY_STEALTH_OPS): {stealth_ops_score.loc[probe_date]:.4f}")
        #         else:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' - 探针日期 {probe_date.strftime('%Y-%m-%d')} 不在当前DataFrame索引中。")
        return stealth_ops_score.astype(np.float32)

    def _diagnose_strategy_shock_and_awe(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_dates: List[pd.Timestamp]) -> pd.Series:
        """
        【V2.2 · 效率校准版】微观诡道二策：诊断“震慑突袭”
        - 核心升级: 将`outcome_intent`（收盘强度意图）与`impact_score`（微观结构效率）进行乘法融合。
                    这意味着只有当收盘强度高且微观结构效率也高时，`outcome_intent`才能保持高位，
                    从而更准确地反映高质量的震慑突袭，避免因表面强势而忽略内在效率不足的问题。
        """
        method_name = "_diagnose_strategy_shock_and_awe"
        required_signals = ['microstructure_efficiency_index_D', 'order_book_clearing_rate_D', 'closing_strength_index_D', 'volume_ratio_D']
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        impact_raw = self._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name=method_name)
        clearing_raw = self._get_safe_series(df, 'order_book_clearing_rate_D', 0.0, method_name=method_name)
        outcome_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name=method_name)
        volume_ratio_raw = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name=method_name)
        # 数据净化步骤
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        outcome_normalized = normalize_score(outcome_raw, df.index, norm_window, debug_info=(False, None, ""))
        impact_score = get_adaptive_mtf_normalized_score(impact_raw.abs(), df.index, ascending=True, tf_weights=tf_weights, debug_info=(False, None, ""))
        clearing_score = get_adaptive_mtf_normalized_score(clearing_raw, df.index, ascending=True, tf_weights=tf_weights, debug_info=(False, None, ""))
        # 归一化量能放大器
        volume_ratio_score = get_adaptive_mtf_normalized_score(volume_ratio_raw, df.index, ascending=True, tf_weights=tf_weights, debug_info=(False, None, ""))
        awe_amplifier = (1 + 0.5 * volume_ratio_score).fillna(1.0)
        # 核心计算：将 outcome_intent 与 impact_score 融合
        # 只有当微观结构效率高时，收盘强度意图才被认为是有效的
        outcome_intent = ((outcome_normalized * 2 - 1) * impact_score).clip(-1, 1) # 修改点
        shock_magnitude = (impact_score * clearing_score).pow(0.5).fillna(0.0)
        base_score = (shock_magnitude * outcome_intent) # 保持乘法，因为outcome_intent已经包含了效率
        shock_and_awe_score = (base_score * awe_amplifier).clip(-1, 1)
        # if is_debug_enabled:
        #     print(f"    -> [微观行为情报探针] 方法 '{method_name}' 调试启动。")
        #     for probe_date in probe_dates:
        #         if probe_date in df.index:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' @ {probe_date.strftime('%Y-%m-%d')}:")
        #             if probe_date in impact_raw.index:
        #                 print(f"       - 原始微观结构效率 (microstructure_efficiency_index_D): {impact_raw.loc[probe_date]:.4f}")
        #             if probe_date in clearing_raw.index:
        #                 print(f"       - 原始订单簿清算率 (order_book_clearing_rate_D): {clearing_raw.loc[probe_date]:.4f}")
        #             if probe_date in outcome_raw.index:
        #                 print(f"       - 原始收盘强度 (closing_strength_index_D): {outcome_raw.loc[probe_date]:.4f}")
        #             if probe_date in volume_ratio_raw.index:
        #                 print(f"       - 原始量比 (volume_ratio_D): {volume_ratio_raw.loc[probe_date]:.4f}")
        #             if probe_date in outcome_normalized.index:
        #                 print(f"       - 归一化收盘强度 (outcome_normalized): {outcome_normalized.loc[probe_date]:.4f}")
        #             if probe_date in impact_score.index:
        #                 print(f"       - 归一化微观结构效率 (impact_score): {impact_score.loc[probe_date]:.4f}")
        #             if probe_date in clearing_score.index:
        #                 print(f"       - 归一化订单簿清算率 (clearing_score): {clearing_score.loc[probe_date]:.4f}")
        #             if probe_date in volume_ratio_score.index:
        #                 print(f"       - 归一化量比 (volume_ratio_score): {volume_ratio_score.loc[probe_date]:.4f}")
        #             if probe_date in awe_amplifier.index:
        #                 print(f"       - 敬畏放大器 (awe_amplifier): {awe_amplifier.loc[probe_date]:.4f}")
        #             if probe_date in outcome_intent.index:
        #                 print(f"       - 结果意图 (outcome_intent): {outcome_intent.loc[probe_date]:.4f}")
        #             if probe_date in shock_magnitude.index:
        #                 print(f"       - 震慑幅度 (shock_magnitude): {shock_magnitude.loc[probe_date]:.4f}")
        #             if probe_date in base_score.index:
        #                 print(f"       - 基础得分 (base_score): {base_score.loc[probe_date]:.4f}")
        #             if probe_date in shock_and_awe_score.index:
        #                 print(f"       - 最终震慑突袭分数 (SCORE_MICRO_STRATEGY_SHOCK_AND_AWE): {shock_and_awe_score.loc[probe_date]:.4f}")
        #         else:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' - 探针日期 {probe_date.strftime('%Y-%m-%d')} 不在当前DataFrame索引中。")
        return shock_and_awe_score.astype(np.float32)

    def _diagnose_strategy_cost_control(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_dates: List[pd.Timestamp]) -> pd.Series:
        """
        【V2.2 · 稳固度强化版】微观诡道三策：诊断“成本控制”
        - 核心重构: 调整了`base_intent_score`和`solidity_score`的融合方式。
                    现在采用简单平均，使得负向的`solidity_score`能更直接、更强烈地拉低最终得分，
                    从而更严格地反映控盘稳固度不佳的情况。
        """
        method_name = "_diagnose_strategy_cost_control"
        required_signals = ['main_force_vwap_guidance_D', 'mf_cost_zone_defense_intent_D', 'control_solidity_index_D']
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        guidance_raw = self._get_safe_series(df, 'main_force_vwap_guidance_D', 0.0, method_name=method_name)
        defense_raw = self._get_safe_series(df, 'mf_cost_zone_defense_intent_D', 0.0, method_name=method_name)
        solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name=method_name)
        # --- 归一化所有输入为[-1, 1]的双极性分数 ---
        guidance_score = get_adaptive_mtf_normalized_bipolar_score(guidance_raw, df.index, tf_weights, debug_info=(False, None, ""))
        defense_score = get_adaptive_mtf_normalized_bipolar_score(defense_raw, df.index, tf_weights, debug_info=(False, None, ""))
        solidity_score = get_adaptive_mtf_normalized_bipolar_score(solidity_raw, df.index, tf_weights, debug_info=(False, None, ""))
        # --- 逻辑重构：从加权平均升级为简单平均，强化负向稳固度的影响 ---
        base_intent_score = (guidance_score * 0.6 + defense_score * 0.4).clip(-1, 1)
        cost_control_score = ((base_intent_score + solidity_score) / 2).clip(-1, 1) # 修改点
        # if is_debug_enabled:
        #     print(f"    -> [微观行为情报探针] 方法 '{method_name}' 调试启动。")
        #     for probe_date in probe_dates:
        #         if probe_date in df.index:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' @ {probe_date.strftime('%Y-%m-%d')}:")
        #             if probe_date in guidance_raw.index:
        #                 print(f"       - 原始主力VWAP引导 (main_force_vwap_guidance_D): {guidance_raw.loc[probe_date]:.4f}")
        #             if probe_date in defense_raw.index:
        #                 print(f"       - 原始主力成本区防守意图 (mf_cost_zone_defense_intent_D): {defense_raw.loc[probe_date]:.4f}")
        #             if probe_date in solidity_raw.index:
        #                 print(f"       - 原始控盘稳固度 (control_solidity_index_D): {solidity_raw.loc[probe_date]:.4f}")
        #             if probe_date in guidance_score.index:
        #                 print(f"       - 归一化主力VWAP引导 (guidance_score): {guidance_score.loc[probe_date]:.4f}")
        #             if probe_date in defense_score.index:
        #                 print(f"       - 归一化主力成本区防守意图 (defense_score): {defense_score.loc[probe_date]:.4f}")
        #             if probe_date in solidity_score.index:
        #                 print(f"       - 归一化控盘稳固度 (solidity_score): {solidity_score.loc[probe_date]:.4f}")
        #             if probe_date in base_intent_score.index:
        #                 print(f"       - 基础意图得分 (base_intent_score): {base_intent_score.loc[probe_date]:.4f}")
        #             if probe_date in cost_control_score.index:
        #                 print(f"       - 最终成本控制分数 (SCORE_MICRO_STRATEGY_COST_CONTROL): {cost_control_score.loc[probe_date]:.4f}")
        #         else:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' - 探针日期 {probe_date.strftime('%Y-%m-%d')} 不在当前DataFrame索引中。")
        return cost_control_score.astype(np.float32)

    def _diagnose_harmony_inflection(self, strategic_intent: pd.Series, is_debug_enabled: bool, probe_dates: List[pd.Timestamp]) -> pd.Series:
        """
        【V1.1 · 探针回溯版】微观和谐拐点诊断器
        - 核心逻辑: 基于微积分思想，对顶层战略意图信号进行二阶求导，捕捉其动态拐点。
        - 核心升级: 优化探针逻辑，使其在打印当日信息时，能自动回溯并展示前两日的关键数据，
                      从而完整地呈现“速度”与“加速度”的计算过程，极大提升了可调试性。
        """
        method_name = "_diagnose_harmony_inflection"
        # 计算速度（一阶导数）
        velocity = strategic_intent.diff().fillna(0)
        # 计算加速度（二阶导数）
        acceleration = velocity.diff().fillna(0)
        # 应用“破晓”逻辑：只有当速度和加速度都为正时，拐点才成立
        bullish_inflection_mask = (velocity > 0) & (acceleration > 0)
        # 计算拐点强度
        inflection_strength = (velocity * acceleration).pow(0.5)
        # 应用掩码
        harmony_inflection_score = pd.Series(np.where(bullish_inflection_mask, inflection_strength, 0), index=strategic_intent.index)
        # Pass debug_info=(False, None, "") to suppress internal printing from utility functions
        # 使用 normalize_score 进行最终的归一化，使其在历史数据中具有可比性
        # 修正 normalize_score 的调用参数，添加 harmony_inflection_score.index
        final_score = normalize_score(harmony_inflection_score, harmony_inflection_score.index, 55, debug_info=(False, None, ""))
        # if is_debug_enabled:
        #     print(f"    -> [微观行为情报探针] 方法 '{method_name}' 调试启动。")
        #     for probe_date in probe_dates:
        #         if probe_date in strategic_intent.index:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' @ {probe_date.strftime('%Y-%m-%d')}:")
        #             # 回溯前两日数据
        #             current_idx = strategic_intent.index.get_loc(probe_date)
        #             for i in range(max(0, current_idx - 2), current_idx + 1):
        #                 date_to_print = strategic_intent.index[i]
        #                 if date_to_print in strategic_intent.index:
        #                     print(f"       - 日期: {date_to_print.strftime('%Y-%m-%d')}")
        #                     print(f"         - 战略意图 (strategic_intent): {strategic_intent.loc[date_to_print]:.4f}")
        #                     print(f"         - 速度 (velocity): {velocity.loc[date_to_print]:.4f}")
        #                     print(f"         - 加速度 (acceleration): {acceleration.loc[date_to_print]:.4f}")
        #                     print(f"         - 看涨拐点掩码 (bullish_inflection_mask): {bullish_inflection_mask.loc[date_to_print]}")
        #                     print(f"         - 拐点强度 (inflection_strength): {inflection_strength.loc[date_to_print]:.4f}")
        #                     print(f"         - 和谐拐点原始分数 (harmony_inflection_score): {harmony_inflection_score.loc[date_to_print]:.4f}")
        #                     if date_to_print == probe_date:
        #                         print(f"       - 最终和谐拐点分数 (SCORE_MICRO_HARMONY_INFLECTION): {final_score.loc[probe_date]:.4f}")
        #         else:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' - 探针日期 {probe_date.strftime('%Y-%m-%d')} 不在当前DataFrame索引中。")
        return final_score.astype(np.float32)

    def _synthesize_strategic_intent(self, stealth_ops: pd.Series, shock_awe: pd.Series, cost_control: pd.Series, divergence: pd.Series, is_debug_enabled: bool, probe_dates: List[pd.Timestamp]) -> pd.Series:
        """
        【V2.1 · 欺骗全局惩罚版】微观战略意图合成器
        - 核心升级: 在最终融合时引入`deception_index_D`的正向部分作为全局惩罚因子。
                    当存在“拉高出货”的欺骗意图时，即使其他微观信号看起来积极，
                    最终的战略意图也将被削弱，以更准确地反映真实的主力意图。
        """
        method_name = "_synthesize_strategic_intent"
        # 1. 计算进攻力量
        offensive_force = (stealth_ops + shock_awe.clip(lower=0)) / 2
        # 2. 构建“控制力门控”调节器
        control_gate = (cost_control + 1) / 2
        # 3. 计算经过门控审核的进攻力量
        gated_offensive_force = offensive_force * control_gate
        # 4. 风险因子（背离）
        risk_factor = divergence
        # 获取欺骗指数作为全局惩罚因子
        deception_raw = self._get_safe_series(self.strategy.df_indicators, 'deception_index_D', 0.0, method_name=method_name)
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        # 只取deception_index_D的正向部分（拉高出货）作为惩罚
        deception_penalty_score = normalize_score(deception_raw.clip(lower=0), self.strategy.df_indicators.index, norm_window, ascending=True, debug_info=(False, None, ""))
        # 欺骗惩罚调节器：1 - deception_penalty_score，欺骗越高，调节器越低
        deception_modulator = (1 - deception_penalty_score).clip(0, 1)
        # 5. 最终博弈：将门控后的进攻力量与风险因子（背离）进行加权融合，并应用欺骗惩罚
        strategic_intent_score = (
            gated_offensive_force * 0.7 +
            risk_factor * 0.3
        ) * deception_modulator.clip(0, 1) # 修改点：应用全局欺骗惩罚
        strategic_intent_score = strategic_intent_score.clip(-1, 1)
        # if is_debug_enabled:
        #     print(f"    -> [微观行为情报探针] 方法 '{method_name}' 调试启动。")
        #     for probe_date in probe_dates:
        #         if probe_date in stealth_ops.index:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' @ {probe_date.strftime('%Y-%m-%d')}:")
        #             if probe_date in stealth_ops.index:
        #                 print(f"       - 隐秘行动 (stealth_ops): {stealth_ops.loc[probe_date]:.4f}")
        #             if probe_date in shock_awe.index:
        #                 print(f"       - 震慑突袭 (shock_awe): {shock_awe.loc[probe_date]:.4f}")
        #             if probe_date in cost_control.index:
        #                 print(f"       - 成本控制 (cost_control): {cost_control.loc[probe_date]:.4f}")
        #             if probe_date in divergence.index:
        #                 print(f"       - 微观背离 (divergence): {divergence.loc[probe_date]:.4f}")
        #             if probe_date in offensive_force.index:
        #                 print(f"       - 进攻力量 (offensive_force): {offensive_force.loc[probe_date]:.4f}")
        #             if probe_date in control_gate.index:
        #                 print(f"       - 控制力门控 (control_gate): {control_gate.loc[probe_date]:.4f}")
        #             if probe_date in gated_offensive_force.index:
        #                 print(f"       - 门控后进攻力量 (gated_offensive_force): {gated_offensive_force.loc[probe_date]:.4f}")
        #             if probe_date in risk_factor.index:
        #                 print(f"       - 风险因子 (risk_factor): {risk_factor.loc[probe_date]:.4f}")
        #             if probe_date in deception_penalty_score.index:
        #                 print(f"       - 欺骗惩罚分数 (deception_penalty_score): {deception_penalty_score.loc[probe_date]:.4f}")
        #             if probe_date in deception_modulator.index:
        #                 print(f"       - 欺骗调节器 (deception_modulator): {deception_modulator.loc[probe_date]:.4f}")
        #             if probe_date in strategic_intent_score.index:
        #                 print(f"       - 最终战略意图分数 (SCORE_MICRO_STRATEGIC_INTENT): {strategic_intent_score.loc[probe_date]:.4f}")
        #         else:
        #             print(f"    -> [微观行为情报探针] 方法 '{method_name}' - 探针日期 {probe_date.strftime('%Y-%m-%d')} 不在当前DataFrame索引中。")
        return strategic_intent_score.astype(np.float32)



