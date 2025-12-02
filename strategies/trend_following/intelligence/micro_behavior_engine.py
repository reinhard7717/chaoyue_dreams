import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
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
        【V3.0 · 诡道三策版】微观行为诊断引擎总指挥
        - 核心升级: 废弃旧的“伪装、试探、效率”三大公理，引入全新的“诡道三策”：
                    1. 隐秘行动 (Stealth Ops): 直接捕捉主力“明修栈道，暗度陈仓”的吸筹行为。
                    2. 震慑突袭 (Shock & Awe): 捕捉主力利用瞬间暴力行为测试或清洗市场的战术。
                    3. 成本控制 (Cost Control): 评估主力引导市场预期和防守自身成本区的能力。
        - 核心保留: 保留了有价值的“微观背离”公理。
        """
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 微观行为引擎在配置中被禁用，跳过分析。")
            return {}
        all_states = {}
        # 借用行为层的MTF权重配置
        p_behavior_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 调用全新的“诡道三策”和保留的“背离”公理 ---
        strategy_stealth_ops = self._diagnose_strategy_stealth_ops(df, default_weights)
        strategy_shock_and_awe = self._diagnose_strategy_shock_and_awe(df, default_weights)
        strategy_cost_control = self._diagnose_strategy_cost_control(df, default_weights)
        axiom_divergence = self._diagnose_axiom_divergence(df, 55) # norm_window 保持旧值
        # --- 更新输出的信号名称 ---
        all_states['SCORE_MICRO_STRATEGY_STEALTH_OPS'] = strategy_stealth_ops
        all_states['SCORE_MICRO_STRATEGY_SHOCK_AND_AWE'] = strategy_shock_and_awe
        all_states['SCORE_MICRO_STRATEGY_COST_CONTROL'] = strategy_cost_control
        all_states['SCORE_MICRO_AXIOM_DIVERGENCE'] = axiom_divergence
        # 引入微观行为层面的看涨/看跌背离信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_MICRO_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_MICRO_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.2 · 探针逻辑重构版】微观行为公理四：诊断“微观背离”
        - 核心重构: 彻底重构了探针逻辑，使其不再依赖于数据集的最后一天。现在探针会遍历
                      `probe_dates` 配置，并为每个在数据集中找到的日期精确打印当日的详细信息，
                      完美适配历史区间调试。
        """
        required_signals = ['SLOPE_5_EMA_5_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        price_trend_raw = self._get_safe_series(df, 'SLOPE_5_EMA_5_D', method_name="_diagnose_axiom_divergence")
        price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df.index, default_weights)
        micro_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0)
        micro_intent_trend_raw = micro_intent.ewm(span=5, adjust=False).mean().diff().fillna(0)
        micro_intent_trend = get_adaptive_mtf_normalized_bipolar_score(micro_intent_trend_raw, df.index, default_weights)
        divergence_score = (micro_intent_trend - price_trend).clip(-1, 1)
        # --- 彻底重构探针逻辑以适配历史回溯 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [微观行为探针] _diagnose_axiom_divergence @ {probe_date_str}")
                print(f"        - 价格趋势 (归一化): {price_trend.loc[probe_ts]:.4f} (原始斜率: {price_trend_raw.loc[probe_ts]:.2f})")
                print(f"        - 微观意图趋势 (归一化): {micro_intent_trend.loc[probe_ts]:.4f} (原始意图: {micro_intent.loc[probe_ts]:.2f})")
                print(f"        - 最终微观背离分: {divergence_score.loc[probe_ts]:.4f}")
        return divergence_score.astype(np.float32)

    def _diagnose_strategy_stealth_ops(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 探针文本优化版】微观诡道一策：诊断“隐秘行动”
        - 核心升级: 引入`wash_trade_intensity_D`（对倒强度）作为“纯度调节器”。
                      高对倒强度将惩罚最终得分，旨在过滤掉虚假的、表演性质的吸筹行为，
                      提升信号的“含金量”。
        - 核心优化: 优化探针输出文本，使其更精确地描述代码逻辑。
        """
        # --- 获取战术证据 ---
        pressure_raw = self._get_safe_series(df, 'large_order_pressure_D', 0.0, method_name="_diagnose_strategy_stealth_ops")
        accumulation_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_diagnose_strategy_stealth_ops")
        # --- 获取纯度证据 ---
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_strategy_stealth_ops")
        # --- 归一化证据 ---
        pressure_score = get_adaptive_mtf_normalized_score(pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        accumulation_score = get_adaptive_mtf_normalized_score(accumulation_raw, df.index, ascending=True, tf_weights=tf_weights)
        # --- 归一化纯度调节器 (对倒强度越高，得分越低，因此ascending=False) ---
        wash_trade_score = get_adaptive_mtf_normalized_score(wash_trade_raw, df.index, ascending=False, tf_weights=tf_weights)
        purity_modulator = wash_trade_score
        # --- 战术合成 ---
        base_score = (pressure_score * accumulation_score).pow(0.5).fillna(0.0)
        stealth_ops_score = (base_score * purity_modulator).fillna(0.0)
        # --- 探针逻辑升级 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [微观行为探针] _diagnose_strategy_stealth_ops @ {probe_date_str}")
                print(f"        - 原始值: 大单压制={pressure_raw.loc[probe_ts]:.2f}, 隐蔽吸筹={accumulation_raw.loc[probe_ts]:.2f}, 对倒强度={wash_trade_raw.loc[probe_ts]:.2f}")
                print(f"        - 归一化分: 压制分={pressure_score.loc[probe_ts]:.4f}, 吸筹分={accumulation_score.loc[probe_ts]:.4f}")
                # 修改代码: 优化探针输出文本
                print(f"        - 纯度调节器 (对倒强度降序归一化): {purity_modulator.loc[probe_ts]:.4f}")
                print(f"        - 最终隐秘行动分 (基础分*纯度): {stealth_ops_score.loc[probe_ts]:.4f}")
        return stealth_ops_score.astype(np.float32)

    def _diagnose_strategy_shock_and_awe(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 数据溯源注释版】微观诡道二策：诊断“震慑突袭”
        - 核心升级: 引入`volume_ratio_D`（量比）作为“量能确认放大器”。
                      高量比会放大最终得分，旨在奖励那些由真金白银驱动的、具备强大“敬畏”效果的突袭。
        - 核心优化: 根据探针反馈，为可能存在数据质量问题的`closing_strength_index_D`增加溯源注释。
        """
        impact_raw = self._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name="_diagnose_strategy_shock_and_awe")
        clearing_raw = self._get_safe_series(df, 'order_book_clearing_rate_D', 0.0, method_name="_diagnose_strategy_shock_and_awe")
        # 新增代码: 增加注释，记录探针发现的数据质量隐患
        # 注意: 探针曾发现此信号出现-0.18等理论范围(0-1)外的值，表明上游数据源可能存在质量问题。
        # 当前的normalize_score具备鲁棒性可处理此问题，但需保持关注。
        outcome_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_diagnose_strategy_shock_and_awe")
        # --- 获取量能证据 ---
        volume_ratio_raw = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_diagnose_strategy_shock_and_awe")
        # 数据净化步骤
        outcome_normalized = normalize_score(outcome_raw, df.index, 55)
        impact_score = get_adaptive_mtf_normalized_score(impact_raw.abs(), df.index, ascending=True, tf_weights=tf_weights)
        clearing_score = get_adaptive_mtf_normalized_score(clearing_raw, df.index, ascending=True, tf_weights=tf_weights)
        # --- 归一化量能放大器 ---
        volume_ratio_score = get_adaptive_mtf_normalized_score(volume_ratio_raw, df.index, ascending=True, tf_weights=tf_weights)
        awe_amplifier = (1 + 0.5 * volume_ratio_score).fillna(1.0)
        # 核心计算
        outcome_intent = (outcome_normalized * 2 - 1).clip(-1, 1)
        shock_magnitude = (impact_score * clearing_score).pow(0.5).fillna(0.0)
        base_score = (shock_magnitude * outcome_intent)
        shock_and_awe_score = (base_score * awe_amplifier).clip(-1, 1)
        # --- 探针逻辑升级 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [微观行为探针] _diagnose_strategy_shock_and_awe @ {probe_date_str}")
                print(f"        - 原始值: 冲击因子={impact_raw.loc[probe_ts]:.2f}, 清扫率={clearing_raw.loc[probe_ts]:.2f}, 收盘偏离={outcome_raw.loc[probe_ts]:.2f}, 量比={volume_ratio_raw.loc[probe_ts]:.2f}")
                print(f"        - 计算节点: 震慑强度={shock_magnitude.loc[probe_ts]:.4f}, 结果意图={outcome_intent.loc[probe_ts]:.4f}, 量能放大器={awe_amplifier.loc[probe_ts]:.4f}")
                print(f"        - 最终震慑突袭分 (基础分*放大器): {shock_and_awe_score.loc[probe_ts]:.4f}")
        return shock_and_awe_score.astype(np.float32)

    def _diagnose_strategy_cost_control(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V2.1 · 加法融合重构版】微观诡道三策：诊断“成本控制”
        - 核心重构: 根据探针反馈，原有的乘法模型存在逻辑缺陷（坏意图*控制不稳=风险减弱）。
                      现重构为加法（平均）模型，能更科学地处理“意图”与“能力”的共振与冲突。
                      1. 将“控盘稳固度”升级为[-1, 1]的双极性评分。
                      2. 最终得分由“基础意图分”和“控盘稳固度分”加权平均得到。
        """
        guidance_raw = self._get_safe_series(df, 'main_force_vwap_guidance_D', 0.0, method_name="_diagnose_strategy_cost_control")
        defense_raw = self._get_safe_series(df, 'mf_cost_zone_defense_intent_D', 0.0, method_name="_diagnose_strategy_cost_control")
        # --- 获取稳固度证据 ---
        solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name="_diagnose_strategy_cost_control")
        # --- 归一化所有输入为[-1, 1]的双极性分数 ---
        guidance_score = get_adaptive_mtf_normalized_bipolar_score(guidance_raw, df.index, tf_weights)
        defense_score = get_adaptive_mtf_normalized_bipolar_score(defense_raw, df.index, tf_weights)
        # 修改代码: 将稳固度也归一化为双极性分数
        solidity_score = get_adaptive_mtf_normalized_bipolar_score(solidity_raw, df.index, tf_weights)
        # --- 逻辑重构：从乘法模型升级为加法（平均）模型 ---
        base_intent_score = (guidance_score * 0.6 + defense_score * 0.4).clip(-1, 1)
        # 修改代码: 核心融合逻辑变更
        cost_control_score = (base_intent_score * 0.7 + solidity_score * 0.3).clip(-1, 1)
        # --- 探针逻辑升级以匹配新模型 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if is_debug_enabled and probe_dates:
            probe_timestamps = pd.to_datetime(probe_dates).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates = [d for d in probe_timestamps if d in df.index]
            for probe_ts in valid_probe_dates:
                probe_date_str = probe_ts.strftime('%Y-%m-%d')
                print(f"      [微观行为探针] _diagnose_strategy_cost_control @ {probe_date_str}")
                print(f"        - 原始值: VWAP引导力={guidance_raw.loc[probe_ts]:.2f}, 成本区防守={defense_raw.loc[probe_ts]:.2f}, 控盘稳固度={solidity_raw.loc[probe_ts]:.2f}")
                print(f"        - 双极性分: 引导分={guidance_score.loc[probe_ts]:.4f}, 防守分={defense_score.loc[probe_ts]:.4f}, 稳固度分={solidity_score.loc[probe_ts]:.4f}")
                print(f"        - 计算节点: 基础意图分={base_intent_score.loc[probe_ts]:.4f}")
                print(f"        - 最终成本控制分 (0.7*意图 + 0.3*稳固度): {cost_control_score.loc[probe_ts]:.4f}")
        return cost_control_score.astype(np.float32)


