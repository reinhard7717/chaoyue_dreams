import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar

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
        # [修改代码行] 借用行为层的MTF权重配置
        p_behavior_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 调用全新的“诡道三策”和保留的“背离”公理 ---
        strategy_stealth_ops = self._diagnose_strategy_stealth_ops(df, default_weights)
        strategy_shock_and_awe = self._diagnose_strategy_shock_and_awe(df, default_weights)
        strategy_cost_control = self._diagnose_strategy_cost_control(df, default_weights)
        axiom_divergence = self._diagnose_axiom_divergence(df, 55) # norm_window 保持旧值
        # --- [修改代码块] 更新输出的信号名称 ---
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
        【V2.0 · 意图趋势背离版】微观行为公理四：诊断“微观背离”
        - 核心升级: 计算范式从“价格结果 vs 资金流量”升级为“价格趋势 vs 微观意图趋势”的深度博弈。
        - 价格端: 使用5日EMA斜率替代单日涨跌幅，聚焦趋势。
        - 意图端: 使用更高阶的 SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT 的趋势，直指主力最原始的意图变化。
        - 新逻辑: 微观背离分 = (微观意图趋势 - 价格趋势)，旨在捕捉价格表象与主力真实微观动作的背道而驰。
        """
        # [修改代码行] 更新依赖信号
        required_signals = ['SLOPE_5_EMA_5_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        # --- 探针初始化 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        last_date_str = df.index[-1].strftime('%Y-%m-%d')
        is_debug_day = is_debug_enabled and (not probe_dates or last_date_str in probe_dates)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # --- 价格趋势 ---
        price_trend_raw = self._get_safe_series(df, 'SLOPE_5_EMA_5_D', method_name="_diagnose_axiom_divergence")
        price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df.index, default_weights)
        # --- 微观意图趋势 ---
        # 从原子状态库中获取更高阶的微观意图信号
        micro_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0)
        micro_intent_trend_raw = micro_intent.ewm(span=5, adjust=False).mean().diff().fillna(0)
        micro_intent_trend = get_adaptive_mtf_normalized_bipolar_score(micro_intent_trend_raw, df.index, default_weights)
        # --- 计算背离 ---
        divergence_score = (micro_intent_trend - price_trend).clip(-1, 1)
        # --- 探针监测 ---
        if is_debug_day:
            print(f"      [微观行为探针] _diagnose_axiom_divergence @ {last_date_str}")
            print(f"        - 价格趋势 (归一化): {price_trend.iloc[-1]:.4f} (原始斜率: {price_trend_raw.iloc[-1]:.2f})")
            print(f"        - 微观意图趋势 (归一化): {micro_intent_trend.iloc[-1]:.4f} (原始意图: {micro_intent.iloc[-1]:.2f})")
            print(f"        - 最终微观背离分: {divergence_score.iloc[-1]:.4f}")
        return divergence_score.astype(np.float32)

    def _diagnose_strategy_stealth_ops(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V1.0 · 隐秘行动版】微观诡道一策：诊断“隐秘行动”
        - 核心逻辑: 替代旧的“伪装”公理，直接捕捉主力“明修栈道，暗度陈仓”的操盘战术。
        - 战术证据: 1. 大单压制强度 (明修栈道); 2. 隐蔽吸筹强度 (暗度陈仓)。
        - 输出: [0, 1] 的单极性分数，分数越高代表主力当天执行隐蔽吸筹战术的证据越确凿。
        """
        # --- 探针初始化 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        last_date_str = df.index[-1].strftime('%Y-%m-%d')
        is_debug_day = is_debug_enabled and (not probe_dates or last_date_str in probe_dates)
        # --- 获取战术证据 ---
        pressure_raw = self._get_safe_series(df, 'large_order_pressure_D', 0.0, method_name="_diagnose_strategy_stealth_ops")
        accumulation_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_diagnose_strategy_stealth_ops")
        # --- 归一化证据 ---
        pressure_score = get_adaptive_mtf_normalized_score(pressure_raw, df.index, ascending=True, tf_weights=tf_weights)
        accumulation_score = get_adaptive_mtf_normalized_score(accumulation_raw, df.index, ascending=True, tf_weights=tf_weights)
        # --- 战术合成 ---
        stealth_ops_score = (pressure_score * accumulation_score).pow(0.5).fillna(0.0)
        # --- 探针监测 ---
        if is_debug_day:
            print(f"      [微观行为探针] _diagnose_strategy_stealth_ops @ {last_date_str}")
            print(f"        - 原始值: 大单压制={pressure_raw.iloc[-1]:.2f}, 隐蔽吸筹={accumulation_raw.iloc[-1]:.2f}")
            print(f"        - 归一化分: 压制分={pressure_score.iloc[-1]:.4f}, 吸筹分={accumulation_score.iloc[-1]:.4f}")
            print(f"        - 最终隐秘行动分: {stealth_ops_score.iloc[-1]:.4f}")
        return stealth_ops_score.astype(np.float32)

    def _diagnose_strategy_shock_and_awe(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V1.0 · 震慑突袭版】微观诡道二策：诊断“震慑突袭”
        - 核心逻辑: 替代旧的“试探”公理，捕捉主力利用资金优势进行瞬间暴力行为以测试或清洗市场的战术。
        - 战术证据: 1. 价格冲击强度 (暴力行为); 2. 盘口清扫率 (决心); 3. 收盘偏离度 (意图)。
        - 输出: [-1, 1] 的双极性分数。正分代表看涨的震慑(如下探回升)，负分代表看跌的震慑(如上冲回落)。
        """
        # --- 探针初始化 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        last_date_str = df.index[-1].strftime('%Y-%m-%d')
        is_debug_day = is_debug_enabled and (not probe_dates or last_date_str in probe_dates)
        # --- 获取战术证据 ---
        impact_raw = self._get_safe_series(df, 'ofi_price_impact_factor_D', 0.0, method_name="_diagnose_strategy_shock_and_awe")
        clearing_raw = self._get_safe_series(df, 'order_book_clearing_rate_D', 0.0, method_name="_diagnose_strategy_shock_and_awe")
        outcome_raw = self._get_safe_series(df, 'closing_price_deviation_score_D', 0.5, method_name="_diagnose_strategy_shock_and_awe")
        # --- 归一化证据 ---
        # 冲击强度，绝对值越大越强
        impact_score = get_adaptive_mtf_normalized_score(impact_raw.abs(), df.index, ascending=True, tf_weights=tf_weights)
        # 清扫率，越高越强
        clearing_score = get_adaptive_mtf_normalized_score(clearing_raw, df.index, ascending=True, tf_weights=tf_weights)
        # 结果意图，双极性
        outcome_intent = (outcome_raw * 2 - 1).clip(-1, 1)
        # --- 战术合成 ---
        # 震慑强度 = (冲击强度 * 清扫率)^0.5
        shock_magnitude = (impact_score * clearing_score).pow(0.5).fillna(0.0)
        # 最终得分 = 震慑强度 * 结果意图
        shock_and_awe_score = (shock_magnitude * outcome_intent).clip(-1, 1)
        # --- 探针监测 ---
        if is_debug_day:
            print(f"      [微观行为探针] _diagnose_strategy_shock_and_awe @ {last_date_str}")
            print(f"        - 原始值: 冲击因子={impact_raw.iloc[-1]:.2f}, 清扫率={clearing_raw.iloc[-1]:.2f}, 收盘偏离={outcome_raw.iloc[-1]:.2f}")
            print(f"        - 归一化与计算: 震慑强度={shock_magnitude.iloc[-1]:.4f}, 结果意图={outcome_intent.iloc[-1]:.4f}")
            print(f"        - 最终震慑突袭分: {shock_and_awe_score.iloc[-1]:.4f}")
        return shock_and_awe_score.astype(np.float32)

    def _diagnose_strategy_cost_control(self, df: pd.DataFrame, tf_weights: Dict) -> pd.Series:
        """
        【V1.0 · 成本控制版】微观诡道三策：诊断“成本控制”
        - 核心逻辑: 替代旧的“效率”公理，评估主力管理自身成本和引导市场预期的能力。
        - 战术证据: 1. 主力VWAP引导力 (引导市场预期的能力); 2. 主力成本区攻防意图 (防守自身成本的意愿)。
        - 输出: [-1, 1] 的双极性分数。正分代表主力控盘能力强，成本管理优秀；负分则相反。
        """
        # --- 探针初始化 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        last_date_str = df.index[-1].strftime('%Y-%m-%d')
        is_debug_day = is_debug_enabled and (not probe_dates or last_date_str in probe_dates)
        # --- 获取战术证据 ---
        guidance_raw = self._get_safe_series(df, 'main_force_vwap_guidance_D', 0.0, method_name="_diagnose_strategy_cost_control")
        defense_raw = self._get_safe_series(df, 'mf_cost_zone_defense_intent_D', 0.0, method_name="_diagnose_strategy_cost_control")
        # --- 归一化证据 (两者本身就是类双极性信号，可直接归一化) ---
        guidance_score = get_adaptive_mtf_normalized_bipolar_score(guidance_raw, df.index, tf_weights)
        defense_score = get_adaptive_mtf_normalized_bipolar_score(defense_raw, df.index, tf_weights)
        # --- 战术合成 ---
        cost_control_score = (guidance_score * 0.6 + defense_score * 0.4).clip(-1, 1)
        # --- 探针监测 ---
        if is_debug_day:
            print(f"      [微观行为探针] _diagnose_strategy_cost_control @ {last_date_str}")
            print(f"        - 原始值: VWAP引导力={guidance_raw.iloc[-1]:.2f}, 成本区防守={defense_raw.iloc[-1]:.2f}")
            print(f"        - 归一化分: 引导分={guidance_score.iloc[-1]:.4f}, 防守分={defense_score.iloc[-1]:.4f}")
            print(f"        - 最终成本控制分: {cost_control_score.iloc[-1]:.4f}")
        return cost_control_score.astype(np.float32)


