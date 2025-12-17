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
        【V3.2 · 和谐拐点升维版】微观行为诊断引擎总指挥
        - 核心升级: 在生成顶层“战略意图”信号后，进一步调用`_diagnose_harmony_inflection`方法，
                    生成终极机会信号 `SCORE_MICRO_HARMONY_INFLECTION`，
                    实现从“诊断”到“预见”的终极升维。
        """
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 微观行为引擎在配置中被禁用，跳过分析。")
            return {}
        all_states = {}
        # 借用行为层的MTF权重配置
        p_behavior_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior_conf.get('mtf_normalization_params'), {})
        # 修正键名 'default_weights' 为 'default'，并使用正确的默认字典结构
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # --- 调用“诡道三策”和“背离”公理 ---
        strategy_stealth_ops = self._diagnose_strategy_stealth_ops(df, default_weights)
        strategy_shock_and_awe = self._diagnose_strategy_shock_and_awe(df, default_weights)
        strategy_cost_control = self._diagnose_strategy_cost_control(df, default_weights)
        axiom_divergence = self._diagnose_axiom_divergence(df, 55)
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
            divergence=axiom_divergence
        )
        all_states['SCORE_MICRO_STRATEGIC_INTENT'] = strategic_intent
        print(f"    -> [微观行为情报校验] 计算“战略意图(SCORE_MICRO_STRATEGIC_INTENT)” 分数：{strategic_intent.mean():.4f}")
        # --- 新增：调用和谐拐点诊断器，生成终极机会信号 ---
        harmony_inflection = self._diagnose_harmony_inflection(strategic_intent) # 新增代码
        all_states['SCORE_MICRO_HARMONY_INFLECTION'] = harmony_inflection # 新增代码
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
        # 从 chip_ultimate_params 获取 tf_fusion_weights，而不是 behavioral_dynamics_params
        p_conf = get_params_block(self.strategy, 'chip_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        price_trend_raw = self._get_safe_series(df, 'SLOPE_5_EMA_5_D', method_name="_diagnose_axiom_divergence")
        price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df.index, tf_weights)
        micro_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 0.0)
        micro_intent_trend_raw = micro_intent.ewm(span=5, adjust=False).mean().diff().fillna(0)
        micro_intent_trend = get_adaptive_mtf_normalized_bipolar_score(micro_intent_trend_raw, df.index, tf_weights)
        divergence_score = (micro_intent_trend - price_trend).clip(-1, 1)
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
        # 修正 normalize_score 的调用参数，添加 df.index
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
       
        return cost_control_score.astype(np.float32)

    def _diagnose_harmony_inflection(self, strategic_intent: pd.Series) -> pd.Series:
        """
        【V1.1 · 探针回溯版】微观和谐拐点诊断器
        - 核心逻辑: 基于微积分思想，对顶层战略意图信号进行二阶求导，捕捉其动态拐点。
        - 核心升级: 优化探针逻辑，使其在打印当日信息时，能自动回溯并展示前两日的关键数据，
                      从而完整地呈现“速度”与“加速度”的计算过程，极大提升了可调试性。
        """
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
        # 使用 normalize_score 进行最终的归一化，使其在历史数据中具有可比性
        # 修正 normalize_score 的调用参数，添加 harmony_inflection_score.index
        final_score = normalize_score(harmony_inflection_score, harmony_inflection_score.index, 55)
        return final_score.astype(np.float32)

    def _synthesize_strategic_intent(self, stealth_ops: pd.Series, shock_awe: pd.Series, cost_control: pd.Series, divergence: pd.Series) -> pd.Series:
        """
        【V2.0 · 控制力门控版】微观战略意图合成器
        - 核心重构: 从简单的加法模型升级为“控制力门控”非线性模型。
                      1. 使用`cost_control`（成本控制）构建一个[0, 1]区间的“控制力门控”调节器。
                      2. 用此门控调节`offensive_force`（进攻力量），得到一个经过可行性审核的“门控后进攻力量”。
                      3. 让“门控后进攻力量”与`divergence`（微观背离）进行最终的加权博弈。
        - 融合公式: (门控后进攻力量 * 0.7 + 微观背离 * 0.3)
        """
        # 1. 计算进攻力量
        offensive_force = (stealth_ops + shock_awe.clip(lower=0)) / 2
        # 2. 构建“控制力门控”调节器
        # 将[-1, 1]的cost_control分映射到[0, 1]的门控调节器
        # 控制力越强(越接近1)，门控越开放(越接近1)；控制力越弱(越接近-1)，门控越关闭(越接近0)
        control_gate = (cost_control + 1) / 2 # 修改代码
        # 3. 计算经过门控审核的进攻力量
        gated_offensive_force = offensive_force * control_gate # 修改代码
        # 4. 最终博弈：让门控后的进攻力量与风险因子（背离）进行加权融合
        risk_factor = divergence
        strategic_intent_score = (
            gated_offensive_force * 0.7 + # 修改代码
            risk_factor * 0.3             # 修改代码
        ).clip(-1, 1)
        return strategic_intent_score.astype(np.float32)



