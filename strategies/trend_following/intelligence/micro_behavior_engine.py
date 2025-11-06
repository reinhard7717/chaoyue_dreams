# 文件: strategies/trend_following/intelligence/micro_behavior_engine.py
# 微观行为诊断引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, get_unified_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

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

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def run_micro_behavior_synthesis(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 · 三大公理重构版】微观行为诊断引擎总指挥
        - 核心流程:
          1. 并行诊断三大公理，生成纯粹的微观行为原子信号。
          2. 融合三大公理，合成终极的微观共振信号。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("微观行为引擎已在配置中禁用，跳过。")
            return {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 步骤一: 诊断三大公理 ---
        axiom_deception = self._diagnose_axiom_deception(df, norm_window)
        axiom_probe = self._diagnose_axiom_probe(df, norm_window)
        axiom_efficiency = self._diagnose_axiom_efficiency(df, norm_window)
        all_states['SCORE_MICRO_AXIOM_DECEPTION'] = axiom_deception
        all_states['SCORE_MICRO_AXIOM_PROBE'] = axiom_probe
        all_states['SCORE_MICRO_AXIOM_EFFICIENCY'] = axiom_efficiency
        # --- 步骤二: 融合三大公理，合成终极信号 ---
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'deception': 0.4, 'probe': 0.3, 'efficiency': 0.3
        })
        # 构造一个融合了所有公理的原始双极性健康分
        bipolar_health = (
            axiom_deception * axiom_weights['deception'] +
            axiom_probe * axiom_weights['probe'] +
            axiom_efficiency * axiom_weights['efficiency']
        ).clip(-1, 1)
        # 分解为互斥的单极性共振分
        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_health)
        all_states['SCORE_MICRO_BULLISH_RESONANCE'] = bullish_resonance
        all_states['SCORE_MICRO_BEARISH_RESONANCE'] = bearish_resonance
        return all_states

    def _get_signal(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        【V1.0 · 新增】信号获取哨兵方法
        - 核心职责: 安全地从DataFrame获取信号。
        - 预警机制: 如果信号不存在，打印明确的警告信息，并返回一个包含默认值的Series，以防止程序崩溃。
        """
        if signal_name not in df.columns:
            print(f"    -> [微观行为引擎警告] 依赖信号 '{signal_name}' 在数据帧中不存在，将使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[signal_name]

    def _diagnose_axiom_deception(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.2 · 健壮性修复版】微观行为公理一：诊断“伪装与欺骗”
        - 核心修复: 使用 _get_signal 方法安全获取所有依赖信号，防止因信号缺失而崩溃。
        - 逻辑修正: 明确使用 'SLOPE_5_short_term_concentration_90pct_D' 作为筹码集中度变化的证据。
        """
        # 核心逻辑：寻找“表象”与“实质”的背离
        # 证据1: 资金流表象 vs 筹码实质
        # 表象：主力资金净流出
        main_force_flow_raw = self._get_signal(df, 'main_force_net_flow_calibrated_D')
        main_force_outflow = -main_force_flow_raw.clip(upper=0)
        # 实质：筹码仍在集中
        # 明确使用短期集中度斜率作为证据
        chip_concentration_slope = self._get_signal(df, 'SLOPE_5_short_term_concentration_90pct_D')
        chip_concentration_increase = chip_concentration_slope.clip(lower=0)
        flow_vs_chip_deception = main_force_outflow * chip_concentration_increase
        # 证据2: 交易颗粒度表象 vs 订单实质
        # 表象：交易颗粒度变小（伪装成散户）
        granularity_slope = self._get_signal(df, 'SLOPE_5_inferred_active_order_size_D')
        granularity_decrease = -granularity_slope.clip(upper=0)
        # 实质：主力控盘度提升
        control_leverage_slope = self._get_signal(df, 'SLOPE_5_main_force_control_leverage_D')
        control_increase = control_leverage_slope.clip(lower=0)
        granularity_vs_control_deception = granularity_decrease * control_increase
        # 融合两大欺骗证据
        raw_deception_score = flow_vs_chip_deception + granularity_vs_control_deception
        # 使用双极归一化进行最终裁决
        deception_score = normalize_to_bipolar(raw_deception_score, df.index, window=norm_window)
        return deception_score.astype(np.float32)

    def _diagnose_axiom_probe(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.1 · 健壮性修复版】微观行为公理二：诊断“试探与确认”
        - 核心修复: 使用 _get_signal 方法安全获取所有依赖信号。
        """
        # 核心逻辑：分析带长影线的K线背后的真实意图
        total_range = (self._get_signal(df, 'high_D') - self._get_signal(df, 'low_D')).replace(0, np.nan)
        # 证据1: 上影线试探 (正分)
        # 表象：长上影线
        upper_shadow_ratio = ((self._get_signal(df, 'high_D') - np.maximum(self._get_signal(df, 'open_D'), self._get_signal(df, 'close_D'))) / total_range).fillna(0)
        # 实质：主力资金并未净流出
        main_force_flow_raw = self._get_signal(df, 'main_force_net_flow_calibrated_D')
        main_force_not_outflow = main_force_flow_raw.clip(lower=0)
        probe_up_score = upper_shadow_ratio * main_force_not_outflow
        # 证据2: 下影线试探 (正分)
        # 表象：长下影线
        lower_shadow_ratio = ((np.minimum(self._get_signal(df, 'open_D'), self._get_signal(df, 'close_D')) - self._get_signal(df, 'low_D')) / total_range).fillna(0)
        # 实质：主力资金净流入
        main_force_inflow = main_force_flow_raw.clip(lower=0)
        probe_down_score = lower_shadow_ratio * main_force_inflow
        # 证据3: 诱多式突破 (负分)
        # 表象：突破近期高点
        breakout_high = (self._get_signal(df, 'close_D') > self._get_signal(df, 'high_D').rolling(21).max().shift(1)).astype(float)
        # 实质：主力资金并未跟进
        main_force_not_inflow = -main_force_flow_raw.clip(upper=0)
        fake_breakout_score = breakout_high * main_force_not_inflow
        # 融合所有试探行为
        raw_probe_score = probe_up_score + probe_down_score - fake_breakout_score
        # 使用双极归一化进行最终裁决
        probe_score = normalize_to_bipolar(raw_probe_score, df.index, window=norm_window)
        return probe_score.astype(np.float32)

    def _diagnose_axiom_efficiency(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.1 · 健壮性修复版】微观行为公理三：诊断“成本与效率”
        - 核心修复: 使用 _get_signal 方法安全获取所有依赖信号。
        """
        # 核心逻辑：衡量“投入”与“产出”的比率
        # 投入：成交额放大程度
        amount_series = self._get_signal(df, 'amount_D')
        amount_ma = amount_series.rolling(norm_window).mean().replace(0, np.nan)
        amount_input = (amount_series / amount_ma).fillna(1.0)
        # 产出：价格变化幅度
        pct_change_series = self._get_signal(df, 'pct_change_D')
        price_output = pct_change_series.abs() * 100 # 乘以100放大
        # 效率 = 产出 / 投入
        # 为了避免除以0，并处理方向，我们使用更稳健的公式
        # 效率分 = 价格变化方向 * (价格变化幅度 - k * 成交额放大程度)
        # 正价格变化，但成交额放大过多，效率分也可能为负（滞涨）
        k = 0.1 # 调节系数
        raw_efficiency_score = np.sign(pct_change_series) * (price_output - k * amount_input)
        # 使用双极归一化进行最终裁决
        efficiency_score = normalize_to_bipolar(raw_efficiency_score, df.index, window=norm_window)
        return efficiency_score.astype(np.float32)
