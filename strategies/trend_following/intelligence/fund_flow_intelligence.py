import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V21.5 · 纯粹原子版】资金流情报分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出资金流领域的原子公理信号和资金流背离信号。
        - 移除信号: SCORE_FUND_FLOW_BULLISH_RESONANCE, SCORE_FUND_FLOW_BEARISH_RESONANCE, BIPOLAR_FUND_FLOW_DOMAIN_HEALTH, SCORE_FUND_FLOW_BOTTOM_REVERSAL, SCORE_FUND_FLOW_TOP_REVERSAL。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 资金流情报引擎在配置中被禁用，跳过分析。")
            return {}
        all_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        axiom_consensus = self._diagnose_axiom_consensus(df, norm_window)
        axiom_conviction = self._diagnose_axiom_conviction(df, norm_window)
        axiom_increment = self._diagnose_axiom_increment(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        all_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        all_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        all_states['SCORE_FF_AXIOM_INCREMENT'] = axiom_increment
        # 引入资金流层面的看涨/看跌背离信号 (保持不变)
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_FUND_FLOW_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】资金流公理四：诊断“资金背离”
        - 核心逻辑: 诊断价格行为与资金流之间的背离。
          - 看涨背离：价格下跌但主力资金净流入。
          - 看跌背离：价格上涨但主力资金净流出。
        """
        price_trend = normalize_to_bipolar(df.get('pct_change_D', pd.Series(0.0, index=df.index)), df.index, norm_window)
        main_force_flow_trend = normalize_to_bipolar(df.get('main_force_net_flow_calibrated_D', pd.Series(0.0, index=df.index)), df.index, norm_window)
        divergence_score = (main_force_flow_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】资金流公理一：诊断“共识与分歧”"""
        main_force_flow = df.get('net_xl_amount_calibrated_D', 0) + df.get('net_lg_amount_calibrated_D', 0)
        retail_flow = df.get('net_md_amount_calibrated_D', 0) + df.get('net_sh_amount_calibrated_D', 0)
        raw_bipolar_series = main_force_flow - retail_flow
        consensus_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return consensus_score.astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】资金流公理二：诊断“信念与决心”"""
        conviction_index = df.get('main_force_conviction_index_D', pd.Series(0.0, index=df.index))
        cost_advantage = df.get('main_force_cost_advantage_D', pd.Series(0.0, index=df.index))
        t0_efficiency = df.get('main_force_t0_efficiency_D', pd.Series(0.5, index=df.index))
        raw_bipolar_series = conviction_index + cost_advantage - (t0_efficiency * 2)
        conviction_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return conviction_score.astype(np.float32)

    def _diagnose_axiom_increment(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】资金流公理三：诊断“存量博弈”"""
        net_flow = df.get('net_flow_calibrated_D', pd.Series(0.0, index=df.index))
        turnover_slope = df.get(f'SLOPE_5_turnover_rate_f_D', pd.Series(0.0, index=df.index))
        raw_bipolar_series = net_flow - (turnover_slope.clip(lower=0) * df.get('circ_mv_D', 1e9) * 0.01)
        increment_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return increment_score.astype(np.float32)
