# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
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
        【V21.3 · 背离公理增强版】资金流情报分析总指挥
        - 核心修复: 修正了输出的共振信号名称，将 'SCORE_FF_*' 修正为 'SCORE_FUND_FLOW_*'，
                      以严格遵守与融合层的情报供应契约。
        - 【新增】引入资金流背离公理。
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
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'consensus': 0.5, 'conviction': 0.3, 'increment': 0.2, 'divergence': 0.0 # [代码修改] 新增divergence权重
        })
        bipolar_health = (
            axiom_consensus * axiom_weights['consensus'] +
            axiom_conviction * axiom_weights['conviction'] +
            axiom_increment * axiom_weights['increment']
        ).clip(-1, 1)
        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_health)
        # 修正信号名称以符合融合层的契约
        all_states['SCORE_FUND_FLOW_BULLISH_RESONANCE'] = bullish_resonance
        all_states['SCORE_FUND_FLOW_BEARISH_RESONANCE'] = bearish_resonance
        # 引入资金流层面的看涨/看跌背离信号
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
        # 证据1: 价格变化趋势
        price_trend = normalize_to_bipolar(df.get('pct_change_D', pd.Series(0.0, index=df.index)), df.index, norm_window)
        # 证据2: 主力资金流趋势
        main_force_flow_trend = normalize_to_bipolar(df.get('main_force_net_flow_calibrated_D', pd.Series(0.0, index=df.index)), df.index, norm_window)
        # 融合：当价格趋势与主力资金流趋势相反时，产生背离信号
        # 看涨背离：价格下跌（负）但资金流入（正）
        # 看跌背离：价格上涨（正）但资金流出（负）
        # 我们可以用 (main_force_flow_trend - price_trend) 来捕捉这种矛盾
        # 价涨资金流出: (负 - 正) = 负 -> 看跌背离
        # 价跌资金流入: (正 - 负) = 正 -> 看涨背离
        divergence_score = (main_force_flow_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】资金流公理一：诊断“共识与分歧”"""
        # 主力资金 = 超大单 + 大单
        main_force_flow = df.get('net_xl_amount_calibrated_D', 0) + df.get('net_lg_amount_calibrated_D', 0)
        # 散户资金 = 中单 + 小单
        retail_flow = df.get('net_md_amount_calibrated_D', 0) + df.get('net_sh_amount_calibrated_D', 0)
        # 构造原始双极性序列：主力净流入 - 散户净流入
        # 正值代表 主力买、散户卖 (共识吸筹)
        # 负值代表 主力卖、散户买 (共识派发)
        raw_bipolar_series = main_force_flow - retail_flow
        # 使用双极归一化引擎进行最终裁决
        consensus_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return consensus_score.astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】资金流公理二：诊断“信念与决心”"""
        # 证据1: 主力信念指数，越高越好
        conviction_index = df.get('main_force_conviction_index_D', pd.Series(0.0, index=df.index))
        # 证据2: 主力成本优势，越高越好
        cost_advantage = df.get('main_force_cost_advantage_D', pd.Series(0.0, index=df.index))
        # 证据3: 主力T+0效率，越低越好 (高T0效率代表套利，信念不坚)
        t0_efficiency = df.get('main_force_t0_efficiency_D', pd.Series(0.5, index=df.index))
        # 构造原始双极性序列
        raw_bipolar_series = conviction_index + cost_advantage - (t0_efficiency * 2) # T0效率乘以2以加大惩罚
        # 使用双极归一化引擎进行最终裁决
        conviction_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return conviction_score.astype(np.float32)

    def _diagnose_axiom_increment(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】资金流公理三：诊断“存量博弈”"""
        # 证据1: 全市场净流入，越高越好
        net_flow = df.get('net_flow_calibrated_D', pd.Series(0.0, index=df.index))
        # 证据2: 换手率，过高或过低都不好，我们关注其变化
        turnover_slope = df.get(f'SLOPE_5_turnover_rate_f_D', pd.Series(0.0, index=df.index))
        # 构造原始双极性序列
        # 核心逻辑：我们希望看到净流入，同时换手率不要急剧放大（可能导致派发）
        raw_bipolar_series = net_flow - (turnover_slope.clip(lower=0) * df.get('circ_mv_D', 1e9) * 0.01) # 用市值惩罚放量
        # 使用双极归一化引擎进行最终裁决
        increment_score = normalize_to_bipolar(raw_bipolar_series, df.index, window=norm_window, sensitivity=1.0)
        return increment_score.astype(np.float32)

















