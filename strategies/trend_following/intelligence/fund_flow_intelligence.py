import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_to_bipolar, bipolar_to_exclusive_unipolar, normalize_score

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [资金流情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V21.6 · 资金流动量版】资金流情报分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出资金流领域的原子公理信号和资金流背离信号。
        - 移除信号: SCORE_FUND_FLOW_BULLISH_RESONANCE, SCORE_FUND_FLOW_BEARISH_RESONANCE, BIPOLAR_FUND_FLOW_DOMAIN_HEALTH, SCORE_FUND_FLOW_BOTTOM_REVERSAL, SCORE_FUND_FLOW_TOP_REVERSAL。
        - 【更新】将 `_diagnose_axiom_increment` 替换为 `_diagnose_axiom_flow_momentum`。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 资金流情报引擎在配置中被禁用，跳过分析。")
            return {}
        all_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        axiom_consensus = self._diagnose_axiom_consensus(df, norm_window)
        axiom_conviction = self._diagnose_axiom_conviction(df, norm_window)
        axiom_flow_momentum = self._diagnose_axiom_flow_momentum(df, norm_window) # 调用新的方法
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        all_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        all_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        all_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum # 更新键名
        # 引入资金流层面的看涨/看跌背离信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_FUND_FLOW_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.0】资金流公理四：诊断“资金背离”
        - 核心逻辑: 诊断价格行为与资金流之间的背离。
          - 看涨背离：价格下跌但主力资金净流入。
          - 看跌背离：价格上涨但主力资金净流出。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        price_trend = normalize_to_bipolar(self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, norm_window)
        main_force_flow_trend = normalize_to_bipolar(self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, norm_window)
        divergence_score = (main_force_flow_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.1 · 博弈烈度增强版】资金流公理一：诊断“共识与分歧”
        - 引入 `mf_retail_battle_intensity` (主力散户博弈烈度) 作为判断资金流共识的重要证据。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        main_force_flow = self._get_safe_series(df, 'net_xl_amount_calibrated_D', 0, method_name="_diagnose_axiom_consensus") + self._get_safe_series(df, 'net_lg_amount_calibrated_D', 0, method_name="_diagnose_axiom_consensus")
        retail_flow = self._get_safe_series(df, 'net_md_amount_calibrated_D', 0, method_name="_diagnose_axiom_consensus") + self._get_safe_series(df, 'net_sh_amount_calibrated_D', 0, method_name="_diagnose_axiom_consensus")
        raw_bipolar_series = main_force_flow - retail_flow
        # 获取主力散户博弈烈度
        battle_intensity_raw = self._get_safe_series(df, 'mf_retail_battle_intensity_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_consensus")
        # 归一化博弈烈度，越高越好，但作为乘数因子，需要映射到 [0, 1]
        battle_intensity_factor = normalize_score(battle_intensity_raw, df_index, window=norm_window, ascending=True).clip(0, 1)
        # 原始共识分数
        consensus_score_base = normalize_to_bipolar(raw_bipolar_series, df_index, window=norm_window, sensitivity=1.0)
        # 融合博弈烈度。高烈度时，放大共识信号；低烈度时，削弱共识信号。
        # 乘数因子 (1 + battle_intensity_factor * 0.5) 可以放大共识，但不会改变方向
        consensus_score = (consensus_score_base * (1 + battle_intensity_factor * 0.5)).clip(-1, 1) # 调整放大系数
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [资金流共识探针] @ {probe_date_for_loop.date()}:")
                print(f"       - main_force_flow: {main_force_flow.loc[probe_date_for_loop]:.4f}")
                print(f"       - retail_flow: {retail_flow.loc[probe_date_for_loop]:.4f}")
                print(f"       - battle_intensity_raw: {battle_intensity_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - battle_intensity_factor: {battle_intensity_factor.loc[probe_date_for_loop]:.4f}")
                print(f"       - consensus_score_base: {consensus_score_base.loc[probe_date_for_loop]:.4f}")
                print(f"       - consensus_score: {consensus_score.loc[probe_date_for_loop]:.4f}")
        return consensus_score.astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.3 · 探针增强与归一化修复版】资金流公理二：诊断“信念与决心”
        - 核心升级: 增加调试探针，打印关键中间值。
        - 核心修复: 对 `conviction_index` 和 `cost_advantage` 进行归一化，避免原始值过大导致截断。
        - 引入 `main_force_price_impact_ratio` (主力价格冲击比率) 作为判断主力信念和效率的重要证据。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        conviction_index_raw = self._get_safe_series(df, 'main_force_conviction_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        t0_efficiency_raw = self._get_safe_series(df, 'main_force_t0_efficiency_D', pd.Series(0.5, index=df_index), method_name="_diagnose_axiom_conviction")
        price_impact_raw = self._get_safe_series(df, 'main_force_price_impact_ratio_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        # 对 conviction_index_raw 和 cost_advantage_raw 进行归一化
        # 赢家信念和成本优势越高越好，所以归一化后应为正
        conviction_index_bipolar = normalize_to_bipolar(conviction_index_raw, df_index, window=norm_window, sensitivity=10.0) # 调整敏感度
        cost_advantage_bipolar = normalize_to_bipolar(cost_advantage_raw, df_index, window=norm_window, sensitivity=100.0) # 调整敏感度
        # t0_efficiency 越高，对信念的负面影响越大，所以归一化后应为负
        t0_efficiency_bipolar = normalize_to_bipolar(t0_efficiency_raw, df_index, window=norm_window, sensitivity=0.5)
        # 价格冲击比率：越高越好，正向贡献
        price_impact_bipolar = normalize_to_bipolar(price_impact_raw, df_index, window=norm_window, sensitivity=10.0) # 归一化价格冲击比率
        # 重新加权融合
        raw_bipolar_series = (
            conviction_index_bipolar * 0.35 +
            cost_advantage_bipolar * 0.35 +
            price_impact_bipolar * 0.2 - # 价格冲击比率权重
            t0_efficiency_bipolar * 0.1 # 降低 t0_efficiency 的权重
        ).clip(-1, 1)
        conviction_score = normalize_to_bipolar(raw_bipolar_series, df_index, window=norm_window, sensitivity=1.0)
        # --- Debugging output for probe date ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [资金流信念探针] @ {probe_date_for_loop.date()}:")
                print(f"       - conviction_index_raw: {conviction_index_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - cost_advantage_raw: {cost_advantage_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - t0_efficiency_raw: {t0_efficiency_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_impact_raw: {price_impact_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - conviction_index_bipolar: {conviction_index_bipolar.loc[probe_date_for_loop]:.4f}")
                print(f"       - cost_advantage_bipolar: {cost_advantage_bipolar.loc[probe_date_for_loop]:.4f}")
                print(f"       - t0_efficiency_bipolar: {t0_efficiency_bipolar.loc[probe_date_for_loop]:.4f}")
                print(f"       - price_impact_bipolar: {price_impact_bipolar.loc[probe_date_for_loop]:.4f}")
                print(f"       - raw_bipolar_series: {raw_bipolar_series.loc[probe_date_for_loop]:.4f}")
                print(f"       - conviction_score: {conviction_score.loc[probe_date_for_loop]:.4f}")
        return conviction_score.astype(np.float32)

    def _diagnose_axiom_flow_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.0 · 资金流动量版】资金流公理三：诊断“资金流动量”
        - 核心逻辑: 衡量主力资金净流量的相对强度和趋势动量。
          - 标准化主力净流量 (NMFNF): 主力净流入额 / 总市值，使其可比。
          - NMFNF的短期 (5日) 和中期 (21日) 斜率，反映资金流的走向和加速。
          - 结合NMFNF的当前值和其动量，形成资金流的整体动量分数。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        # 获取主力净流量和总市值
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        total_market_value = self._get_safe_series(df, 'total_market_value_D', pd.Series(1e9, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        # 计算标准化主力净流量 (NMFNF)，避免除以零
        nmfnf = (main_force_net_flow / total_market_value.replace(0, 1e9)).fillna(0)
        # 归一化NMFNF本身，反映当前资金流的相对强度
        nmfnf_score = normalize_to_bipolar(nmfnf, df_index, window=norm_window, sensitivity=0.001) # 敏感度根据实际数据调整
        # 获取NMFNF的5日和21日斜率，反映资金流的动量和趋势
        slope_5_nmfnf = self._get_safe_series(df, 'SLOPE_5_NMFNF_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        slope_21_nmfnf = self._get_safe_series(df, 'SLOPE_21_NMFNF_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        # 归一化斜率
        slope_5_nmfnf_score = normalize_to_bipolar(slope_5_nmfnf, df_index, window=norm_window, sensitivity=0.0001)
        slope_21_nmfnf_score = normalize_to_bipolar(slope_21_nmfnf, df_index, window=norm_window, sensitivity=0.00005)
        # 融合当前资金流强度和其动量
        # 权重分配：当前强度和短期动量更重要，中期趋势提供确认
        flow_momentum_score = (
            nmfnf_score * 0.4 +
            slope_5_nmfnf_score * 0.35 +
            slope_21_nmfnf_score * 0.25
        ).clip(-1, 1)
        # --- Debugging output for probe date ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [资金流动量探针] @ {probe_date_for_loop.date()}:")
                print(f"       - main_force_net_flow: {main_force_net_flow.loc[probe_date_for_loop]:.4f}")
                print(f"       - total_market_value: {total_market_value.loc[probe_date_for_loop]:.4f}")
                print(f"       - nmfnf: {nmfnf.loc[probe_date_for_loop]:.6f}")
                print(f"       - nmfnf_score: {nmfnf_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - slope_5_nmfnf: {slope_5_nmfnf.loc[probe_date_for_loop]:.6f}")
                print(f"       - slope_21_nmfnf: {slope_21_nmfnf.loc[probe_date_for_loop]:.6f}")
                print(f"       - slope_5_nmfnf_score: {slope_5_nmfnf_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - slope_21_nmfnf_score: {slope_21_nmfnf_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - flow_momentum_score: {flow_momentum_score.loc[probe_date_for_loop]:.4f}")
        return flow_momentum_score.astype(np.float32)

