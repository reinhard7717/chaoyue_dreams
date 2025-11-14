import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar, get_adaptive_mtf_normalized_score

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, data_source: Union[pd.DataFrame, Dict[str, pd.Series]], column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame或字典中获取Series，如果不存在则打印警告并返回默认Series。
        - 核心修复: 兼容处理 pd.DataFrame 和 Dict[str, pd.Series] 两种数据源。
        """
        df_index = self.strategy.df_indicators.index # 获取全局的DataFrame索引
        if isinstance(data_source, pd.DataFrame):
            if column_name not in data_source.columns:
                print(f"    -> [资金流情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            return data_source[column_name]
        elif isinstance(data_source, dict):
            if column_name not in data_source:
                print(f"    -> [资金流情报警告] 方法 '{method_name}' 缺少字典数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            # 确保从字典中取出的也是Series，并且索引对齐
            series = data_source[column_name]
            if isinstance(series, pd.Series):
                return series.reindex(df_index, fill_value=default_value)
            else: # 如果字典中存储的不是Series，则转换为Series
                return pd.Series(series, index=df_index)
        else:
            print(f"    -> [资金流情报警告] 方法 '{method_name}' 接收到未知数据源类型 {type(data_source)}，无法获取 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df_index)

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
        【V1.1 · 多时间维度归一化版】资金流公理四：诊断“资金背离”
        - 核心逻辑: 诊断价格行为与资金流之间的背离。
          - 看涨背离：价格下跌但主力资金净流入。
          - 看跌背离：价格上涨但主力资金净流出。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `price_trend` 和 `main_force_flow_trend` 的归一化方式改为多时间维度自适应归一化。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}) # 借用筹码的MTF权重配置
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, tf_weights)
        main_force_flow_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, tf_weights)
        divergence_score = (main_force_flow_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.8 · 浮点数精度比较修复版】资金流公理一：诊断“共识与分歧”
        - 核心修复: 修正了 `sell_elg_amount` 变量错误地获取 `buy_elg_amount_D` 数据的致命性代码错误。
                      现在 `sell_elg_amount` 将正确获取 `sell_elg_amount_D`，从而确保 `net_elg_amount`
                      以及后续的 `main_force_flow` 和 `lg_xl_net_flow` 计算的准确性。
        - 核心修复: 优化了“拆单吸筹因子”中 `sm_md_net_flow >= lg_xl_net_flow.abs()` 的浮点数比较逻辑，
                      引入 `numpy.isclose` 以处理精度问题，确保当两者数值上足够接近时，条件能正确评估为 True。
        - 核心修复: 修正了 `net_sm_amount_calibrated_D` 等净流入金额的计算逻辑，
                      现在它们将基于原始的买入和卖出金额进行计算，而不是直接从 `df` 中获取。
                      这解决了 `_diagnose_axiom_consensus` 方法中缺少DataFrame数据 'net_sm_amount_calibrated_D' 的警告。
        - 核心强化: 调整了“拆单吸筹因子”的识别条件，使其能更准确地捕捉主力通过小单/中单进行隐蔽吸筹的行为。
        - 核心修正: 确保“拆单吸筹因子”仅在原始信号为正时才对最终共识分数产生贡献，避免了归一化函数可能导致的误判。
        - 引入 `mf_retail_battle_intensity` (主力散户博弈烈度) 作为判断资金流共识的重要证据。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `battle_intensity_factor` 和 `consensus_score_base` 的归一化方式改为多时间维度自适应归一化。
        - 【新增】增加了详细的探针日志，用于诊断 `split_order_accumulation_raw` 赋值失败的问题。
        """
        df_index = df.index
        # 从原始买卖金额计算净流入金额
        buy_sm_amount = self._get_safe_series(df, 'buy_sm_amount_D', 0, method_name="_diagnose_axiom_consensus")
        sell_sm_amount = self._get_safe_series(df, 'sell_sm_amount_D', 0, method_name="_diagnose_axiom_consensus")
        buy_md_amount = self._get_safe_series(df, 'buy_md_amount_D', 0, method_name="_diagnose_axiom_consensus")
        sell_md_amount = self._get_safe_series(df, 'sell_md_amount_D', 0, method_name="_diagnose_axiom_consensus")
        buy_lg_amount = self._get_safe_series(df, 'buy_lg_amount_D', 0, method_name="_diagnose_axiom_consensus")
        sell_lg_amount = self._get_safe_series(df, 'sell_lg_amount_D', 0, method_name="_diagnose_axiom_consensus")
        buy_elg_amount = self._get_safe_series(df, 'buy_elg_amount_D', 0, method_name="_diagnose_axiom_consensus")
        sell_elg_amount = self._get_safe_series(df, 'sell_elg_amount_D', 0, method_name="_diagnose_axiom_consensus")
        # 计算各级别净流入
        net_sm_amount = buy_sm_amount - sell_sm_amount
        net_md_amount = buy_md_amount - sell_md_amount
        net_lg_amount = buy_lg_amount - sell_lg_amount
        net_elg_amount = buy_elg_amount - sell_elg_amount
        # 主力资金流 = 超大单净流入 + 大单净流入
        main_force_flow = net_elg_amount + net_lg_amount
        # 散户资金流 = 中单净流入 + 小单净流入
        retail_flow = net_md_amount + net_sm_amount
        # 原始共识分数：主力资金流 - 散户资金流
        raw_bipolar_series = main_force_flow - retail_flow
        # 获取主力散户博弈烈度
        battle_intensity_raw = self._get_safe_series(df, 'mf_retail_battle_intensity_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_consensus")
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}) # 资金流模块的MTF权重
        # 归一化博弈烈度，越高越好，但作为乘数因子，需要映射到 [0, 1]
        battle_intensity_factor = get_adaptive_mtf_normalized_score(battle_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff).clip(0, 1)
        # --- 新增：拆单吸筹因子 ---
        # 小单净流入 + 中单净流入
        sm_md_net_flow = net_sm_amount + net_md_amount
        # 大单净流入 + 特大单净流入
        lg_xl_net_flow = net_lg_amount + net_elg_amount
        # 价格变化
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_consensus")
        # 拆单吸筹的迹象：
        # 1. 价格下跌或横盘 (pct_change <= 0)
        # 2. 小单/中单净流入为正 (sm_md_net_flow > 0)
        # 3. 大单/特大单净流入不明显或为负 (lg_xl_net_flow <= 0)
        # 4. 且 sm_md_net_flow >= abs(lg_xl_net_flow) (小单中单的净流入能够抵消或对冲大单特大单的流出)
        # 拆单吸筹因子：在价格下跌或横盘时，小单中单净流入与大单特大单净流入的差值，归一化后作为因子
        split_order_accumulation_raw = pd.Series(0.0, index=df_index)
        # [代码修改开始]
        # 优化 Condition 4 的浮点数比较
        condition_4_optimized = (sm_md_net_flow > lg_xl_net_flow.abs()) | np.isclose(sm_md_net_flow, lg_xl_net_flow.abs(), atol=1e-5)
        condition_mask = (pct_change <= 0) & (sm_md_net_flow > 0) & (lg_xl_net_flow <= 0) & condition_4_optimized
        # [代码修改结束]

        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"       - DEBUG: split_order_accumulation_raw BEFORE assignment for probe date: {split_order_accumulation_raw.loc[probe_date_for_loop]}")
                print(f"       - DEBUG: pct_change.loc[probe_date_for_loop]: {pct_change.loc[probe_date_for_loop]}")
                print(f"       - DEBUG: sm_md_net_flow.loc[probe_date_for_loop]: {sm_md_net_flow.loc[probe_date_for_loop]}")
                print(f"       - DEBUG: lg_xl_net_flow.loc[probe_date_for_loop]: {lg_xl_net_flow.loc[probe_date_for_loop]}")
                print(f"       - DEBUG: lg_xl_net_flow.abs().loc[probe_date_for_loop]: {lg_xl_net_flow.abs().loc[probe_date_for_loop]}")
                print(f"       - DEBUG: Condition 1 (pct_change <= 0): {pct_change.loc[probe_date_for_loop] <= 0}")
                print(f"       - DEBUG: Condition 2 (sm_md_net_flow > 0): {sm_md_net_flow.loc[probe_date_for_loop] > 0}")
                print(f"       - DEBUG: Condition 3 (lg_xl_net_flow <= 0): {lg_xl_net_flow.loc[probe_date_for_loop] <= 0}")
                # [代码修改开始]
                print(f"       - DEBUG: Condition 4 (sm_md_net_flow > lg_xl_net_flow.abs()): {sm_md_net_flow.loc[probe_date_for_loop] > lg_xl_net_flow.abs().loc[probe_date_for_loop]}")
                print(f"       - DEBUG: Condition 4 (np.isclose(sm_md_net_flow, lg_xl_net_flow.abs(), atol=1e-5)): {np.isclose(sm_md_net_flow.loc[probe_date_for_loop], lg_xl_net_flow.abs().loc[probe_date_for_loop], atol=1e-5)}")
                print(f"       - DEBUG: combined condition_mask for probe date: {condition_mask.loc[probe_date_for_loop]}")
                # [代码修改结束]
                print(f"       - DEBUG: value to assign if mask is True: {(sm_md_net_flow - lg_xl_net_flow).loc[probe_date_for_loop]}")

        split_order_accumulation_raw.loc[condition_mask] = (sm_md_net_flow - lg_xl_net_flow).loc[condition_mask]

        if probe_dates_str and probe_date_for_loop is not None and probe_date_for_loop in df.index:
            print(f"       - DEBUG: split_order_accumulation_raw AFTER assignment for probe date: {split_order_accumulation_raw.loc[probe_date_for_loop]}")

        # 对拆单吸筹因子进行归一化，使其成为一个 [0, 1] 的正向证据
        normalized_split_factor_series = get_adaptive_mtf_normalized_score(split_order_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights_ff).clip(0, 1)
        # 只有当 split_order_accumulation_raw > 0 时，才使用这个归一化因子，否则为0
        split_order_accumulation_factor = normalized_split_factor_series.where(split_order_accumulation_raw > 0, 0)
        # 原始共识分数
        consensus_score_base = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df_index, tf_weights_ff, sensitivity=1.0)
        # 融合博弈烈度。高烈度时，放大共识信号；低烈度时，削弱共识信号。
        # 乘数因子 (1 + battle_intensity_factor * 0.5) 可以放大共识，但不会改变方向
        # 融合拆单吸筹因子：如果存在拆单吸筹，则对共识分数进行额外加成，尤其是在共识分数偏负时（主力在隐藏吸筹）
        # 调整融合逻辑：当拆单吸筹因子为正时，对共识分数进行正向修正
        # 修正系数可以根据实际情况调整，例如 0.3
        consensus_score = (consensus_score_base * (1 + battle_intensity_factor * 0.5) + split_order_accumulation_factor * 0.3).clip(-1, 1) # 调整放大系数和拆单吸筹系数
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [资金流共识探针] @ {probe_date_for_loop.date()}:")
                print(f"       - main_force_flow: {main_force_flow.loc[probe_date_for_loop]:.4f}")
                print(f"       - retail_flow: {retail_flow.loc[probe_date_for_loop]:.4f}")
                print(f"       - battle_intensity_raw: {battle_intensity_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - battle_intensity_factor: {battle_intensity_factor.loc[probe_date_for_loop]:.4f}")
                print(f"       - sm_md_net_flow: {sm_md_net_flow.loc[probe_date_for_loop]:.4f}")
                print(f"       - lg_xl_net_flow: {lg_xl_net_flow.loc[probe_date_for_loop]:.4f}")
                print(f"       - pct_change: {pct_change.loc[probe_date_for_loop]:.4f}")
                print(f"       - split_order_accumulation_raw: {split_order_accumulation_raw.loc[probe_date_for_loop]:.4f}")
                print(f"       - split_order_accumulation_factor: {split_order_accumulation_factor.loc[probe_date_for_loop]:.4f}")
                print(f"       - consensus_score_base: {consensus_score_base.loc[probe_date_for_loop]:.4f}")
                print(f"       - consensus_score: {consensus_score.loc[probe_date_for_loop]:.4f}")
        return consensus_score.astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.4 · 探针增强与多时间维度归一化版】资金流公理二：诊断“信念与决心”
        - 核心升级: 增加调试探针，打印关键中间值。
        - 核心修复: 对 `conviction_index` 和 `cost_advantage` 进行归一化，避免原始值过大导致截断。
        - 引入 `main_force_price_impact_ratio` (主力价格冲击比率) 作为判断主力信念和效率的重要证据。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        """
        df_index = df.index
        conviction_index_raw = self._get_safe_series(df, 'main_force_conviction_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        t0_efficiency_raw = self._get_safe_series(df, 'main_force_t0_efficiency_D', pd.Series(0.5, index=df_index), method_name="_diagnose_axiom_conviction")
        price_impact_raw = self._get_safe_series(df, 'main_force_price_impact_ratio_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 对 conviction_index_raw 和 cost_advantage_raw 进行归一化
        # 赢家信念和成本优势越高越好，所以归一化后应为正
        conviction_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(conviction_index_raw, df_index, tf_weights_ff, sensitivity=10.0) # 调整敏感度
        cost_advantage_bipolar = get_adaptive_mtf_normalized_bipolar_score(cost_advantage_raw, df_index, tf_weights_ff, sensitivity=100.0) # 调整敏感度
        # t0_efficiency 越高，对信念的负面影响越大，所以归一化后应为负
        t0_efficiency_bipolar = get_adaptive_mtf_normalized_bipolar_score(t0_efficiency_raw, df_index, tf_weights_ff, sensitivity=0.5)
        # 价格冲击比率：越高越好，正向贡献
        price_impact_bipolar = get_adaptive_mtf_normalized_bipolar_score(price_impact_raw, df_index, tf_weights_ff, sensitivity=10.0) # 归一化价格冲击比率
        # 重新加权融合
        raw_bipolar_series = (
            conviction_index_bipolar * 0.35 +
            cost_advantage_bipolar * 0.35 +
            price_impact_bipolar * 0.2 - # 价格冲击比率权重
            t0_efficiency_bipolar * 0.1 # 降低 t0_efficiency 的权重
        ).clip(-1, 1)
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df_index, tf_weights_ff, sensitivity=1.0)
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
        【V2.1 · 资金流动量与多时间维度归一化版】资金流公理三：诊断“资金流动量”
        - 核心逻辑: 衡量主力资金净流量的相对强度和趋势动量。
          - 标准化主力净流量 (NMFNF): 主力净流入额 / 总市值，使其可比。
          - NMFNF的短期 (5日) 和中期 (21日) 斜率，反映资金流的走向和加速。
          - 结合NMFNF的当前值和其动量，形成资金流的整体动量分数。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        """
        df_index = df.index
        # 获取主力净流量和总市值
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        total_market_value = self._get_safe_series(df, 'total_market_value_D', pd.Series(1e9, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        # 计算标准化主力净流量 (NMFNF)，避免除以零
        nmfnf = (main_force_net_flow / total_market_value.replace(0, 1e9)).fillna(0)
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 归一化NMFNF本身，反映当前资金流的相对强度
        nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(nmfnf, df_index, tf_weights_ff, sensitivity=0.001) # 敏感度根据实际数据调整
        # 获取NMFNF的5日和21日斜率，反映资金流的动量和趋势
        slope_5_nmfnf = self._get_safe_series(df, 'SLOPE_5_NMFNF_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        slope_21_nmfnf = self._get_safe_series(df, 'SLOPE_21_NMFNF_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        # 归一化斜率
        slope_5_nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(slope_5_nmfnf, df_index, tf_weights_ff, sensitivity=0.0001)
        slope_21_nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(slope_21_nmfnf, df_index, tf_weights_ff, sensitivity=0.00005)
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

