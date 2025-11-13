# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional

from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar

class ProcessIntelligence:
    """
    【V2.0.0 · 全息四象限引擎】
    - 核心升级: 最终输出分数 meta_score 已升级为 [-1, 1] 的双极区间，完美对齐四象限逻辑。
                +1 代表极强的看涨拐点信号，-1 代表极强的看跌拐点信号。
    - 实现方式: 1. 使用 normalize_to_bipolar 替换 normalize_score 对趋势和加速度进行归一化。
                2. 使用加权平均法替换乘法来融合趋势和加速度，避免负负得正的逻辑错误。
    - 版本: 2.0.0
    """
    def __init__(self, strategy_instance):
        """
        【V3.3.0 · 领域反转生成版】
        - 核心修复: 彻底移除在代码中硬编码的 `genesis_diagnostics` 列表。
        - 核心升级: 确保 `process_intelligence_params.diagnostics` 配置是诊断任务的唯一真相来源，
                      消除了重复执行的严重BUG，并遵循了“配置即代码”的最佳实践。
        - 支持生成原子情报领域的反转信号。
        """
        self.strategy = strategy_instance
        self.params = get_params_block(self.strategy, 'process_intelligence_params', {})
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})
        self.norm_window = get_param_value(self.params.get('norm_window'), 55)
        self.std_window = get_param_value(self.params.get('std_window'), 21)
        self.meta_window = get_param_value(self.params.get('meta_window'), 5)
        self.bipolar_sensitivity = get_param_value(self.params.get('bipolar_sensitivity'), 1.0)
        self.meta_score_weights = get_param_value(self.params.get('meta_score_weights'), [0.6, 0.4])
        self.diagnostics_config = get_param_value(self.params.get('diagnostics'), [])

    def run_process_diagnostics(self, task_type_filter: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        【V3.2.0 · 衰减与反转分析版】运行所有在配置中定义的元分析诊断任务。
        - 核心升级: 新增对 'decay_analysis' 和 'domain_reversal' 任务类型的支持。
        """
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            return {}
        for config in self.diagnostics_config:
            if task_type_filter and config.get('task_type') != task_type_filter:
                continue
            signal_name = config.get('name')
            signal_type = config.get('type')
            if not signal_name:
                continue
            if signal_type in ['meta_analysis', 'strategy_sync']:
                custom_signal_type = config.get('signal_type')
                if custom_signal_type == 'split_meta_analysis':
                    split_states = self._diagnose_split_meta_relationship(df, config)
                    if split_states:
                        all_process_states.update(split_states)
                elif custom_signal_type == 'decay_analysis':
                    decay_states = self._diagnose_signal_decay(df, config)
                    if decay_states:
                        all_process_states.update(decay_states)
                # 新增路由到领域反转诊断器
                elif custom_signal_type == 'domain_reversal':
                    reversal_states = self._diagnose_domain_reversal(df, config)
                    if reversal_states:
                        all_process_states.update(reversal_states)
                else:
                    meta_states = self._diagnose_meta_relationship(df, config)
                    if meta_states:
                        all_process_states.update(meta_states)
        return all_process_states

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        signal_name = config.get('name')
        if signal_name == 'PROCESS_META_MAIN_FORCE_URGENCY':
            return self._calculate_main_force_urgency_relationship(df, config)
        if signal_name == 'PROCESS_META_COST_ADVANTAGE_TREND': # 增加对 PROCESS_META_COST_ADVANTAGE_TREND 的判断
            return self._calculate_cost_advantage_trend_relationship(df, config) # 调用定制化方法
        if signal_name == 'PROCESS_META_MAIN_FORCE_CONTROL': # 新增对 PROCESS_META_MAIN_FORCE_CONTROL 的判断
            return self._calculate_main_force_control_relationship(df, config) # 调用定制化方法
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        df_index = df.index
        relationship_type = config.get('relationship_type', 'consensus')
        def get_signal_series(signal_name: str, source_type: str) -> Optional[pd.Series]:
            series = None
            if source_type == 'atomic_states':
                series = self.strategy.atomic_states.get(signal_name)
            else:
                series = df.get(signal_name)
            if series is None:
                print(f"        -> [过程层警告] 依赖信号 '{signal_name}' (来源: {source_type}) 不存在，无法计算关系。")
            return series
        signal_a = get_signal_series(config.get('signal_A'), config.get('source_A', 'df'))
        signal_b = get_signal_series(config.get('signal_B'), config.get('source_B', 'df'))
        if signal_a is None or signal_b is None:
            return pd.Series(dtype=np.float32)
        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if change_type == 'diff':
                return series.diff(1).fillna(0)
            return ta.percent_return(series, length=1).fillna(0)
        change_a = get_change_series(signal_a, config.get('change_type_A', 'pct'))
        change_b = get_change_series(signal_b, config.get('change_type_B', 'pct'))
        momentum_a = normalize_to_bipolar(change_a, df_index, self.std_window, self.bipolar_sensitivity)
        thrust_b = normalize_to_bipolar(change_b, df_index, self.std_window, self.bipolar_sensitivity)
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        if relationship_type == 'divergence':
            relationship_score = (signal_b_factor_k * thrust_b - momentum_a) / (signal_b_factor_k + 1)
        else:
            relationship_score = (momentum_a + signal_b_factor_k * thrust_b) / (1 + signal_b_factor_k)
        relationship_score = relationship_score.clip(-1, 1)
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        return relationship_score

    def _calculate_cost_advantage_trend_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        print("    -> [过程层] 正在计算 PROCESS_META_COST_ADVANTAGE_TREND (深度博弈四象限版)...")
        df_index = df.index
        std_window = self.std_window
        bipolar_sensitivity = self.bipolar_sensitivity
        price_change = df.get('pct_change_D', pd.Series(0.0, index=df_index))
        main_force_cost_advantage = df.get('main_force_cost_advantage_D', pd.Series(0.0, index=df_index))
        P_change = normalize_to_bipolar(price_change, df_index, std_window, bipolar_sensitivity)
        CA_change = normalize_to_bipolar(main_force_cost_advantage.diff(1).fillna(0), df_index, std_window, bipolar_sensitivity)
        MF_flow = normalize_to_bipolar(df.get('main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index)), df_index, std_window, bipolar_sensitivity)
        Chip_conc = self.strategy.atomic_states.get('SCORE_CHIP_AXIOM_CONCENTRATION', pd.Series(0.0, index=df_index))
        Micro_decep = self.strategy.atomic_states.get('SCORE_MICRO_AXIOM_DECEPTION', pd.Series(0.0, index=df_index))
        Up_eff_unipolar = self.strategy.atomic_states.get('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', pd.Series(0.5, index=df_index))
        Up_eff_bipolar = (Up_eff_unipolar * 2 - 1).clip(-1, 1)
        Vol_apathy_unipolar = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_APATHY', pd.Series(0.5, index=df_index))
        Vol_apathy_bipolar = (Vol_apathy_unipolar * 2 - 1).clip(-1, 1)
        Q1_base = (P_change.clip(lower=0) + CA_change.clip(lower=0)) / 2
        Q1_confirm = (MF_flow.clip(lower=0) + Chip_conc.clip(lower=0) + Up_eff_bipolar.clip(lower=0)) / 3
        Q1_final = Q1_base * Q1_confirm
        Q2_base = (P_change.clip(upper=0).abs() + CA_change.clip(upper=0).abs()) / 2
        MF_flow_bearish = MF_flow.clip(upper=0).abs()
        Chip_conc_bearish = Chip_conc.clip(upper=0).abs()
        Down_eff_bearish = Up_eff_bipolar.clip(upper=0).abs()
        Q2_confirm = (MF_flow_bearish + Chip_conc_bearish + Down_eff_bearish) / 3
        Q2_final = Q2_base * Q2_confirm * -1
        Q3_base = (P_change.clip(upper=0).abs() + CA_change.clip(lower=0)) / 2
        Q3_confirm = (MF_flow.clip(lower=0) + Chip_conc.clip(lower=0) + Micro_decep.clip(lower=0) + Vol_apathy_bipolar.clip(lower=0)) / 4
        Q3_final = Q3_base * Q3_confirm
        Q4_base = (P_change.clip(lower=0) + CA_change.clip(upper=0).abs()) / 2
        MF_flow_bearish_Q4 = MF_flow.clip(upper=0).abs()
        Chip_conc_bearish_Q4 = Chip_conc.clip(upper=0).abs()
        Micro_decep_bearish_Q4 = Micro_decep.clip(upper=0).abs()
        Up_eff_bearish_Q4 = Up_eff_bipolar.clip(upper=0).abs()
        Q4_confirm = (MF_flow_bearish_Q4 + Chip_conc_bearish_Q4 + Micro_decep_bearish_Q4 + Up_eff_bearish_Q4) / 4
        Q4_final = Q4_base * Q4_confirm * -1
        final_score = (Q1_final * 0.4 + Q2_final * 0.3 + Q3_final * 0.2 + Q4_final * 0.1)
        final_score = final_score.clip(-1, 1)
        self.strategy.atomic_states[f"_DEBUG_P_change"] = P_change
        self.strategy.atomic_states[f"_DEBUG_CA_change"] = CA_change
        self.strategy.atomic_states[f"_DEBUG_MF_flow"] = MF_flow
        self.strategy.atomic_states[f"_DEBUG_Chip_conc"] = Chip_conc
        self.strategy.atomic_states[f"_DEBUG_Micro_decep"] = Micro_decep
        self.strategy.atomic_states[f"_DEBUG_Up_eff_bipolar"] = Up_eff_bipolar
        self.strategy.atomic_states[f"_DEBUG_Vol_apathy_bipolar"] = Vol_apathy_bipolar
        self.strategy.atomic_states[f"_DEBUG_Q1_final"] = Q1_final
        self.strategy.atomic_states[f"_DEBUG_Q2_final"] = Q2_final
        self.strategy.atomic_states[f"_DEBUG_Q3_final"] = Q3_final
        self.strategy.atomic_states[f"_DEBUG_Q4_final"] = Q4_final
        print(f"    -> [过程层] PROCESS_META_COST_ADVANTAGE_TREND 计算完成，最新分值: {final_score.iloc[-1]:.4f}")
        return final_score.astype(np.float32)

    def _calculate_main_force_urgency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 深度博弈版】计算“主力紧迫度”的专属关系分数。
        - 核心逻辑: 衡量主力买入成本的抬升速度是否显著快于价格的上涨速度，并结合主力资金净流入进行确认。
        - 证据链:
          1. 价格变化率 (`pct_change_D`)
          2. 主力主动买入成本变化率 (`SLOPE_X_active_winner_avg_cost_D`)
          3. 主力主动买入成本变化加速度 (`ACCEL_X_active_winner_avg_cost_D`)
          4. 主力资金净流入 (`main_force_net_flow_calibrated_D`)
          5. 筹码集中度变化 (`SLOPE_X_short_term_concentration_90pct_D`)
        - 输出: [-1, 1] 的双极性分数，正分代表主力紧迫度高，负分代表紧迫度低或主力不积极。
        - 【修正】调整 `final_urgency_score` 的计算逻辑，确保在价格上涨且主力资金流入时能更准确地反映紧迫度。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_URGENCY (主力紧迫度)...")
        df_index = df.index
        norm_window = self.norm_window
        std_window = self.std_window
        # 1. 获取核心信号
        price_change = df.get('pct_change_D', pd.Series(0.0, index=df_index))
        main_force_cost_change_raw = df.get(f'SLOPE_5_active_winner_avg_cost_D', pd.Series(0.0, index=df_index))
        main_force_cost_accel_raw = df.get(f'ACCEL_5_active_winner_avg_cost_D', pd.Series(0.0, index=df_index))
        main_force_net_flow = df.get('main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index))
        chip_concentration_change = df.get(f'SLOPE_5_short_term_concentration_90pct_D', pd.Series(0.0, index=df_index))
        # 2. 归一化为双极性分数
        price_change_bipolar = normalize_to_bipolar(price_change, df_index, std_window, self.bipolar_sensitivity)
        main_force_cost_change_bipolar = normalize_to_bipolar(main_force_cost_change_raw, df_index, std_window, self.bipolar_sensitivity)
        main_force_cost_accel_bipolar = normalize_to_bipolar(main_force_cost_accel_raw, df_index, std_window, self.bipolar_sensitivity)
        main_force_net_flow_bipolar = normalize_to_bipolar(main_force_net_flow, df_index, std_window, self.bipolar_sensitivity)
        chip_concentration_change_bipolar = normalize_to_bipolar(chip_concentration_change, df_index, std_window, self.bipolar_sensitivity)
        # 3. 核心紧迫度计算：主力成本抬升速度 vs 价格上涨速度
        # 当主力成本抬升速度显著快于价格上涨速度时，紧迫度高
        # 我们可以用 (成本变化率 - 价格变化率) 来衡量这种相对强度
        relative_urgency_speed = (main_force_cost_change_bipolar - price_change_bipolar).clip(-1, 1)
        # 4. 引入加速度作为确认因子：成本抬升的加速度越快，紧迫度越高
        accel_confirmation = main_force_cost_accel_bipolar
        # 5. 主力资金净流入作为乘数因子：只有在主力资金净流入的背景下，紧迫度才有效
        # 将资金流从 [-1, 1] 映射到 [0, 1]，负值变为0，正值保持
        # 修正：如果主力资金净流入为负，则乘数应为负，以压制紧迫度
        main_force_flow_multiplier = main_force_net_flow_bipolar.clip(-1, 1)
        # 6. 筹码集中度变化作为辅助确认：筹码集中度上升，进一步确认主力吸筹意图
        chip_concentration_confirmation = chip_concentration_change_bipolar.clip(lower=0)
        # 7. 综合紧迫度分数
        w_speed = 0.4
        w_accel = 0.3
        w_chip = 0.2
        initial_urgency_score = (
            relative_urgency_speed * w_speed +
            accel_confirmation * w_accel +
            chip_concentration_confirmation * w_chip
        ) / (w_speed + w_accel + w_chip)
        # 8. 最终紧迫度：乘以主力资金净流入乘数
        # 修正：如果主力资金净流入为负，则最终紧迫度应为负
        final_urgency_score = (initial_urgency_score * main_force_flow_multiplier).clip(-1, 1)
        # 9. 进一步的条件判断：只有在价格上涨的背景下，才认为存在“紧迫度”
        # 如果价格下跌，即使主力成本下降很快，也不是我们定义的“紧迫度”
        # 修正：价格下跌时，紧迫度分数应为负值或0
        final_urgency_score = final_urgency_score.mask(price_change_bipolar < 0, final_urgency_score.clip(upper=0))
        self.strategy.atomic_states[f"_DEBUG_relative_urgency_speed"] = relative_urgency_speed
        self.strategy.atomic_states[f"_DEBUG_accel_confirmation"] = accel_confirmation
        self.strategy.atomic_states[f"_DEBUG_main_force_flow_multiplier"] = main_force_flow_multiplier
        self.strategy.atomic_states[f"_DEBUG_chip_concentration_confirmation"] = chip_concentration_confirmation
        self.strategy.atomic_states[f"_DEBUG_initial_urgency_score"] = initial_urgency_score
        self.strategy.atomic_states[f"_DEBUG_final_urgency_score_raw"] = final_urgency_score
        print(f"    -> [过程层] PROCESS_META_MAIN_FORCE_URGENCY 计算完成，最新分值: {final_urgency_score.iloc[-1]:.4f}")
        return final_urgency_score.astype(np.float32)

    def _calculate_main_force_control_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0】计算“主力控盘”的专属关系分数。
        - 核心逻辑: 诊断主力控盘的强度和趋势，基于通达信公式中的“控盘”和“有庄控盘”逻辑。
        - 证据链:
          1. 控盘指标 (VARN1-REF(VARN1,1))/REF(VARN1,1)*1000
          2. 控盘趋势 (控盘>REF(控盘,1) AND 控盘>0)
          3. 主力资金净流入 (main_force_net_flow_calibrated_D)
        - 输出: [-1, 1] 的双极性分数，正分代表主力控盘强且趋势向上，负分代表控盘弱或趋势向下。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_CONTROL (主力控盘)...")
        df_index = df.index
        std_window = self.std_window
        bipolar_sensitivity = self.bipolar_sensitivity
        # 1. 计算 VARN1 (EMA(EMA(CLOSE,13),13))
        ema13 = ta.ema(close=df['close_D'], length=13, append=False)
        varn1 = ta.ema(close=ema13, length=13, append=False)
        # 2. 计算控盘 (VARN1-REF(VARN1,1))/REF(VARN1,1)*1000
        # 避免除以零
        prev_varn1 = varn1.shift(1).replace(0, np.nan)
        kongpan_raw = (varn1 - prev_varn1) / prev_varn1 * 1000
        # 3. 计算有庄控盘 (控盘>REF(控盘,1) AND 控盘>0)
        youzhuang_kongpan = (kongpan_raw > kongpan_raw.shift(1)) & (kongpan_raw > 0)
        # 4. 主力资金净流入作为确认
        main_force_net_flow = df.get('main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index))
        # 5. 归一化为双极性分数
        kongpan_score = normalize_to_bipolar(kongpan_raw, df_index, std_window, bipolar_sensitivity)
        main_force_flow_score = normalize_to_bipolar(main_force_net_flow, df_index, std_window, bipolar_sensitivity)
        # 6. 融合：控盘强度 * 控盘趋势 * 主力资金流
        # 只有在有庄控盘为True时，才考虑控盘强度和资金流
        final_control_score = pd.Series(0.0, index=df_index)
        # 将 youzhuang_kongpan 转换为 float 类型，以便进行乘法运算
        youzhuang_kongpan_float = youzhuang_kongpan.astype(float)
        # 控盘分数和资金流分数都为正时，才贡献正分
        final_control_score = (kongpan_score.clip(lower=0) * main_force_flow_score.clip(lower=0) * youzhuang_kongpan_float).pow(1/3)
        # 如果控盘分数或资金流分数是负的，则贡献负分
        final_control_score = final_control_score.mask(kongpan_score < 0, kongpan_score.clip(upper=0))
        final_control_score = final_control_score.mask(main_force_flow_score < 0, main_force_flow_score.clip(upper=0))
        # 最终分数在 [-1, 1] 之间
        final_control_score = final_control_score.clip(-1, 1)
        self.strategy.atomic_states[f"_DEBUG_kongpan_raw"] = kongpan_raw
        self.strategy.atomic_states[f"_DEBUG_youzhuang_kongpan"] = youzhuang_kongpan
        self.strategy.atomic_states[f"_DEBUG_kongpan_score"] = kongpan_score
        self.strategy.atomic_states[f"_DEBUG_main_force_flow_score"] = main_force_flow_score
        print(f"    -> [过程层] PROCESS_META_MAIN_FORCE_CONTROL 计算完成，最新分值: {final_control_score.iloc[-1]:.4f}")
        return final_control_score.astype(np.float32)

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0.0 · 希格斯场分析法】对“关系分”进行元分析，输出分数。
        - 核心革命: 废除基于线性回归(linreg)的“趋势/加速度”模型，引入全新的“关系位移/关系动量”模型。
                      这使得引擎能更灵敏地捕捉关系的非线性变化和“势”的拐点，更符合A股特性。
        - 核心修复: 再次强调并修正“加速度”计算的致命逻辑错误。加速度是速度(trend)的一阶导数，
                      必须使用 relationship_trend.diff(1) 进行计算。
        """
        signal_name = config.get('name')
        df_index = df.index
        if signal_name == 'PROCESS_META_WINNER_CONVICTION' and 'antidote_signal' in config:
            relationship_score = self._calculate_winner_conviction_relationship(df, config)
        else:
            relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
        intermediate_signal_name = f"PROCESS_ATOMIC_REL_SCORE_{config.get('signal_A')}_VS_{config.get('signal_B')}"
        self.strategy.atomic_states[intermediate_signal_name] = relationship_score.astype(np.float32)
        diagnosis_mode = config.get('diagnosis_mode', 'meta_analysis')
        if diagnosis_mode == 'direct_confirmation':
            meta_score = relationship_score
        else:
            relationship_displacement = relationship_score.diff(self.meta_window).fillna(0)
            relationship_momentum = relationship_displacement.diff(1).fillna(0)
            bipolar_displacement_strength = normalize_to_bipolar(
                series=relationship_displacement,
                target_index=df_index,
                window=self.norm_window,
                sensitivity=self.bipolar_sensitivity
            )
            bipolar_momentum_strength = normalize_to_bipolar(
                series=relationship_momentum,
                target_index=df_index,
                window=self.norm_window,
                sensitivity=self.bipolar_sensitivity
            )
            displacement_weight = self.meta_score_weights[0]
            momentum_weight = self.meta_score_weights[1]
            meta_score = (bipolar_displacement_strength * displacement_weight + bipolar_momentum_strength * momentum_weight)
        if diagnosis_mode == 'gated_meta_analysis':
            gate_condition_config = config.get('gate_condition', {})
            gate_type = gate_condition_config.get('type')
            gate_is_open = pd.Series(True, index=df_index)
            if gate_type == 'price_vs_ma':
                ma_period = gate_condition_config.get('ma_period', 5)
                ma_series = df.get(f'EMA_{ma_period}_D')
                if ma_series is not None:
                    gate_is_open = df['close_D'] < ma_series
            meta_score = meta_score * gate_is_open.astype(float)
        signal_meta = self.score_type_map.get(signal_name, {})
        scoring_mode = signal_meta.get('scoring_mode', 'bipolar')
        if scoring_mode == 'unipolar':
            meta_score = meta_score.clip(lower=0)
        meta_score = meta_score.clip(-1, 1).astype(np.float32)
        return {signal_name: meta_score}

    def _diagnose_split_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V2.0 · 希格斯场分析法】分裂型元关系诊断器
        - 核心升级: 同步采用全新的“关系位移/关系动量”模型进行核心计算。
        """
        states = {}
        output_names = config.get('output_names', {})
        opportunity_signal_name = output_names.get('opportunity')
        risk_signal_name = output_names.get('risk')
        if not opportunity_signal_name or not risk_signal_name:
            print(f"        -> [分裂元分析] 警告: 缺少 'output_names' 配置，无法进行信号分裂。")
            return {}
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
        relationship_displacement = relationship_score.diff(self.meta_window).fillna(0)
        relationship_momentum = relationship_displacement.diff(1).fillna(0)
        bipolar_displacement_strength = normalize_to_bipolar(
            series=relationship_displacement,
            target_index=df.index,
            window=self.norm_window,
            sensitivity=self.bipolar_sensitivity
        )
        bipolar_momentum_strength = normalize_to_bipolar(
            series=relationship_momentum,
            target_index=df.index,
            window=self.norm_window,
            sensitivity=self.bipolar_sensitivity
        )
        displacement_weight = self.meta_score_weights[0]
        momentum_weight = self.meta_score_weights[1]
        meta_score = (bipolar_displacement_strength * displacement_weight + bipolar_momentum_strength * momentum_weight)
        meta_score = meta_score.clip(-1, 1)
        opportunity_part = meta_score.clip(lower=0)
        states[opportunity_signal_name] = opportunity_part.astype(np.float32)
        risk_part = meta_score.clip(upper=0).abs()
        states[risk_signal_name] = risk_part.astype(np.float32)
        return states

    def _calculate_winner_conviction_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 真理探针植入版】“赢家信念”专属关系计算引擎
        - 新增功能: 植入“真理探针”，打印计算过程中的所有中间变量。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        antidote_signal_name = config.get('antidote_signal')
        df_index = df.index
        def get_signal_series(signal_name: str) -> Optional[pd.Series]:
            return df.get(signal_name)
        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if series is None: return pd.Series(dtype=np.float32)
            if change_type == 'diff':
                return series.diff(1).fillna(0)
            return ta.percent_return(series, length=1).fillna(0)
        signal_a = get_signal_series(signal_a_name)
        signal_b = get_signal_series(signal_b_name)
        signal_antidote = get_signal_series(antidote_signal_name)
        if signal_a is None or signal_b is None or signal_antidote is None:
            print(f"        -> [赢家信念] 警告: 缺少原始信号 '{signal_a_name}', '{signal_b_name}' 或 '{antidote_signal_name}'。")
            return pd.Series(dtype=np.float32)
        momentum_a = normalize_to_bipolar(get_change_series(signal_a, config.get('change_type_A')), df_index, self.std_window, self.bipolar_sensitivity)
        momentum_b_raw = normalize_to_bipolar(get_change_series(signal_b, config.get('change_type_B')), df_index, self.std_window, self.bipolar_sensitivity)
        momentum_antidote = normalize_to_bipolar(get_change_series(signal_antidote, config.get('antidote_change_type')), df_index, self.std_window, self.bipolar_sensitivity)
        antidote_k = config.get('antidote_k', 1.0)
        momentum_b_corrected = momentum_b_raw + antidote_k * momentum_antidote
        k = config.get('signal_b_factor_k', 1.0)
        relationship_score = (k * momentum_b_corrected - momentum_a) / (k + 1)
        return relationship_score.clip(-1, 1)

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.0】信号衰减诊断器
        - 核心职责: 专门用于计算单个信号的负向变化（衰减）强度。
        - 数学逻辑: 1. 计算信号的一阶差分。 2. 只保留负值（代表衰减）。 3. 取绝对值。 4. 归一化。
        - 收益: 提供了计算“衰减”的正确且健壮的数学模型，取代了错误的关系诊断模型。
        """
        signal_name = config.get('name')
        source_signal_name = config.get('source_signal')
        source_type = config.get('source_type', 'df')
        df_index = df.index
        if not source_signal_name:
            print(f"        -> [衰减分析] 警告: 缺少 'source_signal' 配置。")
            return {}
        source_series = None
        if source_type == 'atomic_states':
            source_series = self.strategy.atomic_states.get(source_signal_name)
        else:
            source_series = df.get(source_signal_name)
        if source_series is None:
            print(f"        -> [衰减分析] 警告: 缺少源信号 '{source_signal_name}'。")
            return {}
        signal_change = source_series.diff(1).fillna(0)
        decay_magnitude = signal_change.clip(upper=0).abs()
        decay_score = normalize_score(decay_magnitude, df_index, window=self.norm_window, ascending=True)
        return {signal_name: decay_score.astype(np.float32)}

    def _diagnose_domain_reversal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.0】通用领域反转诊断器
        - 核心职责: 接收一个原子情报领域的公理信号列表和权重，计算该领域的双极性健康度，
                      然后从健康度的变化中派生底部反转和顶部反转信号。
        - 命名规范: 输出信号为 PROCESS_META_DOMAIN_BOTTOM_REVERSAL 和 PROCESS_META_DOMAIN_TOP_REVERSAL。
        """
        domain_name = config.get('domain_name')
        axiom_configs = config.get('axioms', [])
        output_bottom_name = config.get('output_bottom_reversal_name')
        output_top_name = config.get('output_top_reversal_name')
        if not domain_name or not axiom_configs or not output_bottom_name or not output_top_name:
            print(f"        -> [领域反转诊断] 警告: 配置不完整，跳过领域 '{domain_name}' 的反转诊断。")
            return {}
        df_index = df.index
        domain_health_components = []
        total_weight = 0.0
        for axiom_config in axiom_configs:
            axiom_name = axiom_config.get('name')
            axiom_weight = axiom_config.get('weight', 0.0)
            axiom_score = self.strategy.atomic_states.get(axiom_name, pd.Series(0.0, index=df_index))
            domain_health_components.append(axiom_score * axiom_weight)
            total_weight += axiom_weight
        if total_weight == 0:
            print(f"        -> [领域反转诊断] 警告: 领域 '{domain_name}' 的公理权重总和为0，无法计算健康度。")
            return {}
        # 计算该领域的双极性健康度
        bipolar_domain_health = (sum(domain_health_components) / total_weight).clip(-1, 1)
        # 从健康度派生反转信号
        # 底部反转信号：当健康度从负值区域开始向上改善时
        bottom_reversal_raw = (bipolar_domain_health.diff(1).clip(lower=0) * (1 - bipolar_domain_health.clip(lower=0))).fillna(0)
        bottom_reversal_score = normalize_score(bottom_reversal_raw, df_index, 21, ascending=True)
        # 顶部反转信号：当健康度从正值区域开始向下恶化时
        top_reversal_raw = (bipolar_domain_health.diff(1).clip(upper=0).abs() * (1 + bipolar_domain_health.clip(upper=0))).fillna(0)
        top_reversal_score = normalize_score(top_reversal_raw, df_index, 21, ascending=True)
        return {
            output_bottom_name: bottom_reversal_score.astype(np.float32),
            output_top_name: top_reversal_score.astype(np.float32)
        }
