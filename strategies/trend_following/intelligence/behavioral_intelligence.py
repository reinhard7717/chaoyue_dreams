# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from strategies.trend_following.intelligence.tactic_engine import TacticEngine
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state, normalize_score, normalize_to_bipolar, calculate_holographic_dynamics, calculate_context_scores

class BehavioralIntelligence:
    """
    【V28.0 · 结构升维版】
    - 核心升级: 废弃了旧的 _calculate_price_health, _calculate_volume_health, _calculate_kline_pattern_health 方法。
                所有健康度计算已统一由全新的 _calculate_structural_behavior_health 引擎负责。
    """
    def __init__(self, strategy_instance):
        """
        初始化行为与模式识别模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        # K线形态识别器可能需要在这里初始化或传入
        self.pattern_recognizer = strategy_instance.pattern_recognizer
        self.tactic_engine = TacticEngine(strategy_instance)

    def run_behavioral_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V3.4 · 职责净化版】行为情报模块总指挥
        - 核心重构: 移除了对 _diagnose_archangel_top_reversal 的调用。
                      “天使长”作为顶层认知信号，其诊断职责已正式移交认知情报模块。
        """
        df = self.strategy.df_indicators
        all_behavioral_states = {}
        internal_atomic_signals = self._generate_all_atomic_signals(df)
        if internal_atomic_signals:
            self.strategy.atomic_states.update(internal_atomic_signals)
            all_behavioral_states.update(internal_atomic_signals)
        ultimate_behavioral_states = self.diagnose_ultimate_behavioral_signals(df, atomic_signals=internal_atomic_signals)
        if ultimate_behavioral_states:
            all_behavioral_states.update(ultimate_behavioral_states)
        return all_behavioral_states

    def diagnose_ultimate_behavioral_signals(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        【V29.1 · 动态引擎换装版】行为终极信号诊断引擎
        - 核心重构: 废弃对通用函数 transmute_health_to_ultimate_signals 的调用，引入“四象限动态分析法”，
                      彻底解决信号命名与逻辑混乱的问题，确保与筹码、资金流模块的哲学统一。
        - 核心升级: 换装全新的 `_calculate_signal_dynamics` 引擎，对健康度进行更精准的动态分析。
        """
        if atomic_signals is None:
            atomic_signals = self._generate_all_atomic_signals(df)
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        resonance_tf_weights = get_param_value(p_synthesis.get('resonance_tf_weights'), {'short': 0.2, 'medium': 0.5, 'long': 0.3})
        # 步骤一：获取原始的双极性健康度字典
        overall_health = self._calculate_structural_behavior_health(df, p_conf)
        self.strategy.atomic_states['_internal_behavior_health_dict'] = overall_health
        # 步骤二：融合多时间周期的双极性健康分
        period_groups = {
            'short': [p for p in periods if p <= 5],
            'medium': [p for p in periods if 5 < p <= 21],
            'long': [p for p in periods if p > 21]
        }
        final_bipolar_health = pd.Series(0.0, index=df.index, dtype=np.float64)
        total_weight = sum(resonance_tf_weights.values())
        if total_weight > 0:
            for tf_name, weight in resonance_tf_weights.items():
                group_periods = period_groups.get(tf_name, [])
                group_scores = [overall_health['s_bull'].get(p, pd.Series(0.0, index=df.index)) for p in group_periods]
                if group_scores:
                    avg_group_score = sum(group_scores) / len(group_scores)
                    final_bipolar_health += avg_group_score * (weight / total_weight)
        final_bipolar_health = final_bipolar_health.clip(-1, 1).astype(np.float32)
        # 步骤三：分离为纯粹的看涨/看跌健康分
        bullish_health = final_bipolar_health.clip(0, 1)
        bearish_health = (final_bipolar_health.clip(-1, 0) * -1)
        states['SCORE_STRUCT_BEHAVIOR_BULLISH_RESONANCE'] = bullish_health
        states['SCORE_STRUCT_BEHAVIOR_BEARISH_RESONANCE'] = bearish_health
        
        # [代码修改开始]
        # 步骤四：使用全新的动态分析引擎计算四象限动态信号
        # --- 基于“看涨健康分”的动态分析 ---
        bull_dynamics = self._calculate_signal_dynamics(bullish_health, p_conf)
        # 看涨加速 = 速度为正 & 加速度为正
        bullish_acceleration = (bull_dynamics['velocity'].clip(0, 1) * bull_dynamics['acceleration'].clip(0, 1)).pow(0.5)
        # 顶部反转风险 = 状态良好 & 速度为正 & 加速度为负 (强弩之末)
        top_reversal_risk = (bull_dynamics['state'].clip(0, 1) * bull_dynamics['velocity'].clip(0, 1) * (bull_dynamics['acceleration'].clip(-1, 0) * -1)).pow(1/3)
        # --- 基于“看跌健康分”的动态分析 ---
        bear_dynamics = self._calculate_signal_dynamics(bearish_health, p_conf)
        # 看跌加速 = 速度为正 & 加速度为正 (下跌加速)
        bearish_acceleration = (bear_dynamics['velocity'].clip(0, 1) * bear_dynamics['acceleration'].clip(0, 1)).pow(0.5)
        # 底部反转机会 = 状态良好(跌幅深) & 速度为正(开始减速) & 加速度为正 (出现拐点)
        bottom_reversal_opportunity = (bear_dynamics['state'].clip(0, 1) * bear_dynamics['velocity'].clip(0, 1) * bear_dynamics['acceleration'].clip(0, 1)).pow(1/3)
        # 步骤五：应用上下文并赋值给命名准确的终极信号
        self.strategy.atomic_states['strategy_instance_ref'] = self.strategy
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        del self.strategy.atomic_states['strategy_instance_ref']
        states['SCORE_STRUCT_BEHAVIOR_BULLISH_ACCELERATION'] = (bullish_acceleration * bottom_context_score).clip(0, 1).astype(np.float32)
        states['SCORE_STRUCT_BEHAVIOR_TOP_REVERSAL'] = (top_reversal_risk * top_context_score).clip(0, 1).astype(np.float32)
        states['SCORE_STRUCT_BEHAVIOR_BEARISH_ACCELERATION'] = (bearish_acceleration * top_context_score).clip(0, 1).astype(np.float32)
        states['SCORE_STRUCT_BEHAVIOR_BOTTOM_REVERSAL'] = (bottom_reversal_opportunity * bottom_context_score).clip(0, 1).astype(np.float32)
        # 步骤六：重铸战术反转信号
        states['SCORE_STRUCT_BEHAVIOR_TACTICAL_REVERSAL'] = (bullish_health * top_reversal_risk).clip(0, 1).astype(np.float32)
        return states

    # ==============================================================================
    # 以下为新增的原子信号中心和降级的原子诊断引擎
    # ==============================================================================
    def _generate_all_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.11 · 压力解析引擎换装版】原子信号中心
        - 核心重构: 使用全新的 `_diagnose_vpa_stagnation_risk` 替换了旧的 `_diagnose_volume_price_dynamics`。
        - 核心新增: 集成全新的 `_diagnose_grand_divergence` 引擎，生成战略级背离信号。
        - 核心升级: 调用全新的权威趋势健康度引擎 `_calculate_trend_health_score` 和权威绝望指数引擎 `_calculate_despair_context_score`，并将其结果存入原子状态。
        - 核心换装: 使用全新的 `_resolve_pressure_absorption_dynamics` 替换旧的压力嬗变逻辑。
        """
        atomic_signals = {}
        params = self.strategy.params
        # 步骤一: 首先计算并存储最基础、最权威的上下文分数
        trend_health_score = self._calculate_trend_health_score(df)
        atomic_signals['SCORE_TREND_HEALTH'] = trend_health_score
        self.strategy.atomic_states['SCORE_TREND_HEALTH'] = trend_health_score # 立即存入，供后续引擎使用
        despair_context_score = self._calculate_despair_context_score(df)
        atomic_signals['SCORE_CONTEXT_DESPAIR'] = despair_context_score
        self.strategy.atomic_states['SCORE_CONTEXT_DESPAIR'] = despair_context_score
        day_quality_score = self._calculate_day_quality_score(df)
        atomic_signals.update(self._diagnose_atomic_bottom_formation(df))
        epic_reversal_states = self._diagnose_atomic_rebound_reversal(df)
        continuation_reversal_states = self._diagnose_atomic_continuation_reversal(df)
        epic_score = epic_reversal_states.get('SCORE_ATOMIC_REBOUND_REVERSAL', pd.Series(0.0, index=df.index))
        continuation_score = continuation_reversal_states.get('SCORE_ATOMIC_CONTINUATION_REVERSAL', pd.Series(0.0, index=df.index))
        final_rebound_score = np.maximum(epic_score, continuation_score)
        atomic_signals['SCORE_ATOMIC_REBOUND_REVERSAL'] = final_rebound_score.astype(np.float32)
        atomic_signals.update(continuation_reversal_states)
        atomic_signals.update(self._diagnose_gap_support(df))
        pressure_signals = self._diagnose_provisional_pressure_risk(df)
        provisional_pressure = pressure_signals.get('PROVISIONAL_GENERAL_PRESSURE_RISK', pd.Series(0.0, index=df.index))
        atomic_signals.update(pressure_signals)
        intent_signals = self._diagnose_upper_shadow_intent(df)
        intent_diagnosis = intent_signals.get('SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS', pd.Series(0.0, index=df.index))
        atomic_signals.update(intent_signals)
        # [代码修改开始]
        # 使用全新的“压力-承接连续谱模型”进行解析
        final_pressure_signals = self._resolve_pressure_absorption_dynamics(provisional_pressure, intent_diagnosis)
        # [代码修改结束]
        atomic_signals.update(final_pressure_signals)
        atomic_signals.update(self._diagnose_advanced_atomic_signals(df))
        atomic_signals.update(self._diagnose_board_patterns(df))
        atomic_signals.update(self._diagnose_liquidity_dynamics(df))
        atomic_signals.update(self._diagnose_contraction_consolidation(df))
        atomic_signals.update(self._diagnose_vpa_stagnation_risk(df))
        upthrust_score_series = self._diagnose_upthrust_distribution(df, params, day_quality_score)
        atomic_signals[upthrust_score_series.name] = upthrust_score_series
        atomic_signals.update(self._diagnose_smart_intraday_trading(df))
        atomic_signals.update(self._diagnose_structural_fault_breakthrough(df))
        atomic_signals.update(self._diagnose_grand_divergence(df))
        return atomic_signals

    def _get_mtf_normalized_score(self, series: pd.Series, ascending: bool = True, tf_weights: Dict[int, float] = None) -> pd.Series:
        """
        【V1.1 · 防御性加固版】多时间框架(MTF)归一化引擎
        - 核心职责: 将单一指标的归一化从固定窗口升维为跨周期的加权融合。
        - 核心加固: 增加对权重字典中非数字值的过滤，防止因配置文件污染导致TypeError。
        - 输入:
          - series: 原始数据序列。
          - ascending: 归一化方向，True表示值越大分数越高。
          - tf_weights: 一个定义了周期及其权重的字典, e.g., {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}。
        - 返回: 一个在[0, 1]区间的、融合了多周期信息的动态分数。
        """
        if tf_weights is None:
            # 如果未提供权重，则使用默认的等权重配置
            tf_weights = {5: 0.2, 13: 0.2, 21: 0.2, 55: 0.2, 89: 0.2}
        
        # [代码修改开始]
        # 防御性编程：过滤掉权重字典中非数字的值，并处理嵌套结构
        if 'weights' in tf_weights and isinstance(tf_weights['weights'], dict):
            # 适配新的、更规范的配置文件结构
            valid_weights = {k: v for k, v in tf_weights['weights'].items() if isinstance(v, (int, float))}
        else:
            # 兼容旧的配置文件结构，并过滤非数字项
            valid_weights = {k: v for k, v in tf_weights.items() if isinstance(v, (int, float))}
        
        if not valid_weights:
             # 如果过滤后没有有效的权重，则退化为使用一个默认周期
            print(f"警告: 在 _get_mtf_normalized_score 中没有找到有效的权重配置。将使用默认周期 55。")
            return normalize_score(series, series.index, 55, ascending)

        final_score = pd.Series(0.0, index=series.index, dtype=np.float32)
        total_weight = sum(valid_weights.values())
        if total_weight <= 0:
            # 如果权重和为0，则退化为使用最长周期的归一化
            return normalize_score(series, series.index, max(valid_weights.keys()), ascending)
        
        for period_str, weight in valid_weights.items():
            try:
                period = int(period_str)
                single_period_score = normalize_score(series, series.index, period, ascending)
                final_score += single_period_score * (weight / total_weight)
            except (ValueError, TypeError) as e:
                print(f"警告: 在 _get_mtf_normalized_score 中跳过无效的周期配置: '{period_str}'. 错误: {e}")
                continue
        # [代码修改结束]
        return final_score.clip(0, 1)

    def _get_mtf_normalized_bipolar_score(self, series: pd.Series, tf_weights: Dict[int, float] = None, sensitivity: float = 1.0) -> pd.Series:
        """
        【V1.1 · 防御性加固版】多时间框架(MTF)双极性归一化引擎
        - 核心职责: _get_mtf_normalized_score 的双极性版本，输出范围为[-1, 1]。
        - 核心加固: 增加对权重字典中非数字值的过滤，防止因配置文件污染导致TypeError。
        """
        if tf_weights is None:
            tf_weights = {5: 0.2, 13: 0.2, 21: 0.2, 55: 0.2, 89: 0.2}
        
        # [代码修改开始]
        # 防御性编程：过滤掉权重字典中非数字的值，并处理嵌套结构
        if 'weights' in tf_weights and isinstance(tf_weights['weights'], dict):
            valid_weights = {k: v for k, v in tf_weights['weights'].items() if isinstance(v, (int, float))}
        else:
            valid_weights = {k: v for k, v in tf_weights.items() if isinstance(v, (int, float))}

        if not valid_weights:
            print(f"警告: 在 _get_mtf_normalized_bipolar_score 中没有找到有效的权重配置。将使用默认周期 55。")
            return normalize_to_bipolar(series, series.index, 55, sensitivity)

        final_score = pd.Series(0.0, index=series.index, dtype=np.float32)
        total_weight = sum(valid_weights.values())
        if total_weight <= 0:
            return normalize_to_bipolar(series, series.index, max(valid_weights.keys()), sensitivity)
        
        for period_str, weight in valid_weights.items():
            try:
                period = int(period_str)
                single_period_score = normalize_to_bipolar(series, series.index, period, sensitivity)
                final_score += single_period_score * (weight / total_weight)
            except (ValueError, TypeError) as e:
                print(f"警告: 在 _get_mtf_normalized_bipolar_score 中跳过无效的周期配置: '{period_str}'. 错误: {e}")
                continue
        # [代码修改结束]
        return final_score.clip(-1, 1)

    def _diagnose_liquidity_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 · MTF升维版】流动性动态诊断引擎
        - 核心升级: 全面采用新的MTF归一化引擎，废弃所有固定的 `norm_window`。
                      现在，价格动能和成交量压力的评估都是基于多时间框架的加权融合，
                      极大地提升了信号的鲁棒性和前瞻性。
        """
        # [代码修改开始]
        states = {}
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 第一步: 定义核心物理量 (已升维至MTF) ---
        price_momentum_axis = self._get_mtf_normalized_bipolar_score(
            df.get('pct_change_D', pd.Series(0.0, index=df.index)),
            tf_weights=default_weights
        )
        volume_ratio = df['volume_D'] / df.get('VOL_MA_21_D', df['volume_D'])
        volume_pressure_axis = self._get_mtf_normalized_bipolar_score(
            volume_ratio,
            tf_weights=default_weights
        )
        # --- 第二步: 构建四象限博弈模型 (逻辑不变，但输入已升维) ---
        p_mom_pos = price_momentum_axis.clip(lower=0)
        p_mom_neg_abs = price_momentum_axis.clip(upper=0).abs()
        v_press_pos = volume_pressure_axis.clip(lower=0)
        v_press_neg_abs = volume_pressure_axis.clip(upper=0).abs()
        lockup_quadrant = p_mom_pos * v_press_neg_abs
        breakthrough_quadrant = p_mom_pos * v_press_pos
        apathy_quadrant = p_mom_neg_abs * v_press_neg_abs
        panic_quadrant = p_mom_neg_abs * v_press_pos
        # --- 第三步: 引入趋势上下文进行“审判” (逻辑不变) ---
        trend_context = self.strategy.atomic_states.get('SCORE_STRUCT_BEHAVIOR_BULLISH_RESONANCE', pd.Series(0.5, index=df.index))
        # --- 第四步: 生成最终信号 (逻辑不变) ---
        bullish_force = (lockup_quadrant + breakthrough_quadrant) * trend_context
        bearish_force = (apathy_quadrant + panic_quadrant) * (1 - trend_context)
        bipolar_liquidity_dynamics = (bullish_force - bearish_force).clip(-1, 1).astype(np.float32)
        states['BEHAVIOR_BIPOLAR_LIQUIDITY_DYNAMICS'] = bipolar_liquidity_dynamics
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = (lockup_quadrant * trend_context).clip(0, 1).astype(np.float32)
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = (panic_quadrant * (1 - trend_context)).clip(0, 1).astype(np.float32)
        states['SCORE_VOL_WEAKENING_DROP'] = (apathy_quadrant * trend_context).clip(0, 1).astype(np.float32)
        return states

    def _diagnose_contraction_consolidation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · MTF升维版】三支柱“收缩盘整”诊断引擎
        - 核心升级: 全面采用新的MTF归一化引擎，废弃所有固定的 `norm_window`。
                      现在，对物理收缩、主力吸筹的评估都基于多时间框架，诊断结果更可靠。
        """
        states = {}
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        volatility_weights = get_param_value(p_mtf.get('volatility_weights'), {21: 0.5, 55: 0.3, 89: 0.2})
        # --- 支柱一: 物理收缩度 (已升维至MTF) ---
        turnover_series = df.get('turnover_rate_f_D', df.get('turnover_rate_D', pd.Series(10.0, index=df.index)))
        volume_contraction_score = self._get_mtf_normalized_score(turnover_series, ascending=False, tf_weights=default_weights)
        bbw_series = df.get('BBW_21_2.0_D', pd.Series(1.0, index=df.index))
        atr_series = df.get('ATR_14_D', pd.Series(1.0, index=df.index)) / df['close_D']
        bbw_contraction_score = self._get_mtf_normalized_score(bbw_series, ascending=False, tf_weights=volatility_weights)
        atr_contraction_score = self._get_mtf_normalized_score(atr_series, ascending=False, tf_weights=volatility_weights)
        price_volatility_score = (bbw_contraction_score * atr_contraction_score).pow(0.5)
        physical_contraction_score = (volume_contraction_score * price_volatility_score).pow(0.5)
        # --- 支柱二: 结构健康度 (逻辑不变) ---
        trend_health_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_TREND_QUALITY', pd.Series(0.5, index=df.index))
        position_health_score = (df['close_D'] > df.get('EMA_55_D', 0)).astype(float)
        structural_health_score = trend_health_score * position_health_score
        # --- 支柱三: 主力吸筹度 (已升维至MTF) ---
        concentration_slope = df.get('SLOPE_5_concentration_90pct_D', pd.Series(0.0, index=df.index))
        chip_accumulation_score = self._get_mtf_normalized_score(concentration_slope, ascending=False, tf_weights=default_weights)
        main_force_flow = df.get('main_force_net_flow_consensus_sum_5d_D', pd.Series(0.0, index=df.index))
        main_force_inflow_score = self._get_mtf_normalized_score(main_force_flow, ascending=True, tf_weights=default_weights)
        main_force_accumulation_score = (chip_accumulation_score * main_force_inflow_score).pow(0.5)
        # --- 最终融合 (逻辑不变) ---
        final_score = physical_contraction_score * structural_health_score * main_force_accumulation_score
        states['SCORE_BEHAVIOR_CONTRACTION_CONSOLIDATION'] = final_score.clip(0, 1).astype(np.float32)
        return states

    def _calculate_structural_behavior_health(self, df: pd.DataFrame, params: dict) -> Dict[str, Dict[int, pd.Series]]:
        """
        【V4.2 · 阿波罗战车版】结构与行为健康度计算核心引擎
        - 核心革命: 签署“阿波罗战车”协议，引入“日内四象限博弈”分析。
                      1. [轨迹四象限] 根据“跳空方向”和“实体方向”，将日内走势划分为四象限，并赋予“轨迹得分”。
                      2. [日内质量分] 融合“轨迹得分”和“影线修正分”，得到对K线质量的最终审判。
                      3. [终极强度融合] 使用“日内质量分”来调制“日间总涨跌幅”，计算出最终的“净有效强度”。
        - 收益: 能够精准解读“高开高走”、“低开高走”等不同日内轨迹的战术含义，评估结果更符合市场博弈的真实情况。
        """
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        sorted_periods = sorted(periods)
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        p_meta = get_param_value(params.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        s_bull, s_bear, d_intensity = {}, {}, {}
        # 实施“阿波罗战车”协议
        # --- 步骤1: 计算日内质量分 (Intraday Quality Score) ---
        gap_up = df['open_D'] > df['pre_close_D']
        gap_down = df['open_D'] < df['pre_close_D']
        body_up = df['close_D'] > df['open_D']
        body_down = df['close_D'] < df['open_D']
        # 1.1 轨迹得分 (Trajectory Score)
        trajectory_score = pd.Series(0.0, index=df.index)
        trajectory_score.loc[gap_up & body_up] = 1.0    # 高开高走
        trajectory_score.loc[gap_down & body_up] = 0.8   # 低开高走
        trajectory_score.loc[gap_up & body_down] = -0.8  # 高开低走
        trajectory_score.loc[gap_down & body_down] = -1.0  # 低开低走
        # 1.2 影线修正分 (Shadow Modifier)
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        lower_shadow = (np.minimum(df['open_D'], df['close_D']) - df['low_D']).clip(lower=0)
        shadow_modifier = ((lower_shadow - upper_shadow) / kline_range).fillna(0)
        # 1.3 融合得到日内质量分
        day_quality_score = (trajectory_score * 0.7 + shadow_modifier * 0.3).clip(-1, 1)
        # --- 步骤2: 计算净有效强度 ---
        quality_adjustment_factor = (1 + day_quality_score) / 2 # 将[-1, 1]映射到[0, 1]
        positive_day_strength_raw = df['pct_change_D'].clip(0)
        negative_day_strength_raw = df['pct_change_D'].clip(upper=0).abs()
        # 结果(50%) + 过程(50%)
        net_effective_bullish_strength = (positive_day_strength_raw * 0.5) + (positive_day_strength_raw * quality_adjustment_factor * 0.5)
        net_effective_bearish_strength = (negative_day_strength_raw * 0.5) + (negative_day_strength_raw * (1 - quality_adjustment_factor) * 0.5)
        # --- 步骤3: 应用“宙斯之雷”协议，归一化“净有效强度” ---
        positive_day_strength = normalize_score(net_effective_bullish_strength, df.index, norm_window) * (net_effective_bullish_strength > 0)
        negative_day_strength = normalize_score(net_effective_bearish_strength, df.index, norm_window) * (net_effective_bearish_strength > 0)
        
        efficiency_holo_bull, efficiency_holo_bear = calculate_holographic_dynamics(df, 'intraday_trend_efficiency_D', norm_window)
        gini_holo_bull, gini_holo_bear = calculate_holographic_dynamics(df, 'intraday_volume_gini_D', norm_window)
        bullish_d_intensity = ((efficiency_holo_bull + gini_holo_bull) / 2.0).astype(np.float32)
        bearish_d_intensity = ((efficiency_holo_bear + gini_holo_bear) / 2.0).astype(np.float32)
        closing_strength_score = normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        closing_weakness_score = 1.0 - closing_strength_score
        bullish_divergence = normalize_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        auction_power = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        trend_efficiency = normalize_score(df.get('intraday_trend_efficiency_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        bearish_divergence = normalize_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, norm_window)
        auction_weakness = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, norm_window)
        trend_inefficiency = 1 - trend_efficiency
        bullish_composite_state = (
            positive_day_strength * closing_strength_score * (1 + bullish_divergence) *
            auction_power * trend_efficiency * bullish_d_intensity
        )**(1/6)
        bearish_composite_state = (
            negative_day_strength * closing_weakness_score * (1 + bearish_divergence) *
            auction_weakness * trend_inefficiency * bearish_d_intensity
        )**(1/6)
        bipolar_composite_state = (bullish_composite_state - bearish_composite_state).clip(-1, 1)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            state_norm_tactical = normalize_to_bipolar(bipolar_composite_state, df.index, p)
            slope_raw = bipolar_composite_state.diff(p).fillna(0)
            slope_norm_tactical = normalize_to_bipolar(slope_raw, df.index, p)
            accel_raw = slope_raw.diff(1).fillna(0)
            accel_norm_tactical = normalize_to_bipolar(accel_raw, df.index, p)
            tactical_health_bipolar = (
                state_norm_tactical * w_state +
                slope_norm_tactical * w_velocity +
                accel_norm_tactical * w_acceleration
            ).clip(-1, 1)
            state_norm_context = normalize_to_bipolar(bipolar_composite_state, df.index, context_p)
            slope_norm_context = normalize_to_bipolar(slope_raw, df.index, context_p)
            accel_norm_context = normalize_to_bipolar(accel_raw, df.index, context_p)
            context_health_bipolar = (
                state_norm_context * w_state +
                slope_norm_context * w_velocity +
                accel_norm_context * w_acceleration
            ).clip(-1, 1)
            final_dynamic_bipolar_health = (tactical_health_bipolar + context_health_bipolar) / 2.0
            s_bull[p] = final_dynamic_bipolar_health.astype(np.float32)
            s_bear[p] = pd.Series(0.0, index=df.index, dtype=np.float32)
        for p in periods:
            d_intensity[p] = pd.Series(1.0, index=df.index, dtype=np.float32)
        return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}

    # 以下方法被降级为私有，作为原子信号的生产者
    def _diagnose_provisional_pressure_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】三维广义抛压诊断引擎
        - 核心逻辑: 融合“行为症状”、“资金病因”和“结构位置”三大维度，形成对抛压的立体诊断。
        """
        states = {}
        signal_name = 'PROVISIONAL_GENERAL_PRESSURE_RISK'
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 支柱一: 行为症状 (Symptom) ---
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow_ratio = ((df['high_D'] - np.maximum(df['open_D'], df['close_D'])) / kline_range).fillna(0)
        upper_shadow_score = self._get_mtf_normalized_score(upper_shadow_ratio, ascending=True, tf_weights=default_weights)
        closing_weakness_score = 1.0 - self._get_mtf_normalized_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        volume_spike_score = self._get_mtf_normalized_score(df['volume_D'], ascending=True, tf_weights=default_weights)
        behavioral_pressure = (upper_shadow_score * closing_weakness_score * volume_spike_score).pow(1/3)
        # --- 支柱二: 资金病因 (Cause) ---
        main_force_dist_score = self._get_mtf_normalized_score(df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        profit_urgency_score = self._get_mtf_normalized_score(df.get('profit_taking_urgency_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        chip_flow_pressure = (main_force_dist_score * profit_urgency_score).pow(0.5)
        # --- 支柱三: 结构位置 (Location) ---
        realized_pressure_score = self._get_mtf_normalized_score(df.get('realized_pressure_intensity_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 价格越接近或超过主峰，位置风险越高
        peak_location_risk = self._get_mtf_normalized_score(df.get('peak_distance_ratio_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        structural_pressure = (realized_pressure_score * peak_location_risk).pow(0.5)
        # --- 最终融合: 三大支柱共振 ---
        final_fused_snapshot_score = (behavioral_pressure * chip_flow_pressure * structural_pressure).pow(1/3)
        states[signal_name] = final_fused_snapshot_score.clip(0, 1).astype(np.float32)
        return states

    def _diagnose_gap_support(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】缺口支撑诊断引擎 (从 _diagnose_kline_patterns 拆分)
        """
        states = {}
        p = get_params_block(self.strategy, 'kline_pattern_params', {})
        p_gap = p.get('gap_support_params', {})
        if not get_param_value(p_gap.get('enabled'), True):
            return states
        persistence_days = get_param_value(p_gap.get('persistence_days'), 10)
        gap_up_mask = df['low_D'] > df['high_D'].shift(1)
        gap_high = df['high_D'].shift(1).where(gap_up_mask).ffill()
        price_fills_gap_mask = df['close_D'] < gap_high
        gap_support_state = create_persistent_state(df=df, entry_event_series=gap_up_mask, persistence_days=persistence_days, break_condition_series=price_fills_gap_mask, state_name='KLINE_STATE_GAP_SUPPORT_ACTIVE')
        support_distance = (df['low_D'] - gap_high).clip(lower=0)
        normalization_base = (df['close_D'] * 0.1).replace(0, np.nan)
        support_strength_score = (support_distance / normalization_base).clip(0, 1).fillna(0)
        states['SCORE_GAP_SUPPORT_ACTIVE'] = (support_strength_score * gap_support_state).astype(np.float32)
        return states

    def _diagnose_advanced_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 赫尔墨斯之翼优化版】诊断高级原子信号
        - 内存优化: 对连涨/连跌天数的计数结果使用`np.int16`存储，减少内存占用。
        - 核心逻辑: 保持高效的向量化连胜/连败计算逻辑不变。
        """
        states = {}
        p = get_params_block(self.strategy, 'advanced_atomic_params', {}) 
        if not get_param_value(p.get('enabled'), True): return states
        # 计算收盘价在当日振幅中的位置，值域[0, 1]
        price_range = (df['high_D'] - df['low_D']).replace(0, 1e-9)
        close_position_in_range = ((df['close_D'] - df['low_D']) / price_range).fillna(0.5)
        states['SCORE_PRICE_POSITION_IN_RANGE'] = close_position_in_range.astype(np.float32)
        # 高效计算连涨/连跌天数
        is_up_day = df['pct_change_D'] > 0
        is_down_day = df['pct_change_D'] < 0
        # 使用groupby和cumcount的经典向量化技巧计算连胜
        up_streak = (is_up_day.groupby((is_up_day != is_up_day.shift()).cumsum()).cumcount() + 1) * is_up_day
        down_streak = (is_down_day.groupby((is_down_day != is_down_day.shift()).cumsum()).cumcount() + 1) * is_down_day
        # 使用更节省内存的整数类型
        states['COUNT_CONSECUTIVE_UP_STREAK'] = up_streak.astype(np.int16)
        states['COUNT_CONSECUTIVE_DOWN_STREAK'] = down_streak.astype(np.int16)
        return states

    def _diagnose_board_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.3 · 前线换装版】诊断天地板/地天板模式
        - 核心升级: 废弃 `auction_conviction_index_D`，换装为 `final_hour_momentum_D`
                      作为更可靠的收盘意图确认因子。
        """
        states = {}
        p = get_params_block(self.strategy, 'board_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = get_param_value(p.get('price_buffer'), 0.005)
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        theoretical_max_range = (limit_up_price - limit_down_price).replace(0, np.nan)
        strength_score = (day_range / theoretical_max_range).clip(0, 1).fillna(0)
        low_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['low_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        close_near_limit_up_score = ((df['close_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        high_near_limit_up_score = ((df['high_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        close_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['close_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # 换装新式武器
        # 使用尾盘动能作为收盘意图的确认因子
        auction_bullish_confirmation = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(0), df.index, 55)
        auction_bearish_confirmation = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, 55)

        states['SCORE_BOARD_EARTH_HEAVEN'] = (strength_score * low_near_limit_down_score * close_near_limit_up_score * (1 + auction_bullish_confirmation)).clip(0, 1).astype(np.float32)
        states['SCORE_BOARD_HEAVEN_EARTH'] = (strength_score * high_near_limit_up_score * close_near_limit_down_score * (1 + auction_bearish_confirmation)).clip(0, 1).astype(np.float32)
        return states

    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, params: dict, day_quality_score: pd.Series) -> pd.Series:
        """
        【V8.0 · 三位一体审判版】上冲派发风险诊断引擎
        - 核心重构: 废弃旧的简单模型，引入“位置-行为-意图”三位一体审判模型。
        - 核心思想: 一次高确定性的上冲派发 = 高危的位置 + 经典的派发行为 + 确凿的主力派发意图。
        - 核心升级: 全面采用MTF归一化引擎，并从资金流/筹码层引入“主力意图”证据链。
        """
        # [代码修改开始]
        p = get_params_block(self.strategy, 'upthrust_distribution_params', {})
        signal_name = 'SCORE_RISK_UPTHRUST_DISTRIBUTION'
        default_series = pd.Series(0.0, index=df.index, name=signal_name, dtype=np.float32)
        if not get_param_value(p.get('enabled'), False):
            return default_series
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 支柱一: 高危位置 (The Setup) ---
        # 证据1.1: 过度拉伸
        overextension_score = self._get_mtf_normalized_score(df.get('BIAS_55_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据1.2: 撞上筹码墙
        resistance_proximity_score = self._get_mtf_normalized_score(df.get('realized_pressure_intensity_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        location_risk_score = (overextension_score * resistance_proximity_score).pow(0.5)
        # --- 支柱二: 派发行为 (The Action) ---
        # 证据2.1: 长上影线
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow_ratio = ((df['high_D'] - np.maximum(df['open_D'], df['close_D'])) / kline_range).fillna(0)
        upper_shadow_score = self._get_mtf_normalized_score(upper_shadow_ratio, ascending=True, tf_weights=default_weights)
        # 证据2.2: 收盘疲弱
        weak_close_score = 1.0 - self._get_mtf_normalized_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据2.3: 成交量放大
        volume_spike_score = self._get_mtf_normalized_score(df['volume_D'], ascending=True, tf_weights=default_weights)
        upthrust_behavior_score = (upper_shadow_score * weak_close_score * volume_spike_score).pow(1/3)
        # --- 支柱三: 主力意图 (The Motive) ---
        # 证据3.1: 主力在拉高派发
        main_force_dist_score = self._get_mtf_normalized_score(df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据3.2: 散户在追高接盘
        retail_fomo_score = self._get_mtf_normalized_score(df.get('retail_chasing_accumulation_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据3.3: 主力当天成功盈利卖出
        main_force_profit_score = self._get_mtf_normalized_score(df.get('main_force_intraday_profit_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        distribution_intent_score = (main_force_dist_score * retail_fomo_score * main_force_profit_score).pow(1/3)
        # --- 最终融合: 三位一体共振 ---
        # 只有当三个支柱同时成立时，风险才会被确认
        final_risk_score = (location_risk_score * upthrust_behavior_score * distribution_intent_score).pow(1/3)
        final_risk_score.name = signal_name
        return final_risk_score.clip(0, 1).astype(np.float32)
        # [代码修改结束]

    def _diagnose_vpa_stagnation_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】三维困境“VPA滞涨”风险诊断引擎
        - 核心思想: 真正的滞涨风险 = 无效的努力 + 剧烈的内耗 + 空头的反击。
        - 核心升级: 融合基础行情、高级行为、高级资金流三大数据层，构建三维立体诊断模型。
        """
        # [代码新增开始]
        states = {}
        signal_name = 'SCORE_RISK_VPA_STAGNATION'
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 支柱一: 无效努力 (Ineffective Effort) ---
        # 证据1.1: 成交量爆发
        vol_ratio_short = df['volume_D'] / df.get('VOL_MA_5_D', df['volume_D'])
        vol_ratio_long = df['volume_D'] / df.get('VOL_MA_21_D', df['volume_D'])
        volume_spike_score = (
            self._get_mtf_normalized_score(vol_ratio_short, ascending=True, tf_weights=default_weights) *
            self._get_mtf_normalized_score(vol_ratio_long, ascending=True, tf_weights=default_weights)
        ).pow(0.5)
        # 证据1.2: 高换手率
        turnover_score = self._get_mtf_normalized_score(df.get('turnover_rate_f_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        ineffective_effort_score = (volume_spike_score * turnover_score).pow(0.5)
        # --- 支柱二: 多头内耗 (Internal Friction) ---
        # 证据2.1: 日内趋势效率低下
        low_efficiency_score = 1.0 - self._get_mtf_normalized_score(df.get('intraday_trend_efficiency_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据2.2: 日内波动剧烈
        high_volatility_score = self._get_mtf_normalized_score(df.get('intraday_volatility_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据2.3: 日内成交量分布不均
        high_gini_score = self._get_mtf_normalized_score(df.get('intraday_volume_gini_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        internal_friction_score = (low_efficiency_score * high_volatility_score * high_gini_score).pow(1/3)
        # --- 支柱三: 空头反击 (Counter-Attack) ---
        # 证据3.1: 收盘疲弱
        weak_close_score = 1.0 - self._get_mtf_normalized_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据3.2: 主力与散户资金流背离 (主力卖, 散户买)
        flow_divergence = df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index))
        mf_sell_retail_buy_score = self._get_mtf_normalized_score(flow_divergence, ascending=False, tf_weights=default_weights) # 负值越大，分数越高
        # 证据3.3: 主力盈利派发
        mf_profit_score = self._get_mtf_normalized_score(df.get('main_force_intraday_profit_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        counter_attack_score = (weak_close_score * mf_sell_retail_buy_score * mf_profit_score).pow(1/3)
        # --- 最终融合: 三维困境共振 ---
        final_risk_score = (ineffective_effort_score * internal_friction_score * counter_attack_score).pow(1/3)
        states[signal_name] = final_risk_score.clip(0, 1).astype(np.float32)
        return states

    def _diagnose_atomic_bottom_formation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 · A股底部四部曲重构版】原子级“底部形态”诊断引擎
        - 核心思想: 一个值得信赖的A股底部，是一个从“绝望中孕育希望，于无声处听惊雷”的完整叙事。
        - 核心升级: 引入“绝望舞台”、“卖盘枯竭”、“新王加冕”、“惊雷闪现”四幕剧模型，
                      融合价格、情绪、量能、筹码、资金流、日内行为等多维度数据，直指A股底部本质。
        """
        # [代码修改开始]
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        long_term_weights = get_param_value(p_mtf.get('volatility_weights'), {21: 0.5, 55: 0.3, 89: 0.2}) # 借用波动率权重作为长期权重
        # --- 第一幕: 绝望的舞台 (The Stage of Despair) ---
        # 证据1.1: 绝对低位
        price_pos_yearly = self._get_mtf_normalized_score(df['close_D'], ascending=True, tf_weights={250: 1.0})
        absolute_low_score = 1.0 - price_pos_yearly
        # 证据1.2: 恐慌乖离
        panic_bias_score = self._get_mtf_normalized_score(df.get('BIAS_55_D', pd.Series(0.0, index=df.index)), ascending=False, tf_weights=default_weights) # 负值越大分数越高
        # 证据1.3: 均值回归倾向
        mean_reversion_score = self._get_mtf_normalized_score(df.get('hurst_120d_D', pd.Series(0.5, index=df.index)), ascending=False, tf_weights={120: 1.0}) # Hurst < 0.5 分数高
        stage_of_despair_score = (absolute_low_score * panic_bias_score * mean_reversion_score).pow(1/3)
        # --- 第二幕: 卖盘的枯竭 (The Exhaustion of Sellers) ---
        # 证据2.1: 波动压缩
        vol_compression_score = self._get_mtf_normalized_score(df.get('BBW_21_2.0_D', pd.Series(1.0, index=df.index)), ascending=False, tf_weights=long_term_weights)
        # 证据2.2: 市场地量
        volume_apathy_score = self._get_mtf_normalized_score(df.get('turnover_rate_f_D', pd.Series(10.0, index=df.index)), ascending=False, tf_weights=long_term_weights)
        # 证据2.3: 缩量下跌行为
        weakening_drop_score = self.strategy.atomic_states.get('SCORE_VOL_WEAKENING_DROP', pd.Series(0.0, index=df.index))
        seller_exhaustion_score = (vol_compression_score * volume_apathy_score * weakening_drop_score).pow(1/3)
        # --- 第三幕: 新王的加冕 (The Coronation of a New King) ---
        # 证据3.1: 主力与散户的权力交接
        power_transfer_score = self._get_mtf_normalized_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights) # 正值越大分数越高
        # 证据3.2: 筹码快速集中
        chip_concentration_slope = df.get('SLOPE_5_concentration_90pct_D', pd.Series(0.0, index=df.index))
        chip_gathering_score = self._get_mtf_normalized_score(-chip_concentration_slope, ascending=True, tf_weights=default_weights) # 负斜率越大分数越高
        # 证据3.3: 主力打压吸筹
        suppressive_accumulation_score = self._get_mtf_normalized_score(df.get('main_force_suppressive_accumulation_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        new_king_score = (power_transfer_score * chip_gathering_score * suppressive_accumulation_score).pow(1/3)
        # --- 第四幕: 惊雷的闪现 (The First Spark of Thunder) ---
        # 证据4.1: 强劲收盘
        strong_close_score = self._get_mtf_normalized_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据4.2: 尾盘偷袭
        auction_power_score = self._get_mtf_normalized_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(lower=0), ascending=True, tf_weights=default_weights)
        # 证据4.3: 聪明钱协同
        smart_money_score = self._get_mtf_normalized_score(df.get('SMART_MONEY_SYNERGY_BUY_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        first_spark_score = (strong_close_score * auction_power_score * smart_money_score).pow(1/3)
        # --- 最终融合: 四幕剧共振 ---
        snapshot_score = (
            stage_of_despair_score *
            seller_exhaustion_score *
            new_king_score *
            first_spark_score
        ).pow(1/4).astype(np.float32)
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_BOTTOM_FORMATION'
        )

    def _diagnose_atomic_rebound_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.0 · 凤凰涅槃重构版】原子级“史诗探底回升”诊断引擎
        - 核心思想: 一次值得捕捉的V型反转，是一场“凤凰涅槃”的史诗，必须经历“绝境深渊”、“绝地反击”、“王权交替”和“涅槃之火”四个完整阶段。
        - 核心升级: 引入全新的四维诊断模型，融合多达12个来自不同数据层的核心指标，对恐慌性抛售后的V型反转进行最全面、最深刻的诊断。
        """
        # [代码修改开始]
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        
        # --- 第一维度: 绝境深渊 (The Great Fall) - 确认恐慌 ---
        # 证据1.1: 深度 - 短期巨大回撤
        rolling_peak_21d = df['high_D'].rolling(window=21, min_periods=10).max()
        drawdown_from_peak = (rolling_peak_21d - df['close_D']) / rolling_peak_21d.replace(0, np.nan)
        depth_score = self._get_mtf_normalized_score(drawdown_from_peak.clip(lower=0), ascending=True, tf_weights=default_weights)
        # 证据1.2: 速度 - 短期下跌速率
        price_roc_5d = df['close_D'].pct_change(5)
        velocity_score = self._get_mtf_normalized_score(price_roc_5d, ascending=False, tf_weights=default_weights) # 负值越大分数越高
        # 证据1.3: 持续性 - 连跌天数
        down_streak_score = self._get_mtf_normalized_score(df.get('COUNT_CONSECUTIVE_DOWN_STREAK', pd.Series(0, index=df.index)), ascending=True, tf_weights=default_weights)
        great_fall_score = (depth_score * velocity_score * down_streak_score).pow(1/3)

        # --- 第二维度: 绝地反击 (The Last Stand) - 确认支撑有效性 ---
        # 证据2.1: 长下影线
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        lower_shadow_ratio = ((np.minimum(df['open_D'], df['close_D']) - df['low_D']) / kline_range).fillna(0)
        lower_shadow_score = self._get_mtf_normalized_score(lower_shadow_ratio, ascending=True, tf_weights=default_weights)
        # 证据2.2: 强劲收盘
        closing_strength_score = self._get_mtf_normalized_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据2.3: 踩在真实支撑上
        realized_support_score = self._get_mtf_normalized_score(df.get('realized_support_intensity_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        last_stand_score = (lower_shadow_score * closing_strength_score * realized_support_score).pow(1/3)

        # --- 第三维度: 王权交替 (The New King's Decree) - 确认主力易主 ---
        # 证据3.1: 主力与散户的资金背离
        power_transfer_score = self._get_mtf_normalized_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据3.2: 主力打压吸筹
        suppressive_accumulation_score = self._get_mtf_normalized_score(df.get('main_force_suppressive_accumulation_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据3.3: 聪明钱协同买入
        smart_money_score = self._get_mtf_normalized_score(df.get('SMART_MONEY_SYNERGY_BUY_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        new_king_score = (power_transfer_score * suppressive_accumulation_score * smart_money_score).pow(1/3)

        # --- 第四维度: 涅槃之火 (The Phoenix Fire) - 确认反转质量 ---
        # 证据4.1: 高效的日内趋势
        efficiency_score = self._get_mtf_normalized_score(df.get('intraday_trend_efficiency_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据4.2: 强劲的尾盘动能
        auction_power_score = self._get_mtf_normalized_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(lower=0), ascending=True, tf_weights=default_weights)
        # 证据4.3: 多头主导日内定价权
        vwap_dominance_score = self._get_mtf_normalized_score(df.get('close_vs_vwap_ratio_D', pd.Series(1.0, index=df.index)), ascending=True, tf_weights=default_weights)
        phoenix_fire_score = (efficiency_score * auction_power_score * vwap_dominance_score).pow(1/3)

        # --- 最终融合: 四维共振 ---
        snapshot_score = (
            great_fall_score *
            last_stand_score *
            new_king_score *
            phoenix_fire_score
        ).pow(1/4).astype(np.float32)

        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_REBOUND_REVERSAL'
        )

    def _diagnose_atomic_continuation_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.1 · 趋势健康度接口升级版】原子级“延续性反转”诊断引擎
        - 核心思想: 一次值得加仓的趋势中继反转，如同一次完美的“空中加油”，必须经历“王牌飞行员”、“平稳对接”和“引擎再点火”三个阶段。
        - 核心升级: 引入全新的三步确认法，融合趋势健康度、龙头地位、主力成本、洗盘行为、聪明钱交易等A股核心特性指标，精准捕捉牛市回调买点。
        """
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        long_term_weights = get_param_value(p_mtf.get('volatility_weights'), {21: 0.5, 55: 0.3, 89: 0.2})
        # --- 第一步: 王牌飞行员 (The Ace Pilot) - 确认趋势王者地位 ---
        # [代码修改开始]
        # 证据1.1: 趋势本身足够健康 (调用全新的权威引擎)
        trend_health_score = self.strategy.atomic_states.get('SCORE_TREND_HEALTH', pd.Series(0.0, index=df.index))
        # [代码修改结束]
        # 证据1.2: 标的是行业龙头
        leader_score = self._get_mtf_normalized_score(df.get('industry_leader_score_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据1.3: 主力拥有成本优势，掌控全局
        cost_advantage_score = self._get_mtf_normalized_score(df.get('cost_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        ace_pilot_score = (trend_health_score * leader_score * cost_advantage_score).pow(1/3)
        # --- 第二步: 平稳对接 (The Stable Docking) - 确认健康回调性质 ---
        # 证据2.1: 波动率与换手率双收缩
        bbw_contraction = self._get_mtf_normalized_score(df.get('BBW_21_2.0_D', pd.Series(1.0, index=df.index)), ascending=False, tf_weights=long_term_weights)
        turnover_contraction = self._get_mtf_normalized_score(df.get('turnover_rate_f_D', pd.Series(10.0, index=df.index)), ascending=False, tf_weights=long_term_weights)
        contraction_score = (bbw_contraction * turnover_contraction).pow(0.5)
        # 证据2.2: 回调由获利盘驱动，而非恐慌盘
        profit_taking_score = self._get_mtf_normalized_score(df.get('short_term_profit_taking_ratio_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据2.3: 主力在回调中并未大规模出逃
        mf_flow_5d = df.get('main_force_net_flow_consensus_sum_5d_D', pd.Series(0.0, index=df.index))
        mf_holding_score = self._get_mtf_normalized_score(mf_flow_5d, ascending=False, tf_weights=default_weights) # 负值越小（越接近0），分数越高
        stable_docking_score = (contraction_score * profit_taking_score * mf_holding_score).pow(1/3)
        # --- 第三步: 引擎再点火 (The Engine Reignites) - 确认王者归来 ---
        # 证据3.1: 聪明钱在反转日重新入场
        smart_money_score = self.strategy.atomic_states.get('SCORE_BEHAVIOR_SMART_INTRADAY_TRADING', pd.Series(0.0, index=df.index))
        # 证据3.2: 主力带着强烈的信念回归
        conviction_score = self._get_mtf_normalized_score(df.get('main_force_conviction_ratio_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据3.3: 上涨高效，阻力已被清洗
        vpa_efficiency_score = self._get_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        engine_reignites_score = (smart_money_score * conviction_score * vpa_efficiency_score).pow(1/3)
        # --- 最终融合: 三步确认共振 ---
        snapshot_score = (
            ace_pilot_score *
            stable_docking_score *
            engine_reignites_score
        ).pow(1/3).astype(np.float32)
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_CONTINUATION_REVERSAL'
        )

    def _diagnose_smart_intraday_trading(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 大师三重奏重构版】“日内聪明钱”诊断引擎
        - 核心思想: 真正的“聪明钱”交易，是一场由“操盘技艺”、“攻击意愿”和“市场控制力”共同谱写的“大师三重奏”。
        - 核心升级: 从描述“现象”升维至刻画“操盘手画像”，融合了日内行为、资金流向、主力信念等A股核心博弈指标，
                      旨在识别由专业机构主导的、具有高度持续性潜力的上涨行为。
        """
        # [代码修改开始]
        states = {}
        signal_name = 'SCORE_BEHAVIOR_SMART_INTRADAY_TRADING'
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 第一章: 操盘技艺 (The Craftsmanship) - 衡量交易的“精巧度” ---
        # 证据1.1: 高执行Alpha (低买高卖)
        execution_alpha_score = self._get_mtf_normalized_score(df.get('intraday_execution_alpha_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据1.2: 高量价效率 (四两拨千斤)
        vpa_efficiency_score = self._get_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据1.3: 低成交量基尼系数 (有计划、非脉冲式买入)
        volume_discipline_score = 1.0 - self._get_mtf_normalized_score(df.get('intraday_volume_gini_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        craftsmanship_score = (execution_alpha_score * vpa_efficiency_score * volume_discipline_score).pow(1/3)
        # --- 第二章: 攻击意愿 (The Will to Attack) - 衡量意图的“坚决度” ---
        # 证据2.1: 强劲收盘 (巩固战果)
        closing_strength_score = self._get_mtf_normalized_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据2.2: 强劲尾盘 (锁定胜局)
        auction_power_score = self._get_mtf_normalized_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(lower=0), ascending=True, tf_weights=default_weights)
        # 证据2.3: 午后强于午前 (持续发力)
        afternoon_power_score = self._get_mtf_normalized_score(df.get('am_pm_vwap_ratio_D', pd.Series(1.0, index=df.index)), ascending=True, tf_weights=default_weights)
        will_to_attack_score = (closing_strength_score * auction_power_score * afternoon_power_score).pow(1/3)
        # --- 第三章: 市场控制力 (The Market Control) - 衡量主导权的“掌控度” ---
        # 证据3.1: 主力信念坚定
        conviction_score = self._get_mtf_normalized_score(df.get('main_force_conviction_ratio_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据3.2: 主力买、散户卖的健康换手
        power_transfer_score = self._get_mtf_normalized_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)).clip(lower=0), ascending=True, tf_weights=default_weights)
        # 证据3.3: 机构协同作战
        synergy_score = self._get_mtf_normalized_score(df.get('SMART_MONEY_SYNERGY_BUY_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        market_control_score = (conviction_score * power_transfer_score * synergy_score).pow(1/3)
        # --- 最终融合: 大师三重奏共振 ---
        snapshot_score = (
            craftsmanship_score *
            will_to_attack_score *
            market_control_score
        ).pow(1/3)
        # 对快照分进行关系元分析，得到最终的动态信号
        final_signal_dict = self._perform_relational_meta_analysis(df=df, snapshot_score=snapshot_score, signal_name=signal_name)
        states.update(final_signal_dict)
        return states

    def _diagnose_structural_fault_breakthrough(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 三峡大坝泄洪模型重构版】“结构性断层突破”诊断引擎
        - 核心思想: 一次完美的断层突破，如同三峡大坝泄洪，必须经历“蓄水成渊”、“开闸泄洪”和“奔流不息”三个完整阶段。
        - 核心升级: 融合了突破前的能量积蓄、突破时的爆发力、突破后的力量格局三大维度，深刻洞察A股筹码博弈的本质。
        """
        states = {}
        signal_name = 'SCORE_STRUCTURAL_FAULT_BREAKTHROUGH'
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})

        # --- 第一章: 蓄水成渊 (The Reservoir) - 确认突破前的能量积蓄 ---
        # 证据1.1: 突破前经历了充分的收缩盘整 (弹簧已压紧)
        consolidation_score = self.strategy.atomic_states.get('SCORE_BEHAVIOR_CONTRACTION_CONSOLIDATION', pd.Series(0.0, index=df.index))
        # 证据1.2: 前方是广阔的筹码真空区 (泄洪区无障碍)
        vacuum_score = self._get_mtf_normalized_score(df.get('chip_fault_vacuum_percent_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据1.3: 股价处于筹码主峰下方，蓄势待发
        proximity_score = self._get_mtf_normalized_score(df.get('peak_distance_ratio_D', pd.Series(1.0, index=df.index)), ascending=False, tf_weights=default_weights) # 距离越小分数越高
        reservoir_score = (consolidation_score * vacuum_score * proximity_score).pow(1/3)

        # --- 第二章: 开闸泄洪 (The Floodgates Open) - 确认突破瞬间的爆发力 ---
        # 证据2.1: 突破动作本身强劲有力
        intensity_score = self._get_mtf_normalized_score(df.get('fault_breakthrough_intensity_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据2.2: 成交量巨幅放出 (水量充沛)
        volume_ratio = df.get('volume_D', pd.Series(0.0, index=df.index)) / df.get('VOL_MA_21_D', pd.Series(1.0, index=df.index))
        power_score = self._get_mtf_normalized_score(volume_ratio, ascending=True, tf_weights=default_weights)
        # 证据2.3: 收盘强势，巩固战果
        quality_score = self._get_mtf_normalized_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), ascending=True, tf_weights=default_weights)
        breakthrough_act_score = (intensity_score * power_score * quality_score).pow(1/3)

        # --- 第三章: 奔流不息 (The Raging Torrent) - 确认主导力量的意图与控制力 ---
        # 证据3.1: 主力资金压倒性净流入 (总工程师亲自开闸)
        intent_score = self._get_mtf_normalized_score(df.get('main_force_net_flow_consensus_D', pd.Series(0.0, index=df.index)).clip(lower=0), ascending=True, tf_weights=default_weights)
        # 证据3.2: 主力买入信念坚定
        conviction_score = self._get_mtf_normalized_score(df.get('main_force_conviction_ratio_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        # 证据3.3: 散户在突破中被清洗出局
        capitulation_score = self._get_mtf_normalized_score(df.get('retail_capitulation_score_D', pd.Series(0.0, index=df.index)), ascending=True, tf_weights=default_weights)
        confirmation_score = (intent_score * conviction_score * capitulation_score).pow(1/3)

        # --- 最终融合: 三峡大坝泄洪模型共振 ---
        snapshot_score = (
            reservoir_score *
            breakthrough_act_score *
            confirmation_score
        ).pow(1/3)
        
        # 对快照分进行关系元分析，得到最终的动态信号
        final_signal_dict = self._perform_relational_meta_analysis(df=df, snapshot_score=snapshot_score, signal_name=signal_name)
        states.update(final_signal_dict)
        return states

    def _diagnose_upper_shadow_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 上影线法庭重构版】上影线意图诊断引擎
        - 核心思想: 将上影线视为一份战报，通过“立案侦查”、“控方陈述(派发)”、“辩方陈述(洗盘)”和“最终裁决”的法庭审判流程，深刻洞察其背后的多空博弈本质。
        - 核心升级: 引入三维质证模型，融合多达6个来自资金流和行为层的核心正反方证据，对上影线意图进行交叉验证和最终裁定。
        """
        # [代码修改开始]
        states = {}
        signal_name = 'SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS'
        p_parent = get_params_block(self.strategy, 'kline_pattern_params', {})
        p = get_params_block(p_parent, 'upper_shadow_intent_params', {})
        if not get_param_value(p.get('enabled'), True):
            states[signal_name] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states
        
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})

        # --- 第一步: 立案侦查 (Filing the Case) - 确认“大案要案” ---
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        upper_shadow_ratio = (upper_shadow / kline_range).fillna(0)
        volume_ratio = df.get('volume_D', 0) / df.get('VOL_MA_21_D', 1)
        
        is_significant_shadow = upper_shadow_ratio > get_param_value(p.get('min_upper_shadow_ratio'), 0.5)
        is_significant_volume = volume_ratio > get_param_value(p.get('min_volume_ratio'), 1.5)
        trigger_mask = is_significant_shadow & is_significant_volume

        # --- 第二步: 控方陈述 (The Prosecution) - 搜集“派发”证据 ---
        # 证据A (动机): 获利了结紧迫
        profit_taking_score = self._get_mtf_normalized_score(df.get('profit_taking_urgency_D', 0), ascending=True, tf_weights=default_weights)
        # 证据B (手法): 主力拉高派发
        rally_dist_score = self._get_mtf_normalized_score(df.get('main_force_rally_distribution_D', 0), ascending=True, tf_weights=default_weights)
        # 证据C (协同): 散户追高接盘
        retail_fomo_score = self._get_mtf_normalized_score(df.get('retail_chasing_accumulation_D', 0), ascending=True, tf_weights=default_weights)
        distribution_score = (profit_taking_score * rally_dist_score * retail_fomo_score).pow(1/3)

        # --- 第三步: 辩方陈述 (The Defense) - 搜集“洗盘”证据 ---
        # 证据A (核心反证): 主力净流入，吸收抛压
        power_transfer_score = self._get_mtf_normalized_score(df.get('flow_divergence_mf_vs_retail_D', 0).clip(lower=0), ascending=True, tf_weights=default_weights)
        # 证据B (信念代价): 主力不惜成本
        cost_insensitivity_score = self._get_mtf_normalized_score(-df.get('main_force_intraday_profit_D', 0), ascending=True, tf_weights=default_weights)
        # 证据C (战后控制): 收盘价依然强势
        closing_control_score = self._get_mtf_normalized_score(df.get('closing_strength_index_D', 0.5), ascending=True, tf_weights=default_weights)
        absorption_score = (power_transfer_score * cost_insensitivity_score * closing_control_score).pow(1/3)

        # --- 第四步: 最终裁决 (The Verdict) ---
        # 最终意图分 = 洗盘可能性 - 派发嫌疑
        final_intent_score = (absorption_score - distribution_score).clip(-1, 1)
        
        # 只有在“大案要案”成立时，才输出裁决结果
        states[signal_name] = (final_intent_score * trigger_mask).astype(np.float32)
        return states

    def _calculate_signal_dynamics(self, series: pd.Series, params: dict) -> Dict[str, pd.Series]:
        """
        【V2.0 · 动力学重构版】信号动态分析引擎
        - 核心思想: 任何信号都具备物理学属性。此引擎为输入的任何信号序列，提供其在“状态(State)”、“速度(Velocity)”和“加速度(Acceleration)”三个维度上的全息画像。
        - 核心升级:
          1. 废弃了旧的 `_calculate_holographic_divergence_behavior`。
          2. 速度(Velocity): 采用多时间框架(MTF)的线性回归斜率加权融合，替代简单的diff，结果更平滑、可靠。
          3. 加速度(Acceleration): 采用对“速度”的一阶差分(diff(1))，更符合物理学定义，精准捕捉动能变化。
        - 返回: 一个包含 'state', 'velocity', 'acceleration' 三个双极性分数的字典。
        """
        # [代码新增开始]
        p_mtf = get_param_value(params.get('mtf_normalization_params'), {})
        velocity_weights = get_param_value(p_mtf.get('velocity_weights'), {3: 0.4, 5: 0.3, 8: 0.2, 13: 0.1})
        norm_window = get_param_value(params.get('relational_meta_analysis_params', {}).get('norm_window'), 55)
        
        # 维度一: 状态 (State) - 信号的当前位置
        # 使用较长周期归一化，以获得稳定的状态评估
        state_score = normalize_to_bipolar(series, series.index, norm_window)
        
        # 维度二: 速度 (Velocity) - 信号的变化速率 (MTF斜率融合)
        holographic_velocity = pd.Series(0.0, index=series.index, dtype=np.float32)
        total_weight = sum(velocity_weights.values())
        if total_weight > 0:
            for period, weight in velocity_weights.items():
                # 使用pandas_ta计算线性回归斜率，更稳健
                slope_series = series.ta.slope(length=period).fillna(0)
                # 对每个周期的斜率进行归一化
                normalized_slope = normalize_to_bipolar(slope_series, series.index, norm_window)
                holographic_velocity += normalized_slope * (weight / total_weight)
        
        # 维度三: 加速度 (Acceleration) - 速度的变化率
        # 加速度是速度的一阶导数
        acceleration_raw = holographic_velocity.diff(1).fillna(0)
        holographic_acceleration = normalize_to_bipolar(acceleration_raw, series.index, norm_window)
        
        return {
            'state': state_score.clip(-1, 1),
            'velocity': holographic_velocity.clip(-1, 1),
            'acceleration': holographic_acceleration.clip(-1, 1)
        }

    def _diagnose_grand_divergence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 背离三位一体版】终极背离诊断引擎
        - 核心思想: A股的背离，是“价格的谎言”与“力量的真相”之间的博弈背离。
        - 核心升级: 废弃任何传统指标，构建“价格动能”与“A股核心力量动能”的直接对抗模型，
                      从博弈论层面量化背离的真实强度。
        """
        states = {}
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        # 使用较短期的权重来捕捉动能变化
        momentum_weights = get_param_value(p_mtf.get('momentum_weights'), {3: 0.5, 5: 0.3, 8: 0.2})

        # --- 第一章: 价格的谎言 (The Lie of Price) - 计算全息价格动能 ---
        # 证据1.1: 速度 - 12日价格变化率
        price_velocity = df.get('ROC_12_D', pd.Series(0.0, index=df.index))
        price_velocity_score = self._get_mtf_normalized_bipolar_score(price_velocity, tf_weights=momentum_weights)
        
        # 证据1.2: 位置 - 年度价格位置
        price_pos_yearly = normalize_score(df['close_D'], df.index, 250, ascending=True)
        price_position_score = (price_pos_yearly * 2 - 1).clip(-1, 1) # 转换为[-1, 1]
        
        # 融合得到全息价格动能
        holographic_price_momentum = (price_velocity_score * 0.6 + price_position_score * 0.4).clip(-1, 1)

        # --- 第二章: 力量的真相 (The Truth of Power) - 计算A股核心力量动能 ---
        # 证据2.1: 主力资金动能
        mf_flow_momentum = df.get('SLOPE_5_main_force_net_flow_consensus_D', pd.Series(0.0, index=df.index))
        mf_flow_momentum_score = self._get_mtf_normalized_bipolar_score(mf_flow_momentum, tf_weights=momentum_weights)
        
        # 证据2.2: 筹码结构动能 (斜率为负是好事，所以要反转)
        chip_structure_momentum = -df.get('SLOPE_5_concentration_90pct_D', pd.Series(0.0, index=df.index))
        chip_structure_momentum_score = self._get_mtf_normalized_bipolar_score(chip_structure_momentum, tf_weights=momentum_weights)
        
        # 证据2.3: 主力信念动能
        mf_conviction_momentum = df.get('SLOPE_5_main_force_conviction_ratio_D', pd.Series(0.0, index=df.index))
        mf_conviction_momentum_score = self._get_mtf_normalized_bipolar_score(mf_conviction_momentum, tf_weights=momentum_weights)

        # 融合得到全息力量动能 (几何平均)
        # 使用 (score + 1) / 2 转换到 [0, 1] 进行几何平均，再转换回 [-1, 1]
        power_bull = (
            ((mf_flow_momentum_score + 1) / 2) *
            ((chip_structure_momentum_score + 1) / 2) *
            ((mf_conviction_momentum_score + 1) / 2)
        ).pow(1/3)
        holographic_power_momentum = (power_bull * 2 - 1).clip(-1, 1)

        # --- 第三章: 博弈的裁决 (The Verdict of the Game) ---
        # 核心公式: 力量动能 - 价格动能
        divergence_score = (holographic_power_momentum - holographic_price_momentum).clip(-1, 1)

        # 分离为看涨背离和看跌背离信号
        states['SCORE_DIVERGENCE_BULLISH'] = divergence_score.clip(lower=0).astype(np.float32)
        states['SCORE_DIVERGENCE_BEARISH'] = (divergence_score.clip(upper=0) * -1).astype(np.float32)
        
        # 也可以提供一个总的背离分数
        states['BIPOLAR_DIVERGENCE_SCORE'] = divergence_score.astype(np.float32)

        # 在原子信号中心调用此方法
        # atomic_signals.update(self._diagnose_grand_divergence(df))
        return states

    def _calculate_trend_health_score(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · 四维一体模型重构版】权威趋势健康度评估引擎
        - 核心思想: 一个健康的A股趋势，是“形态”、“动力”、“博弈”和“气候”四维一体的共振。
        - 核心升级:
          1. 正式取代旧的 _calculate_trend_health_score 和 _diagnose_trend_context。
          2. 骨架(形态): 沿用并优化均线排列和价格位置的评估。
          3. 引擎(动力): 引入主力资金共识、主力信念和资金流背离，评估趋势的“含金量”。
          4. 战场(博弈): 引入主力成本优势和筹码集中趋势，评估趋势的“控盘度”。
          5. 气候(天时): 引入行业地位和板块热度，评估趋势的“顺风度”。
        - 返回: 一个在[0, 1]区间的、能深刻反映A股趋势健康度的单极性分数。
        """
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 维度一: 骨架 (Structure) - 趋势形态 ---
        ema_periods = [5, 13, 21, 55, 89]
        ema_cols = [f'EMA_{p}_D' for p in ema_periods]
        if not all(col in df.columns for col in ema_cols):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        ma_values = np.stack([df[col].values for col in ema_cols], axis=0)
        alignment_score = np.mean(ma_values[:-1] > ma_values[1:], axis=0)
        location_score = (df['close_D'] > df['EMA_55_D']).astype(float)
        structure_score = pd.Series((alignment_score * 0.7 + location_score * 0.3), index=df.index)
        # --- 维度二: 引擎 (Engine) - 核心动力 ---
        mf_flow_score = self._get_mtf_normalized_score(df.get('main_force_net_flow_consensus_sum_5d_D', 0), ascending=True, tf_weights=default_weights)
        mf_conviction_score = self._get_mtf_normalized_score(df.get('main_force_conviction_ratio_D', 0), ascending=True, tf_weights=default_weights)
        power_transfer_score = self._get_mtf_normalized_score(df.get('flow_divergence_mf_vs_retail_D', 0).clip(lower=0), ascending=True, tf_weights=default_weights)
        engine_score = (mf_flow_score * mf_conviction_score * power_transfer_score).pow(1/3)
        # --- 维度三: 战场 (Battlefield) - 博弈环境 ---
        cost_advantage_score = self._get_mtf_normalized_score(df.get('cost_divergence_mf_vs_retail_D', 0), ascending=True, tf_weights=default_weights)
        chip_gathering_score = self._get_mtf_normalized_score(-df.get('SLOPE_5_concentration_90pct_D', 0), ascending=True, tf_weights=default_weights)
        battlefield_score = (cost_advantage_score * chip_gathering_score).pow(0.5)
        # --- 维度四: 气候 (Climate) - 板块天时 ---
        industry_rank = df.get('industry_strength_rank_D', 100) # 假设总共100名
        industry_rank_score = 1.0 - (industry_rank / 100.0).clip(0, 1)
        leader_score = self._get_mtf_normalized_score(df.get('industry_leader_score_D', 0), ascending=True, tf_weights=default_weights)
        theme_score = self._get_mtf_normalized_score(df.get('THEME_HOTNESS_SCORE_D', 0), ascending=True, tf_weights=default_weights)
        climate_score = (industry_rank_score * leader_score * theme_score).pow(1/3)
        # --- 最终融合: 四维一体加权 (A股特性：动力和博弈更重要) ---
        final_health_score = (
            structure_score**0.20 *
            engine_score**0.35 *
            battlefield_score**0.30 *
            climate_score**0.15
        )
        return final_health_score.clip(0, 1).astype(np.float32)

    def _calculate_despair_context_score(self, df: pd.DataFrame) -> pd.Series:
        """
        【V4.0 · 冥河三途模型重构版】权威市场绝望指数引擎
        - 核心思想: 真正的市场绝望，是“深度”、“速度”和“投降”三个维度的共振。
        - 核心升级:
          1. 正式废弃旧的实现，成为一个权威的、可供全局调用的基础环境指标。
          2. 深度(Depth): 融合年度回撤与中期乖离，衡量市场的“痛苦指数”。
          3. 速度(Velocity): 融合中期下跌速率与连跌天数，衡量市场的“恐慌指数”。
          4. 投降(Capitulation): 融合流动性枯竭、收盘绝望和散户抛售，衡量市场的“投降指数”，直指A股博弈本质。
        - 返回: 一个在[0, 1]区间的、代表市场绝望程度的单极性分数，分数越高越绝望。
        """
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 维度一: 深度 (Depth) ---
        # 证据1.1: 从年度高点的回撤幅度
        rolling_peak_250d = df['high_D'].rolling(window=250, min_periods=120).max()
        drawdown_from_peak = (rolling_peak_250d - df['close_D']) / rolling_peak_250d.replace(0, np.nan)
        depth_score = self._get_mtf_normalized_score(drawdown_from_peak.clip(lower=0), ascending=True, tf_weights={250: 1.0})
        # 证据1.2: 向下偏离中期均线的程度
        panic_bias_score = self._get_mtf_normalized_score(df.get('BIAS_55_D', 0), ascending=False, tf_weights=default_weights) # 负值越大分数越高
        depth_dimension_score = (depth_score * panic_bias_score).pow(0.5)
        # --- 维度二: 速度 (Velocity) ---
        # 证据2.1: 中期下跌速率
        price_roc_21d = df['close_D'].pct_change(21)
        velocity_score = self._get_mtf_normalized_score(price_roc_21d, ascending=False, tf_weights=default_weights) # 负值越大分数越高
        # 证据2.2: 连跌天数
        down_streak_score = self._get_mtf_normalized_score(df.get('COUNT_CONSECUTIVE_DOWN_STREAK', 0), ascending=True, tf_weights=default_weights)
        velocity_dimension_score = (velocity_score * down_streak_score).pow(0.5)
        # --- 维度三: 投降 (Capitulation) ---
        # 证据3.1: 流动性枯竭风险 (恐慌性抛售导致买盘真空)
        liquidity_drain_score = self.strategy.atomic_states.get('SCORE_RISK_LIQUIDITY_DRAIN', pd.Series(0.0, index=df.index))
        # 证据3.2: 收盘绝望 (全天无抵抗)
        closing_weakness_score = 1.0 - self._get_mtf_normalized_score(df.get('closing_strength_index_D', 0.5), ascending=True, tf_weights=default_weights)
        # 证据3.3: 散户投降式抛售
        retail_capitulation_score = self._get_mtf_normalized_score(df.get('retail_capitulation_score_D', 0), ascending=True, tf_weights=default_weights)
        capitulation_dimension_score = (liquidity_drain_score * closing_weakness_score * retail_capitulation_score).pow(1/3)
        # --- 最终融合: 冥河三途共振 ---
        final_despair_score = (
            depth_dimension_score *
            velocity_dimension_score *
            capitulation_dimension_score
        ).pow(1/3)
        return final_despair_score.clip(0, 1).astype(np.float32)

    def _perform_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, signal_name: str) -> Dict[str, pd.Series]:
        """
        【V5.0 · 火种-薪柴-天候模型重构版】信号嬗变引擎
        - 核心思想: 一个原子信号(火种)，能否演变成燎原大火，取决于其自身动能(薪柴)和市场环境(天候)的共振。
        - 核心升级:
          1. 废弃了在真空中分析信号的封闭逻辑。
          2. 薪柴(Tinder): 调用权威的 `_calculate_signal_dynamics` 引擎，评估信号自身的状态、速度和加速度。
          3. 天候(Climate): 引入 `SCORE_TREND_HEALTH` (风向) 和 `SCORE_CONTEXT_DESPAIR` (湿度) 作为市场环境调节器。
          4. 融合逻辑: 最终信号 = 火种 * 薪柴 * 天候。任何一个维度的缺失都将导致信号降级，完美体现共振效应。
        - 返回: 一个经过市场环境深度调节的、代表信号“成功概率”的最终分数。
        """
        # [代码修改开始]
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.4)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.3)

        # --- 第一章: 火种 (The Spark) - 信号快照分 ---
        # snapshot_score 本身就是火种的强度，范围 [0, 1]
        spark_score = snapshot_score.clip(0, 1)

        # --- 第二章: 薪柴 (The Tinder) - 信号自身动能 ---
        # 调用权威的动态分析引擎
        signal_dynamics = self._calculate_signal_dynamics(snapshot_score, p_conf)
        # 我们只关心正向的动能，因为我们在评估一个看涨信号的强度
        # 将双极性分数转换为 [0, 1] 的单极性助燃分数
        state_fuel = signal_dynamics['state'].clip(0, 1)
        velocity_fuel = signal_dynamics['velocity'].clip(0, 1)
        acceleration_fuel = signal_dynamics['acceleration'].clip(0, 1)
        # 加权融合得到“薪柴”的综合助燃能力
        tinder_score = (
            state_fuel * w_state +
            velocity_fuel * w_velocity +
            acceleration_fuel * w_acceleration
        ).clip(0, 1)

        # --- 第三章: 天候 (The Climate) - 市场综合环境 ---
        # 证据3.1: 风向 - 趋势是否健康？
        trend_health = self.strategy.atomic_states.get('SCORE_TREND_HEALTH', pd.Series(0.5, index=df.index))
        # 证据3.2: 湿度 - 市场是否绝望？
        despair_level = self.strategy.atomic_states.get('SCORE_CONTEXT_DESPAIR', pd.Series(0.5, index=df.index))
        # 最佳天候 = 趋势健康(1) 且 市场不绝望(0) -> (1 * (1-0)) = 1
        # 最差天候 = 趋势破败(0) 且 市场极度绝望(1) -> (0 * (1-1)) = 0
        climate_modulator = (trend_health * (1.0 - despair_level)).clip(0, 1)

        # --- 最终融合: 三位一体共振 ---
        # 最终信号强度 = 火种强度 * 薪柴助燃能力 * 天候有利程度
        final_score = (spark_score * tinder_score * climate_modulator).clip(0, 1)
        
        states[signal_name] = final_score.astype(np.float32)
        return states
        # [代码修改结束]

    def _supreme_fusion_engine(self, df: pd.DataFrame, signals_to_fuse: Dict[str, pd.Series], params: Dict) -> pd.Series:
        """
        【V1.0 · 新增】最高神谕融合引擎 (宙斯之雷)
        - 核心职责: 对多个动态原子信号进行加权协同融合。
        """
        fusion_weights = get_param_value(params.get('fusion_weights'), {})
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.5)
        valid_signals = []
        weights = []
        for name, weight in fusion_weights.items():
            signal_name_full = f'SCORE_ATOMIC_{name}'
            if signal_name_full in signals_to_fuse and weight > 0:
                # 使用 .values 确保 numpy 操作的性能和对齐
                valid_signals.append(signals_to_fuse[signal_name_full].values)
                weights.append(weight)
        if not valid_signals:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 基础融合：加权几何平均 ---
        stacked_signals = np.stack(valid_signals, axis=0)
        weights_array = np.array(weights)
        # 归一化权重
        total_weight = weights_array.sum()
        if total_weight > 0:
            normalized_weights = weights_array / total_weight
        else:
            normalized_weights = np.full_like(weights_array, 1.0 / len(weights_array))
        # 为避免 log(0) 错误，给信号值增加一个极小量
        safe_signals = np.maximum(stacked_signals, 1e-9)
        log_signals = np.log(safe_signals)
        weighted_log_sum = np.sum(log_signals * normalized_weights[:, np.newaxis], axis=0)
        base_fusion_score = np.exp(weighted_log_sum)
        # --- 协同奖励：当多个信号同时活跃时给予奖励 ---
        # 这里我们简化为取最强的两个信号的乘积作为协同基础
        if stacked_signals.shape[0] >= 2:
            # 沿信号轴排序，取最后两个（最大的）
            sorted_signals = np.sort(stacked_signals, axis=0)
            synergy_base = sorted_signals[-1] * sorted_signals[-2]
            synergy_bonus = (synergy_base**0.5) * synergy_bonus_factor
        else:
            synergy_bonus = 0.0
        # --- 最终融合 ---
        final_score = (base_fusion_score * (1 + synergy_bonus)).clip(0, 1)
        return pd.Series(final_score, index=df.index, dtype=np.float32)

    def _calculate_day_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · K线三维审判模型重构版】单日K线质量评估引擎
        - 核心思想: 一根K线的质量，由“战果的纯度”、“过程的效率”和“力量的归属”三维共振决定。
        - 核心升级:
          1. 战果纯度: 引入收盘强度指数，更精准评估最终战果。
          2. 过程效率: 引入日内趋势效率、VWAP控制权、量价效率，评估操盘水平。
          3. 力量归属: 引入主力资金、主力信念、资金流背离，直指K线背后的驱动者。
        - 返回值: 一个在[-1, 1]区间的双极性分数，深刻反映单日K线的综合看涨/看跌质量。
        """
        p_behavior = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_behavior.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 维度一: 战果的纯度 (Purity of the Outcome) ---
        gap_up = df['open_D'] > df['pre_close_D']
        body_up = df['close_D'] > df['open_D']
        is_bullish_trajectory = gap_up | body_up
        gap_down = df['open_D'] < df['pre_close_D']
        body_down = df['close_D'] < df['open_D']
        is_bearish_trajectory = gap_down | body_down
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        lower_shadow = (np.minimum(df['open_D'], df['close_D']) - df['low_D']).clip(lower=0)
        shadow_bullish_score = (lower_shadow / kline_range).fillna(0)
        shadow_bearish_score = (upper_shadow / kline_range).fillna(0)
        closing_strength_score = self._get_mtf_normalized_score(df.get('closing_strength_index_D', 0.5), ascending=True, tf_weights=default_weights)
        bullish_outcome_score = (is_bullish_trajectory.astype(float) * shadow_bullish_score * closing_strength_score).pow(1/3)
        bearish_outcome_score = (is_bearish_trajectory.astype(float) * shadow_bearish_score * (1.0 - closing_strength_score)).pow(1/3)
        # --- 维度二: 过程的效率 (Efficiency of the Process) ---
        trend_efficiency = self._get_mtf_normalized_score(df.get('intraday_trend_efficiency_D', 0.5), ascending=True, tf_weights=default_weights)
        vwap_dominance = self._get_mtf_normalized_score(df.get('close_vs_vwap_ratio_D', 1.0), ascending=True, tf_weights=default_weights)
        vpa_efficiency = self._get_mtf_normalized_score(df.get('VPA_EFFICIENCY_D', 0), ascending=True, tf_weights=default_weights)
        bullish_process_score = (trend_efficiency * vwap_dominance * vpa_efficiency).pow(1/3)
        bearish_process_score = ((1.0 - trend_efficiency) * (1.0 - vwap_dominance) * (1.0 - vpa_efficiency)).pow(1/3)
        # --- 维度三: 力量的归属 (Attribution of Power) ---
        mf_flow = df.get('main_force_net_flow_consensus_D', 0)
        mf_conviction = df.get('main_force_conviction_ratio_D', 0)
        flow_divergence = df.get('flow_divergence_mf_vs_retail_D', 0)
        bullish_power_score = (
            self._get_mtf_normalized_score(mf_flow.clip(lower=0), ascending=True, tf_weights=default_weights) *
            self._get_mtf_normalized_score(mf_conviction.clip(lower=0), ascending=True, tf_weights=default_weights) *
            self._get_mtf_normalized_score(flow_divergence.clip(lower=0), ascending=True, tf_weights=default_weights)
        ).pow(1/3)
        bearish_power_score = (
            self._get_mtf_normalized_score(mf_flow.clip(upper=0).abs(), ascending=True, tf_weights=default_weights) *
            self._get_mtf_normalized_score(mf_conviction.clip(upper=0).abs(), ascending=True, tf_weights=default_weights) *
            self._get_mtf_normalized_score(flow_divergence.clip(upper=0).abs(), ascending=True, tf_weights=default_weights)
        ).pow(1/3)
        # --- 最终融合: 三维审判 (A股特性: 力量 > 过程 > 战果) ---
        weights = {'outcome': 0.25, 'process': 0.35, 'power': 0.40}
        final_bullish_quality = (
            bullish_outcome_score**weights['outcome'] *
            bullish_process_score**weights['process'] *
            bullish_power_score**weights['power']
        )
        final_bearish_quality = (
            bearish_outcome_score**weights['outcome'] *
            bearish_process_score**weights['process'] *
            bearish_power_score**weights['power']
        )
        day_quality_score = (final_bullish_quality - final_bearish_quality).clip(-1, 1)
        return day_quality_score.astype(np.float32)

    def _resolve_pressure_absorption_dynamics(self, provisional_pressure: pd.Series, intent_diagnosis: pd.Series) -> Dict[str, pd.Series]:
        """
        【V2.0 · 压力-承接连续谱模型重构版】压力与承接动态解析引擎
        - 核心思想: 压力与承接如阴阳两面，相互转化。机会的含金量，正比于其所化解的压力大小。
        - 核心升级:
          1. 废弃旧的二元阈值判断，引入“承接控制因子”的连续谱模型。
          2. 最终风险 = 原始压力 * (1 - 承接控制因子)，量化未被化解的风险。
          3. 最终机会 = 原始压力 * 承接控制因子，量化“弹簧效应”，体现压力被吸收后所蕴含的动能。
        - 产出: 生成更具博弈内涵的 `SCORE_RISK_UNRESOLVED_PRESSURE` 和 `SCORE_OPPORTUNITY_PRESSURE_ABSORPTION`。
        """
        states = {}
        # 步骤一: 将[-1, 1]的意图诊断分，映射为[0, 1]的“承接控制因子”
        # control_factor=1代表多头完全掌控，control_factor=0代表空头完全掌控
        absorption_control_factor = (intent_diagnosis.clip(-1, 1) + 1) / 2.0
        # 步骤二: 计算最终风险。风险 = 原始压力中，未被多头控制住的部分
        final_risk_score = provisional_pressure * (1.0 - absorption_control_factor)
        # 步骤三: 计算最终机会。机会 = 原始压力中，被多头成功吸收的部分 (弹簧效应)
        final_opportunity_score = provisional_pressure * absorption_control_factor
        # 统一命名，更清晰地反映信号内涵
        states['SCORE_RISK_UNRESOLVED_PRESSURE'] = final_risk_score.clip(0, 1).astype(np.float32)
        states['SCORE_OPPORTUNITY_PRESSURE_ABSORPTION'] = final_opportunity_score.clip(0, 1).astype(np.float32)
        return states






