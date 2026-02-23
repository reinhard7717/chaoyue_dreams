# 文件: strategies/trend_following/intelligence/process_intelligence.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
from strategies.trend_following.intelligence.process.calculate_main_force_rally_intent import CalculateMainForceRallyIntent
from strategies.trend_following.intelligence.process.calculate_split_order_accumulation import CalculateSplitOrderAccumulation
from strategies.trend_following.intelligence.process.calculate_price_volume_dynamics import CalculatePriceVolumeDynamics
from strategies.trend_following.intelligence.process.calculate_process_covert_accumulation import CalculateProcessCovertAccumulation
from strategies.trend_following.intelligence.process.calculate_price_momentum_divergence import CalculatePriceMomentumDivergence
from strategies.trend_following.intelligence.process.calculate_winner_conviction_decay import CalculateWinnerConvictionDecay
from strategies.trend_following.intelligence.process.calculate_storm_eye_calm import CalculateStormEyeCalm
from strategies.trend_following.intelligence.process.calculate_winner_conviction_relationship import CalculateWinnerConvictionRelationship
from strategies.trend_following.intelligence.process.calculate_cost_advantage_trend_relationship import CalculateCostAdvantageTrendRelationship
from strategies.trend_following.intelligence.process.calculate_upthrust_washout import CalculateUpthrustWashoutRelationship
from strategies.trend_following.intelligence.process.calculate_main_force_control import CalculateMainForceControlRelationship

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
        self.strategy = strategy_instance
        self.helper = ProcessIntelligenceHelper(strategy_instance)
        self.calculate_main_force_rally_intent_processor = CalculateMainForceRallyIntent(strategy_instance, self.helper)
        self.calculate_split_order_accumulation_processor = CalculateSplitOrderAccumulation(strategy_instance, self.helper)
        self.calculate_price_volume_dynamics_processor = CalculatePriceVolumeDynamics(strategy_instance, self.helper)
        self.calculate_process_covert_accumulation_processor = CalculateProcessCovertAccumulation(strategy_instance, self.helper)
        self.calculate_price_momentum_divergence_processor = CalculatePriceMomentumDivergence(strategy_instance, self.helper)
        self.calculate_winner_conviction_decay_processor = CalculateWinnerConvictionDecay(strategy_instance, self.helper)
        self.calculate_storm_eye_calm_processor = CalculateStormEyeCalm(strategy_instance, self.helper)
        self.calculate_winner_conviction_relationship_processor = CalculateWinnerConvictionRelationship(strategy_instance, self.helper)
        self.calculate_cost_advantage_trend_relationship_processor = CalculateCostAdvantageTrendRelationship(strategy_instance, self.helper)
        self.calculate_upthrust_washout_processor = CalculateUpthrustWashoutRelationship(strategy_instance, self.helper)
        self.calculate_main_force_control_processor = CalculateMainForceControlRelationship(strategy_instance, self.helper) # 新增此行
        self.params = self.helper.params
        self.score_type_map = self.helper.score_type_map
        self.norm_window = self.helper.norm_window
        self.std_window = self.helper.std_window
        self.meta_window = self.helper.meta_window
        self.bipolar_sensitivity = self.helper.bipolar_sensitivity
        self.meta_score_weights = self.helper.meta_score_weights
        self.diagnostics_config = self.helper.diagnostics_config
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        if not hasattr(self.strategy, 'atomic_states'):
            self.strategy.atomic_states = {}

    def _print_debug_output(self, debug_output: Dict):
        """
        统一打印调试信息。
        """
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def _probe_variables(self, method_name: str, df_index: pd.Index, raw_inputs: Dict[str, Any], calc_nodes: Dict[str, Any], final_result: Any):
        """
        【V2.1.0 · 全息探针防御升级版】
        统一管理调试探针。新增了针对 numpy.ndarray 和纯数值的强兼容防御，
        确保即使底层释放了无索引张量，探针仍能精准定轨输出，杜绝执行中断。
        """
        is_debug=get_param_value(self.debug_params.get('enabled'),False) and get_param_value(self.debug_params.get('should_probe'),False)
        if not is_debug or not self.probe_dates:
            return
        probe_dates_dt=[pd.to_datetime(d).tz_localize(None).normalize() for d in self.probe_dates]
        def _get_val(v, target_date):
            if isinstance(v, pd.Series):
                return v.loc[target_date] if target_date in v.index else np.nan
            elif isinstance(v, np.ndarray):
                if len(v)==len(df_index):
                    try:
                        idx=df_index.get_loc(target_date)
                        return v[idx]
                    except (KeyError, IndexError, TypeError):
                        return np.nan
                return np.nan
            elif isinstance(v, (float, int, np.number)):
                return v
            return np.nan
        for date in reversed(df_index):
            if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                debug_output={f"--- {method_name} 全息探针 @ {date.strftime('%Y-%m-%d')} ---": ""}
                debug_output["[原料数据]"]=""
                for k,v in raw_inputs.items():
                    val=_get_val(v, date)
                    debug_output[f"  -> {k}: {val:.4f}" if isinstance(val,(float,np.float32,np.float64)) else f"  -> {k}: {val}"]=""
                debug_output["[计算节点]"]=""
                for k,v in calc_nodes.items():
                    val=_get_val(v, date)
                    debug_output[f"  -> {k}: {val:.4f}" if isinstance(val,(float,np.float32,np.float64)) else f"  -> {k}: {val}"]=""
                final_val=_get_val(final_result, date)
                debug_output["[最终结果]"]=f"  -> OUTPUT: {final_val:.4f}"
                self._print_debug_output(debug_output)
                break

    def _get_safe_series(self, df: pd.DataFrame, col_name: str, default_value: float = np.nan, method_name: str = "") -> pd.Series:
        """
        【V12.0.0 · 零防御暴露版】
        移除原本默认的 fillna 操作，直接返回原始 Series。如果列完全不存在，则强制抛出异常以暴露数据层断层问题。
        """
        if col_name not in df.columns:
            raise ValueError(f"[{method_name}] 致命数据断层: 缺失必需特征列 '{col_name}'，拒绝进行防御性静默填充。")
        return df[col_name].astype(np.float32)

    def _get_mtf_slope_accel_score(self, df: pd.DataFrame, base_signal_name: str, mtf_weights_config: Dict, df_index: pd.Index, method_name: str, ascending: bool = True, bipolar: bool = False) -> pd.Series:
        """【V6.0.0 · MTF动态张量免疫版】注入动态微积分推演，如果 df 中缺失 SLOPE 列，在内存中实时求导计算。"""
        slope_periods_weights = get_param_value(mtf_weights_config.get('slope_periods'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        accel_periods_weights = get_param_value(mtf_weights_config.get('accel_periods'), {"5": 0.6, "13": 0.4})
        all_scores_components = []
        total_combined_weight = 0.0
        def _process_kinematic(kinematic_type: str, weight: float, period: int):
            col_name = f'{kinematic_type}_{period}_{base_signal_name}'
            if col_name in df.columns:
                raw_series = df[col_name].astype(np.float32)
            else:
                if base_signal_name not in df.columns:
                    return 0.0, 0.0
                base_series = df[base_signal_name].astype(np.float32)
                slope = (base_series.diff(period) / period).fillna(0.0)
                if kinematic_type == 'SLOPE':
                    raw_series = slope
                else:
                    raw_series = (slope.diff(period) / period).fillna(0.0)
            if raw_series.isnull().all():
                return 0.0, 0.0
            gated = raw_series.where(raw_series.abs() >= 1e-4, 0.0)
            hab_window = max(21, period * 2)
            shock = self._apply_hab_shock(gated, window=hab_window)
            norm_score = np.tanh(shock)
            if not ascending:
                norm_score = -norm_score
            if not bipolar:
                norm_score = 0.5 * (1.0 + norm_score)
            return norm_score * weight, weight

        for period_str, weight in slope_periods_weights.items():
            score, w = _process_kinematic('SLOPE', weight, int(period_str))
            all_scores_components.append(score)
            total_combined_weight += w
        for period_str, weight in accel_periods_weights.items():
            score, w = _process_kinematic('ACCEL', weight, int(period_str))
            all_scores_components.append(score)
            total_combined_weight += w
        if not all_scores_components or total_combined_weight == 0:
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        fused_score = sum(all_scores_components) / total_combined_weight
        return fused_score.clip(-1, 1).fillna(0.0).astype(np.float32) if bipolar else fused_score.clip(0, 1).fillna(0.0).astype(np.float32)

    def _get_mtf_slope_score(self, df: pd.DataFrame, base_signal_name: str, mtf_weights: Dict, df_index: pd.Index, method_name: str, bipolar: bool = True) -> pd.Series:
        """【V6.0.0 · 纯斜率非线性动能原生版】注入动态微积分推演与绝对零基封印。"""
        fused_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_weight = 0.0
        for period_str, weight in mtf_weights.items():
            try: period = int(period_str)
            except ValueError: continue
            col_name = f'SLOPE_{period}_{base_signal_name}'
            if col_name in df.columns:
                raw_series = df[col_name].astype(np.float32)
            else:
                if base_signal_name not in df.columns: continue
                raw_series = (df[base_signal_name].astype(np.float32).diff(period) / period).fillna(0.0)
            if raw_series.isnull().all(): continue
            gated = raw_series.where(raw_series.abs() >= 1e-4, 0.0)
            shock = self._apply_hab_shock(gated, window=max(21, period * 2))
            score = np.tanh(shock) if bipolar else 0.5 * (1.0 + np.tanh(shock))
            fused_score += score * weight
            total_weight += weight
        if total_weight > 0: fused_score = fused_score / total_weight
        return fused_score.clip(-1, 1).fillna(0.0).astype(np.float32) if bipolar else fused_score.clip(0, 1).fillna(0.0).astype(np.float32)

    def _get_mtf_cohesion_score(self, df: pd.DataFrame, base_signal_names: List[str], mtf_weights_config: Dict, df_index: pd.Index, method_name: str) -> pd.Series:
        """
        【V12.0.0 · 逆向张量协同版】
        直接基于多维信号动能矩阵计算标准差，利用逆向 Tanh 将横向极度离散映射为微观协同。
        应用 HAB 冲击并实施绝对零基封印。
        """
        all_fused_mtf_scores = {}
        for base_signal_name in base_signal_names:
            fused_score = self._get_mtf_slope_accel_score(df, base_signal_name, mtf_weights_config, df_index, method_name, ascending=True, bipolar=False)
            all_fused_mtf_scores[base_signal_name] = fused_score
        if not all_fused_mtf_scores:
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        fused_scores_df = pd.DataFrame(all_fused_mtf_scores, index=df_index)
        min_periods_std = max(1, int(self.meta_window * 0.5))
        instant_std = fused_scores_df.std(axis=1).fillna(0.0)
        smoothed_std = instant_std.rolling(window=self.meta_window, min_periods=min_periods_std).mean().fillna(0.0)
        std_shock = self._apply_hab_shock(smoothed_std, window=34)
        # 逆向压缩：方差越小（冲击度越低），协同度越高
        cohesion_score = 0.5 * (1.0 - np.tanh(std_shock))
        return cohesion_score.clip(0, 1).fillna(0.0).astype(np.float32)

    def _normalize_series(self, *args, **kwargs):
        """【V4.0.0 · 核心禁令】统一归一化引擎已彻底废除。禁止调用此方法，各逻辑节点必须内置领域特化的非线性张力映射。"""
        raise NotImplementedError("致命错误：_normalize_series() 已被废弃，请使用局部 HAB 与 Tanh 映射。")

    def _get_atomic_score(self, df: pd.DataFrame, score_name: str, default_value: float = 0.0) -> pd.Series:
        """
        【V12.0.0 · 原子信号安全门】
        强制提供 Pandas 序列级别的绝对零基封印，防止任何未初始化的原子状态向上传递空值。
        """
        score_series = self.strategy.atomic_states.get(score_name)
        if score_series is None:
            # 采用静默生成，不再打印警告，避免在千万级循环中被 IO 拖垮
            return pd.Series(default_value, index=df.index, dtype=np.float32)
        return score_series.fillna(default_value).astype(np.float32)

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: List[str], method_name: str) -> bool:
        """
        【V2.0.0 · 严格契约校验】
        内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。如果缺失，直接抛出异常而非静默跳过。
        """
        missing_signals=[signal for signal in required_signals if signal not in df.columns and signal not in self.strategy.atomic_states]
        if missing_signals:
            raise ValueError(f"[{method_name}] 启动失败：严重数据断层，缺失核心信号 {missing_signals}。")
        return True

    def _extract_and_validate_config_signals(self, df: pd.DataFrame, config: Dict, method_name: str) -> bool:
        """
        【V13.0.0 · 蓝图审查官 (多维时空与攻防撕裂白名单扩容版)】
        更新版本号。将新增的时空分形共振、日内攻城拔寨、隔夜日内撕裂以及筹码重心动力学等战役级引擎加入免检白名单。
        """
        signal_name=config.get('name','')
        diagnosis_type=config.get('diagnosis_type','meta_relationship')
        if diagnosis_type=='domain_reversal':
            return True
        exempt_signals={
            'PROCESS_META_POWER_TRANSFER','PROCESS_META_PANIC_WASHOUT_ACCUMULATION',
            'PROCESS_META_DECEPTIVE_ACCUMULATION','PROCESS_META_ACCUMULATION_INFLECTION',
            'PROCESS_META_BREAKOUT_ACCELERATION','PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT',
            'PROCESS_META_LOSER_CAPITULATION','PROCESS_META_PROFIT_VS_FLOW',
            'PROCESS_META_STOCK_SECTOR_SYNC','PROCESS_META_HOT_SECTOR_COOLING','PROCESS_META_WASH_OUT_REBOUND',
            'PROCESS_FUSION_TREND_EXHAUSTION_SYNDROME','PROCESS_STRATEGY_DYN_VS_CHIP_DECAY_RISE',
            'PROCESS_META_SMART_MONEY_IGNITION','PROCESS_META_VPA_MF_COHERENCE_RESONANCE',
            'PROCESS_META_MTF_FRACTAL_RESONANCE','PROCESS_META_INTRADAY_SIEGE_EXHAUSTION',
            'PROCESS_META_OVERNIGHT_INTRADAY_TEARING','PROCESS_META_CHIP_CENTER_KINEMATICS'
        }
        if signal_name in exempt_signals:
            return True
        legacy_remap={
            'breakout_confidence_D':'breakout_quality_score_D',
            'closing_strength_index_D':'CLOSING_STRENGTH_D',
            'trend_confirmation_score_D':'uptrend_strength_D',
            'consolidation_quality_grade_D':'consolidation_quality_score_D'
        }
        required_signals=[]
        for key in ['signal_A','signal_B','antidote_signal','source_signal']:
            if config.get(key):
                mapped_val=legacy_remap.get(config[key],config[key])
                config[key]=mapped_val
                required_signals.append(mapped_val)
        if config.get('axioms'):
            for axiom in config.get('axioms',[]):
                if axiom.get('name'):
                    mapped_val=legacy_remap.get(axiom['name'],axiom['name'])
                    axiom['name']=mapped_val
                    required_signals.append(mapped_val)
        required_signals=[
            sig for sig in required_signals 
            if not (sig.startswith('SCORE_') or sig.startswith('PROCESS_') or sig.startswith('FUSION_'))
        ]
        if not required_signals:
            return True
        return self._validate_required_signals(df,required_signals,method_name)

    def run_process_diagnostics(self, df: pd.DataFrame, task_type_filter: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        【V5.7 · 基石信号增强版】过程情报分析总指挥
        - 核心升级: 扩展“基石信号”清单，将 `PROCESS_META_COVERT_ACCUMULATION` 纳入优先计算，
                      确保其在被其他信号依赖时已就绪。
        """
        print("启动【V5.7 · 基石信号增强版】过程情报分析...")
        all_process_states = {}
        # 直接使用 self.params，因为它已在 __init__ 中加载了 process_intelligence_params
        p_conf = self.params
        diagnostics = get_param_value(p_conf.get('diagnostics'), [])
        # 定义需要优先计算的“基石信号”清单，新增 PROCESS_META_COVERT_ACCUMULATION
        priority_signals = [
            'PROCESS_META_POWER_TRANSFER',
            'PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_COVERT_ACCUMULATION'
        ]
        # 依赖前置处理逻辑，遍历基石信号清单
        processed_priority_signals = set()
        for signal_name in priority_signals:
            config = next((d for d in diagnostics if d.get('name') == signal_name), None)
            if config:
                # 使用 _run_meta_analysis 进行计算，以复用其内部逻辑
                score_dict = self._run_meta_analysis(df, config)
                if score_dict:
                    all_process_states.update(score_dict)
                    self.strategy.atomic_states.update(score_dict)
                    latest_score = next(iter(score_dict.values())).iloc[-1]
                    processed_priority_signals.add(signal_name)
        # 主循环处理剩余的诊断任务
        remaining_diagnostics = [d for d in diagnostics if d.get('name') not in processed_priority_signals]
        for diag_config in remaining_diagnostics:
            if task_type_filter:
                if diag_config.get('task_type') != task_type_filter:
                    continue
            diag_name = diag_config.get('name')
            if not diag_name:
                continue
            score = self._run_meta_analysis(df, diag_config)
            if isinstance(score, pd.Series):
                all_process_states[diag_name] = score
                self.strategy.atomic_states[diag_name] = score
            elif isinstance(score, dict):
                all_process_states.update(score)
                self.strategy.atomic_states.update(score)
        print(f"【V5.7 · 基石信号增强版】分析完成，生成 {len(all_process_states)} 个过程元信号。")
        return all_process_states

    def _run_meta_analysis(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.0.0 · 防御性元分析调度版】
        内聚豁免校验逻辑已全部下沉至蓝图审查官，外层逻辑回归极致简洁。
        """
        signal_name = config.get('name', '未知信号')
        if not self._extract_and_validate_config_signals(df, config, f"_run_meta_analysis (for {signal_name})"):
            return {}
        diagnosis_type = config.get('diagnosis_type', 'meta_relationship')
        if diagnosis_type == 'meta_relationship':
            return self._diagnose_meta_relationship(df, config)
        elif diagnosis_type == 'split_meta_relationship':
            return self._diagnose_split_meta_relationship(df, config)
        elif diagnosis_type == 'signal_decay':
            return self._diagnose_signal_decay(df, config)
        elif diagnosis_type == 'domain_reversal':
            return self._diagnose_domain_reversal(df, config)
        else:
            print(f"    -> [过程情报警告] 未知的元分析诊断类型: '{diagnosis_type}'，跳过信号 '{config.get('name')}' 的计算。")
            return {}

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.0.0 · 防御性关系诊断分发器】
        同步在关系分发器层将校验权统一归置于 _extract_and_validate_config_signals。
        """
        signal_name = config.get('name', '未知信号')
        if not self._extract_and_validate_config_signals(df, config, f"_diagnose_meta_relationship (for {signal_name})"):
            return {}
        diagnosis_type = config.get('diagnosis_type', 'meta_relationship')
        if diagnosis_type == 'meta_relationship':
            return self._diagnose_meta_relationship_internal(df, config)
        elif diagnosis_type == 'split_meta_relationship':
            return self._diagnose_split_meta_relationship(df, config)
        elif diagnosis_type == 'signal_decay':
            return self._diagnose_signal_decay(df, config)
        elif diagnosis_type == 'domain_reversal':
            return self._diagnose_domain_reversal(df, config)
        else:
            print(f"    -> [过程情报警告] 未知的元分析诊断类型: '{diagnosis_type}'，跳过信号 '{config.get('name')}' 的计算。")
            return {}

    def _diagnose_meta_relationship_internal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V13.0.0 · 关系诊断分发器 (日内攻城与多维时空挂载版)】
        挂载多时间框架分形共振、日内攻城拔寨衰竭、隔夜撕裂与筹码重心动力学终极防爆引擎。
        """
        signal_name=config.get('name')
        df_index=df.index
        meta_score=pd.Series(dtype=np.float32)
        if signal_name=='PROCESS_META_COST_ADVANTAGE_TREND':
            meta_score=self.calculate_cost_advantage_trend_relationship_processor.calculate(df,config)
        elif signal_name=='PROCESS_STRATEGY_DYN_VS_CHIP_DECAY':
            relationship_score=self._calculate_dyn_vs_chip_relationship(df,config)
            meta_score=self._perform_meta_analysis_on_score(relationship_score,config,df,df_index)
        elif signal_name=='PROCESS_META_PRICE_VS_RETAIL_CAPITULATION':
            relationship_score=self._calculate_price_vs_capitulation_relationship(df,config)
            meta_score=self._perform_meta_analysis_on_score(relationship_score,config,df,df_index)
        elif signal_name=='PROCESS_META_PD_DIVERGENCE_CONFIRM':
            relationship_score=self._calculate_pd_divergence_relationship(df,config)
            meta_score=relationship_score
        elif signal_name=='PROCESS_META_PROFIT_VS_FLOW':
            relationship_score=self._calculate_profit_vs_flow_relationship(df,config)
            meta_score=relationship_score
        elif signal_name=='PROCESS_META_PF_REL_BULLISH_TURN':
            meta_score=self._calculate_pf_relationship(df,config)
        elif signal_name=='PROCESS_META_PC_REL_BULLISH_TURN':
            meta_score=self._calculate_pc_relationship(df,config)
        elif signal_name=='PROCESS_META_PRICE_VOLUME_DYNAMICS':
            meta_score=self.calculate_price_volume_dynamics_processor.calculate(df,config)
        elif signal_name=='PROCESS_META_MAIN_FORCE_RALLY_INTENT':
            meta_score=self.calculate_main_force_rally_intent_processor.calculate(df,config)
        elif signal_name=='PROCESS_META_WINNER_CONVICTION':
            relationship_score=self.calculate_winner_conviction_relationship_processor.calculate(df,config)
            meta_score=self._perform_meta_analysis_on_score(relationship_score,config,df,df_index)
        elif signal_name=='PROCESS_META_LOSER_CAPITULATION':
            meta_score=self._calculate_loser_capitulation(df,config)
        elif signal_name=='PROCESS_STRATEGY_FF_VS_STRUCTURE_LEAD':
            relationship_score=self._calculate_ff_vs_structure_relationship(df,config)
            meta_score=self._perform_meta_analysis_on_score(relationship_score,config,df,df_index)
        elif signal_name=='PROCESS_META_MAIN_FORCE_CONTROL':
            meta_score=self.calculate_main_force_control_processor.calculate(df,config)
        elif signal_name=='PROCESS_META_PANIC_WASHOUT_ACCUMULATION':
            meta_score=self._calculate_panic_washout_accumulation(df,config)
        elif signal_name=='PROCESS_META_DECEPTIVE_ACCUMULATION':
            meta_score=self._calculate_deceptive_accumulation(df,config)
        elif signal_name=='PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY':
            meta_score=self.calculate_split_order_accumulation_processor.calculate(df,config)
        elif signal_name=='PROCESS_META_UPTHRUST_WASHOUT':
            meta_score=self.calculate_upthrust_washout_processor.calculate(df,config)
        elif signal_name=='PROCESS_META_ACCUMULATION_INFLECTION':
            meta_score=self._calculate_accumulation_inflection(df,config)
        elif signal_name=='PROCESS_META_BREAKOUT_ACCELERATION':
            meta_score=self._calculate_breakout_acceleration(df,config)
        elif signal_name=='PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT':
            meta_score=self._calculate_fund_flow_accumulation_inflection(df,config)
        elif signal_name=='PROCESS_META_STOCK_SECTOR_SYNC':
            relationship_score=self._calculate_stock_sector_sync(df,config)
            meta_score=relationship_score
        elif signal_name=='PROCESS_META_HOT_SECTOR_COOLING':
            relationship_score=self._calculate_hot_sector_cooling(df,config)
            meta_score=relationship_score
        elif signal_name=='PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE':
            meta_score=self.calculate_price_momentum_divergence_processor.calculate(df,config)
        elif signal_name=='PROCESS_META_STORM_EYE_CALM':
            meta_score=self.calculate_storm_eye_calm_processor.calculate(df,config)
        elif signal_name=='PROCESS_META_WASH_OUT_REBOUND':
            meta_score=self._calculate_process_wash_out_rebound(df,config)
        elif signal_name=='PROCESS_META_COVERT_ACCUMULATION':
            meta_score=self.calculate_process_covert_accumulation_processor.calculate(df,config)
        elif signal_name=='PROCESS_META_POWER_TRANSFER':
            meta_score=self._calculate_power_transfer(df,config)
        elif signal_name=='PROCESS_FUSION_TREND_EXHAUSTION_SYNDROME':
            meta_score=self._calculate_fusion_trend_exhaustion_syndrome(df,config)
        elif signal_name=='PROCESS_STRATEGY_DYN_VS_CHIP_DECAY_RISE':
            meta_score=self._calculate_dyn_vs_chip_decay_rise(df,config)
        elif signal_name=='PROCESS_META_SMART_MONEY_IGNITION':
            meta_score=self._calculate_smart_money_ignition(df,config)
        elif signal_name=='PROCESS_META_VPA_MF_COHERENCE_RESONANCE':
            meta_score=self._calculate_vpa_mf_coherence_resonance(df,config)
        elif signal_name=='PROCESS_META_MTF_FRACTAL_RESONANCE':
            meta_score=self._calculate_mtf_fractal_resonance(df,config)
        elif signal_name=='PROCESS_META_INTRADAY_SIEGE_EXHAUSTION':
            meta_score=self._calculate_intraday_siege_exhaustion(df,config)
        elif signal_name=='PROCESS_META_OVERNIGHT_INTRADAY_TEARING':
            meta_score=self._calculate_overnight_intraday_tearing(df,config)
        elif signal_name=='PROCESS_META_CHIP_CENTER_KINEMATICS':
            meta_score=self._calculate_chip_center_kinematics(df,config)
        else:
            relationship_score=self._calculate_instantaneous_relationship(df,config)
            if relationship_score.empty:
                return {}
            self.strategy.atomic_states[f"PROCESS_ATOMIC_REL_SCORE_{signal_name}"]=relationship_score.astype(np.float32)
            diagnosis_mode=config.get('diagnosis_mode','direct_confirmation')
            if diagnosis_mode=='direct_confirmation':
                meta_score=relationship_score
            else:
                meta_score=self._perform_meta_analysis_on_score(relationship_score,config,df,df_index)
        if meta_score.empty:
            return {}
        return {signal_name:meta_score}

    def _diagnose_split_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V6.0.0 · 裂变张量全域防线版】
        处理裂变状态输出，确保输出的所有字典 Series 100% 不含 NaN。
        """
        states = {}
        output_names = config.get('output_names', {})
        opportunity_signal_name = output_names.get('opportunity')
        risk_signal_name = output_names.get('risk')
        if not opportunity_signal_name or not risk_signal_name:
            return {}
        relationship_score = self._calculate_price_efficiency_relationship(df, config)
        if relationship_score.empty:
            return {}
        df_index = df.index
        relationship_displacement = relationship_score.diff(self.meta_window).fillna(0.0)
        relationship_momentum = relationship_displacement.diff(1).fillna(0.0)
        bipolar_displacement = np.tanh(self._apply_hab_shock(relationship_displacement, window=self.meta_window*2))
        bipolar_momentum = np.tanh(self._apply_hab_shock(relationship_momentum, window=13))
        displacement_weight = self.meta_score_weights[0]
        momentum_weight = self.meta_score_weights[1]
        meta_score = (bipolar_displacement * displacement_weight + bipolar_momentum * momentum_weight)
        meta_score = np.sign(meta_score) * (np.abs(meta_score) ** 1.5)
        meta_score = meta_score.clip(-1, 1).fillna(0.0)
        # 绝对防漏：分裂后双向补 0.0
        states[opportunity_signal_name] = meta_score.clip(lower=0).astype(np.float32)
        states[risk_signal_name] = meta_score.clip(upper=0).abs().astype(np.float32)
        return states

    def _apply_hab_shock(self, series: pd.Series, window: int = 21) -> pd.Series:
        """【V2.0.0 · HAB存量冲击系统】将绝对数值转化为相对历史存量的冲击度(Z-Score)"""
        roll_mean = series.rolling(window=window, min_periods=1).mean()
        roll_std = series.rolling(window=window, min_periods=1).std().replace(0, 1e-5).fillna(1e-5)
        return ((series - roll_mean) / roll_std).astype(np.float32)

    def _get_kinematic_tensor(self, df: pd.DataFrame, base_col: str, period: int = 13, method_name: str = "") -> pd.Series:
        """
        【V6.0.0 · 动态微积分免疫处理器】
        当配置遗漏导致缺失 SLOPE/ACCEL 列时，自动使用 diff 在内存中实时计算微积分替代，杜绝断层熔断。
        """
        if base_col not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        slope_col = f'SLOPE_{period}_{base_col}'
        accel_col = f'ACCEL_{period}_{base_col}'
        if slope_col in df.columns:
            slope = df[slope_col].astype(np.float32)
        else:
            slope = (df[base_col].diff(period) / period).fillna(0.0).astype(np.float32)
        if accel_col in df.columns:
            accel = df[accel_col].astype(np.float32)
        else:
            accel = (slope.diff(period) / period).fillna(0.0).astype(np.float32)
        raw_tensor = slope + accel * 0.5
        gated_tensor = raw_tensor.where(raw_tensor.abs() >= 1e-4, 0.0)
        return np.tanh(gated_tensor * 20.0).astype(np.float32)

    def _calculate_power_transfer(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V8.0.0 · 权力交接张量终极版 (彻底消除逆势折价陷阱)】
        """
        method_name="_calculate_power_transfer"
        # 已移除对 SLOPE/ACCEL 的强校验，交由 _get_kinematic_tensor 动态处理
        required_signals=['net_mf_amount_D','amount_D','tick_large_order_net_D','tick_chip_transfer_efficiency_D','flow_efficiency_D','intraday_cost_center_migration_D','downtrend_strength_D','chip_concentration_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        tick_large_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name)
        transfer_eff=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        flow_eff=self._get_safe_series(df,'flow_efficiency_D',method_name=method_name)
        cost_migration=self._get_safe_series(df,'intraday_cost_center_migration_D',method_name=method_name)
        downtrend=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        chip_conc=self._get_safe_series(df,'chip_concentration_ratio_D',method_name=method_name)
        mf_ratio=(net_mf/amount).fillna(0.0)
        tick_ratio=(tick_large_net/amount).fillna(0.0)
        mf_shock=np.tanh(self._apply_hab_shock(mf_ratio,21))
        tick_shock=np.tanh(self._apply_hab_shock(tick_ratio,21))
        cost_mig_shock=np.tanh(self._apply_hab_shock(cost_migration, 13))
        power_core=mf_shock*0.5+tick_shock*0.3+cost_mig_shock*0.2
        transfer_eff_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(transfer_eff, 21)))
        flow_eff_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(flow_eff, 21)))
        efficiency_multiplier=1.0+(transfer_eff_norm+flow_eff_norm)/2.0
        chip_diff_shock=np.tanh(self._apply_hab_shock(chip_conc.diff().fillna(0.0),13))
        chip_penetration=chip_diff_shock*efficiency_multiplier
        synergy_amplifier=(1.0 + chip_penetration * np.sign(power_core)).clip(lower=0.1)
        raw_score=power_core*synergy_amplifier
        # 修复逻辑陷阱：只有在逆势做多时才施加下跌折价，不能削弱顺势做空的势能
        trend_discount=pd.Series(1.0,index=df_index).mask((downtrend>0.8) & (raw_score>0),0.6)
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=(np.tanh(final_score)*trend_discount).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'net_mf_amount_D':net_mf,'amount_D':amount,'tick_chip_transfer_efficiency_D':transfer_eff,'intraday_cost_center_migration_D':cost_migration}, calc_nodes={'mf_shock':mf_shock,'tick_shock':tick_shock,'cost_mig_shock':cost_mig_shock,'power_core':power_core,'transfer_eff_norm':transfer_eff_norm,'flow_eff_norm':flow_eff_norm,'efficiency_multiplier':efficiency_multiplier,'chip_diff_shock':chip_diff_shock,'chip_penetration':chip_penetration,'synergy_amplifier':synergy_amplifier,'raw_score':raw_score}, final_result=final_score)
        return final_score

    def _calculate_price_vs_capitulation_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 散户投降背离探针防爆版】
        完全剥离微积分列强校验，末端绝对零基填补。
        """
        method_name = "_calculate_price_vs_capitulation_relationship"
        required_signals = ['pressure_trapped_D', 'INTRADAY_SUPPORT_INTENT_D', 'intraday_low_lock_ratio_D', 'chip_entropy_D']
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        pressure = self._get_safe_series(df, 'pressure_trapped_D', method_name=method_name)
        support = self._get_safe_series(df, 'INTRADAY_SUPPORT_INTENT_D', method_name=method_name)
        low_lock = self._get_safe_series(df, 'intraday_low_lock_ratio_D', method_name=method_name)
        entropy = self._get_safe_series(df, 'chip_entropy_D', method_name=method_name)
        kinematics_p = self._get_kinematic_tensor(df, 'pressure_trapped_D', 13, method_name)
        panic_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(pressure, 21)))
        support_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(support, 34)))
        low_lock_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(low_lock, 21)))
        entropy_shock = np.tanh(self._apply_hab_shock(entropy, 21))
        absorption_resonance = support_norm * (1.0 + low_lock_shock) * (1.0 - entropy_shock.clip(lower=0))
        base_divergence = self._calculate_instantaneous_relationship(df, config)
        raw_score = base_divergence * panic_shock * absorption_resonance * (1.0 + kinematics_p.clip(lower=0))
        final_score = np.sign(raw_score) * (np.abs(raw_score) ** 1.5)
        final_score = np.tanh(final_score).clip(-1, 1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'pressure_trapped_D': pressure, 'INTRADAY_SUPPORT_INTENT_D': support}, calc_nodes={'kinematics_p': kinematics_p, 'panic_shock': panic_shock, 'absorption_resonance': absorption_resonance}, final_result=final_score)
        return final_score

    def _calculate_price_efficiency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 价格效率博弈防爆版】
        完全剥离微积分列强校验，末端绝对零基填补。
        """
        method_name = "_calculate_price_efficiency_relationship"
        required_signals = ['VPA_EFFICIENCY_D', 'net_mf_amount_D', 'shakeout_score_D', 'tick_chip_transfer_efficiency_D', 'high_freq_flow_skewness_D']
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        eff = self._get_safe_series(df, 'VPA_EFFICIENCY_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        shakeout = self._get_safe_series(df, 'shakeout_score_D', method_name=method_name)
        transfer_eff = self._get_safe_series(df, 'tick_chip_transfer_efficiency_D', method_name=method_name)
        flow_skew = self._get_safe_series(df, 'high_freq_flow_skewness_D', method_name=method_name)
        kinematics_eff = self._get_kinematic_tensor(df, 'VPA_EFFICIENCY_D', 13, method_name)
        eff_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(eff, 34)))
        mf_conviction = np.tanh(self._apply_hab_shock(net_mf, 55))
        shakeout_penalty = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(shakeout, 21)))
        transfer_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(transfer_eff, 21)))
        flow_skew_shock = np.tanh(self._apply_hab_shock(flow_skew, 21))
        synergy = eff_shock * mf_conviction * (1.0 + transfer_shock) * (1.0 + kinematics_eff) * (1.0 + flow_skew_shock.clip(lower=0))
        final_score = np.sign(synergy) * (np.abs(synergy) ** 1.5)
        final_score = (final_score * (1.0 - shakeout_penalty ** 1.5)).clip(-1, 1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'VPA_EFFICIENCY_D': eff, 'net_mf_amount_D': net_mf}, calc_nodes={'kinematics_eff': kinematics_eff, 'eff_shock': eff_shock, 'synergy': synergy}, final_result=final_score)
        return final_score

    def _calculate_pd_divergence_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 博弈背离方向性智能杠杆版】
        完全剥离微积分列强校验，末端绝对零基填补。
        """
        method_name="_calculate_pd_divergence_relationship"
        required_signals=['game_intensity_D','weight_avg_cost_D','close_D','intraday_chip_game_index_D','chip_divergence_ratio_D','winner_rate_D','high_freq_flow_kurtosis_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        game=self._get_safe_series(df,'game_intensity_D',method_name=method_name)
        cost=self._get_safe_series(df,'weight_avg_cost_D',method_name=method_name).replace(0,np.nan)
        close_p=self._get_safe_series(df,'close_D',method_name=method_name)
        intra_game=self._get_safe_series(df,'intraday_chip_game_index_D',method_name=method_name)
        chip_div=self._get_safe_series(df,'chip_divergence_ratio_D',method_name=method_name)
        winner=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        hf_kurtosis=self._get_safe_series(df,'high_freq_flow_kurtosis_D',method_name=method_name)
        kinematics_game=self._get_kinematic_tensor(df,'game_intensity_D',13,method_name)
        game_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(game,55)))
        price_adv=np.tanh((close_p-cost)/cost*10.0)
        win_norm=0.5*(1.0+np.tanh(self._apply_hab_shock(winner,21)))
        intra_game_shock=np.tanh(self._apply_hab_shock(intra_game,21))
        chip_div_shock=np.tanh(self._apply_hab_shock(chip_div,21))
        kurtosis_shock=np.tanh(self._apply_hab_shock(hf_kurtosis,21)).clip(lower=0)
        base_divergence=self._calculate_instantaneous_relationship(df,config)
        price_leverage=(1.0-price_adv*np.sign(base_divergence)).clip(lower=0.1)
        tensor_resonance=game_shock*price_leverage*win_norm*(1.0+intra_game_shock.clip(lower=0))*(1.0+chip_div_shock.clip(lower=0))*(1.0+kurtosis_shock)
        raw_score=base_divergence*tensor_resonance*(1.0+kinematics_game.clip(lower=0))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'game_intensity_D':game,'weight_avg_cost_D':cost},calc_nodes={'kinematics_game':kinematics_game,'price_leverage':price_leverage,'tensor_resonance':tensor_resonance,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """【V4.0.0 · 局部方差压迫衰减版】利用本地特征滚动标准差刻画相对衰变严重度，执行非线性激增阻尼。"""
        method_name="_diagnose_signal_decay"
        is_debug_enabled=get_param_value(self.debug_params.get('enabled'),False) and get_param_value(self.debug_params.get('should_probe'),False)
        signal_name=config.get('name')
        if signal_name=='PROCESS_META_WINNER_CONVICTION_DECAY':
            decay_score=self.calculate_winner_conviction_decay_processor.calculate(df,config)
            return {signal_name:decay_score.astype(np.float32)}
        source_signal_name=config.get('source_signal')
        if not source_signal_name:
            return {}
        df_index=df.index
        if config.get('source_type','df')=='atomic_states':
            source_series=self.strategy.atomic_states.get(source_signal_name)
        else:
            if source_signal_name not in df.columns:
                return {}
            source_series=df[source_signal_name].astype(np.float32)
        if source_series is None:
            return {}
        signal_change=source_series.diff(1).fillna(0)
        decay_magnitude=signal_change.clip(upper=0).abs()
        local_std=source_series.rolling(window=21,min_periods=1).std().replace(0,1e-5).fillna(1e-5)
        relative_decay=decay_magnitude/local_std
        decay_score=np.tanh(relative_decay*1.5).clip(0,1)
        if is_debug_enabled and self.probe_dates:
            self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'source':source_series},calc_nodes={'signal_change':signal_change,'relative_decay':relative_decay},final_result=decay_score)
        return {signal_name:decay_score.astype(np.float32)}

    def _diagnose_domain_reversal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V12.0.0 · 领域反转全息诊断入口】
        摒弃孤立公理强校验，通过自适应加权机制执行柔性降级，引入公理群矩阵共振。
        """
        domain_name = config.get('domain_name')
        axiom_configs = config.get('axioms', [])
        output_bottom_name = config.get('output_bottom_reversal_name')
        output_top_name = config.get('output_top_reversal_name')
        if not domain_name or not axiom_configs or not output_bottom_name or not output_top_name:
            return {}
        df_index = df.index
        domain_health_components = []
        total_weight = 0.0
        # 1. 动态加权矩阵坍缩
        for axiom_config in axiom_configs:
            axiom_name = axiom_config.get('name')
            axiom_weight = axiom_config.get('weight', 0.0)
            # 柔性跳过未加载或计算失败的高阶公理信号，阻止系统雪崩
            if axiom_name not in self.strategy.atomic_states:
                continue
            axiom_score = self.strategy.atomic_states.get(axiom_name, pd.Series(0.0, index=df_index))
            domain_health_components.append(axiom_score.fillna(0.0) * axiom_weight)
            total_weight += abs(axiom_weight)
        if total_weight == 0:
            return {}
        # 2. 领域基础健康度 (Bipolar: -1 to 1)
        bipolar_domain_health = (sum(domain_health_components) / total_weight).clip(-1, 1).fillna(0.0).astype(np.float32)
        # 将结果递交至审判庭 _judge_domain_reversal 进行 HAB 冲击判定
        return self._judge_domain_reversal(bipolar_domain_health, config, df)

    def _judge_domain_reversal(self, bipolar_domain_health: pd.Series, config: Dict, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0.0 · 神谕审判庭全域防线版】
        """
        domain_name = config.get('domain_name', '未知领域')
        output_bottom_name = config.get('output_bottom_reversal_name')
        output_top_name = config.get('output_top_reversal_name')
        df_index = df.index
        health_yesterday = bipolar_domain_health.shift(1).fillna(0.0)
        health_change = bipolar_domain_health.diff(1).fillna(0.0)
        shock = self._apply_hab_shock(health_change, window=13)
        bottom_context = (1.0 - health_yesterday).clip(0, 2.0)
        bottom_shock = shock.clip(lower=0)
        bottom_reversal_raw = (bottom_shock * bottom_context) ** 1.5
        bottom_reversal_score = np.tanh(bottom_reversal_raw).clip(0, 1).fillna(0.0)
        top_context = (1.0 + health_yesterday).clip(0, 2.0)
        top_shock = shock.clip(upper=0).abs()
        top_reversal_raw = (top_shock * top_context) ** 1.5
        top_reversal_score = np.tanh(top_reversal_raw).clip(0, 1).fillna(0.0)
        self._probe_variables(
            method_name=f"_judge_domain_reversal ({domain_name})", 
            df_index=df_index, 
            raw_inputs={'bipolar_domain_health': bipolar_domain_health}, 
            calc_nodes={'health_change': health_change, 'shock': shock, 'bottom_context': bottom_context, 'top_context': top_context}, 
            final_result=bottom_reversal_score
        )
        return {
            output_bottom_name: bottom_reversal_score.astype(np.float32),
            output_top_name: top_reversal_score.astype(np.float32)
        }

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V7.0.0 · 恐慌洗盘防爆探针版】
        完全剥离微积分列强校验，末端绝对零基填补。
        """
        method_name="_calculate_panic_washout_accumulation"
        required_signals=['pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D','intraday_trough_filling_degree_D','high_freq_flow_divergence_D','chip_rsi_divergence_D','chip_stability_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        panic_level=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        low_lock_ratio=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorption=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        trough_filling=self._get_safe_series(df,'intraday_trough_filling_degree_D',method_name=method_name)
        hff_div=self._get_safe_series(df,'high_freq_flow_divergence_D',method_name=method_name)
        chip_div=self._get_safe_series(df,'chip_rsi_divergence_D',method_name=method_name)
        chip_stab=self._get_safe_series(df,'chip_stability_D',method_name=method_name)
        panic_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(panic_level,21)))
        low_lock_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(low_lock_ratio,21)))
        absorption_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(absorption,21)))
        trough_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(trough_filling,21)))
        panic_intensity=np.sqrt(panic_shock*low_lock_shock.clip(lower=0))
        absorption_intensity=np.sqrt(absorption_shock*trough_shock.clip(lower=0))
        hff_div_shock=np.tanh(self._apply_hab_shock(hff_div,21))
        chip_div_shock=np.tanh(self._apply_hab_shock(chip_div,21))
        divergence_bonus=1.0+hff_div_shock.clip(lower=0)+chip_div_shock.clip(lower=0)
        base_score=panic_intensity*absorption_intensity*divergence_bonus
        historical_potential_gate=config.get('historical_potential_gate',0.2)
        gate_mask=chip_stab>historical_potential_gate
        final_score=np.tanh(base_score**1.5).where(gate_mask,0.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':panic_level,'intraday_low_lock_ratio_D':low_lock_ratio},calc_nodes={'panic_shock':panic_shock,'panic_intensity':panic_intensity,'absorption_intensity':absorption_intensity,'divergence_bonus':divergence_bonus,'base_score':base_score},final_result=final_score)
        return final_score

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 诡道偏度防爆重构版】
        完全剥离微积分列强校验，末端绝对零基填补。
        """
        method_name="_calculate_deceptive_accumulation"
        required_signals=['stealth_flow_ratio_D','tick_clustering_index_D','intraday_price_distribution_skewness_D','high_freq_flow_skewness_D','price_flow_divergence_D','chip_flow_intensity_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        stealth_flow=self._get_safe_series(df,'stealth_flow_ratio_D',method_name=method_name)
        tick_clustering=self._get_safe_series(df,'tick_clustering_index_D',method_name=method_name)
        price_skew=self._get_safe_series(df,'intraday_price_distribution_skewness_D',method_name=method_name)
        flow_skew=self._get_safe_series(df,'high_freq_flow_skewness_D',method_name=method_name)
        price_flow_div=self._get_safe_series(df,'price_flow_divergence_D',method_name=method_name)
        flow_intensity=self._get_safe_series(df,'chip_flow_intensity_D',method_name=method_name)
        stealth_flow_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(stealth_flow, 21)))
        flow_intensity_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(flow_intensity, 21)))
        tick_clustering_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(tick_clustering, 21)))
        skew_divergence=flow_skew-price_skew
        skew_tension=np.tanh(self._apply_hab_shock(skew_divergence,21)).clip(lower=0)
        stealth_strength=stealth_flow_norm*tick_clustering_norm*flow_intensity_norm
        price_flow_div_shock=np.tanh(self._apply_hab_shock(price_flow_div,21))
        camouflage_index=price_flow_div_shock.clip(lower=0)+skew_tension*0.5
        raw_deception=stealth_strength*(1.0+camouflage_index)
        final_score=np.tanh(raw_deception**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'stealth_flow_ratio_D':stealth_flow,'high_freq_flow_skewness_D':flow_skew},calc_nodes={'stealth_flow_norm':stealth_flow_norm,'skew_divergence':skew_divergence,'skew_tension':skew_tension,'stealth_strength':stealth_strength,'camouflage_index':camouflage_index,'raw_deception':raw_deception},final_result=final_score)
        return final_score

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.0.0 · 吸筹质变临界探针版】
        计算多日吸筹结束，面临质变的临界触发拐点。
        """
        method_name="_calculate_accumulation_inflection"
        required_signals=['PROCESS_META_COVERT_ACCUMULATION','PROCESS_META_DECEPTIVE_ACCUMULATION','PROCESS_META_PANIC_WASHOUT_ACCUMULATION','PROCESS_META_MAIN_FORCE_RALLY_INTENT','chip_convergence_ratio_D','price_vs_ma_21_ratio_D','flow_acceleration_intraday_D','flow_consistency_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        accumulation_window=config.get('accumulation_window',21)
        covert_accum=self._get_atomic_score(df,'PROCESS_META_COVERT_ACCUMULATION',0.0)
        deceptive_accum=self._get_atomic_score(df,'PROCESS_META_DECEPTIVE_ACCUMULATION',0.0)
        panic_accum=self._get_atomic_score(df,'PROCESS_META_PANIC_WASHOUT_ACCUMULATION',0.0)
        rally_intent=self._get_atomic_score(df,'PROCESS_META_MAIN_FORCE_RALLY_INTENT',0.0).clip(lower=0)
        chip_conv=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        price_ma21=self._get_safe_series(df,'price_vs_ma_21_ratio_D',method_name=method_name)
        flow_accel=self._get_safe_series(df,'flow_acceleration_intraday_D',method_name=method_name)
        flow_cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        daily_composite=(covert_accum*0.4+deceptive_accum*0.4+panic_accum*0.2)
        potential_energy=daily_composite.ewm(span=accumulation_window,adjust=False,min_periods=5).mean()
        structural_criticality=(chip_conv*0.6+np.clip(1.0-np.abs(price_ma21-1.0)*10,0,1)*0.4)
        dynamic_ignition=flow_accel.clip(lower=0)*flow_cons*rally_intent
        critical_mass=potential_energy*structural_criticality*dynamic_ignition
        final_score=(1/(1+np.exp(-10*(critical_mass-0.5)))).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'covert_accum':covert_accum,'deceptive_accum':deceptive_accum,'panic_accum':panic_accum,'rally_intent':rally_intent,'chip_convergence_ratio_D':chip_conv,'price_vs_ma_21_ratio_D':price_ma21,'flow_acceleration_intraday_D':flow_accel,'flow_consistency_D':flow_cons},calc_nodes={'daily_composite':daily_composite,'potential_energy':potential_energy,'structural_criticality':structural_criticality,'dynamic_ignition':dynamic_ignition,'critical_mass':critical_mass},final_result=final_score)
        return final_score

    def _calculate_loser_capitulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 绝地反击安全防爆版】
        杜绝所有物理压力释放数值直接相乘，应用统一 HAB-Tanh 管线。
        """
        method_name="_calculate_loser_capitulation"
        required_signals=['pressure_release_index_D','pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        pressure_release=self._get_safe_series(df,'pressure_release_index_D',method_name=method_name)
        pressure_trapped=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorption=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        release_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(pressure_release, 21)))
        trapped_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(pressure_trapped, 21)))
        low_lock_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(low_lock, 21)))
        absorption_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(absorption, 21)))
        panic_extremum=np.sqrt(release_shock*trapped_shock)
        absorption_anchor=np.sqrt(low_lock_shock*absorption_shock)
        final_score=(panic_extremum*absorption_anchor).clip(0,1).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_release_index_D':pressure_release,'pressure_trapped_D':pressure_trapped,'intraday_low_lock_ratio_D':low_lock,'absorption_energy_D':absorption},calc_nodes={'release_shock':release_shock,'trapped_shock':trapped_shock,'absorption_shock':absorption_shock,'panic_extremum':panic_extremum,'absorption_anchor':absorption_anchor},final_result=final_score)
        return final_score

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V9.0.0 · 突破爆发全息游资版】
        完全剥离对微积分特征的契约强约束。
        """
        method_name="_calculate_breakout_acceleration"
        required_signals=['breakout_quality_score_D','industry_strength_rank_D','net_mf_amount_D','flow_consistency_D','tick_abnormal_volume_ratio_D','uptrend_strength_D','T1_PREMIUM_EXPECTATION_D','HM_COORDINATED_ATTACK_D','breakout_penalty_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        breakout=self._get_safe_series(df,'breakout_quality_score_D',method_name=method_name)
        industry=self._get_safe_series(df,'industry_strength_rank_D',method_name=method_name)
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        consistency=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        abnormal_vol=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        uptrend=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        t1_premium=self._get_safe_series(df,'T1_PREMIUM_EXPECTATION_D',method_name=method_name)
        hm_attack=self._get_safe_series(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        penalty=self._get_safe_series(df,'breakout_penalty_score_D',method_name=method_name)
        kinematics_brk=self._get_kinematic_tensor(df,'breakout_quality_score_D',13,method_name)
        kinematics_mf=self._get_kinematic_tensor(df,'net_mf_amount_D',13,method_name)
        breakout_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(breakout,21)))
        mf_shock=np.tanh(self._apply_hab_shock(net_mf,34))
        mf_power=np.sign(mf_shock)*(np.abs(mf_shock)**1.5)
        ind_norm=0.5*(1.0+np.tanh(self._apply_hab_shock(industry,55)))
        abnorm_norm=0.5*(1.0+np.tanh(self._apply_hab_shock(abnormal_vol,21)))
        cons_norm=0.5*(1.0+np.tanh(self._apply_hab_shock(consistency,21)))
        uptrend_norm=0.5*(1.0+np.tanh(self._apply_hab_shock(uptrend,21)))
        t1_shock=np.tanh(self._apply_hab_shock(t1_premium,13))
        hm_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(hm_attack,21)))
        penalty_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(penalty,13)))
        alpha_multiplier=1.0+(t1_shock.clip(lower=0)*0.6+hm_shock*0.4)
        base_tensor=breakout_shock*ind_norm*(1.0+mf_power.clip(lower=0))*(1.0-penalty_shock.clip(lower=0))
        catalyst=(cons_norm*abnorm_norm*uptrend_norm)**1.5
        raw_score=base_tensor*catalyst*(1.0+kinematics_brk.clip(lower=0)+kinematics_mf.clip(lower=0))*alpha_multiplier
        final_score = np.sign(raw_score) * (np.abs(raw_score) ** 1.5)
        final_score=np.tanh(final_score).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'breakout_quality_score_D':breakout,'T1_PREMIUM_EXPECTATION_D':t1_premium,'HM_COORDINATED_ATTACK_D':hm_attack},calc_nodes={'mf_power':mf_power,'catalyst':catalyst,'alpha_multiplier':alpha_multiplier,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 资金流吸筹质变 (黄金坑与跳空动能版)】
        挂载 GAP_MOMENTUM_STRENGTH_D 与 STATE_GOLDEN_PIT_D。
        """
        method_name = "_calculate_fund_flow_accumulation_inflection"
        required_signals = ['accumulation_signal_score_D', 'net_mf_amount_D', 'flow_efficiency_D', 'tick_large_order_net_D', 'intraday_accumulation_confidence_D', 'GAP_MOMENTUM_STRENGTH_D', 'STATE_GOLDEN_PIT_D']
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        acc_score = self._get_safe_series(df, 'accumulation_signal_score_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        flow_eff = self._get_safe_series(df, 'flow_efficiency_D', method_name=method_name)
        large_net = self._get_safe_series(df, 'tick_large_order_net_D', method_name=method_name)
        intra_acc = self._get_safe_series(df, 'intraday_accumulation_confidence_D', method_name=method_name)
        gap_momentum = self._get_safe_series(df, 'GAP_MOMENTUM_STRENGTH_D', method_name=method_name)
        golden_pit = self._get_safe_series(df, 'STATE_GOLDEN_PIT_D', method_name=method_name).clip(0, 1)
        kinematics_mf = self._get_kinematic_tensor(df, 'net_mf_amount_D', 21, method_name)
        kinematics_acc = self._get_kinematic_tensor(df, 'accumulation_signal_score_D', 21, method_name)
        kinematics_gap = self._get_kinematic_tensor(df, 'GAP_MOMENTUM_STRENGTH_D', 5, method_name)
        acc_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(acc_score, 55)))
        mf_shock = np.tanh(self._apply_hab_shock(net_mf, 34))
        eff_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(flow_eff, 21)))
        large_norm = np.tanh(self._apply_hab_shock(large_net, 21))
        intra_acc_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(intra_acc, 21)))
        gap_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(gap_momentum, 13)))
        pit_leverage = 1.0 + golden_pit * 0.5
        ignition_catalyst = (1.0 + gap_shock.clip(lower=0) * (1.0 + kinematics_gap.clip(lower=0)))
        base_ignition = acc_shock * eff_norm * (1.0 + large_norm.clip(lower=0)) * intra_acc_shock * pit_leverage
        synergy_thrust = base_ignition * ignition_catalyst * (1.0 + mf_shock.clip(lower=0) ** 1.5)
        raw_score = synergy_thrust * (1.0 + kinematics_acc.clip(lower=0) + kinematics_mf.clip(lower=0))
        final_score = np.sign(raw_score) * (np.abs(raw_score) ** 1.5)
        final_score = np.tanh(final_score).clip(0, 1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'STATE_GOLDEN_PIT_D': golden_pit, 'GAP_MOMENTUM_STRENGTH_D': gap_momentum}, calc_nodes={'gap_shock': gap_shock, 'pit_leverage': pit_leverage, 'synergy_thrust': synergy_thrust}, final_result=final_score)
        return final_score

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V10.0.0 · 获利压迫与净流对冲 (抛物线神谕极化版)】
        挂载 STATE_PARABOLIC_WARNING_D 状态阵列。高位抛物线崩塌时，赋予派发堰塞湖 1.5 倍毁灭性惩罚杠杆。
        完全剥离对微积分特征的契约强约束。
        """
        method_name = "_calculate_profit_vs_flow_relationship"
        required_signals = ['profit_pressure_D', 'net_mf_amount_D', 'profit_ratio_D', 'flow_consistency_D', 'winner_rate_D', 'intraday_distribution_confidence_D', 'STATE_PARABOLIC_WARNING_D']
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        pressure = self._get_safe_series(df, 'profit_pressure_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        profit_ratio = self._get_safe_series(df, 'profit_ratio_D', method_name=method_name)
        cons = self._get_safe_series(df, 'flow_consistency_D', method_name=method_name)
        winner = self._get_safe_series(df, 'winner_rate_D', method_name=method_name)
        dist_conf = self._get_safe_series(df, 'intraday_distribution_confidence_D', method_name=method_name)
        parabolic_warning = self._get_safe_series(df, 'STATE_PARABOLIC_WARNING_D', method_name=method_name).clip(0, 1)
        kinematics_p = self._get_kinematic_tensor(df, 'profit_pressure_D', 13, method_name)
        pressure_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(pressure, 21)))
        mf_shock = np.tanh(self._apply_hab_shock(net_mf, 34))
        profit_ratio_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(profit_ratio, 21)))
        cons_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(cons, 21)))
        win_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(winner, 55)))
        dist_conf_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(dist_conf, 21)))
        # 状态极化杠杆：抛物线高危状态下，派发压力被直接放大 1.5 倍
        parabolic_leverage = 1.0 + parabolic_warning * 0.5
        pressure_tensor = (pressure_shock * profit_ratio_shock * dist_conf_shock) * (1.0 + kinematics_p.clip(lower=0)) * parabolic_leverage
        support_tensor = (mf_shock.clip(lower=0) * cons_shock) * (1.0 - win_norm * 0.5)
        raw_score = support_tensor - pressure_tensor * 1.5
        final_score = np.sign(raw_score) * (np.abs(raw_score) ** 1.5)
        final_score = np.tanh(final_score).clip(-1, 1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'profit_pressure_D': pressure, 'STATE_PARABOLIC_WARNING_D': parabolic_warning}, calc_nodes={'pressure_tensor': pressure_tensor, 'support_tensor': support_tensor, 'parabolic_leverage': parabolic_leverage}, final_result=final_score)
        return final_score

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V10.0.0 · 个股板块协同 (领军神谕版)】
        挂载 STATE_MARKET_LEADER_D。如果是市场核心龙头的协同，直接给予双倍极化引爆！
        """
        method_name = "_calculate_stock_sector_sync"
        required_signals = ['pct_change_D', 'industry_strength_rank_D', 'net_mf_amount_D', 'flow_consistency_D', 'industry_leader_score_D', 'mid_long_sync_D', 'STATE_MARKET_LEADER_D']
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        pct = self._get_safe_series(df, 'pct_change_D', method_name=method_name)
        rank = self._get_safe_series(df, 'industry_strength_rank_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        cons = self._get_safe_series(df, 'flow_consistency_D', method_name=method_name)
        leader = self._get_safe_series(df, 'industry_leader_score_D', method_name=method_name)
        sync_score = self._get_safe_series(df, 'mid_long_sync_D', method_name=method_name)
        market_leader_state = self._get_safe_series(df, 'STATE_MARKET_LEADER_D', method_name=method_name).clip(0, 1)
        kinematics_rank = self._get_kinematic_tensor(df, 'industry_strength_rank_D', 5, method_name)
        stock_shock = np.tanh(self._apply_hab_shock(pct, 13))
        sector_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(rank, 34)))
        mf_norm = np.tanh(self._apply_hab_shock(net_mf, 21))
        cons_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(cons, 21)))
        leader_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(leader, 21)))
        sync_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(sync_score, 21)))
        sector_tensor = sector_shock * (1.0 + kinematics_rank) * (1.0 + leader_shock)
        flow_tensor = mf_norm * cons_norm * (1.0 + sync_shock)
        flow_synergy = (1.0 + flow_tensor * np.sign(stock_shock)).clip(lower=0.1)
        # 龙头神谕加持
        leader_leverage = 1.0 + market_leader_state
        resonance = stock_shock * sector_tensor * flow_synergy * leader_leverage
        final_score = np.tanh(np.sign(resonance) * (np.abs(resonance) ** 1.5)).clip(-1, 1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'pct_change_D': pct, 'STATE_MARKET_LEADER_D': market_leader_state}, calc_nodes={'flow_tensor': flow_tensor, 'flow_synergy': flow_synergy, 'leader_leverage': leader_leverage, 'resonance': resonance}, final_result=final_score)
        return final_score

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 热门板块退潮防爆版】
        """
        method_name = "_calculate_hot_sector_cooling"
        required_signals = [
            'THEME_HOTNESS_SCORE_D', 'net_mf_amount_D', 'industry_stagnation_score_D', 'outflow_quality_D',
            'SLOPE_13_THEME_HOTNESS_SCORE_D', 'ACCEL_13_THEME_HOTNESS_SCORE_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        hot = self._get_safe_series(df, 'THEME_HOTNESS_SCORE_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        stagnation = self._get_safe_series(df, 'industry_stagnation_score_D', method_name=method_name)
        outflow_q = self._get_safe_series(df, 'outflow_quality_D', method_name=method_name)
        kinematics_hot = self._get_kinematic_tensor(df, 'THEME_HOTNESS_SCORE_D', 13, method_name)
        hot_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(hot, 21)))
        mf_shock = np.tanh(self._apply_hab_shock(net_mf, 13))
        stagnation_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(stagnation, 21)))
        outflow_q_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(outflow_q, 21)))
        outflow_tensor = np.abs(mf_shock.clip(upper=0)) * (1.0 + outflow_q_shock)
        stagnation_boost = 1.0 + stagnation_shock * 2.0
        cooling_resonance = hot_shock * outflow_tensor * stagnation_boost * (1.0 - kinematics_hot.clip(lower=0))
        final_score = np.tanh(cooling_resonance ** 1.5).clip(0, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'THEME_HOTNESS_SCORE_D': hot, 'net_mf_amount_D': net_mf}, calc_nodes={'kinematics_hot': kinematics_hot, 'hot_shock': hot_shock, 'outflow_tensor': outflow_tensor, 'cooling_resonance': cooling_resonance}, final_result=final_score)
        return final_score

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 价资协同双向剥离版】
        完全剥离微积分列强校验，末端绝对零基填补。
        """
        method_name="_calculate_pf_relationship"
        required_signals=['net_mf_amount_D','close_D','price_vs_ma_13_ratio_D','main_force_activity_index_D','flow_momentum_13d_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        close_p=self._get_safe_series(df,'close_D',method_name=method_name)
        price_ma_ratio=self._get_safe_series(df,'price_vs_ma_13_ratio_D',method_name=method_name)
        activity=self._get_safe_series(df,'main_force_activity_index_D',method_name=method_name)
        flow_mom=self._get_safe_series(df,'flow_momentum_13d_D',method_name=method_name)
        kinematics_mf=self._get_kinematic_tensor(df,'net_mf_amount_D',13,method_name)
        mf_shock=np.tanh(self._apply_hab_shock(net_mf,34))
        activity_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(activity,21)))
        flow_mom_shock=np.tanh(self._apply_hab_shock(flow_mom,13))
        price_ma_shock=np.tanh(self._apply_hab_shock(price_ma_ratio,21))
        momentum_p=np.tanh(self._apply_hab_shock(close_p.diff(1).fillna(0),13))
        thrust_f=mf_shock*activity_shock*(1.0+np.abs(kinematics_mf))*(1.0+flow_mom_shock*0.5)
        synergy_amplifier=(1.0+thrust_f*np.sign(momentum_p)).clip(lower=0.1)
        raw_score=momentum_p*synergy_amplifier*(1.0+np.abs(price_ma_shock)*0.5)
        relationship_score=pd.Series(np.sign(raw_score)*(np.abs(raw_score)**1.5),index=df_index).clip(-1,1)
        meta_score=self._perform_meta_analysis_on_score(relationship_score.fillna(0.0),config,df,df_index).fillna(0.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':net_mf,'close_D':close_p},calc_nodes={'momentum_p':momentum_p,'thrust_f':thrust_f,'synergy_amplifier':synergy_amplifier,'relationship_score':relationship_score},final_result=meta_score)
        return meta_score

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 价筹稳态共振全息版】
        完全剥离微积分列强校验，末端绝对零基填补。
        """
        method_name="_calculate_pc_relationship"
        required_signals=['peak_concentration_D','close_D','chip_convergence_ratio_D','high_position_lock_ratio_90_D','chip_stability_change_5d_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        peak_c=self._get_safe_series(df,'peak_concentration_D',method_name=method_name)
        close_p=self._get_safe_series(df,'close_D',method_name=method_name)
        convergence=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        high_lock=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name)
        stab_change=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        kinematics_pc=self._get_kinematic_tensor(df,'peak_concentration_D',13,method_name)
        pc_shock=np.tanh(self._apply_hab_shock(peak_c,34))
        conv_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(convergence,21)))
        lock_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(high_lock,21)))
        stab_change_shock=np.tanh(self._apply_hab_shock(stab_change,13))
        vac_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(vac,34)))
        momentum_p=np.tanh(self._apply_hab_shock(close_p.diff(1).fillna(0),13))
        thrust_c=pc_shock*conv_shock*(1.0+lock_shock)*(1.0+stab_change_shock.clip(lower=0))*(1.0+vac_shock)*(1.0+np.abs(kinematics_pc))
        synergy_amplifier=(1.0+thrust_c*np.sign(momentum_p)).clip(lower=0.1)
        raw_score=momentum_p*synergy_amplifier
        relationship_score=pd.Series(np.sign(raw_score)*(np.abs(raw_score)**1.5),index=df_index).clip(-1,1)
        meta_score=self._perform_meta_analysis_on_score(relationship_score.fillna(0.0),config,df,df_index).fillna(0.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_concentration_D':peak_c,'high_position_lock_ratio_90_D':high_lock},calc_nodes={'momentum_p':momentum_p,'thrust_c':thrust_c,'synergy_amplifier':synergy_amplifier,'relationship_score':relationship_score},final_result=meta_score)
        return meta_score

    def _calculate_ff_vs_structure_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V10.0.0 · 资金结构双极惩罚 (极化稳态神谕版)】
        挂载 STATE_ROBUST_TREND_D。稳态趋势中，结构稳固度极难被大单突发噪音破坏。
        """
        method_name="_calculate_ff_vs_structure_relationship"
        required_signals=['uptrend_strength_D','flow_consistency_D','ma_arrangement_status_D','chip_structure_state_D','industry_stagnation_score_D','large_order_anomaly_D', 'STATE_ROBUST_TREND_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        struct=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        ma_status=self._get_safe_series(df,'ma_arrangement_status_D',method_name=method_name)
        chip_struct=self._get_safe_series(df,'chip_structure_state_D',method_name=method_name)
        stagnation=self._get_safe_series(df,'industry_stagnation_score_D',method_name=method_name)
        anomaly=self._get_safe_series(df,'large_order_anomaly_D',method_name=method_name)
        robust_trend = self._get_safe_series(df, 'STATE_ROBUST_TREND_D', method_name=method_name).clip(0, 1)
        kinematics_struct=self._get_kinematic_tensor(df,'uptrend_strength_D',13,method_name)
        struct_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(struct,34)))
        cons_norm=0.5*(1.0+np.tanh(self._apply_hab_shock(cons,21)))
        stag_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(stagnation,21)))
        anomaly_shock=np.tanh(self._apply_hab_shock(anomaly,13))
        base_divergence=self._calculate_instantaneous_relationship(df,config)
        # 稳态趋势下，异常扰动的惩罚被大幅削弱
        anomaly_penalty = anomaly_shock.clip(lower=0) * 0.5 * (1.0 - robust_trend * 0.8)
        structural_penalty=(1.0-stag_shock*0.5)*(1.0 - anomaly_penalty)
        amplifier=1.0+(struct_shock*cons_norm*(1.0+ma_status*0.5+chip_struct*0.5)*(1.0+np.abs(kinematics_struct)))
        amplifier=amplifier*structural_penalty
        final_score = np.sign(base_divergence * amplifier) * (np.abs(base_divergence * amplifier) ** 1.5)
        final_score = np.tanh(final_score).clip(-1, 1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'uptrend_strength_D':struct, 'STATE_ROBUST_TREND_D': robust_trend},calc_nodes={'struct_shock':struct_shock,'structural_penalty':structural_penalty,'amplifier':amplifier},final_result=final_score)
        return final_score

    def _perform_meta_analysis_on_score(self, relationship_score: pd.Series, config: Dict, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """
        【V6.0.0 · 元动力学全域防线版】
        结合 HAB 冲击测算推演信号位移与动量的绝对极值，执行 Power Law 非线性增益放大。
        贴上绝对零基填充封条，切断 NaN 向上层认知网络传播。
        """
        signal_name = config.get('name')
        relationship_displacement = relationship_score.diff(self.meta_window).fillna(0.0)
        relationship_momentum = relationship_displacement.diff(1).fillna(0.0)
        bipolar_displacement = np.tanh(self._apply_hab_shock(relationship_displacement, window=self.meta_window*2))
        bipolar_momentum = np.tanh(self._apply_hab_shock(relationship_momentum, window=13))
        instant_score_normalized = (relationship_score + 1.0) / 2.0
        weight_momentum = (1.0 - instant_score_normalized).clip(0, 1)
        weight_displacement = 1.0 - weight_momentum
        meta_score = (bipolar_displacement * weight_displacement + bipolar_momentum * weight_momentum)
        meta_score = np.sign(meta_score) * (np.abs(meta_score) ** 1.2)
        if config.get('diagnosis_mode', 'meta_analysis') == 'gated_meta_analysis':
            gate_config = config.get('gate_condition', {})
            if gate_config.get('type') == 'price_vs_ma':
                ma_period = gate_config.get('ma_period', 5)
                ma_col = f'EMA_{ma_period}_D'
                if ma_col in df.columns and 'close_D' in df.columns:
                    gate_is_open = (df['close_D'] < df[ma_col]).astype(float)
                    meta_score = meta_score * gate_is_open
                    
        scoring_mode = self.score_type_map.get(signal_name, {}).get('scoring_mode', 'unipolar')
        if scoring_mode == 'unipolar':
            meta_score = meta_score.clip(lower=0)
        return meta_score.clip(-1, 1).fillna(0.0).astype(np.float32)

    def _calculate_dyn_vs_chip_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 动能筹码异向极化版】
        完全剥离微积分列强校验，末端绝对零基填补。
        """
        method_name="_calculate_dyn_vs_chip_relationship"
        required_signals=['ROC_13_D','winner_rate_D','profit_ratio_D','chip_mean_D','chip_kurtosis_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        roc=self._get_safe_series(df,'ROC_13_D',method_name=method_name)
        win=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        profit=self._get_safe_series(df,'profit_ratio_D',method_name=method_name)
        chip_mean=self._get_safe_series(df,'chip_mean_D',method_name=method_name)
        kurtosis=self._get_safe_series(df,'chip_kurtosis_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        kinematics_roc=self._get_kinematic_tensor(df,'ROC_13_D',13,method_name)
        base_consensus=self._calculate_instantaneous_relationship(df,config)
        profit_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(profit,55)))
        win_shock=np.tanh(self._apply_hab_shock(win,21))
        mean_shock=np.tanh(self._apply_hab_shock(chip_mean,13))
        kurtosis_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(kurtosis,21)))
        vac_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(vac,21)))
        distribution_pressure=1.0+(profit_shock*(1.0+kurtosis_shock)*(1.0+vac_shock))*np.abs(mean_shock)*(1.0+np.abs(kinematics_roc))
        final_score=base_consensus.where(base_consensus>=0,base_consensus*distribution_pressure*(1.0+win_shock.clip(lower=0)*0.5))
        final_score=pd.Series(np.sign(final_score)*(np.abs(final_score)**1.5),index=df_index)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc,'profit_ratio_D':profit},calc_nodes={'profit_shock':profit_shock,'win_shock':win_shock,'distribution_pressure':distribution_pressure},final_result=final_score)
        return final_score

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 瞬时张量全域防线版】
        重构底层A与B特征张量对冲，支持HAB存量意识与Power Law指数级共振。
        强制防丢：遇到任何运算边界全部收敛至 0.0。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        df_index = df.index
        relationship_type = config.get('relationship_type', 'consensus')
        def get_signal_series(signal_name: str, source_type: str) -> Optional[pd.Series]:
            if source_type == 'atomic_states':
                return self.strategy.atomic_states.get(signal_name)
            try:
                return self._get_safe_series(df, signal_name, np.nan, "_calculate_instantaneous_relationship")
            except ValueError:
                return None
                
        signal_a = get_signal_series(signal_a_name, config.get('source_A', 'df'))
        signal_b = get_signal_series(signal_b_name, config.get('source_B', 'df'))
        if signal_a is None or signal_b is None:
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        change_a = signal_a.diff(1).fillna(0.0) if config.get('change_type_A', 'pct') == 'diff' else ta.percent_return(signal_a, length=1).fillna(0.0)
        change_b = signal_b.diff(1).fillna(0.0) if config.get('change_type_B', 'pct') == 'diff' else ta.percent_return(signal_b, length=1).fillna(0.0)
        momentum_a = np.tanh(self._apply_hab_shock(change_a, 13))
        thrust_b = np.tanh(self._apply_hab_shock(change_b, 13))
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        if relationship_type == 'divergence':
            relationship_score = (signal_b_factor_k * thrust_b - momentum_a) / (signal_b_factor_k + 1.0)
        else:
            force_vector_sum = momentum_a + signal_b_factor_k * thrust_b
            magnitude = (np.abs(momentum_a) * np.abs(thrust_b)) ** 0.5
            relationship_score = np.sign(force_vector_sum) * magnitude
        relationship_score = np.sign(relationship_score) * (np.abs(relationship_score) ** 1.5)
        return np.tanh(relationship_score).clip(-1, 1).fillna(0.0).astype(np.float32)

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V10.0.0 · 洗盘反包非线性防爆 (圆弧底神谕版)】
        挂载 STATE_ROUNDING_BOTTOM_D。若大形态已形成圆弧底，洗盘承接的爆发力将被彻底激活。
        """
        method_name="_calculate_process_wash_out_rebound"
        required_signals=['shakeout_score_D','intraday_distribution_confidence_D','pressure_trapped_D','CLOSING_STRENGTH_D','intraday_trough_filling_degree_D','stealth_flow_ratio_D','absorption_energy_D','STATE_ROUNDING_BOTTOM_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        shakeout=self._get_safe_series(df,'shakeout_score_D',method_name=method_name)
        dist_conf=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        panic=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name)
        trough_fill=self._get_safe_series(df,'intraday_trough_filling_degree_D',method_name=method_name)
        stealth=self._get_safe_series(df,'stealth_flow_ratio_D',method_name=method_name)
        absorption=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        rounding_bottom = self._get_safe_series(df, 'STATE_ROUNDING_BOTTOM_D', method_name=method_name).clip(0, 1)
        kinematics_shakeout=self._get_kinematic_tensor(df,'shakeout_score_D',13,method_name)
        shakeout_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(shakeout,21)))
        dist_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(dist_conf,13)))
        panic_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(panic,21)))
        stealth_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(stealth,21)))
        trough_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(trough_fill,21)))
        absorption_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(absorption,21)))
        closing_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(closing,21)))
        # 状态神谕杠杆：圆弧底右侧的洗盘，反身向上的概率呈几何级暴增
        rounding_leverage = 1.0 + rounding_bottom * 0.5
        washout_env=(shakeout_shock*dist_shock*panic_shock)*(1.0+stealth_shock)
        rebound_intent=trough_shock*absorption_shock * rounding_leverage
        raw_score=washout_env*(rebound_intent**1.5)*closing_shock
        final_score=np.tanh(raw_score*2.0*(1.0+kinematics_shakeout.clip(lower=0))).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'shakeout_score_D':shakeout,'STATE_ROUNDING_BOTTOM_D':rounding_bottom},calc_nodes={'washout_env':washout_env,'rounding_leverage':rounding_leverage,'rebound_intent':rebound_intent,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_fusion_trend_exhaustion_syndrome(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 极限坍塌与红军防线版】
        诊断抛物线加速后的衰竭综合征。引入红军对抗防爆机制：如果顶部存在强力的游资协同，强制熔断衰竭信号，防止卖飞主升浪。
        """
        method_name="_calculate_fusion_trend_exhaustion_syndrome"
        required_signals=['STATE_PARABOLIC_WARNING_D','STATE_EMOTIONAL_EXTREME_D','PRICE_ENTROPY_D','profit_pressure_D','HM_COORDINATED_ATTACK_D','intraday_distribution_confidence_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        parabolic=self._get_safe_series(df,'STATE_PARABOLIC_WARNING_D',method_name=method_name).clip(0,1)
        emotional=self._get_safe_series(df,'STATE_EMOTIONAL_EXTREME_D',method_name=method_name).clip(0,1)
        entropy=self._get_safe_series(df,'PRICE_ENTROPY_D',method_name=method_name)
        pressure=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        hm_attack=self._get_safe_series(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        dist_conf=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        entropy_shock=np.tanh(self._apply_hab_shock(entropy,13)).clip(lower=0)
        pressure_shock=np.tanh(self._apply_hab_shock(pressure,21)).clip(lower=0)
        dist_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(dist_conf,13)))
        hm_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(hm_attack,13)))
        state_leverage=1.0+parabolic*1.5+emotional*1.0
        physical_gate=dist_shock*0.6+pressure_shock*0.4
        veto_suppression=(1.0-hm_shock*0.9).clip(lower=0.1)
        raw_score=state_leverage*physical_gate*(1.0+entropy_shock)*veto_suppression
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'STATE_PARABOLIC_WARNING_D':parabolic,'profit_pressure_D':pressure},calc_nodes={'state_leverage':state_leverage,'physical_gate':physical_gate,'veto_suppression':veto_suppression,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_dyn_vs_chip_decay_rise(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1.0 · 张量阻尼看涨版】
        衡量力学下行动能减弱与筹码套牢风险消散的同步反转趋势。寻找强阻尼吸能带来的力学拐点反抽。
        """
        method_name="_calculate_dyn_vs_chip_decay_rise"
        required_signals=['downtrend_strength_D','pressure_trapped_D','absorption_energy_D','chip_kurtosis_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        downtrend=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        pressure=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        absorption=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        kurtosis=self._get_safe_series(df,'chip_kurtosis_D',method_name=method_name)
        kinematics_down=self._get_kinematic_tensor(df,'downtrend_strength_D',13,method_name)
        kinetic_dissipation=np.abs(np.clip(kinematics_down,-1.0,0.0))
        downtrend_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(downtrend,21)))
        pressure_decay_shock=-np.tanh(self._apply_hab_shock(pressure,21))
        absorption_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(absorption,21)))
        kurtosis_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(kurtosis,21)))
        chip_relief=(pressure_decay_shock.clip(lower=0)+absorption_shock)*0.5
        damping_resonance=downtrend_shock*kinetic_dissipation*kurtosis_shock*chip_relief
        final_score=np.tanh(damping_resonance**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'downtrend_strength_D':downtrend,'pressure_trapped_D':pressure},calc_nodes={'kinetic_dissipation':kinetic_dissipation,'chip_relief':chip_relief,'damping_resonance':damping_resonance},final_result=final_score)
        return final_score

    def _calculate_smart_money_ignition(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 游资协同核爆版】
        诊断游资与主力聪明钱的协同点火。融合游资协同攻击、T+1溢价预期。
        受红军底层逻辑约束：绝对严禁在非突破的空头趋势中触发。
        """
        method_name="_calculate_smart_money_ignition"
        required_signals=['HM_COORDINATED_ATTACK_D','T1_PREMIUM_EXPECTATION_D','IS_MARKET_LEADER_D','flow_acceleration_intraday_D','uptrend_strength_D','STATE_BREAKOUT_CONFIRMED_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hm_attack=self._get_safe_series(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        t1_premium=self._get_safe_series(df,'T1_PREMIUM_EXPECTATION_D',method_name=method_name)
        is_leader=self._get_safe_series(df,'IS_MARKET_LEADER_D',method_name=method_name).clip(0,1)
        flow_accel=self._get_safe_series(df,'flow_acceleration_intraday_D',method_name=method_name)
        uptrend=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        breakout=self._get_safe_series(df,'STATE_BREAKOUT_CONFIRMED_D',method_name=method_name).clip(0,1)
        hm_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(hm_attack,13)))
        t1_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(t1_premium,21)))
        accel_shock=np.tanh(self._apply_hab_shock(flow_accel,13)).clip(lower=0)
        uptrend_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(uptrend,21)))
        trend_gate=(uptrend_shock+breakout).clip(0,1)
        leader_leverage=1.0+is_leader*1.5
        attack_kinetic=hm_shock*t1_shock*(1.0+accel_shock)
        raw_score=attack_kinetic*leader_leverage*trend_gate
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_COORDINATED_ATTACK_D':hm_attack,'IS_MARKET_LEADER_D':is_leader},calc_nodes={'trend_gate':trend_gate,'leader_leverage':leader_leverage,'attack_kinetic':attack_kinetic,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_vpa_mf_coherence_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 量价主力物理相干版】
        融合均线相干共振、主力调整后的VPA效率及动能加速度，寻找极高胜率的多周期收敛共振加速点。
        """
        method_name="_calculate_vpa_mf_coherence_resonance"
        required_signals=['MA_COHERENCE_RESONANCE_D','VPA_MF_ADJUSTED_EFF_D','MA_ACCELERATION_EMA_55_D','VPA_ACCELERATION_13D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        ma_coh=self._get_safe_series(df,'MA_COHERENCE_RESONANCE_D',method_name=method_name)
        vpa_mf_eff=self._get_safe_series(df,'VPA_MF_ADJUSTED_EFF_D',method_name=method_name)
        ma_accel=self._get_safe_series(df,'MA_ACCELERATION_EMA_55_D',method_name=method_name)
        vpa_accel=self._get_safe_series(df,'VPA_ACCELERATION_13D',method_name=method_name)
        ma_coh_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(ma_coh,34)))
        vpa_eff_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(vpa_mf_eff,21)))
        ma_accel_shock=np.tanh(self._apply_hab_shock(ma_accel,21)).clip(lower=0)
        vpa_accel_shock=np.tanh(self._apply_hab_shock(vpa_accel,13)).clip(lower=0)
        base_resonance=ma_coh_shock*vpa_eff_shock
        kinetic_catalyst=1.0+ma_accel_shock+vpa_accel_shock
        raw_score=base_resonance*kinetic_catalyst
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_COHERENCE_RESONANCE_D':ma_coh,'VPA_MF_ADJUSTED_EFF_D':vpa_mf_eff},calc_nodes={'base_resonance':base_resonance,'kinetic_catalyst':kinetic_catalyst,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_institutional_sweep(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 机构超大单扫货核爆引擎】
        量化买入特大单(buy_elg)与日内微观筹码转移效率的共振。
        这是量化顶级机构(如国家队/大公募)无视短期波动的左侧/右侧建仓绝对底座信号。
        """
        method_name="_calculate_institutional_sweep"
        required_signals=['buy_elg_amount_D','amount_D','tick_chip_transfer_efficiency_D','flow_consistency_D','net_mf_amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        buy_elg=self._get_safe_series(df,'buy_elg_amount_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        transfer_eff=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        elg_ratio=(buy_elg/amount).fillna(0.0)
        elg_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(elg_ratio,21)))
        transfer_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(transfer_eff,21)))
        cons_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(cons,21)))
        mf_shock=np.tanh(self._apply_hab_shock(net_mf,21))
        kinematics_mf=self._get_kinematic_tensor(df,'net_mf_amount_D',13,method_name)
        base_sweep=elg_shock*transfer_shock*cons_shock*(1.0+mf_shock.clip(lower=0))
        raw_score=base_sweep*(1.0+kinematics_mf.clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'buy_elg_amount_D':buy_elg,'amount_D':amount},calc_nodes={'elg_ratio':elg_ratio,'elg_shock':elg_shock,'transfer_shock':transfer_shock,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_hf_algo_manipulation_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 高频算法诱骗崩塌惩罚引擎】
        量化视角下，主力撤退与诱多绝不是挂单明卖，而是通过算法拆单(Spoofing)体现为高频流的严重极化。
        融合流偏度、峰度与价流背离，对冰山派发实施一剑封喉。
        """
        method_name="_calculate_hf_algo_manipulation_risk"
        required_signals=['high_freq_flow_skewness_D','high_freq_flow_kurtosis_D','large_order_anomaly_D','price_flow_divergence_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hf_skew=self._get_safe_series(df,'high_freq_flow_skewness_D',method_name=method_name)
        hf_kurt=self._get_safe_series(df,'high_freq_flow_kurtosis_D',method_name=method_name)
        anomaly=self._get_safe_series(df,'large_order_anomaly_D',method_name=method_name)
        divergence=self._get_safe_series(df,'price_flow_divergence_D',method_name=method_name)
        skew_shock=np.tanh(self._apply_hab_shock(hf_skew,21)).abs()
        kurt_shock=np.tanh(self._apply_hab_shock(hf_kurt,21)).clip(lower=0)
        anomaly_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(anomaly,13)))
        div_shock=np.tanh(self._apply_hab_shock(divergence,21)).clip(lower=0)
        risk_tensor=skew_shock*(1.0+kurt_shock)*anomaly_shock*(1.0+div_shock)
        final_score=np.tanh(risk_tensor**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'high_freq_flow_skewness_D':hf_skew,'large_order_anomaly_D':anomaly},calc_nodes={'skew_shock':skew_shock,'kurt_shock':kurt_shock,'anomaly_shock':anomaly_shock,'risk_tensor':risk_tensor},final_result=final_score)
        return final_score

    def _calculate_ma_rubber_band_reversal(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 均线张力极值反噬引擎】
        物理学张力在金融市场的终极体现。深度捕捉价格暴涨/暴跌后积累的均值回归胡克拉力。
        利用ADX趋势过滤防止单边被套。负分为向上偏离过大带来的高空跳水预警；正分为向下偏离极值的暴力反抽。
        """
        method_name="_calculate_ma_rubber_band_reversal"
        required_signals=['MA_RUBBER_BAND_EXTENSION_D','MA_POTENTIAL_TENSION_INDEX_D','ADX_14_D','profit_pressure_D','pressure_trapped_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        rubber_ext=self._get_safe_series(df,'MA_RUBBER_BAND_EXTENSION_D',method_name=method_name)
        tension=self._get_safe_series(df,'MA_POTENTIAL_TENSION_INDEX_D',method_name=method_name)
        adx=self._get_safe_series(df,'ADX_14_D',method_name=method_name)
        profit_p=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        trapped_p=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        rubber_shock=np.tanh(self._apply_hab_shock(rubber_ext,21))
        tension_shock=np.tanh(self._apply_hab_shock(tension,21))
        profit_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(profit_p,21)))
        trapped_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(trapped_p,21)))
        trend_suppression=1.0-np.tanh(np.maximum(adx-35.0,0.0)/15.0)
        top_reversal_force=(rubber_shock.clip(lower=0))*tension_shock.clip(lower=0)*profit_shock
        bottom_reversal_force=(rubber_shock.clip(upper=0).abs())*tension_shock.clip(lower=0)*trapped_shock
        raw_score=(bottom_reversal_force-top_reversal_force)*trend_suppression
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_RUBBER_BAND_EXTENSION_D':rubber_ext,'ADX_14_D':adx},calc_nodes={'top_reversal_force':top_reversal_force,'bottom_reversal_force':bottom_reversal_force,'trend_suppression':trend_suppression},final_result=final_score)
        return final_score

    def _calculate_geometric_trend_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 几何流形趋势共振引擎】
        将趋势视为高维刚体。基于线性回归R2、斜率与圆弧曲率，刻画趋势的物理刚性与加速度张力。
        惩罚R2极低的无序波动，奖励曲率向上的二次加速发散。
        """
        method_name="_calculate_geometric_trend_resonance"
        required_signals=['GEOM_REG_R2_D','GEOM_REG_SLOPE_D','GEOM_ARC_CURVATURE_D','GEOM_CHANNEL_POS_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        r2=self._get_safe_series(df,'GEOM_REG_R2_D',method_name=method_name)
        slope=self._get_safe_series(df,'GEOM_REG_SLOPE_D',method_name=method_name)
        curvature=self._get_safe_series(df,'GEOM_ARC_CURVATURE_D',method_name=method_name)
        channel_pos=self._get_safe_series(df,'GEOM_CHANNEL_POS_D',method_name=method_name)
        r2_gate=np.tanh(self._apply_hab_shock(r2,21)).clip(lower=0.1)
        slope_shock=np.tanh(self._apply_hab_shock(slope,21))
        curvature_shock=np.tanh(self._apply_hab_shock(curvature,21))
        channel_norm=(channel_pos-0.5)*2.0
        rigid_tensor=slope_shock*r2_gate
        manifold_dynamics=curvature_shock-channel_norm*0.3
        raw_score=rigid_tensor*(1.0+manifold_dynamics.clip(lower=0))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GEOM_REG_R2_D':r2,'GEOM_ARC_CURVATURE_D':curvature},calc_nodes={'rigid_tensor':rigid_tensor,'manifold_dynamics':manifold_dynamics,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_mtf_fractal_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 多时间框架分形共振引擎】
        量化日线、周线、月线在特定时刻的完美同步对齐。
        结合价格分形维数（趋势平滑度）与上升趋势延续概率，过滤掉日内高频噪音，捕捉宏观级别的共振主升浪。
        """
        method_name="_calculate_mtf_fractal_resonance"
        required_signals=['daily_weekly_sync_D','daily_monthly_sync_D','PRICE_FRACTAL_DIM_D','uptrend_continuation_prob_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        dw_sync=self._get_safe_series(df,'daily_weekly_sync_D',method_name=method_name)
        dm_sync=self._get_safe_series(df,'daily_monthly_sync_D',method_name=method_name)
        fractal_dim=self._get_safe_series(df,'PRICE_FRACTAL_DIM_D',method_name=method_name)
        prob=self._get_safe_series(df,'uptrend_continuation_prob_D',method_name=method_name)
        sync_tensor=0.5*(1.0+np.tanh(self._apply_hab_shock(dw_sync+dm_sync,21)))
        # 分形维数越低越接近单向流，噪音越小，趋势越强
        fractal_smoothness=1.0-np.tanh(self._apply_hab_shock(fractal_dim,34)).clip(lower=0)
        prob_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(prob,21)))
        raw_score=sync_tensor*fractal_smoothness*prob_shock
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'daily_weekly_sync_D':dw_sync,'PRICE_FRACTAL_DIM_D':fractal_dim},calc_nodes={'sync_tensor':sync_tensor,'fractal_smoothness':fractal_smoothness,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_intraday_siege_exhaustion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 日内攻城拔寨衰竭引擎】
        量化日内多空在阻力/支撑位上的反复肉搏与消耗战（Siege Warfare）。
        多次高频测试阻力后若能强势收盘且VWAP向上，代表空头尸体垫脚，城门告破（看涨突破）。
        多次测试支撑但收盘极弱，代表多头承接溃散（看跌崩塌）。
        """
        method_name="_calculate_intraday_siege_exhaustion"
        required_signals=['intraday_resistance_test_count_D','intraday_support_test_count_D','CLOSING_STRENGTH_D','vwap_deviation_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        res_tests=self._get_safe_series(df,'intraday_resistance_test_count_D',method_name=method_name)
        sup_tests=self._get_safe_series(df,'intraday_support_test_count_D',method_name=method_name)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name)
        vwap_dev=self._get_safe_series(df,'vwap_deviation_D',method_name=method_name)
        res_shock=np.tanh(res_tests/3.0) 
        sup_shock=np.tanh(sup_tests/3.0)
        closing_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(closing,21)))
        vwap_shock=np.tanh(self._apply_hab_shock(vwap_dev,21))
        breakout_force=res_shock*closing_shock*(1.0+vwap_shock.clip(lower=0))
        breakdown_force=sup_shock*(1.0-closing_shock)*(1.0+vwap_shock.clip(upper=0).abs())
        raw_score=breakout_force-breakdown_force
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'intraday_resistance_test_count_D':res_tests,'CLOSING_STRENGTH_D':closing},calc_nodes={'breakout_force':breakout_force,'breakdown_force':breakdown_force,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_overnight_intraday_tearing(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 隔夜跳空与日内动能撕裂引擎】
        量化隔夜情绪意志(Gap)与日内真实买盘(OCH)的致命背离。
        高开低走(跳空极高但日内收盘极弱的诱多派发)将被施以毁灭性负面惩罚；
        稳健高开高走(跳空且日内动能极强)将获得巨额的顺势奖励。
        """
        method_name="_calculate_overnight_intraday_tearing"
        required_signals=['GAP_MOMENTUM_STRENGTH_D','OCH_ACCELERATION_D','CLOSING_STRENGTH_D','intraday_price_range_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        gap=self._get_safe_series(df,'GAP_MOMENTUM_STRENGTH_D',method_name=method_name)
        och=self._get_safe_series(df,'OCH_ACCELERATION_D',method_name=method_name)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name)
        range_ratio=self._get_safe_series(df,'intraday_price_range_ratio_D',method_name=method_name)
        gap_shock=np.tanh(self._apply_hab_shock(gap,13))
        och_shock=np.tanh(self._apply_hab_shock(och,13))
        closing_norm=0.5*(1.0+np.tanh(self._apply_hab_shock(closing,21)))
        range_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(range_ratio,21)))
        # 撕裂防线：Gap为正但日内极差(closing_norm < 0.5且och弱)则向量翻负
        tearing_vector=gap_shock*(closing_norm*2.0 - 1.0 + och_shock)
        # 波动率越高，撕裂或共振的杠杆效应越大
        leverage=1.0+(range_shock*np.abs(closing_norm-0.5)*2.0)
        raw_score=tearing_vector*leverage
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GAP_MOMENTUM_STRENGTH_D':gap,'CLOSING_STRENGTH_D':closing},calc_nodes={'gap_shock':gap_shock,'tearing_vector':tearing_vector,'leverage':leverage,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_chip_center_kinematics(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.0.0 · 筹码重心迁徙动力学引擎】
        透视筹码主峰在拉升/下跌途中的迁移速率与震荡幅度。
        股价飙升但重心低波段锁定 = 强庄锁仓不卖 (强势看涨正分)；
        重心极速上移且剧烈波动、高换手 = 主力高位大规模筹码大搬家派发给散户 (极度看跌负分)。
        """
        method_name="_calculate_chip_center_kinematics"
        required_signals=['peak_migration_speed_5d_D','intraday_cost_center_volatility_D','price_to_weight_avg_ratio_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        migration=self._get_safe_series(df,'peak_migration_speed_5d_D',method_name=method_name)
        cost_vol=self._get_safe_series(df,'intraday_cost_center_volatility_D',method_name=method_name)
        price_to_cost=self._get_safe_series(df,'price_to_weight_avg_ratio_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        mig_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(migration,21)))
        vol_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(cost_vol,13)))
        dev_shock=np.tanh(self._apply_hab_shock(price_to_cost,21))
        to_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(turnover,13)))
        # 派发动能：高速度 + 高震荡 + 高换手
        distribution_kinetic=mig_shock*vol_shock*to_shock
        # 锁仓动能：低速度(1-mig) + 高价格偏离
        lock_kinetic=(1.0-mig_shock)*dev_shock.clip(lower=0)
        # 高派发动能结合高价格偏离，产生毁灭性的向下撕扯力
        raw_score=lock_kinetic - (distribution_kinetic*(1.0+dev_shock.clip(lower=0)))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_migration_speed_5d_D':migration,'price_to_weight_avg_ratio_D':price_to_cost},calc_nodes={'distribution_kinetic':distribution_kinetic,'lock_kinetic':lock_kinetic,'raw_score':raw_score},final_result=final_score)
        return final_score





