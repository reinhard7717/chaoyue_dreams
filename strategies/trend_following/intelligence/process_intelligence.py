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
        【V20.0.0 · 蓝图审查官 (生命周期与时间错配终极防线版)】
        更新版本号。将新增的板块生命周期顺风、日内时间非对称陷阱、高位流动性真空轧空与机构结构性清仓四大终极上帝视角引擎加入免检白名单。
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
            'PROCESS_META_OVERNIGHT_INTRADAY_TEARING','PROCESS_META_CHIP_CENTER_KINEMATICS',
            'PROCESS_META_INSTITUTIONAL_SWEEP','PROCESS_META_HF_ALGO_MANIPULATION_RISK',
            'PROCESS_META_MA_RUBBER_BAND_REVERSAL','PROCESS_META_GEOMETRIC_TREND_RESONANCE',
            'PROCESS_META_MA_COMPRESSION_EXPLOSION','PROCESS_META_TOP_TIER_HM_HARVESTING',
            'PROCESS_META_VWAP_MAGNETIC_DIVERGENCE','PROCESS_META_MULTI_PEAK_AVALANCHE_RISK',
            'PROCESS_META_CLOSING_AUCTION_DYNAMICS','PROCESS_META_RETAIL_HERD_CONTRARIAN',
            'PROCESS_META_CHIP_DORMANT_IGNITION','PROCESS_META_INSTITUTIONAL_DUMPING_AVALANCHE',
            'PROCESS_META_VOLATILITY_AWAKENING_BREAKOUT','PROCESS_META_MA_FAN_KINEMATICS_TRACTION',
            'PROCESS_META_CHIP_MA_RESONANCE_SUPPORT','PROCESS_META_SECTOR_LIFECYCLE_TAILWIND',
            'PROCESS_META_TIME_ASYMMETRY_TRAP','PROCESS_META_HIGH_POS_LIQUIDITY_SQUEEZE',
            'PROCESS_META_INSTITUTIONAL_STRUCTURAL_EXIT'
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
        【V20.1.0 · 关系诊断分发器 (局部重构防爆版)】
        将背离处理器收拢至本地，切断外部处理器的 NaN 污染。
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
            meta_score=self._calculate_price_vs_momentum_divergence(df,config)
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
        elif signal_name=='PROCESS_META_INSTITUTIONAL_SWEEP':
            meta_score=self._calculate_institutional_sweep(df,config)
        elif signal_name=='PROCESS_META_HF_ALGO_MANIPULATION_RISK':
            meta_score=self._calculate_hf_algo_manipulation_risk(df,config)
        elif signal_name=='PROCESS_META_MA_RUBBER_BAND_REVERSAL':
            meta_score=self._calculate_ma_rubber_band_reversal(df,config)
        elif signal_name=='PROCESS_META_GEOMETRIC_TREND_RESONANCE':
            meta_score=self._calculate_geometric_trend_resonance(df,config)
        elif signal_name=='PROCESS_META_MA_COMPRESSION_EXPLOSION':
            meta_score=self._calculate_ma_compression_explosion(df,config)
        elif signal_name=='PROCESS_META_TOP_TIER_HM_HARVESTING':
            meta_score=self._calculate_top_tier_hm_harvesting(df,config)
        elif signal_name=='PROCESS_META_VWAP_MAGNETIC_DIVERGENCE':
            meta_score=self._calculate_vwap_magnetic_divergence(df,config)
        elif signal_name=='PROCESS_META_MULTI_PEAK_AVALANCHE_RISK':
            meta_score=self._calculate_multi_peak_avalanche_risk(df,config)
        elif signal_name=='PROCESS_META_SECTOR_LIFECYCLE_TAILWIND':
            meta_score=self._calculate_sector_lifecycle_tailwind(df,config)
        elif signal_name=='PROCESS_META_TIME_ASYMMETRY_TRAP':
            meta_score=self._calculate_time_asymmetry_trap(df,config)
        elif signal_name=='PROCESS_META_HIGH_POS_LIQUIDITY_SQUEEZE':
            meta_score=self._calculate_high_pos_liquidity_squeeze(df,config)
        elif signal_name=='PROCESS_META_INSTITUTIONAL_STRUCTURAL_EXIT':
            meta_score=self._calculate_institutional_structural_exit(df,config)
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
        【V5.1.0 · 断层免疫修复版】恐慌洗盘吸筹引擎
        整合原生的异常爆量与套牢释放，结合低位锁仓实施带血筹码的极限张量猎杀。
        """
        method_name="_calculate_panic_washout_accumulation"
        required_signals=['pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D','intraday_trough_filling_degree_D','high_freq_flow_divergence_D','chip_rsi_divergence_D','chip_stability_D','pressure_release_index_D','tick_abnormal_volume_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        panic_level=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorption=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        trough_fill=self._get_safe_series(df,'intraday_trough_filling_degree_D',method_name=method_name)
        hff_div=self._get_safe_series(df,'high_freq_flow_divergence_D',method_name=method_name)
        chip_div=self._get_safe_series(df,'chip_rsi_divergence_D',method_name=method_name)
        chip_stab=self._get_safe_series(df,'chip_stability_D',method_name=method_name)
        release=self._get_safe_series(df,'pressure_release_index_D',method_name=method_name)
        abnorm_vol=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        panic_shock=0.5*(1.0+np.tanh(_hab(panic_level,21)))
        release_shock=0.5*(1.0+np.tanh(_hab(release,21)))
        vol_shock=0.5*(1.0+np.tanh(_hab(abnorm_vol,13)))
        lock_shock=0.5*(1.0+np.tanh(_hab(low_lock,21)))
        abs_shock=0.5*(1.0+np.tanh(_hab(absorption,21)))
        trough_shock=0.5*(1.0+np.tanh(_hab(trough_fill,21)))
        panic_tensor=(panic_shock*release_shock*vol_shock)**1.5
        absorption_tensor=(lock_shock*abs_shock*trough_shock)**1.5
        hff_shock=np.tanh(_hab(hff_div,21)).clip(lower=0)
        cdiv_shock=np.tanh(_hab(chip_div,21)).clip(lower=0)
        div_bonus=1.0+hff_shock+cdiv_shock
        k_tensor=_kinematics('pressure_trapped_D',panic_level,13)
        base_score=panic_tensor*absorption_tensor*div_bonus*(1.0+k_tensor.clip(lower=0))
        hist_gate=config.get('historical_potential_gate',0.2)
        gate_mask=chip_stab>hist_gate
        final_score=np.tanh(base_score**1.5).where(gate_mask,0.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':panic_level,'pressure_release_index_D':release},calc_nodes={'panic_tensor':panic_tensor,'absorption_tensor':absorption_tensor,'div_bonus':div_bonus,'base_score':base_score},final_result=final_score)
        return final_score

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.1.0 · 断层免疫修复版】诡道吸筹防爆引擎
        引入原生的日内筹码换手烈度与吸筹置信度，拆穿价格偏度与高频流偏度的障眼法。
        """
        method_name="_calculate_deceptive_accumulation"
        required_signals=['stealth_flow_ratio_D','tick_clustering_index_D','intraday_price_distribution_skewness_D','high_freq_flow_skewness_D','price_flow_divergence_D','chip_flow_intensity_D','intraday_chip_turnover_intensity_D','tick_chip_transfer_efficiency_D','intraday_accumulation_confidence_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        stealth=self._get_safe_series(df,'stealth_flow_ratio_D',method_name=method_name)
        cluster=self._get_safe_series(df,'tick_clustering_index_D',method_name=method_name)
        price_skew=self._get_safe_series(df,'intraday_price_distribution_skewness_D',method_name=method_name)
        flow_skew=self._get_safe_series(df,'high_freq_flow_skewness_D',method_name=method_name)
        pf_div=self._get_safe_series(df,'price_flow_divergence_D',method_name=method_name)
        flow_int=self._get_safe_series(df,'chip_flow_intensity_D',method_name=method_name)
        turnover_int=self._get_safe_series(df,'intraday_chip_turnover_intensity_D',method_name=method_name)
        trans_eff=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        acc_conf=self._get_safe_series(df,'intraday_accumulation_confidence_D',method_name=method_name)
        stealth_shock=0.5*(1.0+np.tanh(_hab(stealth,21)))
        cluster_shock=0.5*(1.0+np.tanh(_hab(cluster,21)))
        flow_int_shock=0.5*(1.0+np.tanh(_hab(flow_int,21)))
        trans_shock=0.5*(1.0+np.tanh(_hab(trans_eff,21)))
        turn_shock=0.5*(1.0+np.tanh(_hab(turnover_int,21)))
        conf_shock=0.5*(1.0+np.tanh(_hab(acc_conf,21)))
        pf_div_shock=np.tanh(_hab(pf_div,21)).clip(lower=0)
        skew_mismatch=np.tanh(_hab(flow_skew-price_skew,21)).clip(lower=0)
        stealth_tensor=stealth_shock*cluster_shock*flow_int_shock*trans_shock*conf_shock
        deception_index=pf_div_shock+skew_mismatch*0.5+turn_shock*0.5
        k_tensor=_kinematics('stealth_flow_ratio_D',stealth,13)
        raw_deception=stealth_tensor*(1.0+deception_index)*(1.0+k_tensor.clip(lower=0))
        final_score=np.tanh(raw_deception**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'stealth_flow_ratio_D':stealth,'intraday_chip_turnover_intensity_D':turnover_int},calc_nodes={'stealth_tensor':stealth_tensor,'deception_index':deception_index,'raw_deception':raw_deception},final_result=final_score)
        return final_score

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.1.0 · 断层免疫修复版】吸筹末端质变拐点引擎
        使用盘整质量得分(consolidation)替换虚构指数，精准定位物理奇点临界。
        """
        method_name="_calculate_accumulation_inflection"
        required_signals=['PROCESS_META_COVERT_ACCUMULATION','PROCESS_META_DECEPTIVE_ACCUMULATION','PROCESS_META_PANIC_WASHOUT_ACCUMULATION','PROCESS_META_MAIN_FORCE_RALLY_INTENT','chip_convergence_ratio_D','price_vs_ma_21_ratio_D','flow_acceleration_intraday_D','flow_consistency_D','MA_POTENTIAL_COMPRESSION_RATE_D','MACDh_13_34_8_D','consolidation_quality_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        acc_win=config.get('accumulation_window',21)
        covert=self._get_atomic_score(df,'PROCESS_META_COVERT_ACCUMULATION',0.0)
        decept=self._get_atomic_score(df,'PROCESS_META_DECEPTIVE_ACCUMULATION',0.0)
        panic=self._get_atomic_score(df,'PROCESS_META_PANIC_WASHOUT_ACCUMULATION',0.0)
        rally=self._get_atomic_score(df,'PROCESS_META_MAIN_FORCE_RALLY_INTENT',0.0).clip(lower=0)
        c_conv=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        p_ma21=self._get_safe_series(df,'price_vs_ma_21_ratio_D',method_name=method_name)
        f_accel=self._get_safe_series(df,'flow_acceleration_intraday_D',method_name=method_name)
        f_cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        ma_comp=self._get_safe_series(df,'MA_POTENTIAL_COMPRESSION_RATE_D',method_name=method_name)
        macd=self._get_safe_series(df,'MACDh_13_34_8_D',method_name=method_name)
        consolidation=self._get_safe_series(df,'consolidation_quality_score_D',method_name=method_name)
        daily_comp=covert*0.35+decept*0.35+panic*0.3
        pot_energy=daily_comp.ewm(span=acc_win,adjust=False,min_periods=5).mean()
        conv_shk=0.5*(1.0+np.tanh(_hab(c_conv,34)))
        p_clamp=1.0-np.tanh(_hab(np.abs(p_ma21-1.0),21)).clip(lower=0)
        ma_shk=np.tanh(_hab(ma_comp,34)).clip(lower=0)
        eq_shk=0.5*(1.0+np.tanh(_hab(consolidation,21)))
        acc_shk=np.tanh(_hab(f_accel,13)).clip(lower=0)
        cons_shk=0.5*(1.0+np.tanh(_hab(f_cons,21)))
        macd_shk=np.tanh(_hab(macd,13)).clip(lower=0)
        k_tensor=_kinematics('chip_convergence_ratio_D',c_conv,13)
        struct_mass=conv_shk*p_clamp*ma_shk*eq_shk
        dyn_ignite=acc_shk*cons_shk*rally*(1.0+macd_shk)
        raw_score=pot_energy*struct_mass*dyn_ignite*(1.0+k_tensor.clip(lower=0))
        final_score=(1.0/(1.0+np.exp(-10.0*(raw_score-0.5)))).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'consolidation_quality_score_D':consolidation,'MACDh_13_34_8_D':macd},calc_nodes={'pot_energy':pot_energy,'struct_mass':struct_mass,'dyn_ignite':dyn_ignite,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_loser_capitulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.1.0 · 断层免疫修复版】输家绝地投降引擎
        统合原生的套牢盘与下降趋势动能，结合跌幅释放数据确认带血绝望大底。
        """
        method_name="_calculate_loser_capitulation"
        required_signals=['pressure_release_index_D','pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D','winner_rate_D','downtrend_strength_D','price_to_weight_avg_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        release=self._get_safe_series(df,'pressure_release_index_D',method_name=method_name)
        trapped=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorp=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        winner=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        down=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        price_to_cost=self._get_safe_series(df,'price_to_weight_avg_ratio_D',method_name=method_name)
        loser_rate=100.0-winner
        loss_margin=(1.0-price_to_cost).clip(lower=0)
        rel_shock=0.5*(1.0+np.tanh(_hab(release,21)))
        trap_shock=0.5*(1.0+np.tanh(_hab(trapped,21)))
        lock_shock=0.5*(1.0+np.tanh(_hab(low_lock,21)))
        abs_shock=0.5*(1.0+np.tanh(_hab(absorp,21)))
        marg_shock=0.5*(1.0+np.tanh(_hab(loss_margin,34)))
        down_shock=0.5*(1.0+np.tanh(_hab(down,34)))
        rate_shock=0.5*(1.0+np.tanh(_hab(loser_rate,34)))
        k_tensor=_kinematics('pressure_release_index_D',release,13)
        pain_ext=np.sqrt(rel_shock*trap_shock)*marg_shock*rate_shock*(1.0+down_shock)
        abs_anchor=np.sqrt(lock_shock*abs_shock)
        raw_score=pain_ext*abs_anchor*(1.0+k_tensor.clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'downtrend_strength_D':down,'winner_rate_D':winner},calc_nodes={'pain_ext':pain_ext,'abs_anchor':abs_anchor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 全息物理动力学版】突破加速强攻引擎
        锁定特大买单燃料与筹码流动阻力系数，乘数放大极化突破。
        """
        method_name="_calculate_breakout_acceleration"
        required_signals=['breakout_quality_score_D','industry_strength_rank_D','net_mf_amount_D','flow_consistency_D','tick_abnormal_volume_ratio_D','uptrend_strength_D','T1_PREMIUM_EXPECTATION_D','HM_COORDINATED_ATTACK_D','breakout_penalty_score_D','buy_elg_amount_D','volatility_adjusted_concentration_D','amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        brk=self._get_safe_series(df,'breakout_quality_score_D',method_name=method_name)
        ind=self._get_safe_series(df,'industry_strength_rank_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        abnorm=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        uptrend=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        t1=self._get_safe_series(df,'T1_PREMIUM_EXPECTATION_D',method_name=method_name)
        hm=self._get_safe_series(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        pen=self._get_safe_series(df,'breakout_penalty_score_D',method_name=method_name)
        elg=self._get_safe_series(df,'buy_elg_amount_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        brk_shock=0.5*(1.0+np.tanh(_hab(brk,21)))
        ind_shock=0.5*(1.0+np.tanh(_hab(ind,34)))
        mf_shock=np.tanh(_hab(mf,21)).clip(lower=0)
        cons_shock=0.5*(1.0+np.tanh(_hab(cons,13)))
        abn_shock=0.5*(1.0+np.tanh(_hab(abnorm,13)))
        up_shock=0.5*(1.0+np.tanh(_hab(uptrend,21)))
        t1_shock=np.tanh(_hab(t1,13)).clip(lower=0)
        hm_shock=0.5*(1.0+np.tanh(_hab(hm,13)))
        pen_shock=np.tanh(_hab(pen,13)).clip(lower=0)
        vac_shock=0.5*(1.0+np.tanh(_hab(vac,34)))
        elg_ratio=(elg/amount).fillna(0.0)
        elg_shock=0.5*(1.0+np.tanh(_hab(elg_ratio,13)))
        k_tensor=_kinematics('breakout_quality_score_D',brk,13)
        fuel_multiplier=1.0+(t1_shock*0.5+hm_shock*0.5+elg_shock)
        base_tensor=brk_shock*ind_shock*(1.0+mf_shock**1.5)*(1.0-pen_shock)*vac_shock
        catalyst=(cons_shock*abn_shock*up_shock)**1.5
        raw_score=base_tensor*catalyst*fuel_multiplier*(1.0+k_tensor.clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'breakout_quality_score_D':brk,'buy_elg_amount_D':elg},calc_nodes={'fuel_multiplier':fuel_multiplier,'base_tensor':base_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 全息物理动力学版】资金流吸筹质变引擎
        叠加真金白银大单占比与流量势能矩阵，识别主力由暗入明的分界点。
        """
        method_name="_calculate_fund_flow_accumulation_inflection"
        required_signals=['accumulation_signal_score_D','net_mf_amount_D','flow_efficiency_D','tick_large_order_net_D','intraday_accumulation_confidence_D','GAP_MOMENTUM_STRENGTH_D','STATE_GOLDEN_PIT_D','buy_lg_amount_D','amount_D','flow_persistence_minutes_D','net_energy_flow_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        acc=self._get_safe_series(df,'accumulation_signal_score_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        eff=self._get_safe_series(df,'flow_efficiency_D',method_name=method_name)
        l_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name)
        intra=self._get_safe_series(df,'intraday_accumulation_confidence_D',method_name=method_name)
        gap=self._get_safe_series(df,'GAP_MOMENTUM_STRENGTH_D',method_name=method_name)
        pit=self._get_safe_series(df,'STATE_GOLDEN_PIT_D',method_name=method_name).clip(0,1)
        buy_lg=self._get_safe_series(df,'buy_lg_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        pers=self._get_safe_series(df,'flow_persistence_minutes_D',method_name=method_name)
        energy=self._get_safe_series(df,'net_energy_flow_D',method_name=method_name)
        acc_shk=0.5*(1.0+np.tanh(_hab(acc,34)))
        mf_shk=np.tanh(_hab(mf,21)).clip(lower=0)
        eff_shk=0.5*(1.0+np.tanh(_hab(eff,21)))
        lnet_shk=np.tanh(_hab(l_net,21))
        intra_shk=0.5*(1.0+np.tanh(_hab(intra,21)))
        gap_shk=np.tanh(_hab(gap,13)).clip(lower=0)
        lg_shk=0.5*(1.0+np.tanh(_hab((buy_lg/amt).fillna(0),21)))
        pers_shk=0.5*(1.0+np.tanh(_hab(pers,13)))
        eng_shk=np.tanh(_hab(energy,21)).clip(lower=0)
        k_tensor=_kinematics('accumulation_signal_score_D',acc,13)
        pit_lev=1.0+pit*0.5
        base_ig=acc_shk*eff_shk*(1.0+lnet_shk.clip(lower=0))*intra_shk*lg_shk*pers_shk*pit_lev
        ignite=(1.0+gap_shk)*(1.0+eng_shk)*(1.0+mf_shk**1.5)
        raw_score=base_ig*ignite*(1.0+k_tensor.clip(lower=0))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'buy_lg_amount_D':buy_lg,'net_energy_flow_D':energy},calc_nodes={'base_ig':base_ig,'ignite':ignite,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.1.0 · 断层免疫修复版】获利压迫与净流对冲引擎
        深描特大卖单与派发能量的黑洞级合力砸盘，使用 pressure_profit_D 替代虚构贪婪度。
        """
        method_name="_calculate_profit_vs_flow_relationship"
        required_signals=['profit_pressure_D','net_mf_amount_D','profit_ratio_D','flow_consistency_D','winner_rate_D','intraday_distribution_confidence_D','STATE_PARABOLIC_WARNING_D','distribution_energy_D','sell_elg_amount_D','amount_D','pressure_profit_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        press=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        p_ratio=self._get_safe_series(df,'profit_ratio_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        win=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        dist_conf=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        para=self._get_safe_series(df,'STATE_PARABOLIC_WARNING_D',method_name=method_name).clip(0,1)
        dist_eng=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        sell_elg=self._get_safe_series(df,'sell_elg_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        marg=self._get_safe_series(df,'pressure_profit_D',method_name=method_name)
        p_shk=0.5*(1.0+np.tanh(_hab(press,21)))
        mf_shk=np.tanh(_hab(mf,34))
        r_shk=0.5*(1.0+np.tanh(_hab(p_ratio,21)))
        c_shk=0.5*(1.0+np.tanh(_hab(cons,21)))
        w_shk=0.5*(1.0+np.tanh(_hab(win,55)))
        dconf_shk=0.5*(1.0+np.tanh(_hab(dist_conf,13)))
        deng_shk=0.5*(1.0+np.tanh(_hab(dist_eng,21)))
        sell_shk=0.5*(1.0+np.tanh(_hab((sell_elg/amt).fillna(0),13)))
        marg_shk=0.5*(1.0+np.tanh(_hab(marg,34)))
        k_tensor=_kinematics('profit_pressure_D',press,13)
        para_lev=1.0+para*1.0
        press_tensor=p_shk*r_shk*dconf_shk*deng_shk*sell_shk*marg_shk*para_lev*(1.0+k_tensor.clip(lower=0))
        supp_tensor=(mf_shk.clip(lower=0)*c_shk)*(1.0-w_shk*0.5)
        raw_score=supp_tensor-press_tensor*1.5
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'distribution_energy_D':dist_eng,'sell_elg_amount_D':sell_elg},calc_nodes={'press_tensor':press_tensor,'supp_tensor':supp_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 全息物理动力学版】个股与板块龙脉共振引擎
        聚合板块拉升得分、一阶动量与全网游资攻击阵列，激活绝对顺风局。
        """
        method_name="_calculate_stock_sector_sync"
        required_signals=['pct_change_D','industry_strength_rank_D','net_mf_amount_D','flow_consistency_D','industry_leader_score_D','mid_long_sync_D','STATE_MARKET_LEADER_D','industry_markup_score_D','industry_rank_accel_D','HM_COORDINATED_ATTACK_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        pct=self._get_safe_series(df,'pct_change_D',method_name=method_name)
        rank=self._get_safe_series(df,'industry_strength_rank_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        ldr=self._get_safe_series(df,'industry_leader_score_D',method_name=method_name)
        sync=self._get_safe_series(df,'mid_long_sync_D',method_name=method_name)
        m_ldr=self._get_safe_series(df,'STATE_MARKET_LEADER_D',method_name=method_name).clip(0,1)
        mkup=self._get_safe_series(df,'industry_markup_score_D',method_name=method_name)
        rk_acc=self._get_safe_series(df,'industry_rank_accel_D',method_name=method_name)
        hm=self._get_safe_series(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        pct_shk=np.tanh(_hab(pct,13))
        rk_shk=0.5*(1.0+np.tanh(_hab(rank,34)))
        mf_shk=np.tanh(_hab(mf,21))
        c_shk=0.5*(1.0+np.tanh(_hab(cons,21)))
        ldr_shk=0.5*(1.0+np.tanh(_hab(ldr,21)))
        sync_shk=0.5*(1.0+np.tanh(_hab(sync,21)))
        mk_shk=0.5*(1.0+np.tanh(_hab(mkup,21)))
        acc_shk=np.tanh(_hab(rk_acc,13))
        hm_shk=0.5*(1.0+np.tanh(_hab(hm,13)))
        k_tensor=_kinematics('industry_strength_rank_D',rank,13)
        sect_tensor=rk_shk*mk_shk*(1.0+acc_shk)*ldr_shk*(1.0+k_tensor)
        flow_syn=mf_shk*c_shk*(1.0+sync_shk)*(1.0+hm_shk)
        resonance=pct_shk*sect_tensor*(1.0+flow_syn*np.sign(pct_shk))*(1.0+m_ldr*1.0)
        final_score=np.sign(resonance)*(np.abs(resonance)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'industry_markup_score_D':mkup,'industry_rank_accel_D':rk_acc},calc_nodes={'sect_tensor':sect_tensor,'resonance':resonance,'final_score':final_score},final_result=final_score)
        return final_score

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 全息物理动力学版】热门板块退潮预警引擎
        整合板块物理衰退期得分与特大单无差别抛售量能，粉碎主线死扛幻想。
        """
        method_name="_calculate_hot_sector_cooling"
        required_signals=['THEME_HOTNESS_SCORE_D','net_mf_amount_D','industry_stagnation_score_D','outflow_quality_D','industry_downtrend_score_D','distribution_energy_D','sell_elg_amount_D','amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        hot=self._get_safe_series(df,'THEME_HOTNESS_SCORE_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        stag=self._get_safe_series(df,'industry_stagnation_score_D',method_name=method_name)
        outq=self._get_safe_series(df,'outflow_quality_D',method_name=method_name)
        down=self._get_safe_series(df,'industry_downtrend_score_D',method_name=method_name)
        dist=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        sell_elg=self._get_safe_series(df,'sell_elg_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        hot_shk=0.5*(1.0+np.tanh(_hab(hot,21)))
        mf_shk=np.tanh(_hab(mf,13)).clip(upper=0).abs()
        stag_shk=0.5*(1.0+np.tanh(_hab(stag,21)))
        out_shk=0.5*(1.0+np.tanh(_hab(outq,21)))
        down_shk=0.5*(1.0+np.tanh(_hab(down,21)))
        dist_shk=0.5*(1.0+np.tanh(_hab(dist,21)))
        sell_shk=0.5*(1.0+np.tanh(_hab((sell_elg/amt).fillna(0),13)))
        k_tensor=_kinematics('THEME_HOTNESS_SCORE_D',hot,13)
        outflow_tensor=mf_shk*out_shk*dist_shk*(1.0+sell_shk)
        decay_boost=1.0+stag_shk*1.5+down_shk*2.0
        raw_score=hot_shk*outflow_tensor*decay_boost*(1.0-k_tensor.clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'industry_downtrend_score_D':down,'distribution_energy_D':dist},calc_nodes={'outflow_tensor':outflow_tensor,'decay_boost':decay_boost,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_ff_vs_structure_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 零噪防爆版】资金结构双极惩罚引擎
        清剿NaN断层，阻断伪量化噪音带来的虚假惩罚。
        """
        method_name="_calculate_ff_vs_structure_relationship"
        required_signals=['uptrend_strength_D','flow_consistency_D','ma_arrangement_status_D','chip_structure_state_D','industry_stagnation_score_D','large_order_anomaly_D','STATE_ROBUST_TREND_D','net_mf_amount_D','chip_stability_D','flow_momentum_13d_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _zg(s: pd.Series) -> pd.Series: return pd.Series(np.where(s.abs()<1e-4,0.0,1.0),index=df_index)
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((sf-sf.rolling(w,min_periods=1).mean())/sf.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', sf.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        up=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name).fillna(0.0)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name).fillna(0.0)
        ma_s=self._get_safe_series(df,'ma_arrangement_status_D',method_name=method_name).fillna(0.0)
        chip_s=self._get_safe_series(df,'chip_structure_state_D',method_name=method_name).fillna(0.0)
        stag=self._get_safe_series(df,'industry_stagnation_score_D',method_name=method_name).fillna(0.0)
        anom=self._get_safe_series(df,'large_order_anomaly_D',method_name=method_name).fillna(0.0)
        rob=self._get_safe_series(df,'STATE_ROBUST_TREND_D',method_name=method_name).clip(0,1).fillna(0.0)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name).fillna(0.0)
        stab=self._get_safe_series(df,'chip_stability_D',method_name=method_name).fillna(0.0)
        mom=self._get_safe_series(df,'flow_momentum_13d_D',method_name=method_name).fillna(0.0)
        k_up=_kinematics('uptrend_strength_D',up,13)
        u_shk=np.tanh(_hab('uptrend_strength_D',up,34)).clip(lower=0)*_zg(up)
        c_shk=np.tanh(_hab('flow_consistency_D',cons,21)).clip(lower=0)*_zg(cons)
        s_shk=np.tanh(_hab('industry_stagnation_score_D',stag,21)).clip(lower=0)*_zg(stag)
        a_shk=np.tanh(_hab('large_order_anomaly_D',anom,13)).clip(lower=0)*_zg(anom)
        m_shk=np.tanh(_hab('net_mf_amount_D',mf,21))
        st_shk=np.tanh(_hab('chip_stability_D',stab,21)).clip(lower=0)*_zg(stab)
        mo_shk=np.tanh(_hab('flow_momentum_13d_D',mom,13))
        base_div=self._calculate_instantaneous_relationship(df,config).fillna(0.0)
        a_pen=a_shk.clip(lower=0)*0.5*(1.0-rob*0.8)*(1.0-st_shk)
        s_pen=(1.0-s_shk*0.5)*(1.0-a_pen)*(1.0+m_shk.clip(lower=0)*0.3)*(1.0+mo_shk.clip(lower=0)*0.3)
        amp=1.0+(u_shk*c_shk*(1.0+ma_s*0.5+chip_s*0.5)*(1.0+np.abs(k_up)))*s_pen
        raw=base_div*amp
        final_score=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'uptrend_strength_D':up,'large_order_anomaly_D':anom},calc_nodes={'amp':amp,'raw':raw},final_result=final_score)
        return final_score

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 全息量子动力学版】价筹稳态共振引力引擎
        8步推演：引入筹码混沌度削弱死筹假象，强关联价格动能构建共振张量爆破。
        """
        method_name="_calculate_pc_relationship"
        required_signals=['peak_concentration_D','close_D','chip_convergence_ratio_D','high_position_lock_ratio_90_D','chip_stability_change_5d_D','volatility_adjusted_concentration_D','chip_entropy_D','chip_flow_intensity_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        pk=self._get_safe_series(df,'peak_concentration_D',method_name=method_name)
        cls=self._get_safe_series(df,'close_D',method_name=method_name)
        cvg=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        hl=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name)
        sc=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        fi=self._get_safe_series(df,'chip_flow_intensity_D',method_name=method_name)
        pk_s=np.tanh(_hab('peak_concentration_D',pk,34))
        cv_s=0.5*(1.0+np.tanh(_hab('chip_convergence_ratio_D',cvg,21)))
        hl_s=0.5*(1.0+np.tanh(_hab('high_position_lock_ratio_90_D',hl,21)))
        sc_s=np.tanh(_hab('chip_stability_change_5d_D',sc,13))
        va_s=0.5*(1.0+np.tanh(_hab('volatility_adjusted_concentration_D',vac,34)))
        en_s=1.0-np.tanh(_hab('chip_entropy_D',ent,21)).clip(lower=0)
        fi_s=0.5*(1.0+np.tanh(_hab('chip_flow_intensity_D',fi,21)))
        k_pk=_kinematics('peak_concentration_D',pk,13)
        c_diff=pd.Series(np.where(cls.diff(1).fillna(0).abs()<1e-4,0.0,cls.diff(1).fillna(0)),index=df_index)
        m_p=np.tanh(_hab('close_D',c_diff,13))
        thr=pk_s*cv_s*(1.0+hl_s)*(1.0+sc_s.clip(lower=0))*(1.0+va_s)*(1.0+np.abs(k_pk))*en_s*(1.0+fi_s)
        amp=(1.0+thr*np.sign(m_p)).clip(lower=0.1)
        raw=m_p*amp
        rel=pd.Series(np.sign(raw)*(np.abs(raw)**1.5),index=df_index).clip(-1,1).fillna(0.0)
        final_score=self._perform_meta_analysis_on_score(rel,config,df,df_index).fillna(0.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_concentration_D':pk,'chip_entropy_D':ent},calc_nodes={'thr':thr,'amp':amp},final_result=final_score)
        return final_score

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 零噪防爆版】价资协同双向剥离
        消除NaN病毒，剥离均值底噪，极化验证每一分资金拉抬价格的真实做多系数。
        """
        method_name="_calculate_pf_relationship"
        required_signals=['net_mf_amount_D','close_D','price_vs_ma_13_ratio_D','main_force_activity_index_D','flow_momentum_13d_D','flow_impact_ratio_D','tick_chip_transfer_efficiency_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((sf-sf.rolling(w,min_periods=1).mean())/sf.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', sf.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name).fillna(0.0)
        cls=self._get_safe_series(df,'close_D',method_name=method_name).fillna(0.0)
        pm=self._get_safe_series(df,'price_vs_ma_13_ratio_D',method_name=method_name).fillna(0.0)
        act=self._get_safe_series(df,'main_force_activity_index_D',method_name=method_name).fillna(0.0)
        fm=self._get_safe_series(df,'flow_momentum_13d_D',method_name=method_name).fillna(0.0)
        imp=self._get_safe_series(df,'flow_impact_ratio_D',method_name=method_name).fillna(0.0)
        tr=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name).fillna(0.0)
        m_s=np.tanh(_hab('net_mf_amount_D',mf,34))
        a_s=np.tanh(_hab('main_force_activity_index_D',act,21)).clip(lower=0)
        f_s=np.tanh(_hab('flow_momentum_13d_D',fm,13))
        p_s=np.tanh(_hab('price_vs_ma_13_ratio_D',pm,21))
        i_s=np.tanh(_hab('flow_impact_ratio_D',imp,21)).clip(lower=0)
        t_s=np.tanh(_hab('tick_chip_transfer_efficiency_D',tr,21)).clip(lower=0)
        k_mf=_kinematics('net_mf_amount_D',mf,13)
        c_diff=pd.Series(np.where(cls.diff(1).fillna(0.0).abs()<1e-4,0.0,cls.diff(1).fillna(0.0)),index=df_index)
        m_p=np.tanh(_hab('close_D',c_diff,13))
        thr=m_s*a_s*(1.0+np.abs(k_mf))*(1.0+f_s*0.5)*i_s*(1.0+t_s)
        amp=(1.0+thr*np.sign(m_p)).clip(lower=0.1)
        raw=m_p*amp*(1.0+np.abs(p_s)*0.5)
        rel=pd.Series(np.sign(raw)*(np.abs(raw)**1.5),index=df_index).clip(-1,1).fillna(0.0)
        final_score=self._perform_meta_analysis_on_score(rel,config,df,df_index).fillna(0.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':mf,'flow_impact_ratio_D':imp},calc_nodes={'thr':thr,'amp':amp},final_result=final_score)
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

    def _calculate_price_vs_momentum_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 量子张量防爆版】价势多维背离引擎
        完全在内部重构的价势背离逻辑，强制收拢，彻底免疫外部处理器的 NaN 污染崩塌。
        """
        method_name="_calculate_price_vs_momentum_divergence"
        required_signals=['close_D','ROC_13_D','VPA_EFFICIENCY_D','PRICE_ENTROPY_D','net_mf_amount_D','turnover_rate_f_D','GEOM_REG_SLOPE_D','GEOM_REG_R2_D','BIAS_21_D','GEOM_ARC_CURVATURE_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((sf-sf.rolling(w,min_periods=1).mean())/sf.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', sf.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        cls=self._get_safe_series(df,'close_D',method_name=method_name).fillna(0.0)
        roc=self._get_safe_series(df,'ROC_13_D',method_name=method_name).fillna(0.0)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name).fillna(0.0)
        ent=self._get_safe_series(df,'PRICE_ENTROPY_D',method_name=method_name).fillna(0.0)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name).fillna(0.0)
        to=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name).fillna(0.0)
        slope=self._get_safe_series(df,'GEOM_REG_SLOPE_D',method_name=method_name).fillna(0.0)
        r2=self._get_safe_series(df,'GEOM_REG_R2_D',method_name=method_name).fillna(0.0)
        bias=self._get_safe_series(df,'BIAS_21_D',method_name=method_name).fillna(0.0)
        arc=self._get_safe_series(df,'GEOM_ARC_CURVATURE_D',method_name=method_name).fillna(0.0)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name).fillna(0.0)
        p_vel=np.tanh(_hab('close_D',cls,13))
        m_acc=_kinematics('ROC_13_D',roc,13)
        kinematic_div=p_vel-m_acc
        vpa_shk=np.tanh(_hab('VPA_EFFICIENCY_D',vpa,21))
        ent_shk=np.tanh(_hab('PRICE_ENTROPY_D',ent,13)).clip(lower=0)
        mf_shk=np.tanh(_hab('net_mf_amount_D',mf,21))
        to_shk=0.5*(1.0+np.tanh(_hab('turnover_rate_f_D',to,13)))
        energy_decay=ent_shk*to_shk-vpa_shk*mf_shk
        r2_shk=0.5*(1.0+np.tanh(_hab('GEOM_REG_R2_D',r2,34)))
        slope_shk=np.tanh(_hab('GEOM_REG_SLOPE_D',slope,21))
        arc_shk=np.tanh(_hab('GEOM_ARC_CURVATURE_D',arc,21))
        bias_shk=np.tanh(_hab('BIAS_21_D',bias,21))
        geom_tension=(bias_shk-arc_shk)*r2_shk*np.sign(slope_shk)
        sent_shk=np.tanh(_hab('market_sentiment_score_D',sent,34))
        raw_div=kinematic_div*0.4+energy_decay*0.3+geom_tension*0.3
        raw_score=raw_div*(1.0+np.abs(sent_shk))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc,'VPA_EFFICIENCY_D':vpa},calc_nodes={'kinematic_div':kinematic_div,'energy_decay':energy_decay,'geom_tension':geom_tension,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 量子张量防爆版】张量对冲基础单元
        内置彻底的 NaN 零基封印，确保底层共振计算绝对不崩溃。
        """
        signal_a_name=config.get('signal_A')
        signal_b_name=config.get('signal_B')
        df_index=df.index
        relationship_type=config.get('relationship_type','consensus')
        def _get_sig(name: str, src: str) -> Optional[pd.Series]:
            if src=='atomic_states': return self.strategy.atomic_states.get(name)
            try: return self._get_safe_series(df,name,np.nan,"_calculate_instantaneous_relationship")
            except ValueError: return None
        sa=_get_sig(signal_a_name,config.get('source_A','df'))
        sb=_get_sig(signal_b_name,config.get('source_B','df'))
        if sa is None or sb is None: return pd.Series(0.0,index=df_index,dtype=np.float32)
        sa=sa.fillna(0.0)
        sb=sb.fillna(0.0)
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((sf-sf.rolling(w,min_periods=1).mean())/sf.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', sf.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        ca=sa.diff(1).fillna(0.0) if config.get('change_type_A','pct')=='diff' else sa.pct_change(1).replace([np.inf,-np.inf],0.0).fillna(0.0)
        cb=sb.diff(1).fillna(0.0) if config.get('change_type_B','pct')=='diff' else sb.pct_change(1).replace([np.inf,-np.inf],0.0).fillna(0.0)
        k_a=_kinematics(signal_a_name,sa,13)
        k_b=_kinematics(signal_b_name,sb,13)
        ma=np.tanh(_hab(signal_a_name,ca,13))+k_a*0.5
        tb=np.tanh(_hab(signal_b_name,cb,13))+k_b*0.5
        kf=config.get('signal_b_factor_k',1.0)
        if relationship_type=='divergence':
            rel=(kf*tb-ma)/(kf+1.0)
        else:
            fs=ma+kf*tb
            mag=(np.abs(ma)*np.abs(tb))**0.5
            rel=np.sign(fs)*mag
        rel=np.sign(rel)*(np.abs(rel)**1.5)
        return np.tanh(rel).clip(-1,1).fillna(0.0).astype(np.float32)

    def _calculate_dyn_vs_chip_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 全息量子动力学版】动能筹码异向极化引擎
        8步推演：将筹码混沌度与下行重力加速度挂载，放大顶部套牢发散的死局压迫力。
        """
        method_name="_calculate_dyn_vs_chip_relationship"
        required_signals=['ROC_13_D','winner_rate_D','profit_ratio_D','chip_mean_D','chip_kurtosis_D','volatility_adjusted_concentration_D','downtrend_strength_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        roc=self._get_safe_series(df,'ROC_13_D',method_name=method_name)
        win=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        prof=self._get_safe_series(df,'profit_ratio_D',method_name=method_name)
        mean=self._get_safe_series(df,'chip_mean_D',method_name=method_name)
        kurt=self._get_safe_series(df,'chip_kurtosis_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        down=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        k_roc=_kinematics('ROC_13_D',roc,13)
        bc=self._calculate_instantaneous_relationship(df,config)
        p_shk=0.5*(1.0+np.tanh(_hab('profit_ratio_D',prof,55)))
        w_shk=np.tanh(_hab('winner_rate_D',win,21))
        m_shk=np.tanh(_hab('chip_mean_D',mean,13))
        k_shk=0.5*(1.0+np.tanh(_hab('chip_kurtosis_D',kurt,21)))
        v_shk=0.5*(1.0+np.tanh(_hab('volatility_adjusted_concentration_D',vac,21)))
        d_shk=0.5*(1.0+np.tanh(_hab('downtrend_strength_D',down,21)))
        e_shk=np.tanh(_hab('chip_entropy_D',ent,21)).clip(lower=0)
        dist_pres=1.0+(p_shk*(1.0+k_shk)*(1.0+v_shk)*(1.0+e_shk))*np.abs(m_shk)*(1.0+np.abs(k_roc))*(1.0+d_shk)
        raw=bc.where(bc>=0,bc*dist_pres*(1.0+w_shk.clip(lower=0)*0.5))
        final_score=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc,'downtrend_strength_D':down},calc_nodes={'dist_pres':dist_pres,'bc':bc,'raw':raw},final_result=final_score)
        return final_score

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 全息量子动力学版】洗盘反包诱空反弹引擎
        8步推演：深度融合日内极端锁仓率与VWAP深V拉回，粉碎空头诱骗陷阱。
        """
        method_name="_calculate_process_wash_out_rebound"
        required_signals=['shakeout_score_D','intraday_distribution_confidence_D','pressure_trapped_D','CLOSING_STRENGTH_D','intraday_trough_filling_degree_D','stealth_flow_ratio_D','absorption_energy_D','STATE_ROUNDING_BOTTOM_D','intraday_low_lock_ratio_D','vwap_deviation_D','tick_abnormal_volume_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        shk=self._get_safe_series(df,'shakeout_score_D',method_name=method_name)
        dist=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        pan=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        cls=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name)
        tf=self._get_safe_series(df,'intraday_trough_filling_degree_D',method_name=method_name)
        stl=self._get_safe_series(df,'stealth_flow_ratio_D',method_name=method_name)
        abs_e=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        rnd=self._get_safe_series(df,'STATE_ROUNDING_BOTTOM_D',method_name=method_name).clip(0,1)
        llck=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        vdev=self._get_safe_series(df,'vwap_deviation_D',method_name=method_name)
        t_abn=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        shk_shk=0.5*(1.0+np.tanh(_hab('shakeout_score_D',shk,21)))
        dist_shk=0.5*(1.0+np.tanh(_hab('intraday_distribution_confidence_D',dist,13)))
        pan_shk=0.5*(1.0+np.tanh(_hab('pressure_trapped_D',pan,21)))
        stl_shk=0.5*(1.0+np.tanh(_hab('stealth_flow_ratio_D',stl,21)))
        tf_shk=0.5*(1.0+np.tanh(_hab('intraday_trough_filling_degree_D',tf,21)))
        abs_shk=0.5*(1.0+np.tanh(_hab('absorption_energy_D',abs_e,21)))
        cls_shk=0.5*(1.0+np.tanh(_hab('CLOSING_STRENGTH_D',cls,21)))
        ll_shk=0.5*(1.0+np.tanh(_hab('intraday_low_lock_ratio_D',llck,21)))
        vdev_shk=np.tanh(_hab('vwap_deviation_D',vdev,13)).clip(upper=0).abs()
        tabn_shk=0.5*(1.0+np.tanh(_hab('tick_abnormal_volume_ratio_D',t_abn,13)))
        k_shk=_kinematics('shakeout_score_D',shk,13)
        rnd_lev=1.0+rnd*0.5
        washout_env=(shk_shk*dist_shk*pan_shk)*(1.0+stl_shk)
        rebound_int=tf_shk*abs_shk*ll_shk*(1.0+vdev_shk)*rnd_lev*(1.0+tabn_shk)
        raw_score=washout_env*(rebound_int**1.5)*cls_shk*(1.0+k_shk.clip(lower=0))
        final_score=np.tanh(raw_score*2.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'shakeout_score_D':shk,'intraday_low_lock_ratio_D':llck},calc_nodes={'washout_env':washout_env,'rebound_int':rebound_int,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_fusion_trend_exhaustion_syndrome(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 全息量子动力学版】趋势衰竭综合征引擎
        8步推演：引入筹码混沌度与派发能量共振，叠加多阶微积分与HAB极化防爆，瓦解顶部假象。
        """
        method_name="_calculate_fusion_trend_exhaustion_syndrome"
        required_signals=['STATE_PARABOLIC_WARNING_D','STATE_EMOTIONAL_EXTREME_D','PRICE_ENTROPY_D','profit_pressure_D','HM_COORDINATED_ATTACK_D','intraday_distribution_confidence_D','distribution_energy_D','chip_entropy_D','sell_elg_amount_D','amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        para=self._get_safe_series(df,'STATE_PARABOLIC_WARNING_D',method_name=method_name).clip(0,1)
        emot=self._get_safe_series(df,'STATE_EMOTIONAL_EXTREME_D',method_name=method_name).clip(0,1)
        ent=self._get_safe_series(df,'PRICE_ENTROPY_D',method_name=method_name)
        pres=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        hm=self._get_safe_series(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        dist_c=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        dist_e=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        sell_elg=self._get_safe_series(df,'sell_elg_amount_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        e_shk=np.tanh(_hab('PRICE_ENTROPY_D',ent,13)).clip(lower=0)
        p_shk=np.tanh(_hab('profit_pressure_D',pres,21)).clip(lower=0)
        dc_shk=0.5*(1.0+np.tanh(_hab('intraday_distribution_confidence_D',dist_c,13)))
        hm_shk=0.5*(1.0+np.tanh(_hab('HM_COORDINATED_ATTACK_D',hm,13)))
        de_shk=0.5*(1.0+np.tanh(_hab('distribution_energy_D',dist_e,21)))
        ce_shk=np.tanh(_hab('chip_entropy_D',c_ent,21)).clip(lower=0)
        se_shk=0.5*(1.0+np.tanh(_hab('sell_elg_ratio',(sell_elg/amount).fillna(0),13)))
        k_tensor=_kinematics('profit_pressure_D',pres,13)
        st_lev=1.0+para*1.5+emot*1.0
        phy_gate=dc_shk*0.3+p_shk*0.3+de_shk*0.2+se_shk*0.2
        veto=(1.0-hm_shk*0.9).clip(lower=0.1)
        raw_score=st_lev*phy_gate*(1.0+e_shk)*(1.0+ce_shk)*veto*(1.0+k_tensor.clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'STATE_PARABOLIC_WARNING_D':para,'profit_pressure_D':pres},calc_nodes={'phy_gate':phy_gate,'veto':veto,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_dyn_vs_chip_decay_rise(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 全息量子动力学版】力学筹码阻尼看涨引擎
        8步推演：量化下行动能耗散与恐慌崩塌的终结，引入反转概率与多阶阻尼张量。
        """
        method_name="_calculate_dyn_vs_chip_decay_rise"
        required_signals=['downtrend_strength_D','pressure_trapped_D','absorption_energy_D','chip_kurtosis_D','chip_stability_change_5d_D','reversal_prob_D','intraday_support_test_count_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        down=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        pres=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        abs_e=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        kurt=self._get_safe_series(df,'chip_kurtosis_D',method_name=method_name)
        stb=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        rev=self._get_safe_series(df,'reversal_prob_D',method_name=method_name)
        sup=self._get_safe_series(df,'intraday_support_test_count_D',method_name=method_name)
        k_tensor=_kinematics('downtrend_strength_D',down,13)
        kin_dis=np.abs(np.clip(k_tensor,-1.0,0.0))
        d_shk=0.5*(1.0+np.tanh(_hab('downtrend_strength_D',down,21)))
        p_shk=-np.tanh(_hab('pressure_trapped_D',pres,21))
        a_shk=0.5*(1.0+np.tanh(_hab('absorption_energy_D',abs_e,21)))
        k_shk=0.5*(1.0+np.tanh(_hab('chip_kurtosis_D',kurt,21)))
        st_shk=np.tanh(_hab('chip_stability_change_5d_D',stb,13)).clip(lower=0)
        r_shk=0.5*(1.0+np.tanh(_hab('reversal_prob_D',rev,13)))
        sp_shk=np.tanh(sup/3.0)
        chip_relief=(p_shk.clip(lower=0)+a_shk)*0.5
        damping_resonance=d_shk*kin_dis*k_shk*chip_relief*(1.0+st_shk)*(1.0+r_shk)*(1.0+sp_shk)
        final_score=np.tanh(damping_resonance**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'downtrend_strength_D':down,'pressure_trapped_D':pres},calc_nodes={'kin_dis':kin_dis,'chip_relief':chip_relief,'damping_resonance':damping_resonance},final_result=final_score)
        return final_score

    def _calculate_smart_money_ignition(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 全息量子动力学版】聪明钱协同点火引擎
        8步推演：严密核查特大单底仓(buy_elg)与游资信号的绝对契合度，镇压情绪假突破。
        """
        method_name="_calculate_smart_money_ignition"
        required_signals=['HM_COORDINATED_ATTACK_D','T1_PREMIUM_EXPECTATION_D','IS_MARKET_LEADER_D','flow_acceleration_intraday_D','buy_elg_amount_D','tick_large_order_net_D','amount_D','uptrend_strength_D','STATE_BREAKOUT_CONFIRMED_D','net_energy_flow_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        hm=self._get_safe_series(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        t1=self._get_safe_series(df,'T1_PREMIUM_EXPECTATION_D',method_name=method_name)
        ldr=self._get_safe_series(df,'IS_MARKET_LEADER_D',method_name=method_name).clip(0,1)
        f_acc=self._get_safe_series(df,'flow_acceleration_intraday_D',method_name=method_name)
        elg=self._get_safe_series(df,'buy_elg_amount_D',method_name=method_name)
        t_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        up=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        brk=self._get_safe_series(df,'STATE_BREAKOUT_CONFIRMED_D',method_name=method_name).clip(0,1)
        ne=self._get_safe_series(df,'net_energy_flow_D',method_name=method_name)
        hm_shk=0.5*(1.0+np.tanh(_hab('HM_COORDINATED_ATTACK_D',hm,13)))
        t1_shk=0.5*(1.0+np.tanh(_hab('T1_PREMIUM_EXPECTATION_D',t1,21)))
        acc_shk=np.tanh(_hab('flow_acceleration_intraday_D',f_acc,13)).clip(lower=0)
        up_shk=0.5*(1.0+np.tanh(_hab('uptrend_strength_D',up,21)))
        elg_shk=0.5*(1.0+np.tanh(_hab('buy_elg_ratio',(elg/amt).fillna(0),13)))
        tnet_shk=np.tanh(_hab('tick_net_ratio',(t_net/amt).fillna(0),13)).clip(lower=0)
        ne_shk=0.5*(1.0+np.tanh(_hab('net_energy_flow_D',ne,13)))
        k_hm=_kinematics('HM_COORDINATED_ATTACK_D',hm,13)
        trend_gate=(up_shk+brk).clip(0,1)
        leader_lev=1.0+ldr*1.5
        attack_kinetic=hm_shk*t1_shk*(1.0+acc_shk)*elg_shk*(1.0+tnet_shk)*ne_shk
        raw_score=attack_kinetic*leader_lev*trend_gate*(1.0+k_hm.clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_COORDINATED_ATTACK_D':hm,'buy_elg_amount_D':elg},calc_nodes={'trend_gate':trend_gate,'attack_kinetic':attack_kinetic,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_vpa_mf_coherence_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 全息量子动力学版】量价主力相干共振引擎
        8步推演：将筹码收敛度与资金一致性强行锚定于量价效率，执行相干性几何爆破。
        """
        method_name="_calculate_vpa_mf_coherence_resonance"
        required_signals=['MA_COHERENCE_RESONANCE_D','VPA_MF_ADJUSTED_EFF_D','MA_ACCELERATION_EMA_55_D','VPA_ACCELERATION_13D','chip_convergence_ratio_D','flow_consistency_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        mc=self._get_safe_series(df,'MA_COHERENCE_RESONANCE_D',method_name=method_name)
        ve=self._get_safe_series(df,'VPA_MF_ADJUSTED_EFF_D',method_name=method_name)
        ma=self._get_safe_series(df,'MA_ACCELERATION_EMA_55_D',method_name=method_name)
        va=self._get_safe_series(df,'VPA_ACCELERATION_13D',method_name=method_name)
        cc=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        fc=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        mc_shk=0.5*(1.0+np.tanh(_hab('MA_COHERENCE_RESONANCE_D',mc,34)))
        ve_shk=0.5*(1.0+np.tanh(_hab('VPA_MF_ADJUSTED_EFF_D',ve,21)))
        ma_shk=np.tanh(_hab('MA_ACCELERATION_EMA_55_D',ma,21)).clip(lower=0)
        va_shk=np.tanh(_hab('VPA_ACCELERATION_13D',va,13)).clip(lower=0)
        cc_shk=0.5*(1.0+np.tanh(_hab('chip_convergence_ratio_D',cc,34)))
        fc_shk=0.5*(1.0+np.tanh(_hab('flow_consistency_D',fc,21)))
        vac_shk=0.5*(1.0+np.tanh(_hab('volatility_adjusted_concentration_D',vac,21)))
        k_ve=_kinematics('VPA_MF_ADJUSTED_EFF_D',ve,13)
        base_resonance=mc_shk*ve_shk*cc_shk*fc_shk*vac_shk
        kinetic_catalyst=1.0+ma_shk+va_shk
        raw_score=base_resonance*kinetic_catalyst*(1.0+k_ve.clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_COHERENCE_RESONANCE_D':mc,'VPA_MF_ADJUSTED_EFF_D':ve},calc_nodes={'base_resonance':base_resonance,'kinetic_catalyst':kinetic_catalyst,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_institutional_sweep(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】机构超大单扫货核爆引擎
        融合特大单与大单，附加流动性冲击对冲，三阶微积分与HAB系统。
        """
        method_name="_calculate_institutional_sweep"
        required_signals=['buy_elg_amount_D','buy_lg_amount_D','amount_D','tick_chip_transfer_efficiency_D','flow_consistency_D','net_mf_amount_D','flow_impact_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        buy_elg=self._get_safe_series(df,'buy_elg_amount_D',method_name=method_name)
        buy_lg=self._get_safe_series(df,'buy_lg_amount_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        trans_eff=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        impact=self._get_safe_series(df,'flow_impact_ratio_D',method_name=method_name)
        elg_ratio=((buy_elg+buy_lg*0.5)/amount).fillna(0.0)
        elg_shock=0.5*(1.0+np.tanh(_hab(elg_ratio,21)))
        trans_shock=0.5*(1.0+np.tanh(_hab(trans_eff,34)))
        cons_shock=0.5*(1.0+np.tanh(_hab(cons,21)))
        impact_shock=0.5*(1.0+np.tanh(_hab(impact,13)))
        k_tensor=_kinematics('net_mf_amount_D',net_mf,13)
        base_sweep=elg_shock*trans_shock*cons_shock*impact_shock*(1.0+np.tanh(_hab(net_mf,55)).clip(lower=0))
        raw_score=base_sweep*(1.0+k_tensor.clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'buy_elg_amount_D':buy_elg,'amount_D':amount},calc_nodes={'elg_shock':elg_shock,'k_tensor':k_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_hf_algo_manipulation_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 零噪防爆版】高频算法诱骗崩塌防线
        修复底层负数幂律运算崩溃，强制收敛流偏度、峰度与价流背离底噪信号。
        """
        method_name="_calculate_hf_algo_manipulation_risk"
        required_signals=['high_freq_flow_skewness_D','high_freq_flow_kurtosis_D','large_order_anomaly_D','price_flow_divergence_D','intraday_price_distribution_skewness_D','tick_abnormal_volume_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _zg(s: pd.Series) -> pd.Series: return pd.Series(np.where(s.abs()<1e-4,0.0,1.0),index=df_index)
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((s_f-s_f.rolling(w,min_periods=1).mean())/s_f.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', s_f.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        hf_skew=self._get_safe_series(df,'high_freq_flow_skewness_D',method_name=method_name).fillna(0.0)
        hf_kurt=self._get_safe_series(df,'high_freq_flow_kurtosis_D',method_name=method_name).fillna(0.0)
        anomaly=self._get_safe_series(df,'large_order_anomaly_D',method_name=method_name).fillna(0.0)
        divergence=self._get_safe_series(df,'price_flow_divergence_D',method_name=method_name).fillna(0.0)
        price_skew=self._get_safe_series(df,'intraday_price_distribution_skewness_D',method_name=method_name).fillna(0.0)
        abnorm_vol=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name).fillna(0.0)
        skew_shock=np.tanh(_hab('high_freq_flow_skewness_D',hf_skew,21)).abs()
        kurt_shock=np.tanh(_hab('high_freq_flow_kurtosis_D',hf_kurt,34)).clip(lower=0)
        anomaly_shock=np.tanh(_hab('large_order_anomaly_D',anomaly,13)).clip(lower=0)*_zg(anomaly)
        div_shock=np.tanh(_hab('price_flow_divergence_D',divergence,21)).clip(lower=0)*_zg(divergence)
        skew_mismatch=np.tanh(_hab('skew_mismatch',hf_skew-price_skew,13)).abs()
        abn_shock=np.tanh(_hab('tick_abnormal_volume_ratio_D',abnorm_vol,21)).clip(lower=0)*_zg(abnorm_vol)
        k_tensor=_kinematics('large_order_anomaly_D',anomaly,13)
        risk_tensor=skew_shock*(1.0+kurt_shock)*anomaly_shock*abn_shock*(1.0+div_shock)*(1.0+skew_mismatch)
        raw_score=-(risk_tensor*(1.0+k_tensor.clip(upper=0).abs()))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,0).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'high_freq_flow_skewness_D':hf_skew,'large_order_anomaly_D':anomaly},calc_nodes={'skew_shock':skew_shock,'risk_tensor':risk_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_ma_rubber_band_reversal(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.1.0 · 断层免疫修复版】均线张力极值反噬引擎
        移除越级的周期，使用安全的 BIAS_21_D 测算胡克定律拉力。
        """
        method_name="_calculate_ma_rubber_band_reversal"
        required_signals=['MA_RUBBER_BAND_EXTENSION_D','MA_POTENTIAL_TENSION_INDEX_D','ADX_14_D','profit_pressure_D','pressure_trapped_D','BIAS_21_D','reversal_prob_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        rubber_ext=self._get_safe_series(df,'MA_RUBBER_BAND_EXTENSION_D',method_name=method_name)
        tension=self._get_safe_series(df,'MA_POTENTIAL_TENSION_INDEX_D',method_name=method_name)
        adx=self._get_safe_series(df,'ADX_14_D',method_name=method_name)
        profit_p=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        trapped_p=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        bias_21=self._get_safe_series(df,'BIAS_21_D',method_name=method_name)
        rev_prob=self._get_safe_series(df,'reversal_prob_D',method_name=method_name)
        rubber_shock=np.tanh(_hab(rubber_ext,34))
        tension_shock=np.tanh(_hab(tension,21))
        profit_shock=0.5*(1.0+np.tanh(_hab(profit_p,21)))
        trapped_shock=0.5*(1.0+np.tanh(_hab(trapped_p,21)))
        bias_shock=np.tanh(_hab(bias_21,21))
        rev_shock=0.5*(1.0+np.tanh(_hab(rev_prob,13)))
        k_tensor=_kinematics('MA_RUBBER_BAND_EXTENSION_D',rubber_ext,13)
        trend_suppression=1.0-np.tanh(np.maximum(adx-35.0,0.0)/15.0)
        top_force=(rubber_shock.clip(lower=0))*tension_shock.clip(lower=0)*profit_shock*(1.0+bias_shock.clip(lower=0))*(1.0+rev_shock)
        bottom_force=(rubber_shock.clip(upper=0).abs())*tension_shock.clip(lower=0)*trapped_shock*(1.0+bias_shock.clip(upper=0).abs())*(1.0+rev_shock)
        raw_score=(bottom_force-top_force)*trend_suppression*(1.0+np.abs(k_tensor))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_RUBBER_BAND_EXTENSION_D':rubber_ext,'ADX_14_D':adx},calc_nodes={'top_force':top_force,'bottom_force':bottom_force,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_geometric_trend_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】几何流形趋势共振引擎
        基于线性回归R2、斜率与圆弧曲率，惩罚无序波动，奖励流形加速发散。
        """
        method_name="_calculate_geometric_trend_resonance"
        required_signals=['GEOM_REG_R2_D','GEOM_REG_SLOPE_D','GEOM_ARC_CURVATURE_D','GEOM_CHANNEL_POS_D','PRICE_FRACTAL_DIM_D','trend_confirmation_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        r2=self._get_safe_series(df,'GEOM_REG_R2_D',method_name=method_name)
        slope=self._get_safe_series(df,'GEOM_REG_SLOPE_D',method_name=method_name)
        curvature=self._get_safe_series(df,'GEOM_ARC_CURVATURE_D',method_name=method_name)
        channel_pos=self._get_safe_series(df,'GEOM_CHANNEL_POS_D',method_name=method_name)
        fractal_dim=self._get_safe_series(df,'PRICE_FRACTAL_DIM_D',method_name=method_name)
        trend_conf=self._get_safe_series(df,'trend_confirmation_score_D',method_name=method_name)
        r2_gate=np.tanh(_hab(r2,21)).clip(lower=0.1)
        slope_shock=np.tanh(_hab(slope,34))
        curvature_shock=np.tanh(_hab(curvature,21))
        fractal_smoothness=1.0-np.tanh(_hab(fractal_dim,34)).clip(lower=0)
        conf_shock=0.5*(1.0+np.tanh(_hab(trend_conf,21)))
        channel_norm=(channel_pos-0.5)*2.0
        k_tensor=_kinematics('GEOM_REG_SLOPE_D',slope,13)
        rigid_tensor=slope_shock*r2_gate*fractal_smoothness*conf_shock
        manifold_dynamics=curvature_shock-channel_norm*0.3
        raw_score=rigid_tensor*(1.0+manifold_dynamics.clip(lower=0))*(1.0+k_tensor.clip(lower=0))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GEOM_REG_R2_D':r2,'PRICE_FRACTAL_DIM_D':fractal_dim},calc_nodes={'rigid_tensor':rigid_tensor,'manifold_dynamics':manifold_dynamics,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_mtf_fractal_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】多维时空分形共振引擎
        量化日线、周线、月线级别的完美同步。配合分形降噪捕捉宏观大局。
        """
        method_name="_calculate_mtf_fractal_resonance"
        required_signals=['daily_weekly_sync_D','daily_monthly_sync_D','PRICE_FRACTAL_DIM_D','uptrend_continuation_prob_D','mid_long_sync_D','short_mid_sync_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        dw_sync=self._get_safe_series(df,'daily_weekly_sync_D',method_name=method_name)
        dm_sync=self._get_safe_series(df,'daily_monthly_sync_D',method_name=method_name)
        ml_sync=self._get_safe_series(df,'mid_long_sync_D',method_name=method_name)
        sm_sync=self._get_safe_series(df,'short_mid_sync_D',method_name=method_name)
        fractal_dim=self._get_safe_series(df,'PRICE_FRACTAL_DIM_D',method_name=method_name)
        prob=self._get_safe_series(df,'uptrend_continuation_prob_D',method_name=method_name)
        sync_tensor=0.5*(1.0+np.tanh(_hab(dw_sync+dm_sync+ml_sync+sm_sync,21)))
        fractal_smoothness=1.0-np.tanh(_hab(fractal_dim,34)).clip(lower=0)
        prob_shock=0.5*(1.0+np.tanh(_hab(prob,21)))
        k_tensor=_kinematics('daily_weekly_sync_D',dw_sync,13)
        raw_score=sync_tensor*fractal_smoothness*prob_shock*(1.0+k_tensor.clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'daily_weekly_sync_D':dw_sync,'PRICE_FRACTAL_DIM_D':fractal_dim},calc_nodes={'sync_tensor':sync_tensor,'fractal_smoothness':fractal_smoothness,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_intraday_siege_exhaustion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】日内攻城拔寨衰竭引擎
        透视日内多空在阻力/支撑位上的高频消耗战与攻防强度衰竭。
        """
        method_name="_calculate_intraday_siege_exhaustion"
        required_signals=['intraday_resistance_test_count_D','intraday_support_test_count_D','CLOSING_STRENGTH_D','vwap_deviation_D','resistance_strength_D','support_strength_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        res_tests=self._get_safe_series(df,'intraday_resistance_test_count_D',method_name=method_name)
        sup_tests=self._get_safe_series(df,'intraday_support_test_count_D',method_name=method_name)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name)
        vwap_dev=self._get_safe_series(df,'vwap_deviation_D',method_name=method_name)
        res_str=self._get_safe_series(df,'resistance_strength_D',method_name=method_name)
        sup_str=self._get_safe_series(df,'support_strength_D',method_name=method_name)
        res_shock=np.tanh(res_tests/3.0)*(1.0+np.tanh(_hab(res_str,21)).clip(lower=0))
        sup_shock=np.tanh(sup_tests/3.0)*(1.0+np.tanh(_hab(sup_str,21)).clip(lower=0))
        closing_shock=0.5*(1.0+np.tanh(_hab(closing,21)))
        vwap_shock=np.tanh(_hab(vwap_dev,21))
        k_tensor=_kinematics('CLOSING_STRENGTH_D',closing,13)
        breakout_force=res_shock*closing_shock*(1.0+vwap_shock.clip(lower=0))
        breakdown_force=sup_shock*(1.0-closing_shock)*(1.0+vwap_shock.clip(upper=0).abs())
        raw_score=(breakout_force-breakdown_force)*(1.0+np.abs(k_tensor))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'intraday_resistance_test_count_D':res_tests,'CLOSING_STRENGTH_D':closing},calc_nodes={'breakout_force':breakout_force,'breakdown_force':breakdown_force,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_overnight_intraday_tearing(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 零噪防爆版】隔夜跳空与日内动能撕裂引擎
        加装绝对门控阀值。无跳空不启动，杜绝0Gap产生的虚假幽灵撕裂。
        """
        method_name="_calculate_overnight_intraday_tearing"
        required_signals=['GAP_MOMENTUM_STRENGTH_D','OCH_ACCELERATION_D','CLOSING_STRENGTH_D','intraday_price_range_ratio_D','morning_flow_ratio_D','afternoon_flow_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _zg(s: pd.Series) -> pd.Series: return pd.Series(np.where(s.abs()<1e-3,0.0,1.0),index=df_index)
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((s_f-s_f.rolling(w,min_periods=1).mean())/s_f.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', s_f.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        gap=self._get_safe_series(df,'GAP_MOMENTUM_STRENGTH_D',method_name=method_name).fillna(0.0)
        och=self._get_safe_series(df,'OCH_ACCELERATION_D',method_name=method_name).fillna(0.0)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name).fillna(50.0)
        range_ratio=self._get_safe_series(df,'intraday_price_range_ratio_D',method_name=method_name).fillna(0.0)
        morning=self._get_safe_series(df,'morning_flow_ratio_D',method_name=method_name).fillna(50.0)
        afternoon=self._get_safe_series(df,'afternoon_flow_ratio_D',method_name=method_name).fillna(50.0)
        gap_shock=np.tanh(_hab('GAP_MOMENTUM_STRENGTH_D',gap,13))*_zg(gap)
        och_shock=np.tanh(_hab('OCH_ACCELERATION_D',och,13))
        closing_norm=closing/100.0
        range_shock=np.tanh(_hab('intraday_price_range_ratio_D',range_ratio,21)).clip(lower=0)
        flow_tearing=np.tanh(_hab('flow_tearing', morning-afternoon,13)).clip(lower=0)
        k_tensor=_kinematics('GAP_MOMENTUM_STRENGTH_D',gap,13)
        tearing_vector=gap_shock*(closing_norm*2.0-1.0+och_shock-flow_tearing)
        leverage=1.0+(range_shock*np.abs(closing_norm-0.5)*2.0)
        raw_score=tearing_vector*leverage*(1.0+k_tensor*np.sign(tearing_vector))*_zg(gap)
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GAP_MOMENTUM_STRENGTH_D':gap,'CLOSING_STRENGTH_D':closing},calc_nodes={'tearing_vector':tearing_vector,'leverage':leverage,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_chip_center_kinematics(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 零噪防爆版】筹码重心迁徙动力学引擎
        彻底斩除单向常数底噪，无大位移无判定，根除误报雪崩。
        """
        method_name="_calculate_chip_center_kinematics"
        required_signals=['peak_migration_speed_5d_D','intraday_cost_center_volatility_D','price_to_weight_avg_ratio_D','turnover_rate_f_D','cost_50pct_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _zg(s: pd.Series) -> pd.Series: return pd.Series(np.where(s.abs()<1e-4,0.0,1.0),index=df_index)
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((s_f-s_f.rolling(w,min_periods=1).mean())/s_f.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', s_f.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        migration=self._get_safe_series(df,'peak_migration_speed_5d_D',method_name=method_name).fillna(0.0)
        cost_vol=self._get_safe_series(df,'intraday_cost_center_volatility_D',method_name=method_name).fillna(0.0)
        price_to_cost=self._get_safe_series(df,'price_to_weight_avg_ratio_D',method_name=method_name).fillna(1.0)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name).fillna(0.0)
        cost_50=self._get_safe_series(df,'cost_50pct_D',method_name=method_name).fillna(0.0)
        mig_shock=np.tanh(_hab('peak_migration_speed_5d_D',migration,21)).clip(lower=0)*_zg(migration)
        vol_shock=np.tanh(_hab('intraday_cost_center_volatility_D',cost_vol,13)).clip(lower=0)*_zg(cost_vol)
        dev_shock=np.tanh(_hab('price_to_weight_avg_ratio_D',price_to_cost,21))
        to_shock=np.tanh(_hab('turnover_rate_f_D',turnover,13)).clip(lower=0)*_zg(turnover)
        cost_shock=np.tanh(_hab('cost_50pct_D',cost_50,21))
        k_tensor=_kinematics('cost_50pct_D',cost_50,13)
        distribution_kinetic=mig_shock*vol_shock*to_shock*(1.0+cost_shock.clip(lower=0))
        lock_kinetic=(1.0-mig_shock)*(1.0-vol_shock)*dev_shock.clip(lower=0)
        raw_score=(lock_kinetic-distribution_kinetic)*(1.0+np.abs(k_tensor))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_migration_speed_5d_D':migration,'cost_50pct_D':cost_50},calc_nodes={'distribution_kinetic':distribution_kinetic,'lock_kinetic':lock_kinetic,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_ma_compression_explosion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】均线奇点核爆引擎
        多周期均线极度粘合与筹码高度密集下的绝对势能奇点引爆。
        """
        method_name="_calculate_ma_compression_explosion"
        required_signals=['MA_POTENTIAL_COMPRESSION_RATE_D','chip_convergence_ratio_D','TURNOVER_STABILITY_INDEX_D','MACDh_13_34_8_D','energy_concentration_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        ma_comp=self._get_safe_series(df,'MA_POTENTIAL_COMPRESSION_RATE_D',method_name=method_name)
        chip_conv=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        to_stab=self._get_safe_series(df,'TURNOVER_STABILITY_INDEX_D',method_name=method_name)
        macd_hist=self._get_safe_series(df,'MACDh_13_34_8_D',method_name=method_name)
        energy_conc=self._get_safe_series(df,'energy_concentration_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        comp_shock=np.tanh(_hab(ma_comp,34)).clip(lower=0)
        conv_shock=0.5*(1.0+np.tanh(_hab(chip_conv,34)))
        stab_shock=0.5*(1.0+np.tanh(_hab(to_stab,21)))
        energy_shock=0.5*(1.0+np.tanh(_hab(energy_conc,21)))
        vac_shock=0.5*(1.0+np.tanh(_hab(vac,21)))
        ignition=np.tanh(_hab(macd_hist,13))
        k_tensor=_kinematics('MA_POTENTIAL_COMPRESSION_RATE_D',ma_comp,13)
        potential_singularity=(comp_shock*conv_shock*stab_shock*energy_shock*vac_shock)**1.5
        raw_score=potential_singularity*np.sign(ignition)*(np.abs(ignition)**0.5)*(1.0+k_tensor.clip(lower=0))
        final_score=np.tanh(raw_score*2.0).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_POTENTIAL_COMPRESSION_RATE_D':ma_comp,'chip_convergence_ratio_D':chip_conv},calc_nodes={'comp_shock':comp_shock,'potential_singularity':potential_singularity,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_top_tier_hm_harvesting(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 零噪防爆版】顶级游资收割镰刀引擎
        挂载活性门控。若游资并未出动，强行切断核按钮推演判定，修复负指数防爆。
        """
        method_name="_calculate_top_tier_hm_harvesting"
        required_signals=['HM_ACTIVE_TOP_TIER_D','CLOSING_STRENGTH_D','tick_large_order_net_D','amount_D','outflow_quality_D','distribution_energy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _zg(s: pd.Series) -> pd.Series: return pd.Series(np.where(s.abs()<1e-4,0.0,1.0),index=df_index)
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((s_f-s_f.rolling(w,min_periods=1).mean())/s_f.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', s_f.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        hm_active=self._get_safe_series(df,'HM_ACTIVE_TOP_TIER_D',method_name=method_name).fillna(0.0)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name).fillna(50.0)
        large_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name).fillna(0.0)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        outflow_q=self._get_safe_series(df,'outflow_quality_D',method_name=method_name).fillna(0.0)
        dist_eng=self._get_safe_series(df,'distribution_energy_D',method_name=method_name).fillna(0.0)
        hm_shock=np.tanh(_hab('HM_ACTIVE_TOP_TIER_D',hm_active,13)).clip(lower=0)*_zg(hm_active)
        closing_weakness=(1.0-(closing/100.0)).clip(0,1)*(1.0-np.tanh(_hab('CLOSING_STRENGTH_D',closing,21)).clip(upper=0))
        net_ratio=(large_net/amount).fillna(0.0)
        net_shock=np.tanh(_hab('net_ratio',net_ratio,13))
        outflow_shock=np.tanh(_hab('outflow_quality_D',outflow_q,21)).clip(lower=0)*_zg(outflow_q)
        dist_shock=np.tanh(_hab('distribution_energy_D',dist_eng,21)).clip(lower=0)*_zg(dist_eng)
        k_tensor=_kinematics('tick_large_order_net_D',large_net,13)
        dumping_force=net_shock.clip(upper=0).abs()*(1.0+outflow_shock)*(1.0+dist_shock)
        harvesting_tensor=hm_shock*closing_weakness*dumping_force*(1.0+k_tensor.clip(upper=0).abs())*_zg(hm_active)
        raw_score=-(harvesting_tensor**1.5)
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,0).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_ACTIVE_TOP_TIER_D':hm_active,'tick_large_order_net_D':large_net},calc_nodes={'hm_shock':hm_shock,'closing_weakness':closing_weakness,'dumping_force':dumping_force,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_vwap_magnetic_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.1.0 · 断层免疫修复版】VWAP磁性黑洞引力引擎
        利用历史反转概率的偏离判断抄底或假突破被反噬的必然结果。
        """
        method_name="_calculate_vwap_magnetic_divergence"
        required_signals=['vwap_deviation_D','reversal_prob_D','intraday_main_force_activity_D','intraday_cost_center_migration_D','volume_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        vwap_dev=self._get_safe_series(df,'vwap_deviation_D',method_name=method_name)
        rev_prob=self._get_safe_series(df,'reversal_prob_D',method_name=method_name)
        mf_activity=self._get_safe_series(df,'intraday_main_force_activity_D',method_name=method_name)
        cost_mig=self._get_safe_series(df,'intraday_cost_center_migration_D',method_name=method_name)
        vol_ratio=self._get_safe_series(df,'volume_ratio_D',method_name=method_name)
        dev_shock=np.tanh(_hab(vwap_dev,13))
        corr_shock=0.5*(1.0+np.tanh(_hab(rev_prob,34)))
        mf_shock=np.tanh((mf_activity-50.0)/20.0)
        mig_shock=np.tanh(_hab(cost_mig,21))
        vr_shock=0.5*(1.0+np.tanh(_hab(vol_ratio,13)))
        k_tensor=_kinematics('vwap_deviation_D',vwap_dev,13)
        mismatch_penalty=(1.0-mf_shock*np.sign(dev_shock))*(1.0-mig_shock*np.sign(dev_shock))
        magnetic_pull=-1.0*dev_shock*corr_shock*mismatch_penalty.clip(lower=0.1)*(1.0+vr_shock)
        raw_score=np.sign(magnetic_pull)*(np.abs(magnetic_pull)**1.5)*(1.0+np.abs(k_tensor))
        final_score=np.tanh(raw_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'vwap_deviation_D':vwap_dev,'reversal_prob_D':rev_prob},calc_nodes={'dev_shock':dev_shock,'magnetic_pull':magnetic_pull,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_multi_peak_avalanche_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】多峰筹码断层雪崩引擎
        专治复杂弱势震荡的多峰发散，加入筹码熵值监控双向套牢盘崩盘。
        """
        method_name="_calculate_multi_peak_avalanche_risk"
        required_signals=['is_multi_peak_D','chip_divergence_ratio_D','intraday_distribution_confidence_D','downtrend_strength_D','chip_entropy_D','chip_stability_change_5d_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        multi_peak=self._get_safe_series(df,'is_multi_peak_D',method_name=method_name).clip(0,1)
        divergence=self._get_safe_series(df,'chip_divergence_ratio_D',method_name=method_name)
        dist_conf=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        downtrend=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        entropy=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        stab_chg=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        div_shock=np.tanh(_hab(divergence,21)).clip(lower=0)
        dist_shock=0.5*(1.0+np.tanh(_hab(dist_conf,13)))
        down_shock=0.5*(1.0+np.tanh(_hab(downtrend,21)))
        ent_shock=0.5*(1.0+np.tanh(_hab(entropy,21)))
        stab_shock=np.tanh(_hab(stab_chg,13)).clip(upper=0).abs()
        k_tensor=_kinematics('chip_divergence_ratio_D',divergence,13)
        avalanche_tensor=multi_peak*div_shock*dist_shock*down_shock*ent_shock*(1.0+stab_shock)*(1.0+k_tensor.clip(lower=0))
        raw_score=-(avalanche_tensor**1.5)
        final_score=np.tanh(raw_score*2.0).clip(-1,0).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'is_multi_peak_D':multi_peak,'chip_divergence_ratio_D':divergence},calc_nodes={'div_shock':div_shock,'avalanche_tensor':avalanche_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_sector_lifecycle_tailwind(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】板块生命周期顺风引擎
        融合个股趋势与生命周期四象限共振，大幅奖励预热/拉升周期。
        """
        method_name="_calculate_sector_lifecycle_tailwind"
        required_signals=['industry_preheat_score_D','industry_markup_score_D','industry_stagnation_score_D','industry_downtrend_score_D','uptrend_strength_D','industry_rank_accel_D','industry_strength_rank_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        preheat=self._get_safe_series(df,'industry_preheat_score_D',method_name=method_name)
        markup=self._get_safe_series(df,'industry_markup_score_D',method_name=method_name)
        stagnation=self._get_safe_series(df,'industry_stagnation_score_D',method_name=method_name)
        downtrend=self._get_safe_series(df,'industry_downtrend_score_D',method_name=method_name)
        uptrend=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        rank_accel=self._get_safe_series(df,'industry_rank_accel_D',method_name=method_name)
        rank=self._get_safe_series(df,'industry_strength_rank_D',method_name=method_name)
        pre_shock=0.5*(1.0+np.tanh(_hab(preheat,21)))
        mark_shock=0.5*(1.0+np.tanh(_hab(markup,21)))
        stag_shock=0.5*(1.0+np.tanh(_hab(stagnation,21)))
        down_shock=0.5*(1.0+np.tanh(_hab(downtrend,21)))
        up_shock=0.5*(1.0+np.tanh(_hab(uptrend,21)))
        rank_shock=0.5*(1.0+np.tanh(_hab(rank,34)))
        accel_shock=np.tanh(_hab(rank_accel,13))
        k_tensor=_kinematics('industry_markup_score_D',markup,13)
        sector_vector=(mark_shock*1.0+pre_shock*0.5)-(stag_shock*0.5+down_shock*1.0)
        raw_score=sector_vector*up_shock*(1.0+rank_shock)*(1.0+accel_shock*np.sign(sector_vector))*(1.0+k_tensor.clip(lower=0))
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=np.tanh(final_score*2.0).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'industry_markup_score_D':markup,'industry_downtrend_score_D':downtrend},calc_nodes={'sector_vector':sector_vector,'up_shock':up_shock,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_time_asymmetry_trap(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】日内时间非对称陷阱防爆引擎
        彻底拦截A股经典早盘杀猪盘陷阱，建立流占比时空对冲验证。
        """
        method_name="_calculate_time_asymmetry_trap"
        required_signals=['morning_flow_ratio_D','afternoon_flow_ratio_D','intraday_peak_valley_ratio_D','profit_pressure_D','closing_flow_ratio_D','high_freq_flow_divergence_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        morning=self._get_safe_series(df,'morning_flow_ratio_D',method_name=method_name)
        afternoon=self._get_safe_series(df,'afternoon_flow_ratio_D',method_name=method_name)
        closing=self._get_safe_series(df,'closing_flow_ratio_D',method_name=method_name)
        peak_valley=self._get_safe_series(df,'intraday_peak_valley_ratio_D',method_name=method_name)
        profit_p=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        hf_div=self._get_safe_series(df,'high_freq_flow_divergence_D',method_name=method_name)
        asymmetry=(afternoon+closing*0.5-morning)/100.0
        asym_shock=np.tanh(_hab(asymmetry,13))
        pv_shock=0.5*(1.0+np.tanh(_hab(peak_valley,21)))
        press_shock=0.5*(1.0+np.tanh(_hab(profit_p,21)))
        div_shock=np.tanh(_hab(hf_div,21)).clip(lower=0)
        k_tensor=_kinematics('morning_flow_ratio_D',morning,13)
        trap_power=asym_shock*(1.0+pv_shock*np.sign(-asym_shock))*(1.0+press_shock*np.sign(-asym_shock))*(1.0+div_shock*np.sign(-asym_shock))
        raw_score=np.sign(trap_power)*(np.abs(trap_power)**1.5)*(1.0+np.abs(k_tensor))
        final_score=np.tanh(raw_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'morning_flow_ratio_D':morning,'afternoon_flow_ratio_D':afternoon},calc_nodes={'asym_shock':asym_shock,'trap_power':trap_power,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_high_pos_liquidity_squeeze(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0.0 · 全息物理防爆版】高位流动性真空轧空引擎
        计算无短期筹码干扰的绝对真空区爆发抛物线轧空行情。
        """
        method_name="_calculate_high_pos_liquidity_squeeze"
        required_signals=['price_percentile_position_D','high_position_lock_ratio_90_D','flow_persistence_minutes_D','short_term_chip_ratio_D','uptrend_strength_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _hab(s: pd.Series, w: int) -> pd.Series:
            return ((s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            slope=df.get(f'SLOPE_{w}_{col}', s.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).astype(np.float32)
        percentile=self._get_safe_series(df,'price_percentile_position_D',method_name=method_name)
        lock_ratio=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name)
        persistence=self._get_safe_series(df,'flow_persistence_minutes_D',method_name=method_name)
        short_chip=self._get_safe_series(df,'short_term_chip_ratio_D',method_name=method_name)
        uptrend=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        pos_norm=np.tanh(percentile/50.0) if percentile.max()>1.0 else np.tanh(percentile*2.0)
        lock_shock=0.5*(1.0+np.tanh(_hab(lock_ratio,21)))
        persist_shock=0.5*(1.0+np.tanh(_hab(persistence,13)))
        short_shock=0.5*(1.0+np.tanh(_hab(short_chip,21)))
        up_shock=0.5*(1.0+np.tanh(_hab(uptrend,21)))
        to_shock=1.0-np.tanh(_hab(turnover,21)).clip(lower=0)
        k_tensor=_kinematics('high_position_lock_ratio_90_D',lock_ratio,13)
        squeeze_tensor=lock_shock*persist_shock*(1.0-short_shock)*up_shock*to_shock*pos_norm.clip(lower=0)
        raw_score=(squeeze_tensor**1.5)*(1.0+k_tensor.clip(lower=0))
        final_score=np.tanh(raw_score*2.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'price_percentile_position_D':percentile,'high_position_lock_ratio_90_D':lock_ratio},calc_nodes={'lock_shock':lock_shock,'squeeze_tensor':squeeze_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_institutional_structural_exit(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.2.0 · 零噪防爆版】机构结构性清仓逃顶引擎
        杜绝无卖单状态下的连带惩罚，增设核爆级逃生门控，无真实特大卖单不触发。
        """
        method_name="_calculate_institutional_structural_exit"
        required_signals=['sell_elg_amount_D','sell_lg_amount_D','amount_D','distribution_energy_D','downtrend_strength_D','high_position_lock_ratio_90_D','chip_stability_change_5d_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        def _zg(s: pd.Series) -> pd.Series: return pd.Series(np.where(s.abs()<1e-4,0.0,1.0),index=df_index)
        def _hab(col: str, s: pd.Series, w: int) -> pd.Series:
            hab_col=f'HAB_{w}_{col}'
            if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            return ((s_f-s_f.rolling(w,min_periods=1).mean())/s_f.rolling(w,min_periods=1).std().replace(0,1e-5).fillna(1e-5)).fillna(0.0).astype(np.float32)
        def _kinematics(col: str, s: pd.Series, w: int) -> pd.Series:
            s_f=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
            slope=df.get(f'SLOPE_{w}_{col}', s_f.diff(w)/w).fillna(0.0)
            accel=df.get(f'ACCEL_{w}_{col}', slope.diff(w)/w).fillna(0.0)
            jerk=df.get(f'JERK_{w}_{col}', accel.diff(w)/w).fillna(0.0)
            tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
            return np.tanh(pd.Series(tensor,index=df_index)*10.0).fillna(0.0).astype(np.float32)
        sell_elg=self._get_safe_series(df,'sell_elg_amount_D',method_name=method_name).fillna(0.0)
        sell_lg=self._get_safe_series(df,'sell_lg_amount_D',method_name=method_name).fillna(0.0)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        dist_energy=self._get_safe_series(df,'distribution_energy_D',method_name=method_name).fillna(0.0)
        downtrend=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name).fillna(0.0)
        lock_ratio=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name).fillna(0.0)
        stab_change=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name).fillna(0.0)
        sell_ratio=((sell_elg+sell_lg*0.5)/amount).fillna(0.0)
        sell_shock=np.tanh(_hab('sell_ratio',sell_ratio,13)).clip(lower=0)*_zg(sell_ratio)
        dist_shock=0.5*(1.0+np.tanh(_hab('distribution_energy_D',dist_energy,21)))
        down_shock=0.5*(1.0+np.tanh(_hab('downtrend_strength_D',downtrend,21)))
        lock_diff=lock_ratio.diff(1).fillna(0.0)
        lock_break_shock=np.tanh(_hab('lock_diff',lock_diff,13)).clip(upper=0).abs()
        stab_break_shock=np.tanh(_hab('chip_stability_change_5d_D',stab_change,13)).clip(upper=0).abs()
        k_tensor=_kinematics('sell_ratio',sell_ratio,13)
        exit_tensor=sell_shock*dist_shock*down_shock*(1.0+lock_break_shock)*(1.0+stab_break_shock)
        raw_score=-(exit_tensor**1.5)*(1.0+k_tensor.clip(lower=0))*_zg(sell_ratio)
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**2.5)).clip(-1,0).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'sell_elg_amount_D':sell_elg,'high_position_lock_ratio_90_D':lock_ratio},calc_nodes={'sell_shock':sell_shock,'lock_break_shock':lock_break_shock,'raw_score':raw_score},final_result=final_score)
        return final_score








