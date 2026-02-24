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
        【V20.2.0 · 终极路由解耦版】
        修复由于二次求导(displacement/momentum)导致的持续极值稳态信号被错误归零的漏洞。
        """
        signal_name=config.get('name')
        df_index=df.index
        meta_score=pd.Series(dtype=np.float32)
        if signal_name=='PROCESS_META_COST_ADVANTAGE_TREND':
            meta_score=self.calculate_cost_advantage_trend_relationship_processor.calculate(df,config)
        elif signal_name=='PROCESS_STRATEGY_DYN_VS_CHIP_DECAY':
            meta_score=self._calculate_dyn_vs_chip_relationship(df,config)
        elif signal_name=='PROCESS_META_PRICE_VS_RETAIL_CAPITULATION':
            meta_score=self._calculate_price_vs_capitulation_relationship(df,config)
        elif signal_name=='PROCESS_META_PD_DIVERGENCE_CONFIRM':
            meta_score=self._calculate_pd_divergence_relationship(df,config)
        elif signal_name=='PROCESS_META_PROFIT_VS_FLOW':
            meta_score=self._calculate_profit_vs_flow_relationship(df,config)
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
            meta_score=self._calculate_ff_vs_structure_relationship(df,config)
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
            meta_score=self._calculate_stock_sector_sync(df,config)
        elif signal_name=='PROCESS_META_HOT_SECTOR_COOLING':
            meta_score=self._calculate_hot_sector_cooling(df,config)
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
        if meta_score is None or meta_score.empty:
            return {}
        return {signal_name:meta_score}

    def _diagnose_split_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """【V9.0.0 · 全局矢量化重构】裂变张量引擎"""
        states={}
        output_names=config.get('output_names',{})
        opportunity_signal_name=output_names.get('opportunity')
        risk_signal_name=output_names.get('risk')
        if not opportunity_signal_name or not risk_signal_name: return {}
        relationship_score=self._calculate_price_efficiency_relationship(df,config)
        if relationship_score.empty: return {}
        df_index=df.index
        relationship_displacement=relationship_score.diff(self.meta_window).fillna(0.0)
        relationship_momentum=relationship_displacement.diff(1).fillna(0.0)
        bipolar_displacement=np.tanh(self._apply_hab(df,'disp',relationship_displacement,self.meta_window*2))
        bipolar_momentum=np.tanh(self._apply_hab(df,'mom',relationship_momentum,13))
        displacement_weight=self.meta_score_weights[0]
        momentum_weight=self.meta_score_weights[1]
        meta_score=(bipolar_displacement*displacement_weight+bipolar_momentum*momentum_weight)
        meta_score=np.sign(meta_score)*(np.abs(meta_score)**1.5)
        meta_score=meta_score.clip(-1,1).fillna(0.0)
        states[opportunity_signal_name]=meta_score.clip(lower=0).astype(np.float32)
        states[risk_signal_name]=meta_score.clip(upper=0).abs().astype(np.float32)
        return states

    def _judge_domain_reversal(self, bipolar_domain_health: pd.Series, config: Dict, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V9.0.0 · 全局矢量化重构】神谕审判庭防线"""
        domain_name=config.get('domain_name','未知领域')
        output_bottom_name=config.get('output_bottom_reversal_name')
        output_top_name=config.get('output_top_reversal_name')
        df_index=df.index
        health_yesterday=bipolar_domain_health.shift(1).fillna(0.0)
        health_change=bipolar_domain_health.diff(1).fillna(0.0)
        shock=self._apply_hab(df,'reversal',health_change,13)
        bottom_context=(1.0-health_yesterday).clip(0,2.0)
        bottom_shock=shock.clip(lower=0)
        bottom_reversal_raw=(bottom_shock*bottom_context)**1.5
        bottom_reversal_score=np.tanh(bottom_reversal_raw).clip(0,1).fillna(0.0)
        top_context=(1.0+health_yesterday).clip(0,2.0)
        top_shock=shock.clip(upper=0).abs()
        top_reversal_raw=(top_shock*top_context)**1.5
        top_reversal_score=np.tanh(top_reversal_raw).clip(0,1).fillna(0.0)
        self._probe_variables(method_name=f"_judge_domain_reversal ({domain_name})",df_index=df_index,raw_inputs={'bipolar_domain_health':bipolar_domain_health},calc_nodes={'health_change':health_change,'shock':shock,'bottom_context':bottom_context,'top_context':top_context},final_result=bottom_reversal_score)
        return {output_bottom_name:bottom_reversal_score.astype(np.float32),output_top_name:top_reversal_score.astype(np.float32)}

    def _perform_meta_analysis_on_score(self, relationship_score: pd.Series, config: Dict, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """【V9.0.0 · 全局矢量化重构】元动力学分析"""
        signal_name=config.get('name')
        relationship_displacement=relationship_score.diff(self.meta_window).fillna(0.0)
        relationship_momentum=relationship_displacement.diff(1).fillna(0.0)
        bipolar_displacement=np.tanh(self._apply_hab(df,'disp',relationship_displacement,self.meta_window*2))
        bipolar_momentum=np.tanh(self._apply_hab(df,'mom',relationship_momentum,13))
        instant_score_normalized=(relationship_score+1.0)/2.0
        weight_momentum=(1.0-instant_score_normalized).clip(0,1)
        weight_displacement=1.0-weight_momentum
        meta_score=(bipolar_displacement*weight_displacement+bipolar_momentum*weight_momentum)
        meta_score=np.sign(meta_score)*(np.abs(meta_score)**1.2)
        if config.get('diagnosis_mode','meta_analysis')=='gated_meta_analysis':
            gate_config=config.get('gate_condition',{})
            if gate_config.get('type')=='price_vs_ma':
                ma_period=gate_config.get('ma_period',5)
                ma_col=f'EMA_{ma_period}_D'
                if ma_col in df.columns and 'close_D' in df.columns:
                    gate_is_open=(df['close_D']<df[ma_col]).astype(float)
                    meta_score=meta_score*gate_is_open
        scoring_mode=self.score_type_map.get(signal_name,{}).get('scoring_mode','unipolar')
        if scoring_mode=='unipolar': meta_score=meta_score.clip(lower=0)
        return meta_score.clip(-1,1).fillna(0.0).astype(np.float32)

    def _apply_zg(self, df_index: pd.Index, s: pd.Series) -> pd.Series:
        """【V9.0.0】绝对零基防御门限"""
        return pd.Series(np.where(s.abs()<1e-4,0.0,1.0),index=df_index,dtype=np.float32)

    def _apply_norm(self, s: pd.Series, max_v: float=100.0) -> pd.Series:
        """【V9.0.0】线性边界防爆截断"""
        return (s.fillna(0.0)/max_v).clip(0,1).astype(np.float32)

    def _apply_hab(self, df: pd.DataFrame, col_name: str, s: pd.Series, w: int) -> pd.Series:
        """【V9.0.0】多维HAB历史缓冲池 (带方差正则化防爆)"""
        hab_col=f'HAB_{w}_{col_name}'
        if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
        sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
        roll_mean=sf.rolling(w,min_periods=1).mean()
        roll_std=np.sqrt(sf.rolling(w,min_periods=1).std()**2+1e-5)
        return ((sf-roll_mean)/roll_std).fillna(0.0).astype(np.float32)

    def _apply_kinematics(self, df: pd.DataFrame, col_name: str, s: pd.Series, w: int) -> pd.Series:
        """【V9.0.0】三阶动力学张量引擎 (量纲对齐与软截断)"""
        sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
        slope=df.get(f'SLOPE_{w}_{col_name}',sf.diff(w)/w).fillna(0.0)
        accel=df.get(f'ACCEL_{w}_{col_name}',slope.diff(w)/w).fillna(0.0)
        jerk=df.get(f'JERK_{w}_{col_name}',accel.diff(w)/w).fillna(0.0)
        tensor=np.where(slope.abs()<1e-4,0.0,slope)+np.where(accel.abs()<1e-4,0.0,accel)*0.5+np.where(jerk.abs()<1e-4,0.0,jerk)*0.25
        return np.tanh(pd.Series(tensor,index=s.index)*10.0).fillna(0.0).astype(np.float32)

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

    def _calculate_power_transfer(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息相空间防爆版】微观权力交接张量"""
        method_name="_calculate_power_transfer"
        required_signals=['net_mf_amount_D','amount_D','tick_large_order_net_D','tick_chip_transfer_efficiency_D','flow_efficiency_D','intraday_cost_center_migration_D','downtrend_strength_D','chip_concentration_ratio_D','volatility_adjusted_concentration_D','turnover_rate_f_D']
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
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        mf_ratio=(net_mf/amount).fillna(0.0)
        tick_ratio=(tick_large_net/amount).fillna(0.0)
        mf_shock=np.tanh(self._apply_hab(df,'mf_ratio',mf_ratio,21))
        tick_shock=np.tanh(self._apply_hab(df,'tick_ratio',tick_ratio,21))
        cost_mig_shock=np.tanh(self._apply_hab(df,'intraday_cost_center_migration_D',cost_migration,13))
        to_shock=1.0-np.tanh(self._apply_hab(df,'turnover_rate_f_D',turnover,13)).clip(lower=0)
        vac_amp=0.5*(1.0+np.tanh(self._apply_hab(df,'volatility_adjusted_concentration_D',vac,21)))
        power_core=(mf_shock*0.4+tick_shock*0.2+cost_mig_shock*0.2+to_shock*0.1+vac_amp*0.1)*self._apply_zg(df_index,net_mf)
        transfer_eff_norm=0.5*(1.0+np.tanh(self._apply_hab(df,'tick_chip_transfer_efficiency_D',transfer_eff,21)))
        flow_eff_norm=0.5*(1.0+np.tanh(self._apply_hab(df,'flow_efficiency_D',flow_eff,21)))
        efficiency_multiplier=1.0+(transfer_eff_norm+flow_eff_norm)/2.0
        chip_diff_shock=np.tanh(self._apply_hab(df,'chip_conc_diff',chip_conc.diff().fillna(0.0),13))
        chip_penetration=chip_diff_shock*efficiency_multiplier
        synergy_amplifier=(1.0+chip_penetration*np.sign(power_core)).clip(lower=0.1)
        k_tensor=self._apply_kinematics(df,'net_mf_amount_D',net_mf,13)
        raw_score=power_core*synergy_amplifier*(1.0+np.abs(k_tensor))
        trend_discount=pd.Series(1.0,index=df_index).mask((downtrend>0.8)&(raw_score>0),0.6)
        final_score=np.sign(raw_score)*(np.abs(raw_score)**1.5)
        final_score=(np.tanh(final_score)*trend_discount).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':net_mf,'amount_D':amount},calc_nodes={'power_core':power_core,'synergy_amplifier':synergy_amplifier,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_price_vs_capitulation_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】价格与散户投降背离"""
        method_name="_calculate_price_vs_capitulation_relationship"
        required_signals=['pressure_trapped_D','INTRADAY_SUPPORT_INTENT_D','intraday_low_lock_ratio_D','chip_entropy_D','volatility_adjusted_concentration_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        pressure=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        support=self._get_safe_series(df,'INTRADAY_SUPPORT_INTENT_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        entropy=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        k_p=self._apply_kinematics(df,'pressure_trapped_D',pressure,13)
        panic_base=self._apply_norm(pressure,100.0)*self._apply_zg(df_index,pressure)
        support_mult=np.tanh(self._apply_hab(df,'INTRADAY_SUPPORT_INTENT_D',support,34)).clip(lower=0)
        lock_mult=self._apply_norm(low_lock,1.0)
        ent_penal=1.0-self._apply_norm(entropy,10.0)*0.5
        vac_amp=0.5*(1.0+np.tanh(self._apply_hab(df,'volatility_adjusted_concentration_D',vac,21)))
        to_penal=1.0-np.tanh(self._apply_hab(df,'turnover_rate_f_D',turnover,13)).clip(lower=0)*0.5
        absorption_resonance=1.0+(support_mult+lock_mult+vac_amp)/3.0*ent_penal*to_penal
        base_divergence=self._calculate_instantaneous_relationship(df,config).fillna(0.0)
        raw_score=base_divergence*panic_base*absorption_resonance*(1.0+k_p.clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':pressure},calc_nodes={'panic_base':panic_base,'absorption_resonance':absorption_resonance},final_result=final_score)
        return final_score

    def _calculate_price_efficiency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】价格效率博弈引擎"""
        method_name="_calculate_price_efficiency_relationship"
        required_signals=['VPA_EFFICIENCY_D','net_mf_amount_D','shakeout_score_D','tick_chip_transfer_efficiency_D','high_freq_flow_skewness_D','volatility_adjusted_concentration_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        eff=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        shakeout=self._get_safe_series(df,'shakeout_score_D',method_name=method_name)
        transfer_eff=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        flow_skew=self._get_safe_series(df,'high_freq_flow_skewness_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        k_eff=self._apply_kinematics(df,'VPA_EFFICIENCY_D',eff,13)
        eff_base=np.tanh(eff.abs()*2.0)*np.sign(eff)*self._apply_zg(df_index,eff)
        mf_mult=np.tanh(self._apply_hab(df,'net_mf_amount_D',net_mf,55)).clip(lower=0)
        core_tensor=eff_base*(1.0+mf_mult)
        shake_penal=1.0-self._apply_norm(shakeout,100.0)*0.5
        trans_mult=self._apply_norm(transfer_eff,1e6)
        skew_mult=np.tanh(self._apply_hab(df,'high_freq_flow_skewness_D',flow_skew,21)).clip(lower=0)
        vac_amp=0.5*(1.0+np.tanh(self._apply_hab(df,'volatility_adjusted_concentration_D',vac,21)))
        to_penal=1.0-np.tanh(self._apply_hab(df,'turnover_rate_f_D',turnover,13)).clip(lower=0)*0.5
        amp=1.0+(trans_mult+skew_mult+vac_amp+to_penal)/4.0
        raw_score=core_tensor*amp*shake_penal*(1.0+np.abs(k_eff))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'VPA_EFFICIENCY_D':eff,'net_mf_amount_D':net_mf},calc_nodes={'eff_base':eff_base,'synergy':core_tensor},final_result=final_score)
        return final_score

    def _calculate_pd_divergence_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】博弈背离引擎"""
        method_name="_calculate_pd_divergence_relationship"
        required_signals=['game_intensity_D','weight_avg_cost_D','close_D','intraday_chip_game_index_D','chip_divergence_ratio_D','winner_rate_D','high_freq_flow_kurtosis_D','chip_entropy_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        game=self._get_safe_series(df,'game_intensity_D',method_name=method_name)
        cost=self._get_safe_series(df,'weight_avg_cost_D',method_name=method_name).replace(0,np.nan)
        close_p=self._get_safe_series(df,'close_D',method_name=method_name)
        intra_game=self._get_safe_series(df,'intraday_chip_game_index_D',method_name=method_name)
        chip_div=self._get_safe_series(df,'chip_divergence_ratio_D',method_name=method_name)
        winner=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        hf_kurt=self._get_safe_series(df,'high_freq_flow_kurtosis_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        k_game=self._apply_kinematics(df,'game_intensity_D',game,13)
        game_base=self._apply_norm(game,1.0)*self._apply_zg(df_index,game)
        price_adv=np.tanh((close_p-cost)/cost*10.0).fillna(0.0)
        base_divergence=self._calculate_instantaneous_relationship(df,config).fillna(0.0)
        price_leverage=(1.0-price_adv*np.sign(base_divergence)).clip(lower=0.1)
        amp=1.0+(self._apply_norm(winner,100.0)+self._apply_norm(intra_game,100.0)+self._apply_norm(chip_div,100.0)+np.tanh(self._apply_hab(df,'high_freq_flow_kurtosis_D',hf_kurt,21)).clip(lower=0)+(1.0-self._apply_norm(c_ent,10.0))+self._apply_norm(turnover,10.0))/6.0
        tensor_resonance=game_base*price_leverage*amp
        raw_score=base_divergence*tensor_resonance*(1.0+np.abs(k_game))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'game_intensity_D':game},calc_nodes={'game_base':game_base,'amp':amp,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】恐慌洗盘吸筹引擎"""
        method_name="_calculate_panic_washout_accumulation"
        required_signals=['pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D','intraday_trough_filling_degree_D','high_freq_flow_divergence_D','chip_rsi_divergence_D','chip_stability_D','pressure_release_index_D','tick_abnormal_volume_ratio_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        panic_level=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorption=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        trough_fill=self._get_safe_series(df,'intraday_trough_filling_degree_D',method_name=method_name)
        hff_div=self._get_safe_series(df,'high_freq_flow_divergence_D',method_name=method_name)
        chip_div=self._get_safe_series(df,'chip_rsi_divergence_D',method_name=method_name)
        chip_stab=self._get_safe_series(df,'chip_stability_D',method_name=method_name)
        release=self._get_safe_series(df,'pressure_release_index_D',method_name=method_name)
        abnorm_vol=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        panic_tensor=(0.5*(1.0+np.tanh(self._apply_hab(df,'panic',panic_level,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'release',release,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'abnorm_vol',abnorm_vol,13))))**(1.0/3.0)
        absorption_tensor=(0.5*(1.0+np.tanh(self._apply_hab(df,'low_lock',low_lock,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'abs_energy',absorption,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'trough_fill',trough_fill,21))))**(1.0/3.0)
        div_bonus=1.0+(np.tanh(self._apply_hab(df,'hff_div',hff_div,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'chip_div',chip_div,21)).clip(lower=0)+self._apply_norm(vac,100.0))/3.0
        base_score=panic_tensor*absorption_tensor*div_bonus*(1.0+self._apply_kinematics(df,'pressure_trapped_D',panic_level,13).clip(lower=0))
        final_score=np.tanh(base_score**1.5).where(chip_stab>config.get('historical_potential_gate',0.2),0.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':panic_level},calc_nodes={'panic_tensor':panic_tensor,'absorption_tensor':absorption_tensor},final_result=final_score)
        return final_score

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】诡道吸筹防爆引擎"""
        method_name="_calculate_deceptive_accumulation"
        required_signals=['stealth_flow_ratio_D','tick_clustering_index_D','intraday_price_distribution_skewness_D','high_freq_flow_skewness_D','price_flow_divergence_D','chip_flow_intensity_D','intraday_chip_turnover_intensity_D','tick_chip_transfer_efficiency_D','intraday_accumulation_confidence_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        stealth=self._get_safe_series(df,'stealth_flow_ratio_D',method_name=method_name)
        cluster=self._get_safe_series(df,'tick_clustering_index_D',method_name=method_name)
        price_skew=self._get_safe_series(df,'intraday_price_distribution_skewness_D',method_name=method_name)
        flow_skew=self._get_safe_series(df,'high_freq_flow_skewness_D',method_name=method_name)
        pf_div=self._get_safe_series(df,'price_flow_divergence_D',method_name=method_name)
        flow_int=self._get_safe_series(df,'chip_flow_intensity_D',method_name=method_name)
        turnover_int=self._get_safe_series(df,'intraday_chip_turnover_intensity_D',method_name=method_name)
        trans_eff=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        acc_conf=self._get_safe_series(df,'intraday_accumulation_confidence_D',method_name=method_name)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        stealth_tensor=(0.5*(1.0+np.tanh(self._apply_hab(df,'stealth',stealth,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'cluster',cluster,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'flow_int',flow_int,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'trans_eff',trans_eff,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'acc_conf',acc_conf,21))))**(1.0/5.0)
        deception_index=(np.tanh(self._apply_hab(df,'pf_div',pf_div,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'skew_mismatch',flow_skew-price_skew,21)).clip(lower=0)*0.5+0.5*(1.0+np.tanh(self._apply_hab(df,'turnover_int',turnover_int,21)))*0.5+(1.0-self._apply_norm(vpa.abs(),100.0))*0.5)/2.5
        raw_deception=stealth_tensor*(1.0+deception_index)*(1.0+self._apply_kinematics(df,'stealth_flow_ratio_D',stealth,13).clip(lower=0))
        final_score=np.tanh(raw_deception**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'stealth_flow_ratio_D':stealth},calc_nodes={'stealth_tensor':stealth_tensor,'raw_deception':raw_deception},final_result=final_score)
        return final_score

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】吸筹末端质变拐点引擎"""
        method_name="_calculate_accumulation_inflection"
        required_signals=['PROCESS_META_COVERT_ACCUMULATION','PROCESS_META_DECEPTIVE_ACCUMULATION','PROCESS_META_PANIC_WASHOUT_ACCUMULATION','PROCESS_META_MAIN_FORCE_RALLY_INTENT','chip_convergence_ratio_D','price_vs_ma_21_ratio_D','flow_acceleration_intraday_D','flow_consistency_D','MA_POTENTIAL_COMPRESSION_RATE_D','MACDh_13_34_8_D','consolidation_quality_score_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        covert=self._get_atomic_score(df,'PROCESS_META_COVERT_ACCUMULATION',0.0).clip(lower=0)
        decept=self._get_atomic_score(df,'PROCESS_META_DECEPTIVE_ACCUMULATION',0.0).clip(lower=0)
        panic=self._get_atomic_score(df,'PROCESS_META_PANIC_WASHOUT_ACCUMULATION',0.0).clip(lower=0)
        rally=self._get_atomic_score(df,'PROCESS_META_MAIN_FORCE_RALLY_INTENT',0.0).clip(lower=0)
        c_conv=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        p_ma21=self._get_safe_series(df,'price_vs_ma_21_ratio_D',method_name=method_name).fillna(1.0)
        f_accel=self._get_safe_series(df,'flow_acceleration_intraday_D',method_name=method_name)
        f_cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        ma_comp=self._get_safe_series(df,'MA_POTENTIAL_COMPRESSION_RATE_D',method_name=method_name)
        macd=self._get_safe_series(df,'MACDh_13_34_8_D',method_name=method_name)
        consol=self._get_safe_series(df,'consolidation_quality_score_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        pot_energy=((covert+decept+panic)/3.0).ewm(span=config.get('accumulation_window',21),adjust=False,min_periods=5).mean()
        struct_mass=((1.0+self._apply_norm(c_conv,1.0))*(1.0-np.tanh(self._apply_hab(df,'p_ma21',np.abs(p_ma21-1.0),21)).clip(lower=0)*0.5)*(1.0+self._apply_norm(ma_comp,1.0))*(1.0+self._apply_norm(consol,100.0))*(1.0+self._apply_norm(vac,100.0))).clip(lower=0)**(1.0/5.0)
        dyn_ignite=(np.tanh(self._apply_hab(df,'f_accel',f_accel,13)).clip(lower=0)+self._apply_norm(f_cons,100.0)+rally+np.tanh(self._apply_hab(df,'macd',macd,13)).clip(lower=0))/4.0
        raw_score=pot_energy*struct_mass*(1.0+dyn_ignite)*(1.0+self._apply_kinematics(df,'chip_convergence_ratio_D',c_conv,13).clip(lower=0))*self._apply_zg(df_index,pot_energy)
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).where(raw_score>=0.1,0.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'consolidation_quality_score_D':consol},calc_nodes={'struct_mass':struct_mass,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_loser_capitulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】输家绝地投降引擎"""
        method_name="_calculate_loser_capitulation"
        required_signals=['pressure_release_index_D','pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D','winner_rate_D','downtrend_strength_D','price_to_weight_avg_ratio_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        release=self._get_safe_series(df,'pressure_release_index_D',method_name=method_name)
        trapped=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorp=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        winner=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        down=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        price_to_cost=self._get_safe_series(df,'price_to_weight_avg_ratio_D',method_name=method_name).fillna(1.0)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        core_pain=(self._apply_norm(release,100.0)*self._apply_zg(df_index,release)*self._apply_norm(trapped,100.0))**0.5
        amp=1.0+(self._apply_norm(low_lock,1.0)+self._apply_norm(absorp,100.0)+self._apply_norm((1.0-price_to_cost).clip(lower=0),1.0)+self._apply_norm(100.0-winner,100.0)+self._apply_norm(down,100.0)+(1.0-self._apply_norm(sent,100.0)))/6.0
        raw_score=core_pain*amp*(1.0+self._apply_kinematics(df,'pressure_release_index_D',release,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'downtrend_strength_D':down},calc_nodes={'core_pain':core_pain,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】突破加速强攻引擎"""
        method_name="_calculate_breakout_acceleration"
        required_signals=['breakout_quality_score_D','industry_strength_rank_D','net_mf_amount_D','flow_consistency_D','tick_abnormal_volume_ratio_D','uptrend_strength_D','T1_PREMIUM_EXPECTATION_D','HM_COORDINATED_ATTACK_D','breakout_penalty_score_D','buy_elg_amount_D','volatility_adjusted_concentration_D','amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
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
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        core_tensor=(self._apply_norm(brk,100.0)*self._apply_zg(df_index,brk)*self._apply_norm(ind,100.0)*self._apply_norm(vac,100.0))**(1.0/3.0)
        amp=1.0+(np.tanh(self._apply_hab(df,'mf',mf,21)).clip(lower=0)+self._apply_norm(t1,100.0)+self._apply_norm(hm,100.0)+self._apply_norm((elg/amount).fillna(0.0),0.1)+self._apply_norm(cons,100.0)+self._apply_norm(abnorm,10.0)+self._apply_norm(uptrend,100.0))/7.0
        raw_score=core_tensor*(1.0-self._apply_norm(pen,100.0))*amp*(1.0+self._apply_kinematics(df,'breakout_quality_score_D',brk,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'breakout_quality_score_D':brk},calc_nodes={'core_tensor':core_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】资金流吸筹质变引擎"""
        method_name="_calculate_fund_flow_accumulation_inflection"
        required_signals=['accumulation_signal_score_D','net_mf_amount_D','flow_efficiency_D','tick_large_order_net_D','intraday_accumulation_confidence_D','GAP_MOMENTUM_STRENGTH_D','STATE_GOLDEN_PIT_D','buy_lg_amount_D','amount_D','flow_persistence_minutes_D','net_energy_flow_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        acc=self._get_safe_series(df,'accumulation_signal_score_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        eff=self._get_safe_series(df,'flow_efficiency_D',method_name=method_name)
        l_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name)
        intra=self._get_safe_series(df,'intraday_accumulation_confidence_D',method_name=method_name)
        gap=self._get_safe_series(df,'GAP_MOMENTUM_STRENGTH_D',method_name=method_name)
        pit=self._get_safe_series(df,'STATE_GOLDEN_PIT_D',method_name=method_name).clip(0,1)
        buy_lg=self._get_safe_series(df,'buy_lg_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        pers=self._get_safe_series(df,'flow_persistence_minutes_D',method_name=method_name)
        energy=self._get_safe_series(df,'net_energy_flow_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        core_ig=(self._apply_norm(acc,100.0)*self._apply_zg(df_index,acc)*self._apply_norm(eff,100.0)*self._apply_norm(intra,100.0)*self._apply_norm((buy_lg/amt).fillna(0.0),0.1)*self._apply_norm(pers,100.0))**(1.0/5.0)
        amp=1.0+(np.tanh(self._apply_hab(df,'l_net',l_net,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'gap',gap,13)).clip(lower=0)+np.tanh(self._apply_hab(df,'energy',energy,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'mf',mf,21)).clip(lower=0)+pit*0.5+(1.0-self._apply_norm(c_ent,10.0)))/5.5
        raw_score=core_ig*amp*(1.0+self._apply_kinematics(df,'accumulation_signal_score_D',acc,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'accumulation_signal_score_D':acc},calc_nodes={'core_ig':core_ig,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】获利压迫与净流对冲引擎"""
        method_name="_calculate_profit_vs_flow_relationship"
        required_signals=['profit_pressure_D','net_mf_amount_D','profit_ratio_D','flow_consistency_D','winner_rate_D','intraday_distribution_confidence_D','STATE_PARABOLIC_WARNING_D','distribution_energy_D','sell_elg_amount_D','amount_D','pressure_profit_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        press=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        p_ratio=self._get_safe_series(df,'profit_ratio_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        win=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        dist_conf=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        para=self._get_safe_series(df,'STATE_PARABOLIC_WARNING_D',method_name=method_name).clip(0,1)
        dist_eng=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        sell_elg=self._get_safe_series(df,'sell_elg_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        marg=self._get_safe_series(df,'pressure_profit_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        press_tensor=(self._apply_norm(press,100.0)*self._apply_norm(dist_conf,100.0)*self._apply_norm(dist_eng,100.0))**(1.0/3.0)*(1.0+(self._apply_norm(p_ratio,100.0)+self._apply_norm((sell_elg/amt).fillna(0),0.1)+self._apply_norm(marg,100.0)+para+self._apply_norm(sent,100.0))/5.0)*(1.0+self._apply_kinematics(df,'profit_pressure_D',press,13).clip(lower=0))
        supp_tensor=(np.tanh(self._apply_hab(df,'mf',mf,34)).clip(lower=0)*self._apply_norm(cons,100.0))**0.5*(1.0-self._apply_norm(win,100.0)*0.5)
        raw_score=supp_tensor-press_tensor*1.5
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'profit_pressure_D':press},calc_nodes={'press_tensor':press_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】个股与板块龙脉共振引擎"""
        method_name="_calculate_stock_sector_sync"
        required_signals=['pct_change_D','industry_strength_rank_D','net_mf_amount_D','flow_consistency_D','industry_leader_score_D','mid_long_sync_D','STATE_MARKET_LEADER_D','industry_markup_score_D','industry_rank_accel_D','HM_COORDINATED_ATTACK_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
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
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        sect_tensor=(0.5*(1.0+np.tanh(self._apply_hab(df,'rank',rank,34)))*0.5*(1.0+np.tanh(self._apply_hab(df,'mkup',mkup,21)))*(1.0+np.tanh(self._apply_hab(df,'rk_acc',rk_acc,13)))*0.5*(1.0+np.tanh(self._apply_hab(df,'ldr',ldr,21)))*(1.0+self._apply_kinematics(df,'industry_strength_rank_D',rank,13)))
        flow_syn=np.tanh(self._apply_hab(df,'mf',mf,21))*0.5*(1.0+np.tanh(self._apply_hab(df,'cons',cons,21)))*(1.0+0.5*(1.0+np.tanh(self._apply_hab(df,'sync',sync,21))))*(1.0+0.5*(1.0+np.tanh(self._apply_hab(df,'hm',hm,13))))
        resonance=np.tanh(self._apply_hab(df,'pct',pct,13))*sect_tensor*(1.0+flow_syn*np.sign(np.tanh(self._apply_hab(df,'pct',pct,13))))*(1.0+m_ldr*1.0)*(1.0+self._apply_norm(sent,100.0))
        final_score=np.tanh(np.sign(resonance)*(np.abs(resonance)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pct_change_D':pct},calc_nodes={'sect_tensor':sect_tensor,'raw_score':resonance},final_result=final_score)
        return final_score

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】热门板块退潮预警引擎"""
        method_name="_calculate_hot_sector_cooling"
        required_signals=['THEME_HOTNESS_SCORE_D','net_mf_amount_D','industry_stagnation_score_D','outflow_quality_D','industry_downtrend_score_D','distribution_energy_D','sell_elg_amount_D','amount_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hot=self._get_safe_series(df,'THEME_HOTNESS_SCORE_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        stag=self._get_safe_series(df,'industry_stagnation_score_D',method_name=method_name)
        outq=self._get_safe_series(df,'outflow_quality_D',method_name=method_name)
        down=self._get_safe_series(df,'industry_downtrend_score_D',method_name=method_name)
        dist=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        sell_elg=self._get_safe_series(df,'sell_elg_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        outflow_tensor=np.tanh(self._apply_hab(df,'mf',mf,13)).clip(upper=0).abs()*0.5*(1.0+np.tanh(self._apply_hab(df,'outq',outq,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'dist',dist,21)))*(1.0+0.5*(1.0+np.tanh(self._apply_hab(df,'sell_elg',(sell_elg/amt).fillna(0),13))))
        decay_boost=1.0+0.5*(1.0+np.tanh(self._apply_hab(df,'stag',stag,21)))*1.5+0.5*(1.0+np.tanh(self._apply_hab(df,'down',down,21)))*2.0+(1.0-self._apply_norm(sent,100.0))
        raw_score=0.5*(1.0+np.tanh(self._apply_hab(df,'hot',hot,21)))*outflow_tensor*decay_boost*(1.0-self._apply_kinematics(df,'THEME_HOTNESS_SCORE_D',hot,13).clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'THEME_HOTNESS_SCORE_D':hot},calc_nodes={'outflow_tensor':outflow_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_ff_vs_structure_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】资金结构双极惩罚引擎"""
        method_name="_calculate_ff_vs_structure_relationship"
        required_signals=['uptrend_strength_D','flow_consistency_D','ma_arrangement_status_D','chip_structure_state_D','industry_stagnation_score_D','large_order_anomaly_D','STATE_ROBUST_TREND_D','net_mf_amount_D','chip_stability_D','flow_momentum_13d_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        up=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        ma_s=self._get_safe_series(df,'ma_arrangement_status_D',method_name=method_name)
        chip_s=self._get_safe_series(df,'chip_structure_state_D',method_name=method_name)
        stag=self._get_safe_series(df,'industry_stagnation_score_D',method_name=method_name)
        anom=self._get_safe_series(df,'large_order_anomaly_D',method_name=method_name)
        rob=self._get_safe_series(df,'STATE_ROBUST_TREND_D',method_name=method_name).clip(0,1)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        stab=self._get_safe_series(df,'chip_stability_D',method_name=method_name)
        mom=self._get_safe_series(df,'flow_momentum_13d_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        core_opp=(self._apply_norm(cons,100.0)*np.tanh(self._apply_hab(df,'mf',mf,21)).clip(lower=0))**0.5*pd.Series(np.where(mf>0,1.0,0.0),index=df_index)
        amp=1.0+(1.0+np.tanh(self._apply_hab(df,'mom',mom,13)).clip(lower=0)+self._apply_norm(ma_s,1.0)+self._apply_norm(chip_s,1.0)+self._apply_norm(vac,100.0)+(1.0-self._apply_norm(stag,100.0)*0.5)*(1.0-(1.0-1.0-np.tanh(self._apply_hab(df,'anom',anom,13)).clip(lower=0)*0.5)*0.5*(1.0-rob*0.8)*(1.0-self._apply_norm(stab,100.0)*0.5)))/5.0
        raw=self._calculate_instantaneous_relationship(df,config).fillna(0.0).clip(lower=0)*core_opp*(1.0-self._apply_norm(up,100.0))*amp*(1.0+np.abs(self._apply_kinematics(df,'uptrend_strength_D',up,13)))
        final_score=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'uptrend_strength_D':up},calc_nodes={'core_opp':core_opp,'raw_score':raw},final_result=final_score)
        return final_score

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】价筹稳态共振引力引擎"""
        method_name="_calculate_pc_relationship"
        required_signals=['peak_concentration_D','close_D','chip_convergence_ratio_D','high_position_lock_ratio_90_D','chip_stability_change_5d_D','volatility_adjusted_concentration_D','chip_entropy_D','chip_flow_intensity_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        pk=self._get_safe_series(df,'peak_concentration_D',method_name=method_name)
        cls=self._get_safe_series(df,'close_D',method_name=method_name)
        cvg=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        hl=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name)
        sc=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        fi=self._get_safe_series(df,'chip_flow_intensity_D',method_name=method_name)
        c_diff=pd.Series(np.where(cls.diff(1).fillna(0).abs()<1e-4,0.0,cls.diff(1).fillna(0)),index=df_index)
        force_vector=np.tanh(self._apply_hab(df,'c_diff',c_diff,13))*self._apply_zg(df_index,c_diff)+self._apply_norm(pk,100.0)*self._apply_zg(df_index,pk)*0.5
        amp=(1.0+(self._apply_norm(cvg,1.0)+self._apply_norm(hl,1.0)+1.0+np.tanh(self._apply_hab(df,'sc',sc,13)).clip(lower=0)+self._apply_norm(vac,100.0)+self._apply_norm(fi,100.0))/5.0)*(1.0-self._apply_norm(ent,10.0)*0.5)*(1.0+np.abs(self._apply_kinematics(df,'peak_concentration_D',pk,13)))
        raw=force_vector*amp
        final_score=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_concentration_D':pk},calc_nodes={'force_vector':force_vector,'raw_score':raw},final_result=final_score)
        return final_score

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】价资协同双向剥离"""
        method_name="_calculate_pf_relationship"
        required_signals=['net_mf_amount_D','close_D','price_vs_ma_13_ratio_D','main_force_activity_index_D','flow_momentum_13d_D','flow_impact_ratio_D','tick_chip_transfer_efficiency_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        cls=self._get_safe_series(df,'close_D',method_name=method_name)
        pm=self._get_safe_series(df,'price_vs_ma_13_ratio_D',method_name=method_name).fillna(1.0)
        act=self._get_safe_series(df,'main_force_activity_index_D',method_name=method_name)
        fm=self._get_safe_series(df,'flow_momentum_13d_D',method_name=method_name)
        imp=self._get_safe_series(df,'flow_impact_ratio_D',method_name=method_name)
        tr=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        c_diff=pd.Series(np.where(cls.diff(1).fillna(0.0).abs()<1e-4,0.0,cls.diff(1).fillna(0.0)),index=df_index)
        force_vector=np.tanh(self._apply_hab(df,'c_diff_pf',c_diff,13))*self._apply_zg(df_index,c_diff)+np.tanh(self._apply_hab(df,'mf_base',mf,34))*self._apply_zg(df_index,mf)
        amp=1.0+(1.0+self._apply_norm(act,100.0)+1.0+np.tanh(self._apply_hab(df,'fm',fm,13)).clip(lower=0)+1.0+np.tanh(self._apply_hab(df,'imp',imp,21)).clip(lower=0)+1.0+np.tanh(self._apply_hab(df,'tr',tr,21)).clip(lower=0)+self._apply_norm(vpa.abs(),100.0))/5.0
        raw=force_vector*amp*(1.0+np.tanh(self._apply_hab(df,'pm',pm,21)).abs()*0.5)*(1.0+np.abs(self._apply_kinematics(df,'net_mf_amount_D',mf,13)))
        final_score=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':mf},calc_nodes={'force_vector':force_vector,'raw_score':raw},final_result=final_score)
        return final_score

    def _calculate_price_vs_momentum_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】价势多维背离引擎"""
        method_name="_calculate_price_vs_momentum_divergence"
        required_signals=['close_D','ROC_13_D','VPA_EFFICIENCY_D','PRICE_ENTROPY_D','net_mf_amount_D','turnover_rate_f_D','GEOM_REG_SLOPE_D','GEOM_REG_R2_D','BIAS_21_D','GEOM_ARC_CURVATURE_D','market_sentiment_score_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        cls=self._get_safe_series(df,'close_D',method_name=method_name)
        roc=self._get_safe_series(df,'ROC_13_D',method_name=method_name)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        ent=self._get_safe_series(df,'PRICE_ENTROPY_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        to=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        slope=self._get_safe_series(df,'GEOM_REG_SLOPE_D',method_name=method_name)
        r2=self._get_safe_series(df,'GEOM_REG_R2_D',method_name=method_name)
        bias=self._get_safe_series(df,'BIAS_21_D',method_name=method_name)
        arc=self._get_safe_series(df,'GEOM_ARC_CURVATURE_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        p_vel=np.tanh(self._apply_hab(df,'c_diff_vel',pd.Series(np.where(cls.diff(1).fillna(0.0).abs()<1e-4,0.0,cls.diff(1).fillna(0.0)),index=df_index),13))
        kinematic_div=self._apply_kinematics(df,'ROC_13_D',roc,13)-p_vel
        energy_div=(np.tanh(self._apply_hab(df,'vpa',vpa,21))+np.tanh(self._apply_hab(df,'mf',mf,21)))/2.0-p_vel
        geom_tension=(np.tanh(self._apply_hab(df,'arc',arc,21))-np.tanh(self._apply_hab(df,'bias',bias,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'r2',r2,34)))
        raw_div=kinematic_div*0.3+energy_div*0.3+geom_tension*0.4
        raw_score=raw_div*(1.0+np.abs(np.tanh(self._apply_hab(df,'sent',sent,34))))*(1.0+np.tanh(self._apply_hab(df,'ent',ent,13)).clip(lower=0))*(1.0+self._apply_norm(vac,100.0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc},calc_nodes={'kinematic_div':kinematic_div,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】张量对冲基础单元"""
        signal_a_name,signal_b_name=config.get('signal_A'),config.get('signal_B')
        df_index=df.index
        relationship_type=config.get('relationship_type','consensus')
        def _get_sig(name: str, src: str) -> Optional[pd.Series]:
            if src=='atomic_states': return self.strategy.atomic_states.get(name)
            try: return self._get_safe_series(df,name,np.nan,"_calculate_instantaneous_relationship")
            except ValueError: return None
        sa=_get_sig(signal_a_name,config.get('source_A','df'))
        sb=_get_sig(signal_b_name,config.get('source_B','df'))
        if sa is None or sb is None: return pd.Series(0.0,index=df_index,dtype=np.float32)
        sa,sb=sa.fillna(0.0),sb.fillna(0.0)
        ca=sa.diff(1).fillna(0.0) if config.get('change_type_A','pct')=='diff' else sa.pct_change(1).replace([np.inf,-np.inf],0.0).fillna(0.0)
        cb=sb.diff(1).fillna(0.0) if config.get('change_type_B','pct')=='diff' else sb.pct_change(1).replace([np.inf,-np.inf],0.0).fillna(0.0)
        ma=np.tanh(self._apply_hab(df,f'{signal_a_name}_rel',ca,13))+self._apply_kinematics(df,signal_a_name,sa,13)*0.5
        tb=np.tanh(self._apply_hab(df,f'{signal_b_name}_rel',cb,13))+self._apply_kinematics(df,signal_b_name,sb,13)*0.5
        kf=config.get('signal_b_factor_k',1.0)
        if relationship_type=='divergence': rel=(kf*tb-ma)/(kf+1.0)
        else: rel=np.sign(ma+kf*tb)*(np.abs(ma)*np.abs(tb))**0.5
        return np.tanh(np.sign(rel)*(np.abs(rel)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)

    def _calculate_dyn_vs_chip_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】动能筹码异向极化引擎"""
        method_name="_calculate_dyn_vs_chip_relationship"
        required_signals=['ROC_13_D','winner_rate_D','profit_ratio_D','chip_mean_D','chip_kurtosis_D','volatility_adjusted_concentration_D','downtrend_strength_D','chip_entropy_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        roc=self._get_safe_series(df,'ROC_13_D',method_name=method_name)
        win=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        prof=self._get_safe_series(df,'profit_ratio_D',method_name=method_name)
        mean=self._get_safe_series(df,'chip_mean_D',method_name=method_name)
        kurt=self._get_safe_series(df,'chip_kurtosis_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        down=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        bc=self._calculate_instantaneous_relationship(df,config).fillna(0.0)
        amp=1.0+(self._apply_norm(prof,100.0)+self._apply_norm(win,100.0)+np.tanh(self._apply_hab(df,'mean',mean,13)).abs()+self._apply_norm(kurt,100.0)+self._apply_norm(vac,100.0)+self._apply_norm(down,100.0)+self._apply_norm(ent,10.0)+(1.0-self._apply_norm(sent,100.0)))/8.0
        raw_score=(-bc).where(bc<0,0.0)*amp*(1.0+np.abs(self._apply_kinematics(df,'ROC_13_D',roc,13)))*pd.Series(np.where(roc<0.0,1.0,0.0),index=df_index)
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc},calc_nodes={'amp':amp,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】洗盘反包诱空反弹引擎"""
        method_name="_calculate_process_wash_out_rebound"
        required_signals=['shakeout_score_D','intraday_distribution_confidence_D','pressure_trapped_D','CLOSING_STRENGTH_D','intraday_trough_filling_degree_D','stealth_flow_ratio_D','absorption_energy_D','STATE_ROUNDING_BOTTOM_D','intraday_low_lock_ratio_D','vwap_deviation_D','tick_abnormal_volume_ratio_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
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
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        washout_tensor=(self._apply_norm(shk,100.0)*self._apply_zg(df_index,shk)*self._apply_norm(dist,100.0)*self._apply_norm(pan,100.0))**(1.0/3.0)
        rebound_tensor=(self._apply_norm(tf,100.0)*self._apply_norm(abs_e,100.0)*self._apply_norm(llck,1.0)*self._apply_norm(t_abn,10.0))**(1.0/4.0)
        amp=1.0+(self._apply_norm(stl,1.0)+np.tanh(self._apply_hab(df,'vdev',vdev,13)).clip(upper=0).abs()+self._apply_norm(cls,100.0)+rnd*0.5+(1.0-self._apply_norm(c_ent,10.0)))/5.0
        raw_score=washout_tensor*rebound_tensor*amp*(1.0+self._apply_kinematics(df,'shakeout_score_D',shk,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'shakeout_score_D':shk},calc_nodes={'washout_tensor':washout_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_fusion_trend_exhaustion_syndrome(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】趋势衰竭综合征引擎"""
        method_name="_calculate_fusion_trend_exhaustion_syndrome"
        required_signals=['STATE_PARABOLIC_WARNING_D','STATE_EMOTIONAL_EXTREME_D','PRICE_ENTROPY_D','profit_pressure_D','HM_COORDINATED_ATTACK_D','intraday_distribution_confidence_D','distribution_energy_D','chip_entropy_D','sell_elg_amount_D','amount_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
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
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        phy_gate=0.5*(1.0+np.tanh(self._apply_hab(df,'dist_c',dist_c,13)))*0.3+np.tanh(self._apply_hab(df,'pres',pres,21)).clip(lower=0)*0.3+0.5*(1.0+np.tanh(self._apply_hab(df,'dist_e',dist_e,21)))*0.2+0.5*(1.0+np.tanh(self._apply_hab(df,'sell',(sell_elg/amount).fillna(0),13)))*0.2
        veto=(1.0-0.5*(1.0+np.tanh(self._apply_hab(df,'hm',hm,13)))*0.9).clip(lower=0.1)*(1.0-self._apply_norm(vpa.abs(),100.0)*0.5)
        raw_score=(1.0+para*1.5+emot*1.0)*phy_gate*(1.0+np.tanh(self._apply_hab(df,'ent',ent,13)).clip(lower=0))*(1.0+np.tanh(self._apply_hab(df,'c_ent',c_ent,21)).clip(lower=0))*veto*(1.0+self._apply_kinematics(df,'profit_pressure_D',pres,13).clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'STATE_PARABOLIC_WARNING_D':para},calc_nodes={'phy_gate':phy_gate,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_dyn_vs_chip_decay_rise(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】力学筹码阻尼看涨引擎"""
        method_name="_calculate_dyn_vs_chip_decay_rise"
        required_signals=['downtrend_strength_D','pressure_trapped_D','absorption_energy_D','chip_kurtosis_D','chip_stability_change_5d_D','reversal_prob_D','intraday_support_test_count_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        down=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        pres=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        abs_e=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        kurt=self._get_safe_series(df,'chip_kurtosis_D',method_name=method_name)
        stb=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        rev=self._get_safe_series(df,'reversal_prob_D',method_name=method_name)
        sup=self._get_safe_series(df,'intraday_support_test_count_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        core_decay=self._apply_norm(down,100.0)*self._apply_norm(pres,100.0)*self._apply_zg(df_index,down)*self._apply_zg(df_index,pres)
        amp=1.0+(np.abs(np.clip(self._apply_kinematics(df,'downtrend_strength_D',down,13),-1.0,0.0))+self._apply_norm(abs_e,100.0)+self._apply_norm(kurt,100.0)+np.tanh(self._apply_hab(df,'stb',stb,13)).clip(lower=0)+self._apply_norm(rev,100.0)+self._apply_norm(sup,10.0)+self._apply_norm(vac,100.0))/7.0
        raw_score=core_decay*amp
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'downtrend_strength_D':down},calc_nodes={'core_decay':core_decay,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_smart_money_ignition(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】聪明钱协同点火引擎"""
        method_name="_calculate_smart_money_ignition"
        required_signals=['HM_COORDINATED_ATTACK_D','T1_PREMIUM_EXPECTATION_D','IS_MARKET_LEADER_D','flow_acceleration_intraday_D','buy_elg_amount_D','tick_large_order_net_D','amount_D','uptrend_strength_D','STATE_BREAKOUT_CONFIRMED_D','net_energy_flow_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
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
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        attack_kinetic=(0.5*(1.0+np.tanh(self._apply_hab(df,'hm',hm,13)))*0.5*(1.0+np.tanh(self._apply_hab(df,'t1',t1,21)))*(1.0+np.tanh(self._apply_hab(df,'f_acc',f_acc,13)).clip(lower=0))*0.5*(1.0+np.tanh(self._apply_hab(df,'elg',(elg/amt).fillna(0),13)))*(1.0+np.tanh(self._apply_hab(df,'t_net',(t_net/amt).fillna(0),13)).clip(lower=0))*0.5*(1.0+np.tanh(self._apply_hab(df,'ne',ne,13))))**(1.0/6.0)
        amp=1.0+(ldr*1.5+self._apply_norm(sent,100.0))/2.0
        raw_score=attack_kinetic*amp*(0.5*(1.0+np.tanh(self._apply_hab(df,'up',up,21)))+brk).clip(0,1)*(1.0+self._apply_kinematics(df,'HM_COORDINATED_ATTACK_D',hm,13).clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_COORDINATED_ATTACK_D':hm},calc_nodes={'attack_kinetic':attack_kinetic,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_vpa_mf_coherence_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】量价主力相干共振引擎"""
        method_name="_calculate_vpa_mf_coherence_resonance"
        required_signals=['MA_COHERENCE_RESONANCE_D','VPA_MF_ADJUSTED_EFF_D','MA_ACCELERATION_EMA_55_D','VPA_ACCELERATION_13D','chip_convergence_ratio_D','flow_consistency_D','volatility_adjusted_concentration_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        mc=self._get_safe_series(df,'MA_COHERENCE_RESONANCE_D',method_name=method_name)
        ve=self._get_safe_series(df,'VPA_MF_ADJUSTED_EFF_D',method_name=method_name)
        ma=self._get_safe_series(df,'MA_ACCELERATION_EMA_55_D',method_name=method_name)
        va=self._get_safe_series(df,'VPA_ACCELERATION_13D',method_name=method_name)
        cc=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        fc=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        core_tensor=(np.tanh(self._apply_hab(df,'mc',mc,34)).clip(lower=0)*np.tanh(self._apply_hab(df,'ve',ve,21)).clip(lower=0)*self._apply_zg(df_index,mc)*self._apply_zg(df_index,ve))**0.5
        amp=1.0+(self._apply_norm(cc,1.0)+self._apply_norm(fc,100.0)+self._apply_norm(vac,100.0)+np.tanh(self._apply_hab(df,'ma',ma,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'va',va,13)).clip(lower=0)+(1.0-self._apply_norm(c_ent,10.0)))/6.0
        raw_score=core_tensor*amp*(1.0+self._apply_kinematics(df,'VPA_MF_ADJUSTED_EFF_D',ve,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_COHERENCE_RESONANCE_D':mc},calc_nodes={'core_tensor':core_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_mtf_fractal_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】多维时空分形共振引擎"""
        method_name="_calculate_mtf_fractal_resonance"
        required_signals=['daily_weekly_sync_D','daily_monthly_sync_D','PRICE_FRACTAL_DIM_D','uptrend_continuation_prob_D','mid_long_sync_D','short_mid_sync_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        dw_sync=self._get_safe_series(df,'daily_weekly_sync_D',method_name=method_name)
        dm_sync=self._get_safe_series(df,'daily_monthly_sync_D',method_name=method_name)
        ml_sync=self._get_safe_series(df,'mid_long_sync_D',method_name=method_name)
        sm_sync=self._get_safe_series(df,'short_mid_sync_D',method_name=method_name)
        fractal_dim=self._get_safe_series(df,'PRICE_FRACTAL_DIM_D',method_name=method_name)
        prob=self._get_safe_series(df,'uptrend_continuation_prob_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        sync_tensor=(0.5*(1.0+np.tanh(self._apply_hab(df,'sync_sum',dw_sync+dm_sync+ml_sync+sm_sync,21))))
        amp=1.0+(1.0-np.tanh(self._apply_hab(df,'frac',fractal_dim,34)).clip(lower=0)+0.5*(1.0+np.tanh(self._apply_hab(df,'prob',prob,21)))+self._apply_norm(sent,100.0))/3.0
        raw_score=sync_tensor*amp*(1.0+self._apply_kinematics(df,'daily_weekly_sync_D',dw_sync,13).clip(lower=0))
        final_score=np.tanh(raw_score**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'daily_weekly_sync_D':dw_sync},calc_nodes={'sync_tensor':sync_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_intraday_siege_exhaustion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】日内攻城拔寨衰竭引擎"""
        method_name="_calculate_intraday_siege_exhaustion"
        required_signals=['intraday_resistance_test_count_D','intraday_support_test_count_D','CLOSING_STRENGTH_D','vwap_deviation_D','resistance_strength_D','support_strength_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        res_tests=self._get_safe_series(df,'intraday_resistance_test_count_D',method_name=method_name)
        sup_tests=self._get_safe_series(df,'intraday_support_test_count_D',method_name=method_name)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name)
        vwap_dev=self._get_safe_series(df,'vwap_deviation_D',method_name=method_name)
        res_str=self._get_safe_series(df,'resistance_strength_D',method_name=method_name)
        sup_str=self._get_safe_series(df,'support_strength_D',method_name=method_name)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        closing_shock=0.5*(1.0+np.tanh(self._apply_hab(df,'closing',closing,21)))
        vwap_shock=np.tanh(self._apply_hab(df,'vwap',vwap_dev,21))
        breakout_force=np.tanh(res_tests/3.0)*(1.0+np.tanh(self._apply_hab(df,'res_str',res_str,21)).clip(lower=0))*closing_shock*(1.0+vwap_shock.clip(lower=0))*(1.0+self._apply_norm(vpa.abs(),100.0))
        breakdown_force=np.tanh(sup_tests/3.0)*(1.0+np.tanh(self._apply_hab(df,'sup_str',sup_str,21)).clip(lower=0))*(1.0-closing_shock)*(1.0+vwap_shock.clip(upper=0).abs())*(1.0-self._apply_norm(vpa.abs(),100.0))
        raw_score=(breakout_force-breakdown_force)*(1.0+np.abs(self._apply_kinematics(df,'CLOSING_STRENGTH_D',closing,13)))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'intraday_resistance_test_count_D':res_tests},calc_nodes={'breakout_force':breakout_force,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_overnight_intraday_tearing(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】隔夜日内动能撕裂引擎"""
        method_name="_calculate_overnight_intraday_tearing"
        required_signals=['GAP_MOMENTUM_STRENGTH_D','OCH_ACCELERATION_D','CLOSING_STRENGTH_D','intraday_price_range_ratio_D','morning_flow_ratio_D','afternoon_flow_ratio_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        gap=self._get_safe_series(df,'GAP_MOMENTUM_STRENGTH_D',method_name=method_name)
        och=self._get_safe_series(df,'OCH_ACCELERATION_D',method_name=method_name)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name).fillna(50.0)
        range_ratio=self._get_safe_series(df,'intraday_price_range_ratio_D',method_name=method_name)
        morning=self._get_safe_series(df,'morning_flow_ratio_D',method_name=method_name).fillna(50.0)
        afternoon=self._get_safe_series(df,'afternoon_flow_ratio_D',method_name=method_name).fillna(50.0)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        gap_shock=np.tanh(self._apply_hab(df,'gap',gap,13))
        intraday_vector=(closing/100.0*2.0-1.0)+np.tanh(self._apply_hab(df,'och',och,13))-np.tanh(self._apply_hab(df,'tearing',morning-afternoon,13)).clip(lower=0)*np.sign(closing/100.0*2.0-1.0)
        tearing_vector=(gap_shock.abs()*self._apply_zg(df_index,gap))*intraday_vector
        leverage=1.0+(np.tanh(self._apply_hab(df,'range',range_ratio,21)).clip(lower=0)*np.abs(closing/100.0-0.5)*2.0)*(1.0+self._apply_norm(sent,100.0))
        raw_score=tearing_vector*leverage*(1.0+self._apply_kinematics(df,'GAP_MOMENTUM_STRENGTH_D',gap,13).abs())*self._apply_zg(df_index,gap)
        is_resonance=(np.sign(gap_shock)*np.sign(intraday_vector)).clip(lower=0)
        raw_score=raw_score*(1.0-is_resonance*0.5)
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GAP_MOMENTUM_STRENGTH_D':gap},calc_nodes={'intraday_vector':intraday_vector,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_chip_center_kinematics(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】筹码重心迁徙动力学引擎"""
        method_name="_calculate_chip_center_kinematics"
        required_signals=['peak_migration_speed_5d_D','intraday_cost_center_volatility_D','price_to_weight_avg_ratio_D','turnover_rate_f_D','cost_50pct_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        migration=self._get_safe_series(df,'peak_migration_speed_5d_D',method_name=method_name)
        cost_vol=self._get_safe_series(df,'intraday_cost_center_volatility_D',method_name=method_name)
        price_to_cost=self._get_safe_series(df,'price_to_weight_avg_ratio_D',method_name=method_name).fillna(1.0)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        cost_50=self._get_safe_series(df,'cost_50pct_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        mig_shock=np.tanh(self._apply_hab(df,'mig',migration,21)).clip(lower=0)*self._apply_zg(df_index,migration)
        vol_shock=np.tanh(self._apply_hab(df,'vol',cost_vol,13)).clip(lower=0)*self._apply_zg(df_index,cost_vol)
        distribution_kinetic=mig_shock*vol_shock*(np.tanh(self._apply_hab(df,'to',turnover,13)).clip(lower=0)*self._apply_zg(df_index,turnover))*(1.0+np.tanh(self._apply_hab(df,'cost',cost_50,21)).clip(lower=0))*(1.0+self._apply_norm(c_ent,10.0))
        lock_kinetic=(1.0-mig_shock)*(1.0-vol_shock)*np.tanh(self._apply_hab(df,'p_to_c',price_to_cost,21)).clip(lower=0)*(1.0-self._apply_norm(c_ent,10.0))
        raw_score=(lock_kinetic-distribution_kinetic)*(1.0+np.abs(self._apply_kinematics(df,'cost_50pct_D',cost_50,13)))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_migration_speed_5d_D':migration},calc_nodes={'distribution_kinetic':distribution_kinetic,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_institutional_sweep(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】机构超大单扫货核爆引擎"""
        method_name="_calculate_institutional_sweep"
        required_signals=['buy_elg_amount_D','buy_lg_amount_D','amount_D','tick_chip_transfer_efficiency_D','flow_consistency_D','net_mf_amount_D','flow_impact_ratio_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        buy_elg=self._get_safe_series(df,'buy_elg_amount_D',method_name=method_name)
        buy_lg=self._get_safe_series(df,'buy_lg_amount_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        trans_eff=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        impact=self._get_safe_series(df,'flow_impact_ratio_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        elg_ratio=((buy_elg+buy_lg*0.5)/amount).fillna(0.0)
        core_tensor=(self._apply_norm(elg_ratio,0.1)*self._apply_zg(df_index,elg_ratio)*(1.0+np.tanh(self._apply_hab(df,'mf',net_mf,55)).clip(lower=0)))**0.5
        amp=1.0+(self._apply_norm(trans_eff,1e6)+self._apply_norm(cons,100.0)+self._apply_norm(impact,10.0)+self._apply_norm(sent,100.0))/4.0
        raw_score=core_tensor*amp*(1.0+self._apply_kinematics(df,'net_mf_amount_D',net_mf,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'buy_elg_amount_D':buy_elg},calc_nodes={'core_tensor':core_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_hf_algo_manipulation_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】高频算法诱骗崩塌防线"""
        method_name="_calculate_hf_algo_manipulation_risk"
        required_signals=['high_freq_flow_skewness_D','high_freq_flow_kurtosis_D','large_order_anomaly_D','price_flow_divergence_D','intraday_price_distribution_skewness_D','tick_abnormal_volume_ratio_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hf_skew=self._get_safe_series(df,'high_freq_flow_skewness_D',method_name=method_name)
        hf_kurt=self._get_safe_series(df,'high_freq_flow_kurtosis_D',method_name=method_name)
        anomaly=self._get_safe_series(df,'large_order_anomaly_D',method_name=method_name)
        divergence=self._get_safe_series(df,'price_flow_divergence_D',method_name=method_name)
        price_skew=self._get_safe_series(df,'intraday_price_distribution_skewness_D',method_name=method_name)
        abnorm_vol=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        core_risk=(np.tanh(self._apply_hab(df,'skew',hf_skew,21)).abs()*self._apply_zg(df_index,hf_skew)*self._apply_norm(anomaly,1.0))**0.5
        amp_tensor=1.0+(np.tanh(self._apply_hab(df,'div',divergence,21)).clip(lower=0)*self._apply_zg(df_index,divergence)+np.tanh(self._apply_hab(df,'kurt',hf_kurt,34)).clip(lower=0)+np.tanh(self._apply_hab(df,'mis',hf_skew-price_skew,13)).abs()+np.tanh(self._apply_hab(df,'abn',abnorm_vol,21)).clip(lower=0)+(1.0-self._apply_norm(vac,100.0)))/5.0
        raw_score=(core_risk*amp_tensor*(1.0+self._apply_kinematics(df,'large_order_anomaly_D',anomaly,13).clip(upper=0).abs()))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'high_freq_flow_skewness_D':hf_skew},calc_nodes={'core_risk':core_risk,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_ma_rubber_band_reversal(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】均线张力极值反噬引擎"""
        method_name="_calculate_ma_rubber_band_reversal"
        required_signals=['MA_RUBBER_BAND_EXTENSION_D','MA_POTENTIAL_TENSION_INDEX_D','ADX_14_D','profit_pressure_D','pressure_trapped_D','BIAS_21_D','reversal_prob_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        rubber_ext=self._get_safe_series(df,'MA_RUBBER_BAND_EXTENSION_D',method_name=method_name)
        tension=self._get_safe_series(df,'MA_POTENTIAL_TENSION_INDEX_D',method_name=method_name)
        adx=self._get_safe_series(df,'ADX_14_D',method_name=method_name)
        profit_p=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        trapped_p=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        bias_21=self._get_safe_series(df,'BIAS_21_D',method_name=method_name)
        rev_prob=self._get_safe_series(df,'reversal_prob_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        rubber_shock=np.tanh(self._apply_hab(df,'ext',rubber_ext,34))
        rs_aligned=pd.Series(np.where(np.sign(rubber_shock)==np.sign(rubber_ext),rubber_shock,0.0),index=df_index)
        tension_shock=np.tanh(self._apply_hab(df,'tension',tension,21))
        profit_shock=0.5*(1.0+np.tanh(self._apply_hab(df,'profit_p',profit_p,21)))
        trapped_shock=0.5*(1.0+np.tanh(self._apply_hab(df,'trapped_p',trapped_p,21)))
        bias_shock=np.tanh(self._apply_hab(df,'bias',bias_21,21))
        rev_shock=0.5*(1.0+np.tanh(self._apply_hab(df,'rev',rev_prob,13)))
        trend_suppression=1.0-np.tanh(np.maximum(adx-35.0,0.0)/15.0)
        top_force=(rs_aligned.clip(lower=0))*tension_shock.clip(lower=0)*profit_shock*(1.0+bias_shock.clip(lower=0))*(1.0+rev_shock)*(1.0-self._apply_norm(c_ent,10.0))
        bottom_force=(rs_aligned.clip(upper=0).abs())*tension_shock.clip(lower=0)*trapped_shock*(1.0+bias_shock.clip(upper=0).abs())*(1.0+rev_shock)*(1.0-self._apply_norm(c_ent,10.0))
        raw_score=(bottom_force-top_force)*trend_suppression*(1.0+np.abs(self._apply_kinematics(df,'MA_RUBBER_BAND_EXTENSION_D',rubber_ext,13)))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_RUBBER_BAND_EXTENSION_D':rubber_ext},calc_nodes={'top_force':top_force,'bottom_force':bottom_force,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_geometric_trend_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】几何流形趋势共振引擎"""
        method_name="_calculate_geometric_trend_resonance"
        required_signals=['GEOM_REG_R2_D','GEOM_REG_SLOPE_D','GEOM_ARC_CURVATURE_D','GEOM_CHANNEL_POS_D','PRICE_FRACTAL_DIM_D','trend_confirmation_score_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        r2=self._get_safe_series(df,'GEOM_REG_R2_D',method_name=method_name)
        slope=self._get_safe_series(df,'GEOM_REG_SLOPE_D',method_name=method_name)
        curvature=self._get_safe_series(df,'GEOM_ARC_CURVATURE_D',method_name=method_name)
        channel_pos=self._get_safe_series(df,'GEOM_CHANNEL_POS_D',method_name=method_name)
        fractal_dim=self._get_safe_series(df,'PRICE_FRACTAL_DIM_D',method_name=method_name)
        trend_conf=self._get_safe_series(df,'trend_confirmation_score_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        core_tensor=np.tanh(self._apply_hab(df,'slope',slope,34))*np.maximum(self._apply_norm(r2,1.0),0.1)*self._apply_zg(df_index,slope)
        manifold_dynamics=np.tanh(self._apply_hab(df,'curve',curvature,21))-(channel_pos.fillna(0.0)-0.5)*2.0*0.3
        amp=1.0+(1.0-self._apply_norm(fractal_dim,2.0)+self._apply_norm(trend_conf,100.0)+manifold_dynamics.clip(lower=0)+self._apply_norm(vac,100.0))/4.0
        raw_score=core_tensor*amp*(1.0+self._apply_kinematics(df,'GEOM_REG_SLOPE_D',slope,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GEOM_REG_SLOPE_D':slope},calc_nodes={'core_tensor':core_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_ma_compression_explosion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】均线奇点核爆引擎"""
        method_name="_calculate_ma_compression_explosion"
        required_signals=['MA_POTENTIAL_COMPRESSION_RATE_D','chip_convergence_ratio_D','TURNOVER_STABILITY_INDEX_D','MACDh_13_34_8_D','energy_concentration_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        ma_comp=self._get_safe_series(df,'MA_POTENTIAL_COMPRESSION_RATE_D',method_name=method_name)
        chip_conv=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        to_stab=self._get_safe_series(df,'TURNOVER_STABILITY_INDEX_D',method_name=method_name)
        macd_hist=self._get_safe_series(df,'MACDh_13_34_8_D',method_name=method_name)
        energy_conc=self._get_safe_series(df,'energy_concentration_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        core_tensor=(self._apply_norm(ma_comp,1.0)*self._apply_zg(df_index,ma_comp)*self._apply_norm(chip_conv,1.0))**0.5
        amp=1.0+(self._apply_norm(to_stab,1.0)+self._apply_norm(energy_conc,100.0)+self._apply_norm(vac,100.0))/3.0
        ignition=np.tanh(self._apply_hab(df,'macd',macd_hist,13))
        raw_score=core_tensor*amp*np.sign(ignition)*(np.abs(ignition)**0.5)*(1.0+self._apply_kinematics(df,'MA_POTENTIAL_COMPRESSION_RATE_D',ma_comp,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_POTENTIAL_COMPRESSION_RATE_D':ma_comp},calc_nodes={'core_tensor':core_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_top_tier_hm_harvesting(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】顶级游资收割镰刀引擎"""
        method_name="_calculate_top_tier_hm_harvesting"
        required_signals=['HM_ACTIVE_TOP_TIER_D','CLOSING_STRENGTH_D','tick_large_order_net_D','amount_D','outflow_quality_D','distribution_energy_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hm_active=self._get_safe_series(df,'HM_ACTIVE_TOP_TIER_D',method_name=method_name)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name).fillna(50.0)
        large_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        outflow_q=self._get_safe_series(df,'outflow_quality_D',method_name=method_name)
        dist_eng=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        net_ratio=(large_net/amount).fillna(0.0)
        core_harvest=(self._apply_norm(hm_active,100.0)*self._apply_zg(df_index,hm_active)*np.tanh(self._apply_hab(df,'net_r',net_ratio,13)).clip(upper=0).abs()*self._apply_zg(df_index,net_ratio.clip(upper=0)))**0.5
        amp=1.0+(1.0-self._apply_norm(closing,100.0)+self._apply_norm(outflow_q,100.0)+self._apply_norm(dist_eng,100.0)+(1.0-self._apply_norm(sent,100.0)))/4.0
        raw_score=core_harvest*amp*(1.0+self._apply_kinematics(df,'tick_large_order_net_D',large_net,13).clip(upper=0).abs())*self._apply_zg(df_index,hm_active)
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_ACTIVE_TOP_TIER_D':hm_active},calc_nodes={'core_harvest':core_harvest,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_vwap_magnetic_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】VWAP磁性黑洞引力引擎"""
        method_name="_calculate_vwap_magnetic_divergence"
        required_signals=['vwap_deviation_D','reversal_prob_D','intraday_main_force_activity_D','intraday_cost_center_migration_D','volume_ratio_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        vwap_dev=self._get_safe_series(df,'vwap_deviation_D',method_name=method_name)
        rev_prob=self._get_safe_series(df,'reversal_prob_D',method_name=method_name)
        mf_activity=self._get_safe_series(df,'intraday_main_force_activity_D',method_name=method_name).fillna(50.0)
        cost_mig=self._get_safe_series(df,'intraday_cost_center_migration_D',method_name=method_name)
        vol_ratio=self._get_safe_series(df,'volume_ratio_D',method_name=method_name).fillna(1.0)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        dev_shock=np.tanh(self._apply_hab(df,'vwap',vwap_dev,13))
        dev_aligned=pd.Series(np.where(np.sign(dev_shock)==np.sign(vwap_dev),dev_shock,0.0),index=df_index)
        core_pull=-1.0*dev_aligned*self._apply_norm(rev_prob,100.0)
        mismatch_penalty=(1.0-(self._apply_norm(mf_activity,100.0)*2.0-1.0)*np.sign(dev_aligned))*(1.0-np.tanh(self._apply_hab(df,'mig',cost_mig,21))*np.sign(dev_aligned))
        raw_score=core_pull*mismatch_penalty.clip(lower=0.1)*(1.0+self._apply_norm(vol_ratio,10.0))*(1.0+np.abs(self._apply_kinematics(df,'vwap_deviation_D',vwap_dev,13)))*(1.0-self._apply_norm(c_ent,10.0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'vwap_deviation_D':vwap_dev},calc_nodes={'core_pull':core_pull,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_multi_peak_avalanche_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】多峰筹码断层雪崩引擎"""
        method_name="_calculate_multi_peak_avalanche_risk"
        required_signals=['is_multi_peak_D','chip_divergence_ratio_D','intraday_distribution_confidence_D','downtrend_strength_D','chip_entropy_D','chip_stability_change_5d_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        multi_peak=self._get_safe_series(df,'is_multi_peak_D',method_name=method_name).clip(0,1)
        divergence=self._get_safe_series(df,'chip_divergence_ratio_D',method_name=method_name)
        dist_conf=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        downtrend=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        entropy=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        stab_chg=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        core_tensor=multi_peak*np.maximum(self._apply_norm(dist_conf,100.0),self._apply_norm(downtrend,100.0))*self._apply_zg(df_index,multi_peak)
        amp=1.0+(self._apply_norm(divergence,100.0)+self._apply_norm(entropy,10.0)+np.tanh(self._apply_hab(df,'stab',stab_chg.diff(1).fillna(0),13)).clip(upper=0).abs()+(1.0-self._apply_norm(vac,100.0)))/4.0
        raw_score=core_tensor*amp*(1.0+self._apply_kinematics(df,'chip_divergence_ratio_D',divergence,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'is_multi_peak_D':multi_peak},calc_nodes={'core_tensor':core_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_sector_lifecycle_tailwind(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】板块生命周期顺风引擎"""
        method_name="_calculate_sector_lifecycle_tailwind"
        required_signals=['industry_preheat_score_D','industry_markup_score_D','industry_stagnation_score_D','industry_downtrend_score_D','uptrend_strength_D','industry_rank_accel_D','industry_strength_rank_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        preheat=self._get_safe_series(df,'industry_preheat_score_D',method_name=method_name)
        markup=self._get_safe_series(df,'industry_markup_score_D',method_name=method_name)
        stagnation=self._get_safe_series(df,'industry_stagnation_score_D',method_name=method_name)
        downtrend=self._get_safe_series(df,'industry_downtrend_score_D',method_name=method_name)
        uptrend=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        rank_accel=self._get_safe_series(df,'industry_rank_accel_D',method_name=method_name)
        rank=self._get_safe_series(df,'industry_strength_rank_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        sector_vector=(0.5*(1.0+np.tanh(self._apply_hab(df,'mk',markup,21)))*1.0+0.5*(1.0+np.tanh(self._apply_hab(df,'pr',preheat,21)))*0.5)-(0.5*(1.0+np.tanh(self._apply_hab(df,'st',stagnation,21)))*0.5+0.5*(1.0+np.tanh(self._apply_hab(df,'dt',downtrend,21)))*1.0)
        raw_score=sector_vector*0.5*(1.0+np.tanh(self._apply_hab(df,'up',uptrend,21)))*(1.0+0.5*(1.0+np.tanh(self._apply_hab(df,'rk',rank,34))))*(1.0+np.tanh(self._apply_hab(df,'acc',rank_accel,13))*np.sign(sector_vector))*(1.0+self._apply_norm(sent,100.0))*(1.0+self._apply_kinematics(df,'industry_markup_score_D',markup,13).clip(lower=0))
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)*2.0).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'industry_markup_score_D':markup},calc_nodes={'sector_vector':sector_vector,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_time_asymmetry_trap(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】时间非对称陷阱防爆引擎"""
        method_name="_calculate_time_asymmetry_trap"
        required_signals=['morning_flow_ratio_D','afternoon_flow_ratio_D','intraday_peak_valley_ratio_D','profit_pressure_D','closing_flow_ratio_D','high_freq_flow_divergence_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        morning=self._get_safe_series(df,'morning_flow_ratio_D',method_name=method_name)
        afternoon=self._get_safe_series(df,'afternoon_flow_ratio_D',method_name=method_name)
        closing=self._get_safe_series(df,'closing_flow_ratio_D',method_name=method_name)
        peak_valley=self._get_safe_series(df,'intraday_peak_valley_ratio_D',method_name=method_name)
        profit_p=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        hf_div=self._get_safe_series(df,'high_freq_flow_divergence_D',method_name=method_name)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        asym_shock=np.tanh(self._apply_hab(df,'asym',(afternoon+closing*0.5-morning)/100.0,13))
        trap_power=asym_shock*(1.0+0.5*(1.0+np.tanh(self._apply_hab(df,'pv',peak_valley,21)))*np.sign(-asym_shock))*(1.0+0.5*(1.0+np.tanh(self._apply_hab(df,'pp',profit_p,21)))*np.sign(-asym_shock))*(1.0+np.tanh(self._apply_hab(df,'hd',hf_div,21)).clip(lower=0)*np.sign(-asym_shock))*(1.0-self._apply_norm(vpa.abs(),100.0))
        raw_score=np.sign(trap_power)*(np.abs(trap_power)**1.5)*(1.0+np.abs(self._apply_kinematics(df,'morning_flow_ratio_D',morning,13)))
        final_score=np.tanh(raw_score).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'morning_flow_ratio_D':morning},calc_nodes={'asym_shock':asym_shock,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_high_pos_liquidity_squeeze(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】高位流动性真空轧空引擎"""
        method_name="_calculate_high_pos_liquidity_squeeze"
        required_signals=['price_percentile_position_D','high_position_lock_ratio_90_D','flow_persistence_minutes_D','short_term_chip_ratio_D','uptrend_strength_D','turnover_rate_f_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        percentile=self._get_safe_series(df,'price_percentile_position_D',method_name=method_name)
        lock_ratio=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name)
        persistence=self._get_safe_series(df,'flow_persistence_minutes_D',method_name=method_name)
        short_chip=self._get_safe_series(df,'short_term_chip_ratio_D',method_name=method_name)
        uptrend=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        pos_norm=np.tanh(percentile/50.0) if percentile.max()>1.0 else np.tanh(percentile*2.0)
        squeeze_tensor=0.5*(1.0+np.tanh(self._apply_hab(df,'lock',lock_ratio,21)))*0.5*(1.0+np.tanh(self._apply_hab(df,'per',persistence,13)))*(1.0-0.5*(1.0+np.tanh(self._apply_hab(df,'short',short_chip,21))))*0.5*(1.0+np.tanh(self._apply_hab(df,'up',uptrend,21)))*(1.0-np.tanh(self._apply_hab(df,'to',turnover,21)).clip(lower=0))*pos_norm.clip(lower=0)*(1.0-self._apply_norm(c_ent,10.0))
        raw_score=(squeeze_tensor**1.5)*(1.0+self._apply_kinematics(df,'high_position_lock_ratio_90_D',lock_ratio,13).clip(lower=0))
        final_score=np.tanh(raw_score*2.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'price_percentile_position_D':percentile},calc_nodes={'squeeze_tensor':squeeze_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score

    def _calculate_institutional_structural_exit(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V9.0.0 · 全息量子防爆版】机构结构性清仓逃顶引擎"""
        method_name="_calculate_institutional_structural_exit"
        required_signals=['sell_elg_amount_D','sell_lg_amount_D','amount_D','distribution_energy_D','downtrend_strength_D','high_position_lock_ratio_90_D','chip_stability_change_5d_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        sell_elg=self._get_safe_series(df,'sell_elg_amount_D',method_name=method_name)
        sell_lg=self._get_safe_series(df,'sell_lg_amount_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        dist_energy=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        downtrend=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        lock_ratio=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name)
        stab_change=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        sell_ratio=((sell_elg+sell_lg*0.5)/amount).fillna(0.0)
        core_tensor=pd.Series(np.where(sell_ratio>0.05,1.0,0.0),index=df_index)*self._apply_norm(sell_ratio,0.15)
        amp=1.0+(self._apply_norm(dist_energy,100.0)+self._apply_norm(downtrend,100.0)+np.tanh(self._apply_hab(df,'lock_diff',lock_ratio.diff(1).fillna(0),13)).clip(upper=0).abs()+np.tanh(self._apply_hab(df,'stab_c',stab_change,13)).clip(upper=0).abs()+(1.0-self._apply_norm(sent,100.0)))/5.0
        raw_score=core_tensor*amp*(1.0+self._apply_kinematics(df,'sell_ratio',sell_ratio,13).clip(upper=0).abs())*self._apply_zg(df_index,sell_ratio)
        final_score=np.tanh(np.sign(raw_score)*(np.abs(raw_score)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'sell_ratio':sell_ratio},calc_nodes={'core_tensor':core_tensor,'raw_score':raw_score},final_result=final_score)
        return final_score







