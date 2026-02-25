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
        """【V12.0.0 · 流形简并防爆版】裂变张量引擎"""
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
        """【V12.0.0 · 流形简并防爆版】神谕审判庭防线"""
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
        """【V12.0.0 · 流形简并防爆版】元动力学分析"""
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
        """【V12.0.0】绝对零基防御门限 (Zero-Gate)"""
        return pd.Series(np.where(s.abs()<1e-4,0.0,1.0),index=df_index,dtype=np.float32)

    def _apply_norm(self, s: pd.Series, max_v: float=100.0) -> pd.Series:
        """【V12.0.0】线性边界防爆截断"""
        return (s.fillna(0.0)/max_v).clip(0,1).astype(np.float32)

    def _apply_hab(self, df: pd.DataFrame, col_name: str, s: pd.Series, w: int) -> pd.Series:
        """【V12.0.0】多维HAB历史缓冲池 (带方差正则化防爆)"""
        hab_col=f'HAB_{w}_{col_name}'
        if hab_col in df.columns: return df[hab_col].fillna(0.0).astype(np.float32)
        sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
        rm=sf.rolling(window=w,min_periods=1).mean()
        rs=np.sqrt(sf.rolling(window=w,min_periods=1).std()**2+1e-5)
        return ((sf-rm)/rs).fillna(0.0).astype(np.float32)

    def _apply_kinematics(self, df: pd.DataFrame, col_name: str, s: pd.Series, w: int) -> pd.Series:
        """【V12.0.0】三阶动力学张量引擎 (量纲对齐与软截断)"""
        sf=s.ffill().fillna(0.0).replace([np.inf,-np.inf],0.0)
        sl=df.get(f'SLOPE_{w}_{col_name}',sf.diff(w)/w).fillna(0.0)
        ac=df.get(f'ACCEL_{w}_{col_name}',sl.diff(w)/w).fillna(0.0)
        jk=df.get(f'JERK_{w}_{col_name}',ac.diff(w)/w).fillna(0.0)
        ts=np.where(sl.abs()<1e-4,0.0,sl)+np.where(ac.abs()<1e-4,0.0,ac)*0.5+np.where(jk.abs()<1e-4,0.0,jk)*0.25
        return np.tanh(pd.Series(ts,index=s.index)*10.0).fillna(0.0).astype(np.float32)

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V65.0.0 · 方差正则化防爆版】信号衰减总线
        修改要点: 升级 local_std 采用 np.sqrt(std**2 + 1e-5) 进行正则化，彻底杜绝一字板或停牌期间方差为 0 导致的 NaN 崩塌。
        """
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
        signal_change=source_series.diff(1).fillna(0.0)
        decay_magnitude=signal_change.clip(upper=0.0).abs()
        # [V65.0.0] 彻底解决除以零问题，引入统一的方差正则化模型，防爆处理
        local_std=np.sqrt(source_series.rolling(window=21,min_periods=1).std().fillna(0.0)**2 + 1e-5).fillna(1e-5)
        relative_decay=(decay_magnitude/local_std).fillna(0.0)
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
        """
        【V50.0.0 · 绝对物理极性收官版】微观权力交接张量
        用途: 捕捉真实主力资金流对股价主导权的微观接管。
        修改要点: 彻底剥夺历史冲击(HAB)对主核极性的决定权！将当期资金流的物理方向(净流入/流出)作为唯一的极性标尺。
        """
        method_name="_calculate_power_transfer"
        required_signals=['net_mf_amount_D','amount_D','tick_large_order_net_D','tick_chip_transfer_efficiency_D','flow_efficiency_D','intraday_cost_center_migration_D','downtrend_strength_D','chip_concentration_ratio_D','volatility_adjusted_concentration_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        tick_large_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name)
        transfer_eff=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        flow_eff=self._get_safe_series(df,'flow_efficiency_D',method_name=method_name)
        cost_migration=self._get_safe_series(df,'intraday_cost_center_migration_D',method_name=method_name)
        downtrend=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        chip_conc=self._get_safe_series(df,'chip_concentration_ratio_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        mf_ratio=(net_mf/amt).fillna(0.0)
        tick_ratio=(tick_large_net/amt).fillna(0.0)
        # [V50.0.0 核心重构] 极性完全由绝对比例决定。50.0 放大感光度，2%净流即可激活强信号。
        mf_core=np.tanh(mf_ratio*50.0)
        tick_core=np.tanh(tick_ratio*50.0)
        absolute_core=(mf_core*0.6+tick_core*0.4)*self._apply_zg(df_index,net_mf)
        hab_amp=1.0+(np.tanh(self._apply_hab(df,'mf_r',mf_ratio,21)).abs()+np.tanh(self._apply_hab(df,'t_r',tick_ratio,21)).abs())/2.0
        core=absolute_core*hab_amp
        amp=1.0+(np.tanh(self._apply_hab(df,'mig',cost_migration,13)).abs()+(1.0-np.tanh(turnover.fillna(0.0)/10.0))+self._apply_norm(vac,100.0)+self._apply_norm(transfer_eff,1e6)+np.abs(np.tanh(flow_eff.fillna(0.0)/5.0)))/5.0
        c_pen=np.tanh(self._apply_hab(df,'c_diff',chip_conc.diff().fillna(0.0),13))*amp
        synergy=(1.0+c_pen*np.sign(core)).clip(lower=0.1)
        raw=core*synergy*(1.0+np.abs(self._apply_kinematics(df,'mf_ratio_scaled',mf_ratio*100.0,13)))
        td=pd.Series(1.0,index=df_index).mask((downtrend.fillna(0.0)>80.0)&(raw>0),0.6)
        res=(np.tanh(np.sign(raw)*(np.abs(raw)**1.5))*td).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':net_mf},calc_nodes={'absolute_core':absolute_core,'hab_amp':hab_amp,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_price_vs_capitulation_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 物理极性显式重构版】价格与散户投降背离
        修改要点: 彻底废除底层 base_div 的黑盒依赖，采用显式的价格与套牢盘动力学极性相乘，防范极性倒置。
        """
        method_name="_calculate_price_vs_capitulation_relationship"
        required_signals=['pressure_trapped_D','INTRADAY_SUPPORT_INTENT_D','intraday_low_lock_ratio_D','chip_entropy_D','volatility_adjusted_concentration_D','turnover_rate_f_D','close_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        pressure=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        support=self._get_safe_series(df,'INTRADAY_SUPPORT_INTENT_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        entropy=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        cls=self._get_safe_series(df,'close_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        # [V65.0.0] 显式动力学提取
        p_kin=self._apply_kinematics(df,'close_D_scaled',(cls/cls.rolling(21,min_periods=1).mean().fillna(1.0)).fillna(1.0)*10.0,13)
        t_kin=self._apply_kinematics(df,'pressure_trapped_D_scaled',pressure.fillna(0.0)*100.0,13)
        # 散户投降机会 (价跌 p_kin<0 且 套牢盘降 t_kin<0) -> 负负得正
        opp_core=np.tanh(-p_kin.clip(upper=0))*np.tanh(-t_kin.clip(upper=0))
        # 散户追高陷阱 (价涨 p_kin>0 且 套牢盘增 t_kin>0) -> 正正得正，加负号变风险
        risk_core=np.tanh(p_kin.clip(lower=0))*np.tanh(t_kin.clip(lower=0))
        core=(opp_core-risk_core)*self._apply_zg(df_index,pressure)
        amp=1.0+(np.tanh(self._apply_hab(df,'sup',support,34)).clip(lower=0)+self._apply_norm(low_lock,1.0)+(1.0-self._apply_norm(entropy,10.0))+self._apply_norm(vac,100.0)+(1.0-np.tanh(turnover.fillna(0.0)/10.0)))/5.0
        raw=core*amp*(1.0+np.abs(t_kin))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':pressure,'close_D':cls},calc_nodes={'p_kin':p_kin,'t_kin':t_kin,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_price_efficiency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V38.0.0】价格效率博弈引擎
        修改要点: 对 VPA_EFFICIENCY_D 采取 tanh(vpa * 5.0) 释放感光度，防止 /100.0 带来的失真湮灭。
        """
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
        # [V38.0.0] 修复 VPA 失真
        core=np.tanh(eff.fillna(0.0)*5.0)*self._apply_zg(df_index,eff)
        amp=1.0+(np.tanh(self._apply_hab(df,'mf',net_mf,55)).clip(lower=0)+(1.0-self._apply_norm(shakeout,100.0))+self._apply_norm(transfer_eff,1e6)+np.tanh(self._apply_hab(df,'skew',flow_skew,21)).clip(lower=0)+self._apply_norm(vac,100.0)+(1.0-np.tanh(turnover.fillna(0.0)/10.0)))/6.0
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'VPA_EFFICIENCY_D_scaled',eff.fillna(0.0),13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'VPA_EFFICIENCY_D':eff},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_pd_divergence_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V38.0.0】博弈背离引擎
        修改要点: 底层依赖瞬时对冲单元已彻底切断错误极性，保障博弈极值正常映射。
        """
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
        base_div=self._calculate_instantaneous_relationship(df,config).fillna(0.0)
        core=base_div*self._apply_norm(game,1.0)*self._apply_zg(df_index,game)
        price_adv=np.tanh((close_p-cost)/cost*10.0).fillna(0.0)
        p_lev=np.maximum((1.0-price_adv*np.sign(base_div)).clip(lower=0.1),0.1)
        amp=1.0+(self._apply_norm(winner,100.0)+self._apply_norm(intra_game,100.0)+self._apply_norm(chip_div,100.0)+np.tanh(self._apply_hab(df,'kurt',hf_kurt,21)).clip(lower=0)+(1.0-self._apply_norm(c_ent,10.0))+self._apply_norm(turnover,10.0))/6.0
        raw=core*p_lev*amp*(1.0+np.abs(self._apply_kinematics(df,'game_intensity_D_scaled',game.fillna(0.0),13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'game_intensity_D':game},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】恐慌洗盘吸筹引擎
        用途: 捕捉恐慌暴跌中主力暗中承接带血筹码的行为。
        修改要点: 解决极小比率的湮灭问题，并在 kinematics 使用 _scaled 阻断饱和。
        """
        method_name="_calculate_panic_washout_accumulation"
        required_signals=['pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D','intraday_trough_filling_degree_D','high_freq_flow_divergence_D','chip_rsi_divergence_D','chip_stability_D','pressure_release_index_D','tick_abnormal_volume_ratio_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        panic=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorption=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        trough_fill=self._get_safe_series(df,'intraday_trough_filling_degree_D',method_name=method_name)
        hff_div=self._get_safe_series(df,'high_freq_flow_divergence_D',method_name=method_name)
        chip_div=self._get_safe_series(df,'chip_rsi_divergence_D',method_name=method_name)
        chip_stab=self._get_safe_series(df,'chip_stability_D',method_name=method_name)
        release=self._get_safe_series(df,'pressure_release_index_D',method_name=method_name)
        abnorm_vol=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        p_core=np.tanh(panic.fillna(0.0)*100.0)*self._apply_zg(df_index,panic)
        a_core=np.tanh(absorption.fillna(0.0)*100.0)*self._apply_zg(df_index,absorption)
        core=p_core*a_core
        amp=1.0+(np.tanh(release.fillna(0.0)*100.0)+self._apply_norm(abnorm_vol,10.0)+self._apply_norm(low_lock,1.0)+self._apply_norm(trough_fill,100.0)+np.tanh(self._apply_hab(df,'hff',hff_div,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'cdiv',chip_div,21)).clip(lower=0)+self._apply_norm(vac,100.0))/7.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'pressure_trapped_D_scaled',panic.fillna(0.0)*100.0,13).clip(lower=0))
        res=np.tanh(raw**1.5).where(chip_stab.fillna(0.0)>config.get('historical_potential_gate',0.2),0.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':panic,'absorption_energy_D':absorption},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V38.0.0】诡道吸筹防爆引擎
        修改要点: 修复 VPA_EFFICIENCY_D 绝对值过小导致被强行截断的失真。
        """
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
        acc_norm=acc_conf.where(acc_conf<=1.0,acc_conf/100.0).fillna(0.0)
        core=np.tanh(stealth.fillna(0.0)*100.0)*acc_norm*self._apply_zg(df_index,stealth)*self._apply_zg(df_index,acc_conf)
        amp=1.0+(self._apply_norm(cluster,1.0)+self._apply_norm(flow_int,100.0)+self._apply_norm(trans_eff,1e6)+np.tanh(self._apply_hab(df,'pf_div',pf_div,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'smis',flow_skew.fillna(0.0)-price_skew.fillna(0.0),21)).clip(lower=0)+self._apply_norm(turnover_int,100.0)+(1.0-np.abs(np.tanh(vpa.fillna(0.0)*5.0))))/7.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'stealth_flow_ratio_D_scaled',stealth.fillna(0.0)*100.0,13).clip(lower=0))
        res=np.tanh(raw**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'stealth_flow_ratio_D':stealth,'intraday_accumulation_confidence_D':acc_conf},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】吸筹末端质变拐点引擎
        用途: 聚合微观吸筹信号，判断量变到质变的主升浪起爆点。
        修改要点: 强制约束集中度指标并追加 _scaled 绕过污染缓存。
        """
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
        pot=((covert+decept+panic)/3.0).ewm(span=config.get('accumulation_window',21),adjust=False,min_periods=5).mean()
        core=pot*self._apply_zg(df_index,pot)
        amp=1.0+(self._apply_norm(consol,100.0)+self._apply_norm(c_conv,1.0)+(1.0-np.tanh(self._apply_hab(df,'pma',np.abs(p_ma21-1.0),21)).clip(lower=0)*0.5)+self._apply_norm(ma_comp,1.0)+self._apply_norm(vac,100.0)+np.tanh(self._apply_hab(df,'fa',f_accel,13)).clip(lower=0)+self._apply_norm(f_cons,100.0)+rally+np.tanh(self._apply_hab(df,'md',macd,13)).clip(lower=0))/9.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'chip_convergence_ratio_D_scaled',c_conv.fillna(0.0),13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).where(raw>=0.1,0.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'consolidation_quality_score_D':consol},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_loser_capitulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】输家绝地投降引擎
        用途: 捕捉套牢盘割肉带来的筹码洗盘结束点。
        修改要点: 对 release 极小比率使用 _scaled 绕过缓存。
        """
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
        core=np.tanh(release.fillna(0.0)*100.0)*np.tanh(trapped.fillna(0.0)*100.0)*self._apply_zg(df_index,release)*self._apply_zg(df_index,trapped)
        amp=1.0+(self._apply_norm(low_lock,1.0)+np.tanh(absorp.fillna(0.0)*10.0)+self._apply_norm((1.0-price_to_cost).clip(lower=0),1.0)+self._apply_norm(100.0-winner.fillna(0.0),100.0)+self._apply_norm(down,100.0)+(1.0-self._apply_norm(sent,100.0)))/6.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'pressure_release_index_D_scaled',release.fillna(0.0)*100.0,13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_release_index_D':release,'pressure_trapped_D':trapped},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】突破加速强攻引擎
        用途: 捕捉真实有效的突破确认。
        修改要点: 取消滞后趋势的强绑定，使用 _scaled 防止突破分高频饱和。
        """
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
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        core=self._apply_norm(brk,100.0)*self._apply_zg(df_index,brk)
        ind_norm=ind.where(ind<=1.0,ind/100.0).fillna(0.0)
        amp=1.0+(self._apply_norm(uptrend,100.0)+ind_norm+self._apply_norm(vac,100.0)+np.tanh(self._apply_hab(df,'mf',mf,21)).clip(lower=0)+self._apply_norm(t1,100.0)+self._apply_norm(hm,100.0)+self._apply_norm((elg.fillna(0.0)/amt).fillna(0.0),0.1)+self._apply_norm(cons,100.0)+self._apply_norm(abnorm,10.0))/9.0
        raw=core*(1.0-self._apply_norm(pen,100.0))*amp*(1.0+self._apply_kinematics(df,'breakout_quality_score_D_scaled',brk.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'breakout_quality_score_D':brk,'uptrend_strength_D':uptrend},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V50.0.0 · 完美抛光版】资金流吸筹质变引擎
        用途: 捕捉持续资金流入后的质变起爆点。
        修改要点: 修复 flow_efficiency_D 及 confidence 的标度失真。
        """
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
        eff_norm=np.abs(np.tanh(eff.fillna(0.0)/5.0))
        intra_norm=intra.where(intra<=1.0,intra/100.0).fillna(0.0)
        core=self._apply_norm(acc,100.0)*self._apply_zg(df_index,acc)
        amp=1.0+(eff_norm+intra_norm+self._apply_norm((buy_lg.fillna(0.0)/amt).fillna(0.0),0.1)+self._apply_norm(pers,100.0)+np.tanh(self._apply_hab(df,'l_net',l_net,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'gap',gap,13)).clip(lower=0)+np.tanh(self._apply_hab(df,'eng',energy,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'mf',mf,21)).clip(lower=0)+pit.fillna(0.0)*0.5+(1.0-self._apply_norm(c_ent,10.0)))/10.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'accumulation_signal_score_D_scaled',acc.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'accumulation_signal_score_D':acc,'flow_efficiency_D':eff},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】获利压迫与净流对冲引擎
        用途: 权衡高位获利盘抛压与主力资金主动承接强度的背离状况。
        修改要点: 强制使用 _scaled 防止压迫指标被原始大标度缓存抹平差异。
        """
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
        press_core=self._apply_norm(press,100.0)*self._apply_norm(dist_conf,100.0)*self._apply_zg(df_index,press)*self._apply_zg(df_index,dist_conf)
        press_amp=1.0+(self._apply_norm(dist_eng,100.0)+self._apply_norm(p_ratio,100.0)+self._apply_norm((sell_elg/amt).fillna(0),0.1)+self._apply_norm(marg,100.0)+para.fillna(0.0)+self._apply_norm(sent,100.0))/7.0
        press_total=press_core*press_amp*(1.0+self._apply_kinematics(df,'profit_pressure_D_scaled',press.fillna(0.0)/100.0,13).clip(lower=0))
        mf_ratio=(mf/amt).fillna(0.0)
        supp_core=np.tanh(mf_ratio*20.0).clip(lower=0)*self._apply_norm(cons,100.0)*self._apply_zg(df_index,mf.clip(lower=0))*self._apply_zg(df_index,cons)
        supp_amp=1.0+(1.0-self._apply_norm(win,100.0))
        supp_total=supp_core*supp_amp
        raw=supp_total-press_total*1.5
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'profit_pressure_D':press,'net_mf_amount_D':mf},calc_nodes={'press_total':press_total,'supp_total':supp_total,'raw_score':raw},final_result=res)
        return res

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】个股与板块龙脉共振引擎
        用途: 验证个股是否受真实板块强度支撑。
        修改要点: 行业排名强制使用 _scaled 解除绝对值带来的常年满分。
        """
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
        rank_norm=rank.where(rank<=1.0,rank/100.0).fillna(0.0)
        core=np.tanh(pct.fillna(0.0)/10.0)*rank_norm*self._apply_zg(df_index,pct)*self._apply_zg(df_index,rank)
        amp=1.0+(self._apply_norm(mkup,100.0)+(1.0+np.tanh(rk_acc.fillna(0.0)/10.0))*0.5+self._apply_norm(ldr,100.0)+np.tanh(self._apply_hab(df,'mf',mf,21)).clip(lower=0)+self._apply_norm(cons,100.0)+self._apply_norm(sync,100.0)+self._apply_norm(hm,100.0)+m_ldr.fillna(0.0)+self._apply_norm(sent,100.0))/10.0
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'industry_strength_rank_D_scaled',rank_norm,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pct_change_D':pct,'industry_strength_rank_D':rank},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 逻辑与门控恢复版】热门板块退潮预警引擎
        修改要点: 拨乱反正！必须同时具备[高热度]与[强资金出逃]，缺一不可。恢复 dump_energy 对主核的一票否决权。
        """
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
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        hot_state=self._apply_norm(hot,100.0)
        # [V65.0.0 极性修复] 恢复资金出逃能量与高热度状态的强制绑定(AND关系)，防止无风起浪的假阳性
        dump_energy=np.maximum.reduce([self._apply_norm(outq,100.0),self._apply_norm(dist,100.0),self._apply_norm((sell_elg.fillna(0.0)/amt).fillna(0.0),0.1),np.tanh(self._apply_hab(df,'mf',mf,13)).clip(upper=0).abs()])
        core=hot_state*dump_energy*self._apply_zg(df_index,hot_state*dump_energy)
        amp=1.0+(self._apply_norm(stag,100.0)+self._apply_norm(down,100.0)+(1.0-self._apply_norm(sent,100.0)))/3.0
        raw=core*amp*(1.0-self._apply_kinematics(df,'THEME_HOTNESS_SCORE_D_scaled',hot.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(raw**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'THEME_HOTNESS_SCORE_D':hot,'net_mf_amount_D':mf},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_ff_vs_structure_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 契约对齐版】资金领跑结构发令引擎
        修改要点: 拨乱反正！恢复提取资金净流入 (inflow_force)，并叠加滞后的弱结构进行机会共振。
        """
        method_name="_calculate_ff_vs_structure_relationship"
        required_signals=['uptrend_strength_D','flow_consistency_D','ma_arrangement_status_D','chip_structure_state_D','industry_stagnation_score_D','large_order_anomaly_D','STATE_ROBUST_TREND_D','net_mf_amount_D','chip_stability_D','flow_momentum_13d_D','volatility_adjusted_concentration_D','amount_D']
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
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        mf_ratio=(mf/amt).fillna(0.0)
        base_div=self._calculate_instantaneous_relationship(df,config).fillna(0.0).clip(lower=0)
        # [V65.0.0 极性修复] 资金领跑发令是一次正向机会 (Opportunity)，提取强流入(mf_ratio > 0)
        inflow_force=np.tanh(mf_ratio*50.0).clip(lower=0)
        struct_lag=1.0-self._apply_norm(up,100.0)
        core=base_div*inflow_force*struct_lag*self._apply_zg(df_index,inflow_force)
        amp=1.0+(self._apply_norm(cons,100.0)+np.tanh(self._apply_hab(df,'mom',mom,13)).clip(lower=0)+self._apply_norm(ma_s,1.0)+self._apply_norm(chip_s,1.0)+self._apply_norm(vac,100.0)+(1.0-self._apply_norm(stag,100.0)*0.5)+np.tanh(self._apply_hab(df,'anm',anom,13)).clip(lower=0)+rob.fillna(0.0)+(1.0-self._apply_norm(stab,100.0)))/9.0
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'uptrend_strength_D_scaled',up.fillna(0.0)/100.0,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'uptrend_strength_D':up,'net_mf_amount_D':mf},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】价筹稳态共振引力引擎
        用途: 量化价格拉升与筹码峰聚集的共振协同度。
        修改要点: 强制追加 _scaled 确保集中度加速不被死锁缓存劫持。
        """
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
        c_diff=pd.Series(np.where(cls.diff(1).fillna(0.0).abs()<1e-4,0.0,cls.diff(1).fillna(0.0)),index=df_index)
        price_force=np.tanh(c_diff/cls.replace(0,np.nan).fillna(1.0)*10.0)
        pk_norm=self._apply_norm(pk,100.0)
        core=price_force*pd.Series(np.where(price_force>0,pk_norm,1.0-pk_norm),index=df_index)*self._apply_zg(df_index,price_force)
        amp=1.0+(self._apply_norm(cvg,1.0)+self._apply_norm(hl,100.0)+np.tanh(self._apply_hab(df,'sc',sc,13)).clip(lower=0)+self._apply_norm(vac,100.0)+self._apply_norm(fi,100.0)+(1.0-self._apply_norm(ent,10.0)))/6.0
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'peak_concentration_D_scaled',pk.fillna(0.0)/100.0,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_concentration_D':pk,'close_D':cls},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V35.0.0】价资协同双向剥离
        修改要点: 修正 VPA_EFFICIENCY_D 标度失真。
        """
        method_name="_calculate_pf_relationship"
        required_signals=['net_mf_amount_D','close_D','price_vs_ma_13_ratio_D','main_force_activity_index_D','flow_momentum_13d_D','flow_impact_ratio_D','tick_chip_transfer_efficiency_D','VPA_EFFICIENCY_D','amount_D']
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
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        c_diff=pd.Series(np.where(cls.diff(1).fillna(0.0).abs()<1e-4,0.0,cls.diff(1).fillna(0.0)),index=df_index)
        mf_ratio=(mf/amt).fillna(0.0)
        price_force=np.tanh((c_diff/cls.replace(0,np.nan).fillna(1.0)).fillna(0.0)*20.0)
        mf_force=np.tanh(mf_ratio*50.0)
        core=(price_force*0.5+mf_force*0.5)*self._apply_zg(df_index,np.abs(price_force)+np.abs(mf_force))
        amp=1.0+(self._apply_norm(act,100.0)+np.tanh(self._apply_hab(df,'fm',fm,13)).clip(lower=0)+np.tanh(self._apply_hab(df,'imp',imp,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'tr',tr,21)).clip(lower=0)+np.abs(np.tanh(vpa.fillna(0.0)*5.0)))/5.0
        raw=core*amp*(1.0+np.tanh(self._apply_hab(df,'pm',pm,21)).abs()*0.5)*(1.0+np.abs(self._apply_kinematics(df,'mf_ratio_scaled',mf_ratio*100.0,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':mf,'close_D':cls},calc_nodes={'price_force':price_force,'mf_force':mf_force,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_price_vs_momentum_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 极性拨乱反正版】价势多维背离引擎
        修改要点: 物理极性彻底纠正！使用 roc_kin - p_kin。顶背离时(价涨量缩)必然输出负向高分(Risk=-1.0)。
        """
        method_name="_calculate_price_vs_momentum_divergence"
        required_signals=['close_D','ROC_13_D','VPA_EFFICIENCY_D','PRICE_ENTROPY_D','net_mf_amount_D','turnover_rate_f_D','GEOM_REG_SLOPE_D','GEOM_REG_R2_D','BIAS_21_D','GEOM_ARC_CURVATURE_D','market_sentiment_score_D','volatility_adjusted_concentration_D','amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        cls=self._get_safe_series(df,'close_D',method_name=method_name)
        roc=self._get_safe_series(df,'ROC_13_D',method_name=method_name)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        ent=self._get_safe_series(df,'PRICE_ENTROPY_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        to=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        slope=self._get_safe_series(df,'GEOM_REG_SLOPE_D',method_name=method_name)
        r2=self._get_safe_series(df,'GEOM_REG_R2_D',method_name=method_name)
        bias=self._get_safe_series(df,'BIAS_21_D',method_name=method_name)
        arc=self._get_safe_series(df,'GEOM_ARC_CURVATURE_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        cls_smooth=cls.rolling(13,min_periods=1).mean().replace(0,np.nan).fillna(1.0)
        p_kin=self._apply_kinematics(df,'close_D_scaled',(cls/cls_smooth).fillna(1.0)*10.0,13)
        roc_kin=self._apply_kinematics(df,'ROC_13_D_scaled',roc.fillna(0.0)/100.0,13)
        # [V65.0.0 极性修复] 顶背离: 价涨(p_kin>0)量缩(roc_kin<0) => kinematic_div < 0 (触发 Bipolar 风险态)
        kinematic_div=roc_kin - p_kin
        p_vel=np.tanh(pd.Series(np.where(cls.diff(1).fillna(0.0).abs()<1e-4,0.0,cls.diff(1).fillna(0.0)),index=df_index)/cls.replace(0,np.nan).fillna(1.0)*10.0)
        energy_div=(np.tanh(vpa.fillna(0.0)*5.0)+np.tanh((mf/amt).fillna(0.0)*20.0))/2.0 - p_vel
        geom_tension=(np.tanh(self._apply_hab(df,'arc',arc,21))-np.tanh(self._apply_hab(df,'bias',bias,21)))*0.5*(1.0+self._apply_norm(r2,1.0))
        raw_div=(kinematic_div*0.4+energy_div*0.3+geom_tension*0.3)
        core=raw_div*self._apply_zg(df_index,roc)
        amp=1.0+(np.abs(np.tanh(roc.fillna(0.0)/10.0))+np.abs(self._apply_norm(sent,100.0))+self._apply_norm(ent,10.0)+self._apply_norm(vac,100.0))/4.0
        raw=core*amp
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc,'VPA_EFFICIENCY_D':vpa},calc_nodes={'kinematic_div':kinematic_div,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · Z-Score 跨越零基防爆版】张量对冲基础单元
        修改要点: 废除 rolling.mean() 除法，改用 Z-Score (HAB) 将异构数据安全压制到 [-3, 3] 区间进行动力学求导，防止绝对值在 0 附近振荡导致的除零爆炸。
        """
        sa_name=config.get('signal_A')
        sb_name=config.get('signal_B')
        df_index=df.index
        rel_type=config.get('relationship_type','consensus')
        def _get_sig(name: str, src: str) -> Optional[pd.Series]:
            if src=='atomic_states': return self.strategy.atomic_states.get(name)
            try: return self._get_safe_series(df,name,np.nan,"_calculate_instantaneous_relationship")
            except ValueError: return None
        sa=_get_sig(sa_name,config.get('source_A','df'))
        sb=_get_sig(sb_name,config.get('source_B','df'))
        if sa is None or sb is None: return pd.Series(0.0,index=df_index,dtype=np.float32)
        sa=sa.fillna(0.0)
        sb=sb.fillna(0.0)
        ca=sa.diff(1).fillna(0.0) if config.get('change_type_A','pct')=='diff' else sa.pct_change(1).replace([np.inf,-np.inf],0.0).fillna(0.0)
        cb=sb.diff(1).fillna(0.0) if config.get('change_type_B','pct')=='diff' else sb.pct_change(1).replace([np.inf,-np.inf],0.0).fillna(0.0)
        # [V65.0.0] Z-Score 防爆平滑，杜绝资金流等过零信号导致除以极小数爆炸
        sa_z=self._apply_hab(df, f'{sa_name}_z', sa, 21)
        sb_z=self._apply_hab(df, f'{sb_name}_z', sb, 21)
        k_a=self._apply_kinematics(df,f'{sa_name}_z_scaled',sa_z/5.0,13)
        k_b=self._apply_kinematics(df,f'{sb_name}_z_scaled',sb_z/5.0,13)
        ma=np.tanh(self._apply_hab(df,f'{sa_name}_rel',ca,13))+k_a*0.5
        tb=np.tanh(self._apply_hab(df,f'{sb_name}_rel',cb,13))+k_b*0.5
        kf=config.get('signal_b_factor_k',1.0)
        if rel_type=='divergence': rel=(kf*tb-ma)/(kf+1.0)
        else: rel=np.sign(ma+kf*tb)*(np.abs(ma)*np.abs(tb))**0.5
        return np.tanh(np.sign(rel)*(np.abs(rel)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)

    def _calculate_dyn_vs_chip_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 物理极性收官版】力学筹码共振看跌引擎
        修改要点: 拨乱反正！引入牛市绝对否决权。当 ROC_13_D >= 0 时，属于主升浪洗盘，强行将风险警报彻底归零，严防被错杀。
        """
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
        core=(-bc).where(bc<0,0.0)*self._apply_zg(df_index,bc)
        amp=1.0+(self._apply_norm(prof,100.0)+self._apply_norm(win,100.0)+np.tanh(self._apply_hab(df,'mn',mean,13)).abs()+self._apply_norm(kurt,100.0)+self._apply_norm(vac,100.0)+self._apply_norm(down,100.0)+self._apply_norm(ent,10.0)+(1.0-self._apply_norm(sent,100.0)))/8.0
        # [V65.0.0 极性修复] 牛市绝对否决权。如果 ROC 大于等于 0（身处上升通道），直接归 0 抹杀做空警报
        roc_penalty=pd.Series(np.where(roc.fillna(0.0)<0.0,1.0,0.0),index=df_index)
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'ROC_13_D_scaled',roc.fillna(0.0)/100.0,13)))*roc_penalty
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 感光度全域统一版】洗盘反包诱空反弹引擎
        修改要点: 修复 pressure_trapped_D (pan) 和 stealth_flow_ratio_D (stl) 的微缩失真漏网，统一使用 np.tanh(x * 100.0) 激活感光度。
        """
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
        shk_norm=self._apply_norm(shk,100.0)
        tf_norm=tf.where(tf<=1.0,tf/100.0).fillna(0.0)
        abs_norm=np.tanh(abs_e.fillna(0.0)*100.0)
        reb_norm=(tf_norm+abs_norm)/2.0
        core=shk_norm*reb_norm*self._apply_zg(df_index,shk_norm*reb_norm)
        # [V65.0.0 标度修复] 防止极小比率被除以 100 彻底湮灭
        pan_norm=np.tanh(pan.fillna(0.0)*100.0)
        stl_norm=np.tanh(stl.fillna(0.0)*100.0)
        amp=1.0+(self._apply_norm(dist,100.0)+pan_norm+self._apply_norm(llck,1.0)+self._apply_norm(t_abn,10.0)+stl_norm+np.tanh(self._apply_hab(df,'vdev',vdev,13)).clip(upper=0).abs()+self._apply_norm(cls,100.0)+rnd.fillna(0.0)+(1.0-self._apply_norm(c_ent,10.0)))/9.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'shakeout_score_D_scaled',shk.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'shakeout_score_D':shk,'intraday_trough_filling_degree_D':tf,'absorption_energy_D':abs_e},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_fusion_trend_exhaustion_syndrome(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V49.0.0 · 标度统合版】趋势衰竭综合征引擎
        用途: 诊断极端抛物线拉升后的趋势衰竭与派发共振。
        修改要点: 统一将 profit_pressure_D 恢复为标准的 100 标度归一化；VPA严格实施极性单向整流，仅多头放量能压制预警。
        """
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
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        base_state=np.maximum(self._apply_norm(para,1.0),self._apply_norm(emot,1.0))**2
        # [V49.0.0 标度修复] 统一使用 _apply_norm 进行 100 标度归一
        dist_force=np.maximum(self._apply_norm(pres,100.0),self._apply_norm(dist_e,100.0))
        core=base_state*dist_force*self._apply_zg(df_index,base_state*dist_force)
        amp=1.0+(self._apply_norm(dist_c,100.0)+np.tanh((sell_elg.fillna(0.0)/amt).fillna(0.0)*10.0)+self._apply_norm(ent,10.0)+self._apply_norm(c_ent,10.0))/4.0
        # [V49.0.0 单向整流] 仅正向的高效拉升 VPA 才具备否决衰竭风险的资格
        vpa_bull=np.tanh(vpa.fillna(0.0)*5.0).clip(lower=0)
        veto=(1.0-self._apply_norm(hm,100.0)*0.9).clip(lower=0.1)*(1.0-vpa_bull*0.5)
        raw=core*amp*veto*(1.0+self._apply_kinematics(df,'profit_pressure_D_scaled',pres.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(raw**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'STATE_PARABOLIC_WARNING_D':para,'distribution_energy_D':dist_e},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_dyn_vs_chip_decay_rise(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】力学筹码阻尼看涨引擎
        用途: 捕捉下跌趋势极值区域的动能衰竭（减速）与套牢盘松动共振点。
        修改要点: 强制追加 _scaled 确保绝对大数(68.65)的减速得到正确反馈。
        """
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
        down_k=self._apply_kinematics(df,'downtrend_strength_D_scaled',down.fillna(0.0)/100.0,13)
        decay_force=np.abs(down_k.clip(upper=0))
        core=self._apply_norm(down,100.0)*np.tanh(pres.fillna(0.0)*100.0)*decay_force*self._apply_zg(df_index,down)*self._apply_zg(df_index,decay_force)
        amp=1.0+(np.tanh(abs_e.fillna(0.0)*100.0)+self._apply_norm(kurt,100.0)+np.tanh(self._apply_hab(df,'stb',stb,13)).clip(lower=0)+self._apply_norm(rev,100.0)+self._apply_norm(sup,10.0)+self._apply_norm(vac,100.0))/6.0
        raw=core*amp
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'downtrend_strength_D':down,'pressure_trapped_D':pres},calc_nodes={'decay_force':decay_force,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_smart_money_ignition(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】聪明钱协同点火引擎
        用途: 捕捉顶级游资和特大单的极速共振扫货点火。
        修改要点: 强制追加 _scaled 保障 0-100 热度攻击特征的正确微分计算。
        """
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
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        up=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        brk=self._get_safe_series(df,'STATE_BREAKOUT_CONFIRMED_D',method_name=method_name).clip(0,1)
        ne=self._get_safe_series(df,'net_energy_flow_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        core=self._apply_norm(hm,100.0)*self._apply_zg(df_index,hm)
        amp=1.0+(self._apply_norm(up,100.0)+self._apply_norm(t1,100.0)+self._apply_norm(f_acc,100.0)+self._apply_norm((elg.fillna(0.0)/amt).fillna(0.0),0.1)+self._apply_norm((t_net.fillna(0.0)/amt).fillna(0.0).clip(lower=0),0.1)+self._apply_norm(ne,100.0)+ldr.fillna(0.0)+self._apply_norm(sent,100.0)+brk.fillna(0.0))/9.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'HM_COORDINATED_ATTACK_D_scaled',hm.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(raw**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_COORDINATED_ATTACK_D':hm,'uptrend_strength_D':up},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_vpa_mf_coherence_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V38.0.0】量价主力相干共振引擎：修复 VPA 标度失真"""
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
        core=np.tanh(mc.fillna(0.0)/2.0).clip(lower=0)*self._apply_zg(df_index,mc)
        # [V38.0.0] 恢复 VPA (* 5.0) 极小标度放大感光度
        amp=1.0+(np.tanh(ve.fillna(0.0)*5.0).clip(lower=0)+self._apply_norm(cc,1.0)+self._apply_norm(fc,100.0)+self._apply_norm(vac,100.0)+np.tanh(ma.fillna(0.0)/2.0).clip(lower=0)+np.tanh(va.fillna(0.0)/2.0).clip(lower=0)+(1.0-self._apply_norm(c_ent,10.0)))/7.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'VPA_MF_ADJUSTED_EFF_D_scaled',ve.fillna(0.0),13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_COHERENCE_RESONANCE_D':mc,'VPA_MF_ADJUSTED_EFF_D':ve},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_mtf_fractal_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】多维时空分形共振引擎
        用途: 捕捉多时间框架(MTF)同步向上带来的几何分形大主升起点。
        修改要点: 强阻断未缩放缓存。
        """
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
        core=self._apply_norm(dw_sync.fillna(0.0)+dm_sync.fillna(0.0)+ml_sync.fillna(0.0)+sm_sync.fillna(0.0),400.0)*self._apply_zg(df_index,dw_sync)
        amp=1.0+((1.0-self._apply_norm(fractal_dim,2.0))+self._apply_norm(prob,100.0)+self._apply_norm(sent,100.0))/3.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'daily_weekly_sync_D_scaled',dw_sync.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(raw**1.5).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'daily_weekly_sync_D':dw_sync},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_intraday_siege_exhaustion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 物理裁决极性修正版】日内攻城拔寨衰竭引擎
        修改要点: 彻底纠正极性倒置漏洞！向上测试阻力(res_tests)本身不代表看涨，若收盘拉胯则说明多头"攻城衰竭"，必须记负分。通过提取 close_vector 统一赋予盘口波动的最终极性。
        """
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
        res_base=self._apply_norm(res_tests,10.0)*self._apply_zg(df_index,res_tests)
        sup_base=self._apply_norm(sup_tests,10.0)*self._apply_zg(df_index,sup_tests)
        closing_shock=self._apply_norm(closing,100.0)
        vwap_shock=np.tanh(self._apply_hab(df,'vwap',vwap_dev,21))
        vpa_norm=np.abs(np.tanh(vpa.fillna(0.0)*5.0))
        b_amp=1.0+(self._apply_norm(res_str,100.0)+closing_shock+vwap_shock.clip(lower=0)+vpa_norm)/4.0
        d_amp=1.0+(self._apply_norm(sup_str,100.0)+(1.0-closing_shock)+vwap_shock.clip(upper=0).abs()+(1.0-vpa_norm))/4.0
        # [V65.0.0 极性修复] 用收盘强度的双极状态 (-1.0 到 1.0) 作为绝对裁决极性
        close_vector=closing_shock*2.0-1.0
        # 激烈的交火作为绝对动能强度，由收盘向量决定最终赢家
        total_siege_power=res_base*b_amp+sup_base*d_amp
        core=total_siege_power*close_vector*self._apply_zg(df_index,total_siege_power)
        raw=core*(1.0+np.abs(self._apply_kinematics(df,'CLOSING_STRENGTH_D_scaled',closing.fillna(0.0)/100.0,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'intraday_resistance_test_count_D':res_tests,'CLOSING_STRENGTH_D':closing},calc_nodes={'total_siege_power':total_siege_power,'close_vector':close_vector,'raw_score':raw},final_result=res)
        return res

    def _calculate_overnight_intraday_tearing(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】隔夜日内动能撕裂引擎
        用途: 捕捉跳空高开且日内全天真实动能共振（或天地板诱多）带来的极性撕裂。
        修改要点: 追加 _scaled 阻断跳空动能大数缓存饱和。
        """
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
        gap_base=np.tanh(gap.fillna(0.0).abs()/10.0)
        diff=afternoon.fillna(0.0)+closing.fillna(0.0)*0.5-morning.fillna(0.0)*1.5
        intraday_vector=np.tanh(diff/100.0)
        core=gap_base*intraday_vector*self._apply_zg(df_index,gap)*self._apply_zg(df_index,diff)
        risk_amp=self._apply_norm(range_ratio,100.0)+np.tanh(self._apply_hab(df,'och',och,13)).clip(lower=0)
        opp_amp=self._apply_norm(sent,100.0)
        amp=1.0+pd.Series(np.where(core<0,risk_amp/2.0,opp_amp),index=df_index)
        raw=core*amp*(1.0+self._apply_kinematics(df,'GAP_MOMENTUM_STRENGTH_D_scaled',gap.fillna(0.0)/10.0,13).abs())
        is_res=(np.sign(gap.fillna(0.0))*np.sign(intraday_vector)).clip(lower=0)
        raw=raw*(1.0-is_res*0.5)
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GAP_MOMENTUM_STRENGTH_D':gap,'morning_flow_ratio_D':morning,'afternoon_flow_ratio_D':afternoon},calc_nodes={'gap_base':gap_base,'intraday_vector':intraday_vector,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_chip_center_kinematics(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 物理极性重构版】筹码重心迁徙动力学引擎
        修改要点: 拨乱反正！真正的“高位锁仓”应当是【低换手】。将锁仓的主核改为奖励低换手 (1.0 - to_active)，修复高换手被误判为锁仓的逻辑死锁。
        """
        method_name="_calculate_chip_center_kinematics"
        required_signals=['peak_migration_speed_5d_D','intraday_cost_center_volatility_D','price_to_weight_avg_ratio_D','turnover_rate_f_D','cost_50pct_D','chip_entropy_D','close_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        migration=self._get_safe_series(df,'peak_migration_speed_5d_D',method_name=method_name)
        cost_vol=self._get_safe_series(df,'intraday_cost_center_volatility_D',method_name=method_name)
        price_to_cost=self._get_safe_series(df,'price_to_weight_avg_ratio_D',method_name=method_name).fillna(1.0)
        turnover=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        cost_50=self._get_safe_series(df,'cost_50pct_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        cls=self._get_safe_series(df,'close_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        mig_base=self._apply_norm(migration,10.0)*self._apply_zg(df_index,migration)
        vol_shock=self._apply_norm(cost_vol,100.0)
        # [V65.0.0 极性修复] 物理分离：派发需要高换手(to_active)，锁仓伴随低换手(to_locked)
        to_active=np.tanh(turnover.fillna(0.0)/2.0)
        to_locked=(1.0-to_active)*self._apply_zg(df_index,turnover) 
        dist_amp=1.0+(vol_shock+to_active+np.tanh(self._apply_hab(df,'cost',cost_50,21)).clip(lower=0)+self._apply_norm(c_ent,10.0))/4.0
        distribution=mig_base*dist_amp
        price_dev=np.maximum(price_to_cost.fillna(1.0)-1.0,0.0)
        # [V65.0.0] 锁仓主核：无迁移 + 高溢价 + 极低换手
        lock_core=(1.0-mig_base)*self._apply_norm(price_dev,0.2)*to_locked*self._apply_zg(df_index,price_dev)
        lock_amp=1.0+((1.0-vol_shock)+np.tanh(self._apply_hab(df,'ptc',price_to_cost,21)).clip(lower=0)+(1.0-self._apply_norm(c_ent,10.0)))/3.0
        lock=lock_core*lock_amp
        cost_norm=(cost_50.fillna(0.0)/cls).fillna(1.0)*10.0
        raw=(lock.fillna(0.0)-distribution.fillna(0.0))*(1.0+np.abs(self._apply_kinematics(df,'cost_50_norm',cost_norm,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_migration_speed_5d_D':migration,'turnover_rate_f_D':turnover},calc_nodes={'lock_core':lock_core,'mig_base':mig_base,'to_locked':to_locked,'raw_score':raw},final_result=res)
        return res

    def _calculate_ma_compression_explosion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 契约绝对约束收官版】均线奇点核爆引擎
        修改要点: 由于本引擎签署的是 Opportunity (Unipolar) 契约，输出域必须被物理锁死在 [0, 1] 之间。将底层的 .clip(-1, 1) 修正为 .clip(0, 1) 以符合绝对架构规范。
        """
        method_name="_calculate_ma_compression_explosion"
        required_signals=['MA_POTENTIAL_COMPRESSION_RATE_D','chip_convergence_ratio_D','TURNOVER_STABILITY_INDEX_D','MACDh_13_34_8_D','energy_concentration_D','volatility_adjusted_concentration_D','close_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        ma_comp=self._get_safe_series(df,'MA_POTENTIAL_COMPRESSION_RATE_D',method_name=method_name)
        chip_conv=self._get_safe_series(df,'chip_convergence_ratio_D',method_name=method_name)
        to_stab=self._get_safe_series(df,'TURNOVER_STABILITY_INDEX_D',method_name=method_name)
        macd=self._get_safe_series(df,'MACDh_13_34_8_D',method_name=method_name)
        energy_conc=self._get_safe_series(df,'energy_concentration_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        cls=self._get_safe_series(df,'close_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        macd_ratio=(macd.fillna(0.0)/cls)*100.0
        ignition=np.tanh(macd_ratio/5.0)*self._apply_zg(df_index,macd.clip(lower=0))
        core=self._apply_norm(ma_comp,1.0)*ignition*self._apply_zg(df_index,ma_comp)
        amp=1.0+(self._apply_norm(chip_conv,1.0)+self._apply_norm(to_stab,1.0)+self._apply_norm(energy_conc,100.0)+self._apply_norm(vac,100.0))/4.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'MA_POTENTIAL_COMPRESSION_RATE_D_scaled',ma_comp,13).clip(lower=0))
        # [V65.0.0 契约约束] 机会信号绝对锁死在 [0, 1] 区间
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_POTENTIAL_COMPRESSION_RATE_D':ma_comp,'MACDh_13_34_8_D':macd},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_top_tier_hm_harvesting(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】顶级游资收割镰刀引擎
        用途: 捕捉一线游资极其活跃拉抬、但伴随异常特大单净流出的天地板收割陷阱。
        修改要点: 阻断 net_ratio 放大100倍后在动力学缓存中的截胡错误。
        """
        method_name="_calculate_top_tier_hm_harvesting"
        required_signals=['HM_ACTIVE_TOP_TIER_D','CLOSING_STRENGTH_D','tick_large_order_net_D','amount_D','outflow_quality_D','distribution_energy_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hm_active=self._get_safe_series(df,'HM_ACTIVE_TOP_TIER_D',method_name=method_name)
        closing=self._get_safe_series(df,'CLOSING_STRENGTH_D',method_name=method_name).fillna(50.0)
        large_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        outflow_q=self._get_safe_series(df,'outflow_quality_D',method_name=method_name)
        dist_eng=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        net_ratio=(large_net/amt).fillna(0.0)
        dump_force=np.tanh(net_ratio*10.0).clip(upper=0).abs()
        hm_norm=self._apply_norm(hm_active,100.0)
        core=hm_norm*dump_force*self._apply_zg(df_index,hm_norm*dump_force)
        amp=1.0+((1.0-self._apply_norm(closing,100.0))+self._apply_norm(outflow_q,100.0)+self._apply_norm(dist_eng,100.0)+(1.0-self._apply_norm(sent,100.0)))/4.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'net_ratio_scaled',net_ratio*100.0,13).clip(upper=0).abs())
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_ACTIVE_TOP_TIER_D':hm_active,'tick_large_order_net_D':large_net},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_vwap_magnetic_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V42.0.0 · 无量纲不变版】VWAP磁性黑洞引力引擎
        用途: 诊断价格在高度乖离 VWAP 后的必然物理均值拉回引力。
        修改要点: 修复 vwap_dev 的过度感光，将 * 5.0 改为 / 5.0，恢复正常 2%~10% 偏离区间的灰度分辨率。
        """
        method_name="_calculate_vwap_magnetic_divergence"
        required_signals=['vwap_deviation_D','reversal_prob_D','intraday_main_force_activity_D','intraday_cost_center_migration_D','volume_ratio_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        vwap_dev=self._get_safe_series(df,'vwap_deviation_D',method_name=method_name)
        rev_prob=self._get_safe_series(df,'reversal_prob_D',method_name=method_name)
        mf_act=self._get_safe_series(df,'intraday_main_force_activity_D',method_name=method_name).fillna(50.0)
        cost_mig=self._get_safe_series(df,'intraday_cost_center_migration_D',method_name=method_name)
        vol_ratio=self._get_safe_series(df,'volume_ratio_D',method_name=method_name).fillna(1.0)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        # [V42.0.0 标度修复] vwap_dev 为百分点数值 (如 2.2 表示 2.2%)。除以 5.0，使得 5% 的绝对偏离产生 0.76 的引力
        dev_base=np.tanh(vwap_dev.fillna(0.0)/5.0)
        core=-1.0*dev_base*self._apply_zg(df_index,dev_base)
        mismatch=(1.0-(self._apply_norm(mf_act,100.0)*2.0-1.0)*np.sign(dev_base))*(1.0-np.tanh(self._apply_hab(df,'mig',cost_mig,21))*np.sign(dev_base))
        amp=1.0+(self._apply_norm(rev_prob,100.0)+self._apply_norm(vol_ratio,10.0)+(1.0-self._apply_norm(c_ent,10.0)))/3.0
        raw=core*mismatch.clip(lower=0.1)*amp*(1.0+np.abs(self._apply_kinematics(df,'vwap_deviation_D_scaled',vwap_dev.fillna(0.0)/5.0,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'vwap_deviation_D':vwap_dev,'reversal_prob_D':rev_prob},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_multi_peak_avalanche_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】多峰筹码断层雪崩引擎
        用途: 诊断多筹码峰结构下的断层雪崩风险。
        修改要点: 追加 _scaled 解除绝对指标缓存。
        """
        method_name="_calculate_multi_peak_avalanche_risk"
        required_signals=['is_multi_peak_D','chip_divergence_ratio_D','intraday_distribution_confidence_D','downtrend_strength_D','chip_entropy_D','chip_stability_change_5d_D','volatility_adjusted_concentration_D','pressure_trapped_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        mp=self._get_safe_series(df,'is_multi_peak_D',method_name=method_name).clip(0,1)
        div=self._get_safe_series(df,'chip_divergence_ratio_D',method_name=method_name)
        dist=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        down=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        stab_chg=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        trap=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        trap_norm=np.tanh(trap.fillna(0.0)*100.0)
        env_risk=np.maximum(self._apply_norm(dist,100.0),self._apply_norm(down,100.0))
        core=mp*env_risk*trap_norm*self._apply_zg(df_index,mp*env_risk)
        amp=1.0+(self._apply_norm(div,100.0)+self._apply_norm(ent,10.0)+np.tanh(self._apply_hab(df,'stab',stab_chg.diff(1).fillna(0.0),13)).clip(upper=0).abs()+(1.0-self._apply_norm(vac,100.0)))/4.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'chip_divergence_ratio_D_scaled',div.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'is_multi_peak_D':mp,'downtrend_strength_D':down,'pressure_trapped_D':trap},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_sector_lifecycle_tailwind(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】板块生命周期顺风引擎
        用途: 个股顺应/逆背板块大周期的叠加放大/削弱器。
        修改要点: 追加 _scaled 解除绝对大数缓存。
        """
        method_name="_calculate_sector_lifecycle_tailwind"
        required_signals=['industry_preheat_score_D','industry_markup_score_D','industry_stagnation_score_D','industry_downtrend_score_D','uptrend_strength_D','industry_rank_accel_D','industry_strength_rank_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        preheat=self._get_safe_series(df,'industry_preheat_score_D',method_name=method_name)
        markup=self._get_safe_series(df,'industry_markup_score_D',method_name=method_name)
        stag=self._get_safe_series(df,'industry_stagnation_score_D',method_name=method_name)
        down=self._get_safe_series(df,'industry_downtrend_score_D',method_name=method_name)
        up=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        rk_acc=self._get_safe_series(df,'industry_rank_accel_D',method_name=method_name)
        rank=self._get_safe_series(df,'industry_strength_rank_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        vector=(self._apply_norm(markup,100.0)*1.0+self._apply_norm(preheat,100.0)*0.5)-(self._apply_norm(stag,100.0)*0.5+self._apply_norm(down,100.0)*1.0)
        core=vector*self._apply_zg(df_index,vector)
        amp=1.0+(self._apply_norm(up,100.0)+self._apply_norm(rank,100.0)+np.tanh(rk_acc.fillna(0.0)/10.0)*np.sign(vector)+self._apply_norm(sent,100.0))/4.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'industry_markup_score_D_scaled',markup.fillna(0.0)/100.0,13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)*2.0).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'industry_markup_score_D':markup},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_time_asymmetry_trap(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V38.0.0】时间非对称陷阱防爆引擎
        修改要点: 修复 vpa.abs() * 5.0 正确放大 VPA 弱小量级，消除 /2.0 带来的致盲。
        """
        method_name="_calculate_time_asymmetry_trap"
        required_signals=['morning_flow_ratio_D','afternoon_flow_ratio_D','intraday_peak_valley_ratio_D','profit_pressure_D','closing_flow_ratio_D','high_freq_flow_divergence_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        morning=self._get_safe_series(df,'morning_flow_ratio_D',method_name=method_name)
        afternoon=self._get_safe_series(df,'afternoon_flow_ratio_D',method_name=method_name)
        closing=self._get_safe_series(df,'closing_flow_ratio_D',method_name=method_name)
        pv=self._get_safe_series(df,'intraday_peak_valley_ratio_D',method_name=method_name)
        pp=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        hd=self._get_safe_series(df,'high_freq_flow_divergence_D',method_name=method_name)
        vpa=self._get_safe_series(df,'VPA_EFFICIENCY_D',method_name=method_name)
        diff=afternoon.fillna(0.0)+closing.fillna(0.0)*0.5-morning.fillna(0.0)*1.5
        core=np.tanh(diff/100.0)*self._apply_zg(df_index,diff)
        risk_amp=self._apply_norm(pv,10.0)+self._apply_norm(pp,100.0)+np.tanh(self._apply_hab(df,'hd',hd,21)).clip(lower=0)
        # [V38.0.0] vpa 是极小值，*5.0 放大量纲
        opp_amp=np.tanh(vpa.fillna(0.0).abs()*5.0)*3.0
        amp=1.0+pd.Series(np.where(core<0,risk_amp/3.0,opp_amp/2.0),index=df_index)
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'morning_flow_ratio_D_scaled',morning.fillna(0.0)/100.0,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'morning_flow_ratio_D':morning,'afternoon_flow_ratio_D':afternoon},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_high_pos_liquidity_squeeze(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V35.0.0 · 极限轧空解绑版】高位流动性真空轧空引擎
        修改要点: 彻底解除滞后的均线趋势(uptrend_strength_D)约束。将“高换手”视为打破真空的物理绞肉机，强制压制轧空核心！
        """
        method_name="_calculate_high_pos_liquidity_squeeze"
        required_signals=['price_percentile_position_D','high_position_lock_ratio_90_D','flow_persistence_minutes_D','short_term_chip_ratio_D','uptrend_strength_D','turnover_rate_f_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        perc=self._get_safe_series(df,'price_percentile_position_D',method_name=method_name)
        lock=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name)
        per=self._get_safe_series(df,'flow_persistence_minutes_D',method_name=method_name)
        short=self._get_safe_series(df,'short_term_chip_ratio_D',method_name=method_name)
        up=self._get_safe_series(df,'uptrend_strength_D',method_name=method_name)
        to=self._get_safe_series(df,'turnover_rate_f_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        pos_norm=perc.where(perc<=1.0,perc/100.0).fillna(0.0)
        lock_norm=lock.where(lock<=1.0,lock/100.0).fillna(0.0)
        to_penalty=np.tanh(to.fillna(0.0)/15.0)
        # [V35.0.0] 解绑 uptrend_strength，保障首日妖股启动异动；保留对高换手的绝对压制
        core=lock_norm*pos_norm.clip(lower=0)*(1.0-to_penalty)*self._apply_zg(df_index,lock)
        amp=1.0+(self._apply_norm(up,100.0)+self._apply_norm(per,100.0)+(1.0-self._apply_norm(short,1.0))+(1.0-self._apply_norm(c_ent,10.0)))/4.0
        raw=(core*amp)**1.5*(1.0+self._apply_kinematics(df,'high_position_lock_ratio_90_D_scaled',lock_norm,13).clip(lower=0))
        res=np.tanh(raw*2.0).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'price_percentile_position_D':perc,'high_position_lock_ratio_90_D':lock,'turnover_rate_f_D':to},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_institutional_structural_exit(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V34.0.0】机构结构性清仓逃顶引擎
        用途: 捕捉机构在高位通过特大卖单实施的毁灭性出货撤离。
        修改要点: sell_ratio 放大百倍后计算 kinematics 时，强制追加 _scaled 绕开缓存，保障斜率侦测。
        """
        method_name="_calculate_institutional_structural_exit"
        required_signals=['sell_elg_amount_D','sell_lg_amount_D','amount_D','distribution_energy_D','downtrend_strength_D','high_position_lock_ratio_90_D','chip_stability_change_5d_D','market_sentiment_score_D','price_percentile_position_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        sell_elg=self._get_safe_series(df,'sell_elg_amount_D',method_name=method_name)
        sell_lg=self._get_safe_series(df,'sell_lg_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        dist=self._get_safe_series(df,'distribution_energy_D',method_name=method_name)
        down=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        lock=self._get_safe_series(df,'high_position_lock_ratio_90_D',method_name=method_name)
        stab=self._get_safe_series(df,'chip_stability_change_5d_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        pos=self._get_safe_series(df,'price_percentile_position_D',method_name=method_name)
        sell_ratio=((sell_elg.fillna(0.0)+sell_lg.fillna(0.0)*0.5)/amt).fillna(0.0)
        sell_energy=self._apply_norm(sell_ratio,0.05)
        pos_norm=pos.where(pos<=1.0,pos/100.0).fillna(0.0)
        lock_norm=lock.where(lock<=1.0,lock/100.0).fillna(0.0)
        high_state=np.maximum(lock_norm,pos_norm.clip(lower=0))
        sell_gate=pd.Series(np.where(sell_ratio>0.02,1.0,0.0),index=df_index)
        core=high_state*sell_energy*sell_gate*self._apply_zg(df_index,high_state*sell_energy)
        amp=1.0+(self._apply_norm(dist,100.0)+self._apply_norm(down,100.0)+np.tanh(self._apply_hab(df,'lkd',lock.diff(1).fillna(0.0),13)).clip(upper=0).abs()+np.tanh(self._apply_hab(df,'stb',stab,13)).clip(upper=0).abs()+(1.0-np.tanh(sent.fillna(0.0)/5.0).abs()))/5.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'sell_ratio_scaled',sell_ratio*100.0,13).clip(upper=0).abs())
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'sell_ratio':sell_ratio,'price_percentile_position_D':pos},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_institutional_sweep(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V31.0.0】机构超大单扫货核爆引擎
        修改要点: 恢复 10% 的高门槛扫货满分标度。
        """
        method_name="_calculate_institutional_sweep"
        required_signals=['buy_elg_amount_D','buy_lg_amount_D','amount_D','tick_chip_transfer_efficiency_D','flow_consistency_D','net_mf_amount_D','flow_impact_ratio_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        buy_elg=self._get_safe_series(df,'buy_elg_amount_D',method_name=method_name)
        buy_lg=self._get_safe_series(df,'buy_lg_amount_D',method_name=method_name)
        amt=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        tr=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        cons=self._get_safe_series(df,'flow_consistency_D',method_name=method_name)
        mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        imp=self._get_safe_series(df,'flow_impact_ratio_D',method_name=method_name)
        sent=self._get_safe_series(df,'market_sentiment_score_D',method_name=method_name)
        buy_ratio=((buy_elg.fillna(0.0)+buy_lg.fillna(0.0)*0.5)/amt).fillna(0.0)
        # [V31.0.0] 恢复 0.1(10%) 满分标度
        core=self._apply_norm(buy_ratio,0.1)*self._apply_zg(df_index,buy_ratio)
        amp=1.0+(np.tanh(self._apply_hab(df,'mf',mf,55)).clip(lower=0)+self._apply_norm(tr,1e6)+self._apply_norm(cons,100.0)+self._apply_norm(imp,10.0)+self._apply_norm(sent,100.0))/5.0
        # [V31.0.0] 动力学防爆
        raw=core*amp*(1.0+self._apply_kinematics(df,'buy_ratio',buy_ratio*100.0,13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'buy_elg_amount_D':buy_elg,'net_mf_amount_D':mf},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_hf_algo_manipulation_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V31.0.0】高频算法诱骗崩塌防线
        修改要点: 偏度极易达到 20，分母设为 50 以保留灰度特征。
        """
        method_name="_calculate_hf_algo_manipulation_risk"
        required_signals=['high_freq_flow_skewness_D','high_freq_flow_kurtosis_D','large_order_anomaly_D','price_flow_divergence_D','intraday_price_distribution_skewness_D','tick_abnormal_volume_ratio_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        skew=self._get_safe_series(df,'high_freq_flow_skewness_D',method_name=method_name)
        kurt=self._get_safe_series(df,'high_freq_flow_kurtosis_D',method_name=method_name)
        anom=self._get_safe_series(df,'large_order_anomaly_D',method_name=method_name)
        div=self._get_safe_series(df,'price_flow_divergence_D',method_name=method_name)
        pskew=self._get_safe_series(df,'intraday_price_distribution_skewness_D',method_name=method_name)
        abn=self._get_safe_series(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        # [V31.0.0] 偏度极易达到 20，分母设为 50 以保留灰度特征
        skew_base=np.tanh(skew.fillna(0.0).abs()/50.0)
        anom_base=self._apply_norm(anom,1.0)
        core=skew_base*anom_base*self._apply_zg(df_index,skew_base*anom_base)
        amp=1.0+(np.tanh(self._apply_hab(df,'div',div,21)).clip(lower=0)+np.tanh(self._apply_hab(df,'kurt',kurt,34)).clip(lower=0)+np.tanh(self._apply_hab(df,'mis',skew.fillna(0.0)-pskew.fillna(0.0),13)).abs()+np.tanh(self._apply_hab(df,'abn',abn,21)).clip(lower=0)+(1.0-self._apply_norm(vac,100.0)))/5.0
        raw=core*amp*(1.0+self._apply_kinematics(df,'large_order_anomaly_D',anom,13).clip(upper=0).abs())
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(0,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'high_freq_flow_skewness_D':skew,'large_order_anomaly_D':anom},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res

    def _calculate_ma_rubber_band_reversal(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V65.0.0 · 缓存阻断收官版】均线张力极值反噬引擎
        修改要点: 强制追加 _scaled 后缀，切断底层对未缩放原始斜率的错误缓存，恢复动力学引擎对乖离率的灰度感知。同时修复 trap_p 感光度。
        """
        method_name="_calculate_ma_rubber_band_reversal"
        required_signals=['MA_RUBBER_BAND_EXTENSION_D','MA_POTENTIAL_TENSION_INDEX_D','ADX_14_D','profit_pressure_D','pressure_trapped_D','BIAS_21_D','reversal_prob_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        ext=self._get_safe_series(df,'MA_RUBBER_BAND_EXTENSION_D',method_name=method_name)
        tension=self._get_safe_series(df,'MA_POTENTIAL_TENSION_INDEX_D',method_name=method_name)
        adx=self._get_safe_series(df,'ADX_14_D',method_name=method_name)
        profit_p=self._get_safe_series(df,'profit_pressure_D',method_name=method_name)
        trap_p=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        bias=self._get_safe_series(df,'BIAS_21_D',method_name=method_name)
        rev=self._get_safe_series(df,'reversal_prob_D',method_name=method_name)
        c_ent=self._get_safe_series(df,'chip_entropy_D',method_name=method_name)
        supp=1.0-np.tanh(np.maximum(adx.fillna(0.0)-35.0,0.0)/15.0)
        c_ent_n=self._apply_norm(c_ent,10.0)
        trap_norm=np.tanh(trap_p.fillna(0.0)*100.0)
        t_base=np.tanh(ext.fillna(0.0)/10.0).clip(lower=0)*self._apply_zg(df_index,ext.clip(lower=0))
        t_amp=1.0+(self._apply_norm(tension,100.0)+self._apply_norm(profit_p,100.0)+np.tanh(self._apply_hab(df,'bias',bias,21)).clip(lower=0)+self._apply_norm(rev,100.0)+(1.0-c_ent_n))/5.0
        b_base=np.tanh(ext.fillna(0.0)/10.0).clip(upper=0).abs()*self._apply_zg(df_index,ext.clip(upper=0))
        b_amp=1.0+(self._apply_norm(tension,100.0)+trap_norm+np.tanh(self._apply_hab(df,'bias',bias,21)).clip(upper=0).abs()+self._apply_norm(rev,100.0)+(1.0-c_ent_n))/5.0
        # [V65.0.0 缓存阻断] 追加 _scaled 后缀，击穿底层错误缓存，激活真实灰度求导
        raw=(b_base*b_amp-t_base*t_amp)*supp*(1.0+np.abs(self._apply_kinematics(df,'MA_RUBBER_BAND_EXTENSION_D_scaled',ext.fillna(0.0)/100.0,13)))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_RUBBER_BAND_EXTENSION_D':ext},calc_nodes={'top_base':t_base,'bot_base':b_base,'raw_score':raw},final_result=res)
        return res

    def _calculate_geometric_trend_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V42.0.0 · 无量纲不变版】几何流形趋势共振引擎
        用途: 通过流形曲率回归精确感知中线趋势爆发。
        修改要点: 将绝对几何斜率 (GEOM_REG_SLOPE_D) 除以当期收盘价转化为百分比，彻底解决百元股与仙股的量纲错位。
        """
        method_name="_calculate_geometric_trend_resonance"
        required_signals=['GEOM_REG_R2_D','GEOM_REG_SLOPE_D','GEOM_ARC_CURVATURE_D','GEOM_CHANNEL_POS_D','PRICE_FRACTAL_DIM_D','trend_confirmation_score_D','volatility_adjusted_concentration_D','close_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        r2=self._get_safe_series(df,'GEOM_REG_R2_D',method_name=method_name)
        slope=self._get_safe_series(df,'GEOM_REG_SLOPE_D',method_name=method_name)
        curv=self._get_safe_series(df,'GEOM_ARC_CURVATURE_D',method_name=method_name)
        pos=self._get_safe_series(df,'GEOM_CHANNEL_POS_D',method_name=method_name)
        frac=self._get_safe_series(df,'PRICE_FRACTAL_DIM_D',method_name=method_name)
        conf=self._get_safe_series(df,'trend_confirmation_score_D',method_name=method_name)
        vac=self._get_safe_series(df,'volatility_adjusted_concentration_D',method_name=method_name)
        cls=self._get_safe_series(df,'close_D',method_name=method_name).replace(0,np.nan).fillna(1.0)
        # [V42.0.0 标度修复] 将绝对斜率转化为相对于收盘价的无量纲日均涨幅百分比
        slope_ratio=(slope.fillna(0.0)/cls)*100.0
        core=np.tanh(slope_ratio/10.0)*self._apply_norm(r2,1.0)*self._apply_zg(df_index,slope)*self._apply_zg(df_index,r2)
        dyn=np.tanh(self._apply_hab(df,'curv',curv,21))-(self._apply_norm(pos,1.0)-0.5)*2.0*0.3
        amp=1.0+((1.0-self._apply_norm(frac,2.0))+self._apply_norm(conf,100.0)+dyn.clip(lower=0)+self._apply_norm(vac,100.0))/4.0
        # 动力学引擎也必须采用无量纲比例
        raw=core*amp*(1.0+self._apply_kinematics(df,'GEOM_REG_SLOPE_D_scaled',slope_ratio/10.0,13).clip(lower=0))
        res=np.tanh(np.sign(raw)*(np.abs(raw)**1.5)).clip(-1,1).fillna(0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GEOM_REG_SLOPE_D':slope,'GEOM_REG_R2_D':r2},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return res




