# 文件: strategies/trend_following/intelligence/process_intelligence.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any
from numba import njit
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

# ==========================================
# 外部 Numba 硬件级加速函数 (Just-In-Time)
# ==========================================
@njit(fastmath=True, cache=True)
def _jit_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n == 0: return out
    sum_val = 0.0
    for i in range(n):
        sum_val += arr[i]
        if i >= window:
            sum_val -= arr[i - window]
        count = min(i + 1, window)
        out[i] = sum_val / count
    return out

@njit(fastmath=True, cache=True)
def _jit_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n == 0: return out
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(n):
        val = arr[i]
        sum_val += val
        sum_sq += val * val
        if i >= window:
            old_val = arr[i - window]
            sum_val -= old_val
            sum_sq -= old_val * old_val
        count = min(i + 1, window)
        if count > 1:
            mean = sum_val / count
            var = (sum_sq - count * mean * mean) / (count - 1)
            if var < 0.0: var = 0.0
            out[i] = np.sqrt(var)
        else:
            out[i] = 0.0
    return out

@njit(fastmath=True, cache=True)
def _jit_ema(arr: np.ndarray, span: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n == 0: return out
    alpha = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * alpha + out[i-1] * (1.0 - alpha)
    return out

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
        【V334.0.0 · 毒瘤切除与规范化版】
        - 核心修复: 移除了在内存中强行篡改 Bipolar 信号 penalty_weight 符号的荒谬“热修复”。
        """
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
        self.calculate_main_force_control_processor = CalculateMainForceControlRelationship(strategy_instance, self.helper)
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

    def _judge_domain_reversal(self, bipolar_domain_health: np.ndarray, config: Dict, df: pd.DataFrame) -> Dict[str, pd.Series]:
        domain_name=config.get('domain_name','未知领域')
        output_bottom_name=config.get('output_bottom_reversal_name')
        output_top_name=config.get('output_top_reversal_name')
        df_index=df.index
        arr=bipolar_domain_health
        health_yesterday=np.zeros_like(arr)
        health_yesterday[1:]=arr[:-1]
        health_change=np.zeros_like(arr)
        health_change[1:]=arr[1:]-arr[:-1]
        shock=self._apply_hab(df,'reversal',health_change,13)
        bottom_context=np.clip(1.0-health_yesterday,0.0,2.0)
        bottom_shock=np.clip(shock,0.0,None)
        bottom_reversal_raw=(bottom_shock*bottom_context)**1.5
        bottom_reversal_score=np.clip(np.tanh(bottom_reversal_raw),0.0,1.0)
        top_context=np.clip(1.0+health_yesterday,0.0,2.0)
        top_shock=np.abs(np.clip(shock,None,0.0))
        top_reversal_raw=(top_shock*top_context)**1.5
        top_reversal_score=np.clip(np.tanh(top_reversal_raw),0.0,1.0)
        self._probe_variables(method_name=f"_judge_domain_reversal ({domain_name})",df_index=df_index,raw_inputs={'bipolar_domain_health':arr},calc_nodes={'health_change':health_change,'shock':shock,'bottom_context':bottom_context,'top_context':top_context},final_result=bottom_reversal_score)
        return {output_bottom_name:pd.Series(bottom_reversal_score,index=df_index,dtype=np.float32),output_top_name:pd.Series(top_reversal_score,index=df_index,dtype=np.float32)}

    def _diagnose_split_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        states={}
        output_names=config.get('output_names',{})
        opportunity_signal_name=output_names.get('opportunity')
        risk_signal_name=output_names.get('risk')
        if not opportunity_signal_name or not risk_signal_name: return {}
        relationship_score=self._calculate_price_efficiency_relationship(df,config)
        if relationship_score.empty: return {}
        arr=relationship_score.to_numpy(dtype=np.float32)
        disp=np.zeros_like(arr)
        disp[self.meta_window:]=arr[self.meta_window:]-arr[:-self.meta_window]
        mom=np.zeros_like(disp)
        mom[1:]=disp[1:]-disp[:-1]
        bipolar_displacement=np.tanh(self._apply_hab(df,'disp',disp,self.meta_window*2))
        bipolar_momentum=np.tanh(self._apply_hab(df,'mom',mom,13))
        meta_score=(bipolar_displacement*self.meta_score_weights[0]+bipolar_momentum*self.meta_score_weights[1])
        meta_score=np.sign(meta_score)*(np.abs(meta_score)**1.5)
        meta_score=np.clip(meta_score,-1.0,1.0)
        states[opportunity_signal_name]=pd.Series(np.clip(meta_score,0.0,None),index=df.index,dtype=np.float32)
        states[risk_signal_name]=pd.Series(np.abs(np.clip(meta_score,None,0.0)),index=df.index,dtype=np.float32)
        return states

    def _perform_meta_analysis_on_score(self, relationship_score: pd.Series, config: Dict, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        signal_name=config.get('name')
        arr=relationship_score.to_numpy(dtype=np.float32)
        disp=np.zeros_like(arr)
        disp[self.meta_window:]=arr[self.meta_window:]-arr[:-self.meta_window]
        mom=np.zeros_like(disp)
        mom[1:]=disp[1:]-disp[:-1]
        bipolar_displacement=np.tanh(self._apply_hab(df,'disp',disp,self.meta_window*2))
        bipolar_momentum=np.tanh(self._apply_hab(df,'mom',mom,13))
        instant_score_normalized=(arr+1.0)/2.0
        weight_momentum=np.clip(1.0-instant_score_normalized,0.0,1.0)
        weight_displacement=1.0-weight_momentum
        meta_score=(bipolar_displacement*weight_displacement+bipolar_momentum*weight_momentum)
        meta_score=np.sign(meta_score)*(np.abs(meta_score)**1.2)
        if config.get('diagnosis_mode','meta_analysis')=='gated_meta_analysis':
            gate_config=config.get('gate_condition',{})
            if gate_config.get('type')=='price_vs_ma':
                ma_period=gate_config.get('ma_period',5)
                ma_col=f'EMA_{ma_period}_D'
                if ma_col in df.columns and 'close_D' in df.columns:
                    gate_is_open=(df['close_D'].to_numpy(dtype=np.float32)<df[ma_col].to_numpy(dtype=np.float32)).astype(np.float32)
                    meta_score=meta_score*gate_is_open
        scoring_mode=self.score_type_map.get(signal_name,{}).get('scoring_mode','unipolar')
        if scoring_mode=='unipolar': meta_score=np.clip(meta_score,0.0,None)
        return pd.Series(np.clip(meta_score,-1.0,1.0),index=df_index,dtype=np.float32)

    def _get_safe_array(self, df: pd.DataFrame, col_name: str, method_name: str = "") -> np.ndarray:
        """
        【V327.0.0 · 因果律防泄露版】
        修改思路：阻断 bfill() 导致的未来函数数据泄露(Lookahead Bias)。仅保留 ffill() 维持时序连贯性，对期初缺失值采用 fillna(0.0) 安全填充，彻底斩断时空穿越。
        """
        if col_name not in df.columns:
            raise ValueError(f"[{method_name}] 致命数据断层: 缺失必需特征列 '{col_name}'。")
        return np.nan_to_num(df[col_name].ffill().fillna(0.0).to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _get_atomic_array(self, df: pd.DataFrame, score_name: str, default_value: float = 0.0) -> np.ndarray:
        """
        【V327.0.0 · 物理平滑缝合版】
        修改思路：强制加入 ffill().fillna(default_value)，消除原子信号期初从 NaN 跃迁至有效值时产生的无穷大虚假物理加速度（Jerk），维持三阶动力学张量的绝对平滑。
        """
        score_series=self.strategy.atomic_states.get(score_name)
        if score_series is None or score_series.empty:
            return np.full(len(df),default_value,dtype=np.float32)
        return np.nan_to_num(score_series.ffill().fillna(default_value).to_numpy(dtype=np.float32),nan=default_value,posinf=default_value,neginf=default_value)

    def _safe_div(self, a: np.ndarray, b: np.ndarray, fill_val: float = 0.0) -> np.ndarray:
        """
        【V326.0.0 · 微观奇点免疫版】极速矩阵安全除法
        修改要点：防范底层清洗遗漏的 1.42e-16 级浮点幽灵。原逻辑仅拦截绝对 0.0，现采用 np.abs(b)<1e-6 强力拦截微观奇点，防止超大数值撕裂 Tanh 梯度拓扑。
        """
        mask = (np.abs(b) < 1e-6) | np.isnan(b)
        b_safe = np.where(mask, 1.0, b)
        return np.where(mask, fill_val, a / b_safe).astype(np.float32)

    def _apply_zg(self, arr: np.ndarray) -> np.ndarray:
        """【V300.0.0】绝对零基防御门限 Numpy版"""
        return np.where(np.abs(np.nan_to_num(arr, nan=0.0)) < 1e-4, 0.0, 1.0).astype(np.float32)

    def _apply_norm(self, arr: np.ndarray, max_v: float = 100.0) -> np.ndarray:
        """【V300.0.0】线性边界防爆截断 Numpy版"""
        return np.clip(np.nan_to_num(arr, nan=0.0) / max_v, 0.0, 1.0).astype(np.float32)

    def _apply_norm_adaptive(self, arr: np.ndarray) -> np.ndarray:
        """
        【V332.0.0 · 物理量纲绝对硬化版】
        用途：自适应标度遗留桩点。
        修改要点：再次重申，彻底拔除导致特征时空断层与未来函数泄露的动态百分比探测。所有上游核心逻辑现已全部使用显式的 `_apply_norm(..., scale)`。此处仅保留基础的 [0,1] 安全截断防爆垫。
        """
        sf=np.nan_to_num(arr,nan=0.0,posinf=0.0,neginf=0.0)
        return np.clip(sf, 0.0, 1.0).astype(np.float32)

    def _apply_hab(self, df: pd.DataFrame, col_name: str, arr: np.ndarray, w: int) -> np.ndarray:
        """【V300.0.0】Numba 加速多维HAB历史缓冲池"""
        hab_col = f'HAB_{w}_{col_name}'
        if hab_col in df.columns:
            return np.nan_to_num(df[hab_col].to_numpy(dtype=np.float32), nan=0.0)
        sf = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        rm = _jit_rolling_mean(sf, w)
        rs = np.sqrt(_jit_rolling_std(sf, w)**2 + 1e-5)
        return self._safe_div(sf - rm, rs, 0.0)

    def _apply_kinematics(self, df: pd.DataFrame, col_name: str, arr: np.ndarray, w: int) -> np.ndarray:
        """
        【V326.0.0 · 因果律锚定版】三阶动力学张量引擎
        修改要点：修复高阶差分计算时越界减去 0.0 虚无区间的因果律穿越BUG。严格要求二阶加速度 ac 切片推移至 2w，三阶 jk 切片推移至 3w，彻底将动力学网络底噪清零。
        """
        sf = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(sf)
        sl_series = df.get(f'SLOPE_{w}_{col_name}')
        if sl_series is not None:
            sl = np.nan_to_num(sl_series.to_numpy(dtype=np.float32), nan=0.0)
        else:
            sl = np.zeros(n, dtype=np.float32)
            if n > w: sl[w:] = (sf[w:] - sf[:-w]) / w
        ac_series = df.get(f'ACCEL_{w}_{col_name}')
        if ac_series is not None:
            ac = np.nan_to_num(ac_series.to_numpy(dtype=np.float32), nan=0.0)
        else:
            ac = np.zeros(n, dtype=np.float32)
            if n > 2*w: ac[2*w:] = (sl[2*w:] - sl[w:-w]) / w
        jk_series = df.get(f'JERK_{w}_{col_name}')
        if jk_series is not None:
            jk = np.nan_to_num(jk_series.to_numpy(dtype=np.float32), nan=0.0)
        else:
            jk = np.zeros(n, dtype=np.float32)
            if n > 3*w: jk[3*w:] = (ac[3*w:] - ac[2*w:-w]) / w
        ts = np.where(np.abs(sl) < 1e-4, 0.0, sl) + np.where(np.abs(ac) < 1e-4, 0.0, ac) * 0.5 + np.where(np.abs(jk) < 1e-4, 0.0, jk) * 0.25
        return np.nan_to_num(np.clip(np.tanh(np.clip(ts, -20.0, 20.0) * 10.0), -1.0, 1.0), nan=0.0).astype(np.float32)

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        method_name="_diagnose_signal_decay"
        is_debug_enabled=get_param_value(self.debug_params.get('enabled'),False) and get_param_value(self.debug_params.get('should_probe'),False)
        signal_name=config.get('name')
        if signal_name=='PROCESS_META_WINNER_CONVICTION_DECAY':
            decay_score=self.calculate_winner_conviction_decay_processor.calculate(df,config)
            return {signal_name:decay_score.astype(np.float32)}
        source_signal_name=config.get('source_signal')
        if not source_signal_name: return {}
        df_index=df.index
        if config.get('source_type','df')=='atomic_states': 
            source_series=self.strategy.atomic_states.get(source_signal_name)
        else:
            if source_signal_name not in df.columns: return {}
            source_series=df[source_signal_name].astype(np.float32)
        if source_series is None: return {}
        arr=np.nan_to_num(source_series.to_numpy(dtype=np.float32),nan=0.0)
        diff_arr=np.zeros_like(arr)
        diff_arr[1:]=arr[1:]-arr[:-1]
        decay_magnitude=np.abs(np.clip(diff_arr,a_min=None,a_max=0.0))
        local_std=np.sqrt(_jit_rolling_std(arr,21)**2+1e-5)
        relative_decay=self._safe_div(decay_magnitude,local_std,0.0)
        decay_score=np.clip(np.tanh(relative_decay*1.5),0.0,1.0)
        if is_debug_enabled and self.probe_dates:
            self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'source':arr},calc_nodes={'signal_change':diff_arr,'relative_decay':relative_decay},final_result=decay_score)
        return {signal_name:pd.Series(decay_score,index=df_index,dtype=np.float32)}

    def _diagnose_domain_reversal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        domain_name = config.get('domain_name')
        axiom_configs = config.get('axioms', [])
        output_bottom_name = config.get('output_bottom_reversal_name')
        output_top_name = config.get('output_top_reversal_name')
        if not domain_name or not axiom_configs or not output_bottom_name or not output_top_name: return {}
        df_index = df.index
        n = len(df)
        bipolar_domain_health = np.zeros(n, dtype=np.float32)
        total_weight = 0.0
        for axiom_config in axiom_configs:
            axiom_name = axiom_config.get('name')
            axiom_weight = axiom_config.get('weight', 0.0)
            if axiom_name not in self.strategy.atomic_states: continue
            axiom_score = self.strategy.atomic_states.get(axiom_name)
            if axiom_score is not None:
                arr = np.nan_to_num(axiom_score.to_numpy(dtype=np.float32), nan=0.0)
                bipolar_domain_health += arr * axiom_weight
                total_weight += abs(axiom_weight)
        if total_weight == 0: return {}
        bipolar_domain_health = np.clip(bipolar_domain_health / total_weight, -1.0, 1.0)
        return self._judge_domain_reversal(bipolar_domain_health, config, df)

    def _calculate_power_transfer(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V330.0.0 · 权力转移防微观死锁版】
        用途：诊断微观主力和大单的权力交接。
        修改要点：移除 absolute_core 对 net_mf 的单一 0 值死锁依赖。实盘中，极有可能出现全市场净资金为 0，但其中超大单疯狂抢筹、中小单对等抛售的“完美换庄”场景。现改为对 net_mf 和 tick_large_net 的绝对值之和进行联合零基验证，释放隐藏的换庄信号。
        """
        method_name="_calculate_power_transfer"
        required_signals=['net_mf_amount_D','amount_D','tick_large_order_net_D','tick_chip_transfer_efficiency_D','flow_efficiency_D','intraday_cost_center_migration_D','downtrend_strength_D','chip_concentration_ratio_D','volatility_adjusted_concentration_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        net_mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        tick_large_net=self._get_safe_array(df,'tick_large_order_net_D',method_name=method_name)
        transfer_eff=self._get_safe_array(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        flow_eff=self._get_safe_array(df,'flow_efficiency_D',method_name=method_name)
        cost_migration=self._get_safe_array(df,'intraday_cost_center_migration_D',method_name=method_name)
        downtrend=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        chip_conc=self._get_safe_array(df,'chip_concentration_ratio_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        turnover=self._get_safe_array(df,'turnover_rate_f_D',method_name=method_name)
        amt_safe=np.where(amt==0.0,1.0,amt)
        mf_ratio=self._safe_div(net_mf,amt_safe,0.0)
        tick_ratio=self._safe_div(tick_large_net,amt_safe,0.0)
        mf_core=np.tanh(mf_ratio*50.0)
        tick_core=np.tanh(tick_ratio*50.0)
        # [V330.0.0] 联合零基防御，释放隐性主力的微观换庄信号
        absolute_core=(mf_core*0.6+tick_core*0.4)*self._apply_zg(np.abs(net_mf)+np.abs(tick_large_net))
        hab_amp=1.0+(np.abs(np.tanh(self._apply_hab(df,'mf_r',mf_ratio,21)))+np.abs(np.tanh(self._apply_hab(df,'t_r',tick_ratio,21))))/2.0
        core=absolute_core*hab_amp
        amp=1.0+(np.abs(np.tanh(self._apply_hab(df,'mig',cost_migration,13)))+(1.0-np.tanh(turnover/10.0))+self._apply_norm(vac,100.0)+self._apply_norm(transfer_eff,1e6)+np.abs(np.tanh(flow_eff/5.0)))/5.0
        c_diff=np.zeros_like(chip_conc)
        c_diff[1:]=chip_conc[1:]-chip_conc[:-1]
        c_pen=np.tanh(self._apply_hab(df,'c_diff',c_diff,13))*amp
        synergy=np.maximum(1.0+c_pen*np.sign(core),0.1)
        raw=core*synergy*(1.0+np.abs(self._apply_kinematics(df,'mf_ratio_scaled',mf_ratio*100.0,13)))
        downtrend_penalty=np.clip((downtrend-50.0)/30.0,0.0,1.0)
        raw_is_positive=np.clip(raw*10.0,0.0,1.0)
        td=1.0-0.4*downtrend_penalty*raw_is_positive
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5))*td,-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':net_mf},calc_nodes={'absolute_core':absolute_core,'hab_amp':hab_amp,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_price_vs_capitulation_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V333.0.0 · 物理背离绝对正交版】
        用途：诊断价格与散户投降套牢盘的微观背离。
        修改要点：拨乱反正！修复了原代码的逻辑错位。价格上涨且恐慌下降是正常的健康趋势，不应触发信号。
        现严格遵循背离的第一性原理：
        1. 价格下跌(p_kin<0) 但 恐慌盘衰竭(t_kin<0) -> 散户绝望割肉，抛压真空，生成极致机会(opp_core)。
        2. 价格上涨(p_kin>0) 但 恐慌盘剧增(t_kin>0) -> 主力高位派发，散户追高被套，生成致命风险(risk_core)。
        同向矢量物理对冲，彻底激活这套双极核武。
        """
        method_name="_calculate_price_vs_capitulation_relationship"
        required_signals=['pressure_trapped_D','INTRADAY_SUPPORT_INTENT_D','intraday_low_lock_ratio_D','chip_entropy_D','volatility_adjusted_concentration_D','turnover_rate_f_D','close_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        pressure=self._get_safe_array(df,'pressure_trapped_D',method_name=method_name)
        support=self._get_safe_array(df,'INTRADAY_SUPPORT_INTENT_D',method_name=method_name)
        low_lock=self._get_safe_array(df,'intraday_low_lock_ratio_D',method_name=method_name)
        entropy=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        turnover=self._get_safe_array(df,'turnover_rate_f_D',method_name=method_name)
        cls=self._get_safe_array(df,'close_D',method_name=method_name)
        
        cls_safe=np.where(cls==0.0,1.0,cls)
        cls_rm=_jit_rolling_mean(cls_safe,21)
        p_kin=self._apply_kinematics(df,'close_D_scaled',self._safe_div(cls,cls_rm,1.0)*10.0,13)
        t_kin=self._apply_kinematics(df,'pressure_trapped_D_scaled',pressure*100.0,13)
        
        # [V333.0.0] 极性第一性原理重构：背离识别
        # Opportunity (机会，取正分)：价格下跌(p_kin < 0) 且 恐慌盘衰竭(t_kin < 0)
        opp_core=np.tanh(-np.clip(p_kin,a_min=None,a_max=0.0)) * np.tanh(-np.clip(t_kin,a_min=None,a_max=0.0))
        # Risk (风险，取负分基础)：价格上涨(p_kin > 0) 且 恐慌盘剧增(t_kin > 0)
        risk_core=np.tanh(np.clip(p_kin,a_min=0.0,a_max=None)) * np.tanh(np.clip(t_kin,a_min=0.0,a_max=None))
        
        state_magnitude=np.tanh(pressure*100.0)
        # 机会取正，风险取负，完美对齐 Bipolar 标准输出极性
        core=(opp_core-risk_core)*state_magnitude*self._apply_zg(pressure)
        
        amp=1.0+(np.maximum(np.tanh(self._apply_hab(df,'sup',support,34)),0.0)+self._apply_norm(low_lock,1.0)+(1.0-self._apply_norm(entropy,10.0))+self._apply_norm(vac,100.0)+(1.0-np.tanh(turnover/10.0)))/5.0
        raw=core*amp*(1.0+np.abs(t_kin))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':pressure,'close_D':cls},calc_nodes={'p_kin':p_kin,'t_kin':t_kin,'state_magnitude':state_magnitude,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_price_efficiency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name="_calculate_price_efficiency_relationship"
        required_signals=['VPA_EFFICIENCY_D','net_mf_amount_D','shakeout_score_D','tick_chip_transfer_efficiency_D','high_freq_flow_skewness_D','volatility_adjusted_concentration_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        eff=self._get_safe_array(df,'VPA_EFFICIENCY_D',method_name=method_name)
        net_mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        shakeout=self._get_safe_array(df,'shakeout_score_D',method_name=method_name)
        transfer_eff=self._get_safe_array(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        flow_skew=self._get_safe_array(df,'high_freq_flow_skewness_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        turnover=self._get_safe_array(df,'turnover_rate_f_D',method_name=method_name)
        core=np.tanh(eff*5.0)*self._apply_zg(eff)
        amp=1.0+(np.maximum(np.tanh(self._apply_hab(df,'mf',net_mf,55)),0.0)+(1.0-self._apply_norm(shakeout,100.0))+self._apply_norm(transfer_eff,1e6)+np.maximum(np.tanh(self._apply_hab(df,'skew',flow_skew,21)),0.0)+self._apply_norm(vac,100.0)+(1.0-np.tanh(turnover/10.0)))/6.0
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'VPA_EFFICIENCY_D_scaled',eff,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'VPA_EFFICIENCY_D':eff},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_pd_divergence_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V332.0.0 · 博弈极性校准版】
        用途：博弈背离确认引擎。
        修改要点：坚守 Bipolar 规则（正为机会，负为风险），底层 _calculate_instantaneous_relationship 的顶背离本就是负值，底背离本就是正值。绝对禁止人为添加负号反转导致主脑判决颠倒。
        """
        method_name="_calculate_pd_divergence_relationship"
        required_signals=['game_intensity_D','weight_avg_cost_D','close_D','intraday_chip_game_index_D','chip_divergence_ratio_D','winner_rate_D','high_freq_flow_kurtosis_D','chip_entropy_D','turnover_rate_f_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        game=self._get_safe_array(df,'game_intensity_D',method_name=method_name)
        cost=self._get_safe_array(df,'weight_avg_cost_D',method_name=method_name)
        close_p=self._get_safe_array(df,'close_D',method_name=method_name)
        intra_game=self._get_safe_array(df,'intraday_chip_game_index_D',method_name=method_name)
        chip_div=self._get_safe_array(df,'chip_divergence_ratio_D',method_name=method_name)
        winner=self._get_safe_array(df,'winner_rate_D',method_name=method_name)
        hf_kurt=self._get_safe_array(df,'high_freq_flow_kurtosis_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        turnover=self._get_safe_array(df,'turnover_rate_f_D',method_name=method_name)
        # [V332.0.0] 恢复原汁原味的 Bipolar 矢量输出，不进行翻转
        base_div=self._calculate_instantaneous_relationship(df,config).to_numpy(dtype=np.float32)
        core=base_div*self._apply_norm(game,1.0)*self._apply_zg(game)
        price_adv=np.tanh(self._safe_div(close_p-cost,cost,0.0)*10.0)
        # Bipolar放大器：顶背离(base_div<0)且处于高位(price_adv>0)时放大；底背离(base_div>0)且处于低位(price_adv<0)时放大
        p_lev=np.maximum(1.0-price_adv*np.sign(base_div),0.1)
        amp=1.0+(self._apply_norm(winner,100.0)+self._apply_norm(intra_game,100.0)+self._apply_norm(chip_div,100.0)+np.maximum(np.tanh(self._apply_hab(df,'kurt',hf_kurt,21)),0.0)+(1.0-self._apply_norm(c_ent,10.0))+self._apply_norm(turnover,10.0))/6.0
        raw=core*p_lev*amp*(1.0+np.abs(self._apply_kinematics(df,'game_intensity_D_scaled',game,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'game_intensity_D':game},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V327.0.0 · 量纲降维对齐版】
        修改思路：修复 chip_stability_D [0,100] 与历史门控阈值 (0.2) 之间的量纲错配。引入 _apply_norm_adaptive 强制降维至 [0,1] 区间，杜绝在主跌溃散期由于量纲失效而误判为洗盘的致命漏洞。
        """
        method_name="_calculate_panic_washout_accumulation"
        required_signals=['pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D','intraday_trough_filling_degree_D','high_freq_flow_divergence_D','chip_rsi_divergence_D','chip_stability_D','pressure_release_index_D','tick_abnormal_volume_ratio_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        panic=self._get_safe_array(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_array(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorption=self._get_safe_array(df,'absorption_energy_D',method_name=method_name)
        trough_fill=self._get_safe_array(df,'intraday_trough_filling_degree_D',method_name=method_name)
        hff_div=self._get_safe_array(df,'high_freq_flow_divergence_D',method_name=method_name)
        chip_div=self._get_safe_array(df,'chip_rsi_divergence_D',method_name=method_name)
        chip_stab=self._get_safe_array(df,'chip_stability_D',method_name=method_name)
        release=self._get_safe_array(df,'pressure_release_index_D',method_name=method_name)
        abnorm_vol=self._get_safe_array(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        p_core=np.tanh(panic*100.0)*self._apply_zg(panic)
        a_core=np.tanh(absorption*100.0)*self._apply_zg(absorption)
        core=p_core*a_core
        amp=1.0+(np.tanh(release*100.0)+self._apply_norm(abnorm_vol,10.0)+self._apply_norm(low_lock,1.0)+self._apply_norm(trough_fill,100.0)+np.maximum(np.tanh(self._apply_hab(df,'hff',hff_div,21)),0.0)+np.maximum(np.tanh(self._apply_hab(df,'cdiv',chip_div,21)),0.0)+self._apply_norm(vac,100.0))/7.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'pressure_trapped_D_scaled',panic*100.0,13),0.0))
        res_raw=np.clip(np.tanh(raw**1.5),0.0,1.0)
        chip_stab_norm=self._apply_norm_adaptive(chip_stab)
        res=np.where(chip_stab_norm>config.get('historical_potential_gate',0.2),res_raw,0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':panic,'absorption_energy_D':absorption},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V331.0.0 · 物理量纲硬化版】"""
        method_name="_calculate_deceptive_accumulation"
        required_signals=['stealth_flow_ratio_D','tick_clustering_index_D','intraday_price_distribution_skewness_D','high_freq_flow_skewness_D','price_flow_divergence_D','chip_flow_intensity_D','intraday_chip_turnover_intensity_D','tick_chip_transfer_efficiency_D','intraday_accumulation_confidence_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        stealth=self._get_safe_array(df,'stealth_flow_ratio_D',method_name=method_name)
        cluster=self._get_safe_array(df,'tick_clustering_index_D',method_name=method_name)
        price_skew=self._get_safe_array(df,'intraday_price_distribution_skewness_D',method_name=method_name)
        flow_skew=self._get_safe_array(df,'high_freq_flow_skewness_D',method_name=method_name)
        pf_div=self._get_safe_array(df,'price_flow_divergence_D',method_name=method_name)
        flow_int=self._get_safe_array(df,'chip_flow_intensity_D',method_name=method_name)
        turnover_int=self._get_safe_array(df,'intraday_chip_turnover_intensity_D',method_name=method_name)
        trans_eff=self._get_safe_array(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        acc_conf=self._get_safe_array(df,'intraday_accumulation_confidence_D',method_name=method_name)
        vpa=self._get_safe_array(df,'VPA_EFFICIENCY_D',method_name=method_name)
        # [V331.0.0] 显式物理量纲约束
        acc_norm=self._apply_norm(acc_conf, 1.0)
        stealth_norm=np.tanh(stealth*100.0)
        core=(stealth_norm*0.7 + acc_norm*0.3)*self._apply_zg(stealth)
        amp=1.0+(self._apply_norm(cluster,1.0)+self._apply_norm(flow_int,100.0)+self._apply_norm(trans_eff,1e6)+np.maximum(np.tanh(self._apply_hab(df,'pf_div',pf_div,21)),0.0)+np.maximum(np.tanh(self._apply_hab(df,'smis',flow_skew-price_skew,21)),0.0)+self._apply_norm(turnover_int,100.0)+(1.0-np.abs(np.tanh(vpa*5.0))))/7.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'stealth_flow_ratio_D_scaled',stealth*100.0,13),0.0))
        res=np.clip(np.tanh(raw**1.5),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'stealth_flow_ratio_D':stealth,'intraday_accumulation_confidence_D':acc_conf},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V328.0.0 · 绝地反击阈值软化版】
        用途：识别多日累积吸筹后，即将由量变到质变的拉升拐点。
        修改要点：废除对 downtrend_strength_D 的刚性清零惩罚。底部吸筹拐点本身就孕育在暴跌后的深水区，原逻辑 down_penalty=0 会将深水区的真实信号无情抹杀。现将惩罚公式升级为连续线性衰减，并赋予 0.1 的底座生存权重。
        """
        method_name="_calculate_accumulation_inflection"
        required_signals=['PROCESS_META_COVERT_ACCUMULATION','PROCESS_META_DECEPTIVE_ACCUMULATION','PROCESS_META_PANIC_WASHOUT_ACCUMULATION','PROCESS_META_MAIN_FORCE_RALLY_INTENT','chip_convergence_ratio_D','price_vs_ma_21_ratio_D','flow_acceleration_intraday_D','flow_consistency_D','MA_POTENTIAL_COMPRESSION_RATE_D','MACDh_13_34_8_D','consolidation_quality_score_D','volatility_adjusted_concentration_D','downtrend_strength_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        covert=np.maximum(self._get_atomic_array(df,'PROCESS_META_COVERT_ACCUMULATION',0.0),0.0)
        decept=np.maximum(self._get_atomic_array(df,'PROCESS_META_DECEPTIVE_ACCUMULATION',0.0),0.0)
        panic=np.maximum(self._get_atomic_array(df,'PROCESS_META_PANIC_WASHOUT_ACCUMULATION',0.0),0.0)
        rally=np.maximum(self._get_atomic_array(df,'PROCESS_META_MAIN_FORCE_RALLY_INTENT',0.0),0.0)
        c_conv=self._get_safe_array(df,'chip_convergence_ratio_D',method_name=method_name)
        p_ma21=self._get_safe_array(df,'price_vs_ma_21_ratio_D',method_name=method_name)
        p_ma21=np.where(p_ma21==0.0,1.0,p_ma21)
        f_accel=self._get_safe_array(df,'flow_acceleration_intraday_D',method_name=method_name)
        f_cons=self._get_safe_array(df,'flow_consistency_D',method_name=method_name)
        ma_comp=self._get_safe_array(df,'MA_POTENTIAL_COMPRESSION_RATE_D',method_name=method_name)
        macd=self._get_safe_array(df,'MACDh_13_34_8_D',method_name=method_name)
        consol=self._get_safe_array(df,'consolidation_quality_score_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        pot=_jit_ema((covert+decept+panic)/3.0,config.get('accumulation_window',21))
        down_penalty=np.clip(1.0-self._apply_norm(down,100.0),0.1,1.0)
        inflection_gate=np.where((f_accel>0.0)|(macd>0.0)|(c_conv>0.5),1.0,0.1).astype(np.float32)
        core=pot*self._apply_zg(pot)*inflection_gate
        amp=1.0+(self._apply_norm(consol,100.0)+self._apply_norm(c_conv,1.0)+(1.0-np.maximum(np.tanh(self._apply_hab(df,'pma',np.abs(p_ma21-1.0),21)),0.0)*0.5)+self._apply_norm(ma_comp,1.0)+self._apply_norm(vac,100.0)+np.maximum(np.tanh(self._apply_hab(df,'fa',f_accel,13)),0.0)+self._apply_norm(f_cons,100.0)+rally+np.maximum(np.tanh(self._apply_hab(df,'md',macd,13)),0.0))/9.0
        raw=core*amp*down_penalty*(1.0+np.maximum(self._apply_kinematics(df,'chip_convergence_ratio_D_scaled',c_conv,13),0.0))
        res_raw=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        res=np.where(raw>=0.1,res_raw,0.0).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'consolidation_quality_score_D':consol,'downtrend_strength_D':down},calc_nodes={'core':core,'amp':amp,'down_penalty':down_penalty,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_loser_capitulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name="_calculate_loser_capitulation"
        required_signals=['pressure_release_index_D','pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D','winner_rate_D','downtrend_strength_D','price_to_weight_avg_ratio_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        release=self._get_safe_array(df,'pressure_release_index_D',method_name=method_name)
        trapped=self._get_safe_array(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_array(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorp=self._get_safe_array(df,'absorption_energy_D',method_name=method_name)
        winner=self._get_safe_array(df,'winner_rate_D',method_name=method_name)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        price_to_cost=self._get_safe_array(df,'price_to_weight_avg_ratio_D',method_name=method_name)
        price_to_cost=np.where(price_to_cost==0.0,1.0,price_to_cost)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        core=np.tanh(release*100.0)*np.tanh(trapped*100.0)*self._apply_zg(release)*self._apply_zg(trapped)
        amp=1.0+(self._apply_norm(low_lock,1.0)+np.tanh(absorp*10.0)+self._apply_norm(np.maximum(1.0-price_to_cost,0.0),1.0)+self._apply_norm(100.0-winner,100.0)+self._apply_norm(down,100.0)+(1.0-self._apply_norm(sent,100.0)))/6.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'pressure_release_index_D_scaled',release*100.0,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_release_index_D':release,'pressure_trapped_D':trapped},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V331.0.0 · 物理量纲硬化版】"""
        method_name="_calculate_breakout_acceleration"
        required_signals=['breakout_quality_score_D','industry_strength_rank_D','net_mf_amount_D','flow_consistency_D','tick_abnormal_volume_ratio_D','uptrend_strength_D','T1_PREMIUM_EXPECTATION_D','HM_COORDINATED_ATTACK_D','breakout_penalty_score_D','buy_elg_amount_D','volatility_adjusted_concentration_D','amount_D','downtrend_strength_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        brk=self._get_safe_array(df,'breakout_quality_score_D',method_name=method_name)
        ind=self._get_safe_array(df,'industry_strength_rank_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        cons=self._get_safe_array(df,'flow_consistency_D',method_name=method_name)
        abnorm=self._get_safe_array(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        uptrend=self._get_safe_array(df,'uptrend_strength_D',method_name=method_name)
        t1=self._get_safe_array(df,'T1_PREMIUM_EXPECTATION_D',method_name=method_name)
        hm=self._get_safe_array(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        pen=self._get_safe_array(df,'breakout_penalty_score_D',method_name=method_name)
        elg=self._get_safe_array(df,'buy_elg_amount_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        down_penalty=np.clip(1.0-self._apply_norm(down,100.0),0.1,1.0)
        core=self._apply_norm(brk,100.0)*self._apply_zg(brk)
        # [V331.0.0] 显式物理量纲约束
        ind_norm=self._apply_norm(ind, 1.0)
        amp=1.0+(self._apply_norm(uptrend,100.0)+ind_norm+self._apply_norm(vac,100.0)+np.maximum(np.tanh(self._apply_hab(df,'mf',mf,21)),0.0)+self._apply_norm(t1,100.0)+self._apply_norm(hm,100.0)+self._apply_norm(self._safe_div(elg,amt,0.0),0.1)+self._apply_norm(cons,100.0)+self._apply_norm(abnorm,10.0))/9.0
        raw=core*(1.0-self._apply_norm(pen,100.0))*amp*down_penalty*(1.0+np.maximum(self._apply_kinematics(df,'breakout_quality_score_D_scaled',brk/100.0,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'breakout_quality_score_D':brk,'uptrend_strength_D':uptrend},calc_nodes={'core':core,'amp':amp,'down_penalty':down_penalty,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V331.0.0 · 物理量纲硬化版】"""
        method_name="_calculate_fund_flow_accumulation_inflection"
        required_signals=['accumulation_signal_score_D','net_mf_amount_D','flow_efficiency_D','tick_large_order_net_D','intraday_accumulation_confidence_D','GAP_MOMENTUM_STRENGTH_D','STATE_GOLDEN_PIT_D','buy_lg_amount_D','amount_D','flow_persistence_minutes_D','net_energy_flow_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        acc=self._get_safe_array(df,'accumulation_signal_score_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        eff=self._get_safe_array(df,'flow_efficiency_D',method_name=method_name)
        l_net=self._get_safe_array(df,'tick_large_order_net_D',method_name=method_name)
        intra=self._get_safe_array(df,'intraday_accumulation_confidence_D',method_name=method_name)
        gap=self._get_safe_array(df,'GAP_MOMENTUM_STRENGTH_D',method_name=method_name)
        pit=self._get_safe_array(df,'STATE_GOLDEN_PIT_D',method_name=method_name)
        pit=np.clip(pit,0.0,1.0)
        buy_lg=self._get_safe_array(df,'buy_lg_amount_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        pers=self._get_safe_array(df,'flow_persistence_minutes_D',method_name=method_name)
        energy=self._get_safe_array(df,'net_energy_flow_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        eff_norm=np.abs(np.tanh(eff/5.0))
        # [V331.0.0] 显式物理量纲约束
        intra_norm=self._apply_norm(intra, 1.0)
        acc_base=np.maximum(self._apply_norm(acc,100.0), intra_norm*0.8)
        core=acc_base*np.maximum(eff_norm,0.1)*self._apply_zg(acc_base)
        amp=1.0+(eff_norm+intra_norm+self._apply_norm(self._safe_div(buy_lg,amt,0.0),0.1)+self._apply_norm(pers,100.0)+np.maximum(np.tanh(self._apply_hab(df,'l_net',l_net,21)),0.0)+np.maximum(np.tanh(self._apply_hab(df,'gap',gap,13)),0.0)+np.maximum(np.tanh(self._apply_hab(df,'eng',energy,21)),0.0)+np.maximum(np.tanh(self._apply_hab(df,'mf',mf,21)),0.0)+pit*0.5+(1.0-self._apply_norm(c_ent,10.0)))/10.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'accumulation_signal_score_D_scaled',acc/100.0,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'accumulation_signal_score_D':acc,'flow_efficiency_D':eff,'intraday_accumulation_confidence_D':intra},calc_nodes={'acc_base':acc_base,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V327.0.0 · 物理并联防死锁版】
        修改思路：打破原有的零值连乘死锁。将微观日内派发置信度与宏观派发能量并联，同时赋予 press_core 基础的 0.5 底座权重，确保宏观获利盘巨大死压不会因为某单一微观特征缺失而惨遭 0 值抹杀。
        """
        method_name="_calculate_profit_vs_flow_relationship"
        required_signals=['profit_pressure_D','net_mf_amount_D','profit_ratio_D','flow_consistency_D','winner_rate_D','intraday_distribution_confidence_D','STATE_PARABOLIC_WARNING_D','distribution_energy_D','sell_elg_amount_D','amount_D','pressure_profit_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        press=self._get_safe_array(df,'profit_pressure_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        p_ratio=self._get_safe_array(df,'profit_ratio_D',method_name=method_name)
        cons=self._get_safe_array(df,'flow_consistency_D',method_name=method_name)
        win=self._get_safe_array(df,'winner_rate_D',method_name=method_name)
        dist_conf=self._get_safe_array(df,'intraday_distribution_confidence_D',method_name=method_name)
        para=self._get_safe_array(df,'STATE_PARABOLIC_WARNING_D',method_name=method_name)
        para=np.clip(para,0.0,1.0)
        dist_eng=self._get_safe_array(df,'distribution_energy_D',method_name=method_name)
        sell_elg=self._get_safe_array(df,'sell_elg_amount_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        marg=self._get_safe_array(df,'pressure_profit_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        dist_force=np.maximum(self._apply_norm(dist_conf,100.0),self._apply_norm(dist_eng,100.0))
        press_core=self._apply_norm(press,100.0)*(0.5+0.5*dist_force)*self._apply_zg(press)
        press_amp=1.0+(self._apply_norm(dist_eng,100.0)+self._apply_norm(p_ratio,100.0)+self._apply_norm(self._safe_div(sell_elg,amt,0.0),0.1)+self._apply_norm(marg,100.0)+para+self._apply_norm(sent,100.0))/7.0
        press_total=press_core*press_amp*(1.0+np.maximum(self._apply_kinematics(df,'profit_pressure_D_scaled',press/100.0,13),0.0))
        mf_ratio=self._safe_div(mf,amt,0.0)
        supp_core=np.maximum(np.tanh(mf_ratio*20.0),0.0)*self._apply_norm(cons,100.0)*self._apply_zg(np.maximum(mf,0.0))*self._apply_zg(cons)
        supp_amp=1.0+(1.0-self._apply_norm(win,100.0))
        supp_total=supp_core*supp_amp
        raw=supp_total-press_total*1.5
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'profit_pressure_D':press,'net_mf_amount_D':mf},calc_nodes={'press_total':press_total,'supp_total':supp_total,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V331.0.0 · 物理量纲硬化版】"""
        method_name="_calculate_stock_sector_sync"
        required_signals=['pct_change_D','industry_strength_rank_D','net_mf_amount_D','flow_consistency_D','industry_leader_score_D','mid_long_sync_D','STATE_MARKET_LEADER_D','industry_markup_score_D','industry_rank_accel_D','HM_COORDINATED_ATTACK_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        pct=self._get_safe_array(df,'pct_change_D',method_name=method_name)
        rank=self._get_safe_array(df,'industry_strength_rank_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        cons=self._get_safe_array(df,'flow_consistency_D',method_name=method_name)
        ldr=self._get_safe_array(df,'industry_leader_score_D',method_name=method_name)
        sync=self._get_safe_array(df,'mid_long_sync_D',method_name=method_name)
        m_ldr=self._get_safe_array(df,'STATE_MARKET_LEADER_D',method_name=method_name)
        m_ldr=np.clip(m_ldr,0.0,1.0)
        mkup=self._get_safe_array(df,'industry_markup_score_D',method_name=method_name)
        rk_acc=self._get_safe_array(df,'industry_rank_accel_D',method_name=method_name)
        hm=self._get_safe_array(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        # [V331.0.0] 显式物理量纲约束
        rank_norm=self._apply_norm(rank, 1.0)
        core=np.tanh(pct/10.0)*rank_norm*self._apply_zg(pct)*self._apply_zg(rank)
        amp=1.0+(self._apply_norm(mkup,100.0)+(1.0+np.maximum(np.tanh(rk_acc/10.0),0.0))*0.5+self._apply_norm(ldr,100.0)+np.maximum(np.tanh(self._apply_hab(df,'mf',mf,21)),0.0)+self._apply_norm(cons,100.0)+self._apply_norm(sync,100.0)+self._apply_norm(hm,100.0)+m_ldr+self._apply_norm(sent,100.0))/10.0
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'industry_strength_rank_D_scaled',rank_norm,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pct_change_D':pct,'industry_strength_rank_D':rank},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V306.0.0】热点板块退潮预警
        修改要点：拆除动能缩减时的 maximum 限制，令退热负斜率(热度加速冷却)直接转化为大于1的乘级放大倍数，精准击杀高位加速退潮的风险点。
        """
        method_name="_calculate_hot_sector_cooling"
        required_signals=['THEME_HOTNESS_SCORE_D','net_mf_amount_D','industry_stagnation_score_D','outflow_quality_D','industry_downtrend_score_D','distribution_energy_D','sell_elg_amount_D','amount_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hot=self._get_safe_array(df,'THEME_HOTNESS_SCORE_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        stag=self._get_safe_array(df,'industry_stagnation_score_D',method_name=method_name)
        outq=self._get_safe_array(df,'outflow_quality_D',method_name=method_name)
        down=self._get_safe_array(df,'industry_downtrend_score_D',method_name=method_name)
        dist=self._get_safe_array(df,'distribution_energy_D',method_name=method_name)
        sell_elg=self._get_safe_array(df,'sell_elg_amount_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        hot_state=self._apply_norm(hot,100.0)
        dump_energy=np.maximum.reduce([self._apply_norm(outq,100.0),self._apply_norm(dist,100.0),self._apply_norm(self._safe_div(sell_elg,amt,0.0),0.1),np.abs(np.clip(np.tanh(self._apply_hab(df,'mf',mf,13)),None,0.0))])
        core=hot_state*dump_energy*self._apply_zg(hot_state*dump_energy)
        amp=1.0+(self._apply_norm(stag,100.0)+self._apply_norm(down,100.0)+(1.0-self._apply_norm(sent,100.0)))/3.0
        hot_k=self._apply_kinematics(df,'THEME_HOTNESS_SCORE_D_scaled',hot/100.0,13)
        raw=core*amp*(1.0-np.minimum(hot_k,0.0))
        res=np.clip(np.tanh(raw**1.5),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'THEME_HOTNESS_SCORE_D':hot,'net_mf_amount_D':mf},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_ff_vs_structure_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V331.0.0 · 第一性原理背离解耦版】
        修改要点：彻底斩断由导数求交带来的 0 值死锁。采用第一性原理：绝对买力(inflow_force)强劲且相对结构滞后(struct_lag)本身就是最直接的底部量价背离起爆点。
        """
        method_name="_calculate_ff_vs_structure_relationship"
        required_signals=['uptrend_strength_D','flow_consistency_D','ma_arrangement_status_D','chip_structure_state_D','industry_stagnation_score_D','large_order_anomaly_D','STATE_ROBUST_TREND_D','net_mf_amount_D','chip_stability_D','flow_momentum_13d_D','volatility_adjusted_concentration_D','amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        up=self._get_safe_array(df,'uptrend_strength_D',method_name=method_name)
        cons=self._get_safe_array(df,'flow_consistency_D',method_name=method_name)
        ma_s=self._get_safe_array(df,'ma_arrangement_status_D',method_name=method_name)
        chip_s=self._get_safe_array(df,'chip_structure_state_D',method_name=method_name)
        stag=self._get_safe_array(df,'industry_stagnation_score_D',method_name=method_name)
        anom=self._get_safe_array(df,'large_order_anomaly_D',method_name=method_name)
        rob=self._get_safe_array(df,'STATE_ROBUST_TREND_D',method_name=method_name)
        rob=np.clip(rob,0.0,1.0)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        stab=self._get_safe_array(df,'chip_stability_D',method_name=method_name)
        mom=self._get_safe_array(df,'flow_momentum_13d_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        mf_ratio=self._safe_div(mf,amt,0.0)
        inflow_force=np.maximum(np.tanh(mf_ratio*50.0),0.0)
        struct_lag=1.0-self._apply_norm(up,100.0)
        core=inflow_force*struct_lag*self._apply_zg(inflow_force)
        amp=1.0+(self._apply_norm(cons,100.0)+np.maximum(np.tanh(self._apply_hab(df,'mom',mom,13)),0.0)+self._apply_norm(ma_s,1.0)+self._apply_norm(chip_s,1.0)+self._apply_norm(vac,100.0)+(1.0-self._apply_norm(stag,100.0)*0.5)+np.maximum(np.tanh(self._apply_hab(df,'anm',anom,13)),0.0)+rob+(1.0-self._apply_norm(stab,100.0)))/9.0
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'uptrend_strength_D_scaled',up/100.0,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'uptrend_strength_D':up,'net_mf_amount_D':mf},calc_nodes={'inflow_force':inflow_force,'struct_lag':struct_lag,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V331.0.0 · 价筹共振对称重构版】
        修改要点：修复熊市断层崩塌被硬性抹零的严重盲点！废除 1.0-pk_norm 畸形抑制，直接使 pk_norm 对多空双向进行等效对称放大，忠实反映高密度筹码破位引发的动能共振。
        """
        method_name="_calculate_pc_relationship"
        required_signals=['peak_concentration_D','close_D','chip_convergence_ratio_D','high_position_lock_ratio_90_D','chip_stability_change_5d_D','volatility_adjusted_concentration_D','chip_entropy_D','chip_flow_intensity_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        pk=self._get_safe_array(df,'peak_concentration_D',method_name=method_name)
        cls=self._get_safe_array(df,'close_D',method_name=method_name)
        cvg=self._get_safe_array(df,'chip_convergence_ratio_D',method_name=method_name)
        hl=self._get_safe_array(df,'high_position_lock_ratio_90_D',method_name=method_name)
        sc=self._get_safe_array(df,'chip_stability_change_5d_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        fi=self._get_safe_array(df,'chip_flow_intensity_D',method_name=method_name)
        c_diff=np.zeros_like(cls)
        c_diff[1:]=cls[1:]-cls[:-1]
        price_force=np.tanh(self._safe_div(c_diff,cls,0.0)*10.0)
        pk_norm=self._apply_norm(pk,100.0)
        # [V331.0.0] 对称共振：高密度筹码峰无论是向上突破还是向下破位，都将爆发出巨大共振势能
        core=price_force*pk_norm*self._apply_zg(price_force)
        amp=1.0+(self._apply_norm(cvg,1.0)+self._apply_norm(hl,100.0)+np.maximum(np.tanh(self._apply_hab(df,'sc',sc,13)),0.0)+self._apply_norm(vac,100.0)+self._apply_norm(fi,100.0)+(1.0-self._apply_norm(ent,10.0)))/6.0
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'peak_concentration_D_scaled',pk/100.0,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_concentration_D':pk,'close_D':cls},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V303.0.0 · 传参修复版】价资协同双向剥离
        修改要点: 彻底移除废弃的 df_index 传参，对齐 _apply_zg 纯数组签名，解决 TypeError 宕机。
        """
        method_name="_calculate_pf_relationship"
        required_signals=['net_mf_amount_D','close_D','price_vs_ma_13_ratio_D','main_force_activity_index_D','flow_momentum_13d_D','flow_impact_ratio_D','tick_chip_transfer_efficiency_D','VPA_EFFICIENCY_D','amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        cls=self._get_safe_array(df,'close_D',method_name=method_name)
        pm=self._get_safe_array(df,'price_vs_ma_13_ratio_D',method_name=method_name)
        pm=np.where(pm==0.0,1.0,pm)
        act=self._get_safe_array(df,'main_force_activity_index_D',method_name=method_name)
        fm=self._get_safe_array(df,'flow_momentum_13d_D',method_name=method_name)
        imp=self._get_safe_array(df,'flow_impact_ratio_D',method_name=method_name)
        tr=self._get_safe_array(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        vpa=self._get_safe_array(df,'VPA_EFFICIENCY_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        c_diff=np.zeros_like(cls)
        c_diff[1:]=cls[1:]-cls[:-1]
        mf_ratio=self._safe_div(mf,amt,0.0)
        price_force=np.tanh(self._safe_div(c_diff,cls,0.0)*20.0)
        mf_force=np.tanh(mf_ratio*50.0)
        # [V303.0.0 热修复] 彻底移除 df_index 传参
        core=(price_force*0.5+mf_force*0.5)*self._apply_zg(np.abs(price_force)+np.abs(mf_force))
        amp=1.0+(self._apply_norm(act,100.0)+np.maximum(np.tanh(self._apply_hab(df,'fm',fm,13)),0.0)+np.maximum(np.tanh(self._apply_hab(df,'imp',imp,21)),0.0)+np.maximum(np.tanh(self._apply_hab(df,'tr',tr,21)),0.0)+np.abs(np.tanh(vpa*5.0)))/5.0
        raw=core*amp*(1.0+np.abs(np.tanh(self._apply_hab(df,'pm',pm,21)))*0.5)*(1.0+np.abs(self._apply_kinematics(df,'mf_ratio_scaled',mf_ratio*100.0,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':mf,'close_D':cls},calc_nodes={'price_force':price_force,'mf_force':mf_force,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_price_vs_momentum_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V333.0.0 · 第一性原理极性绝对纠偏版】
        用途：诊断价格趋势位移与动能张量的多维非线性背离。
        修改要点：强制覆盖！消除极性反噬 BUG。在 Bipolar 机制中，风险必须为负。现利用第一性原理：顶背离(roc_kin - p_kin)天然生成负向张量，底背离天然生成正向张量。采用严密的 np.minimum/maximum 绝对门控，彻底杜绝高位超买区报出正分机会的“指鹿为马”。
        """
        method_name="_calculate_price_vs_momentum_divergence"
        required_signals=['close_D','ROC_13_D','VPA_EFFICIENCY_D','PRICE_ENTROPY_D','net_mf_amount_D','turnover_rate_f_D','GEOM_REG_SLOPE_D','GEOM_REG_R2_D','BIAS_21_D','GEOM_ARC_CURVATURE_D','market_sentiment_score_D','volatility_adjusted_concentration_D','amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        cls=self._get_safe_array(df,'close_D',method_name=method_name)
        roc=self._get_safe_array(df,'ROC_13_D',method_name=method_name)
        vpa=self._get_safe_array(df,'VPA_EFFICIENCY_D',method_name=method_name)
        ent=self._get_safe_array(df,'PRICE_ENTROPY_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        to=self._get_safe_array(df,'turnover_rate_f_D',method_name=method_name)
        slope=self._get_safe_array(df,'GEOM_REG_SLOPE_D',method_name=method_name)
        r2=self._get_safe_array(df,'GEOM_REG_R2_D',method_name=method_name)
        bias=self._get_safe_array(df,'BIAS_21_D',method_name=method_name)
        arc=self._get_safe_array(df,'GEOM_ARC_CURVATURE_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        
        cls_smooth=np.where(_jit_rolling_mean(cls,13)==0.0,1.0,_jit_rolling_mean(cls,13))
        p_kin=self._apply_kinematics(df,'close_D_scaled',self._safe_div(cls,cls_smooth,1.0)*10.0,13)
        roc_kin=self._apply_kinematics(df,'ROC_13_D_scaled',roc/100.0,13)
        
        # [V333.0.0] 极性第一性原理对齐：
        # 顶背离(风险)：动能减弱(roc_kin<0) 减去 价格上涨(p_kin>0) = 天然的负向张量
        top_div=np.where((p_kin>0.0)&(roc_kin<0.0), roc_kin-p_kin, 0.0) 
        # 底背离(机会)：动能增强(roc_kin>0) 减去 价格下跌(p_kin<0) = 天然的正向张量
        bot_div=np.where((p_kin<0.0)&(roc_kin>0.0), roc_kin-p_kin, 0.0) 
        kinematic_div = top_div + bot_div
        
        c_diff=np.zeros_like(cls)
        c_diff[1:]=cls[1:]-cls[:-1]
        p_vel=np.tanh(self._safe_div(c_diff,cls,0.0)*10.0)
        e_vel=(np.tanh(vpa*5.0)+np.tanh(self._safe_div(mf,amt,0.0)*50.0))/2.0
        
        # 同理，量价能量背离极性同步对齐
        e_top_div=np.where((p_vel>0.0)&(e_vel<0.0), e_vel-p_vel, 0.0) # 负向张量
        e_bot_div=np.where((p_vel<0.0)&(e_vel>0.0), e_vel-p_vel, 0.0) # 正向张量
        energy_div = e_top_div + e_bot_div
        
        # 高位张力过大是绝对风险，直接生成负分惩罚
        geom_tension=(np.tanh(self._apply_hab(df,'bias',bias,21))-np.tanh(self._apply_hab(df,'arc',arc,21)))*0.5*(1.0+self._apply_norm(r2,1.0))
        geom_penalty=-np.maximum(geom_tension, 0.0)
        
        raw_div=(kinematic_div*0.4 + energy_div*0.3 + geom_penalty*0.3)
        
        # [V333.0.0] 绝对物理门控：多头超买区(roc>0)只允许输出负数风险，空头超卖区(roc<=0)只允许输出正数机会
        div_physics_gate=np.where(roc>0.0, np.minimum(raw_div, 0.0), np.maximum(raw_div, 0.0)).astype(np.float32)
        orbit_activation=np.tanh(np.abs(roc)/20.0)
        
        core=div_physics_gate*self._apply_zg(roc)*orbit_activation
        amp=1.0+(np.abs(np.tanh(roc/10.0))+np.abs(self._apply_norm(sent,100.0))+self._apply_norm(ent,10.0)+self._apply_norm(vac,100.0))/4.0
        raw=core*amp
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc,'VPA_EFFICIENCY_D':vpa},calc_nodes={'kinematic_div':kinematic_div,'div_physics_gate':div_physics_gate,'orbit_activation':orbit_activation,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        sa_name=config.get('signal_A')
        sb_name=config.get('signal_B')
        df_index=df.index
        rel_type=config.get('relationship_type','consensus')
        def _get_sig(name: str, src: str) -> Optional[np.ndarray]:
            if src=='atomic_states': 
                s=self.strategy.atomic_states.get(name)
                return s.to_numpy(dtype=np.float32) if s is not None else None
            try: return self._get_safe_array(df,name,method_name="_calculate_instantaneous_relationship")
            except ValueError: return None
        sa=_get_sig(sa_name,config.get('source_A','df'))
        sb=_get_sig(sb_name,config.get('source_B','df'))
        if sa is None or sb is None: return pd.Series(0.0,index=df_index,dtype=np.float32)
        ca=np.zeros_like(sa)
        if config.get('change_type_A','pct')=='diff': ca[1:]=sa[1:]-sa[:-1]
        else: ca[1:]=self._safe_div(sa[1:]-sa[:-1],sa[:-1],0.0)
        cb=np.zeros_like(sb)
        if config.get('change_type_B','pct')=='diff': cb[1:]=sb[1:]-sb[:-1]
        else: cb[1:]=self._safe_div(sb[1:]-sb[:-1],sb[:-1],0.0)
        sa_z=self._apply_hab(df,f'{sa_name}_z',sa,21)
        sb_z=self._apply_hab(df,f'{sb_name}_z',sb,21)
        k_a=self._apply_kinematics(df,f'{sa_name}_z_scaled',sa_z/5.0,13)
        k_b=self._apply_kinematics(df,f'{sb_name}_z_scaled',sb_z/5.0,13)
        ma=np.tanh(self._apply_hab(df,f'{sa_name}_rel',ca,13))+k_a*0.5
        tb=np.tanh(self._apply_hab(df,f'{sb_name}_rel',cb,13))+k_b*0.5
        kf=config.get('signal_b_factor_k',1.0)
        if rel_type=='divergence': rel=(kf*tb-ma)/(kf+1.0)
        else: rel=np.sign(ma+kf*tb)*np.sqrt(np.abs(ma)*np.abs(tb))
        res=np.clip(np.tanh(np.sign(rel)*(np.abs(rel)**1.5)),-1.0,1.0)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_dyn_vs_chip_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V332.0.0 · 抛物线防隐身修正版】
        用途：力学筹码共振看跌风险(Unipolar)。
        修改要点：修正旧版对强多头趋势中的 0 值硬性抹杀。在极端的逼空主升浪中(roc>0且roc_k>0)，如果底层筹码已经发生严重高位派发背离(bc<0)，现改为保留基础输出。若动能进一步恶化则触发 1.5 倍催化剂，确保顶部筹码核爆绝不隐身。
        """
        method_name="_calculate_dyn_vs_chip_relationship"
        required_signals=['ROC_13_D','winner_rate_D','profit_ratio_D','chip_mean_D','chip_kurtosis_D','volatility_adjusted_concentration_D','downtrend_strength_D','chip_entropy_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        roc=self._get_safe_array(df,'ROC_13_D',method_name=method_name)
        win=self._get_safe_array(df,'winner_rate_D',method_name=method_name)
        prof=self._get_safe_array(df,'profit_ratio_D',method_name=method_name)
        mean=self._get_safe_array(df,'chip_mean_D',method_name=method_name)
        kurt=self._get_safe_array(df,'chip_kurtosis_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        bc=self._calculate_instantaneous_relationship(df,config).to_numpy(dtype=np.float32)
        # Unipolar 风险：由于系统框架通过负权重转化为风险，这里强制取正分送出
        core=np.where(bc<0.0,-bc,0.0)*self._apply_zg(bc)
        amp=1.0+(self._apply_norm(prof,100.0)+self._apply_norm(win,100.0)+np.abs(np.tanh(self._apply_hab(df,'mn',mean,13)))+self._apply_norm(kurt,100.0)+self._apply_norm(vac,100.0)+self._apply_norm(down,100.0)+self._apply_norm(ent,10.0)+(1.0-self._apply_norm(sent,100.0)))/8.0
        roc_k=self._apply_kinematics(df,'ROC_13_D_scaled',roc/100.0,13)
        # [V332.0.0] 取消硬锁，替换为恶化催化剂，防止逼空期盲区
        roc_catalyst=np.where((roc<0.0)|(roc_k<0.0),1.5,1.0).astype(np.float32)
        raw=core*amp*(1.0+np.abs(roc_k))*roc_catalyst
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc},calc_nodes={'roc_k':roc_k,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V331.0.0 · 物理量纲硬化版】"""
        method_name="_calculate_process_wash_out_rebound"
        required_signals=['shakeout_score_D','intraday_distribution_confidence_D','pressure_trapped_D','CLOSING_STRENGTH_D','intraday_trough_filling_degree_D','stealth_flow_ratio_D','absorption_energy_D','STATE_ROUNDING_BOTTOM_D','intraday_low_lock_ratio_D','vwap_deviation_D','tick_abnormal_volume_ratio_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        shk=self._get_safe_array(df,'shakeout_score_D',method_name=method_name)
        dist=self._get_safe_array(df,'intraday_distribution_confidence_D',method_name=method_name)
        pan=self._get_safe_array(df,'pressure_trapped_D',method_name=method_name)
        cls=self._get_safe_array(df,'CLOSING_STRENGTH_D',method_name=method_name)
        tf=self._get_safe_array(df,'intraday_trough_filling_degree_D',method_name=method_name)
        stl=self._get_safe_array(df,'stealth_flow_ratio_D',method_name=method_name)
        abs_e=self._get_safe_array(df,'absorption_energy_D',method_name=method_name)
        rnd=self._get_safe_array(df,'STATE_ROUNDING_BOTTOM_D',method_name=method_name)
        rnd=np.clip(rnd,0.0,1.0)
        llck=self._get_safe_array(df,'intraday_low_lock_ratio_D',method_name=method_name)
        vdev=self._get_safe_array(df,'vwap_deviation_D',method_name=method_name)
        t_abn=self._get_safe_array(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        # [V331.0.0] 显式物理量纲约束
        cls_norm=self._apply_norm(cls, 1.0)
        shk_norm=self._apply_norm(shk,100.0)
        tf_norm=self._apply_norm(tf, 1.0)
        abs_norm=np.tanh(abs_e*100.0)
        reb_norm=(tf_norm+abs_norm)/2.0
        base_shk=np.maximum(shk_norm,reb_norm*0.5)
        core=base_shk*reb_norm*self._apply_zg(base_shk*reb_norm)
        pan_norm=np.tanh(pan*100.0)
        stl_norm=np.tanh(stl*100.0)
        amp=1.0+(self._apply_norm(dist,100.0)+pan_norm+self._apply_norm(llck,1.0)+self._apply_norm(t_abn,10.0)+stl_norm+np.abs(np.clip(np.tanh(self._apply_hab(df,'vdev',vdev,13)),None,0.0))+cls_norm+rnd+(1.0-self._apply_norm(c_ent,10.0)))/9.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'shakeout_score_D_scaled',shk_norm,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'shakeout_score_D':shk,'intraday_trough_filling_degree_D':tf,'CLOSING_STRENGTH_D':cls},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_fusion_trend_exhaustion_syndrome(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name="_calculate_fusion_trend_exhaustion_syndrome"
        required_signals=['STATE_PARABOLIC_WARNING_D','STATE_EMOTIONAL_EXTREME_D','PRICE_ENTROPY_D','profit_pressure_D','HM_COORDINATED_ATTACK_D','intraday_distribution_confidence_D','distribution_energy_D','chip_entropy_D','sell_elg_amount_D','amount_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        para=self._get_safe_array(df,'STATE_PARABOLIC_WARNING_D',method_name=method_name)
        para=np.clip(para,0.0,1.0)
        emot=self._get_safe_array(df,'STATE_EMOTIONAL_EXTREME_D',method_name=method_name)
        emot=np.clip(emot,0.0,1.0)
        ent=self._get_safe_array(df,'PRICE_ENTROPY_D',method_name=method_name)
        pres=self._get_safe_array(df,'profit_pressure_D',method_name=method_name)
        hm=self._get_safe_array(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        dist_c=self._get_safe_array(df,'intraday_distribution_confidence_D',method_name=method_name)
        dist_e=self._get_safe_array(df,'distribution_energy_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        sell_elg=self._get_safe_array(df,'sell_elg_amount_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        vpa=self._get_safe_array(df,'VPA_EFFICIENCY_D',method_name=method_name)
        base_state=np.maximum(self._apply_norm(para,1.0),self._apply_norm(emot,1.0))**2
        dist_force=np.maximum(self._apply_norm(pres,100.0),self._apply_norm(dist_e,100.0))
        core=base_state*dist_force*self._apply_zg(base_state*dist_force)
        amp=1.0+(self._apply_norm(dist_c,100.0)+np.tanh(self._safe_div(sell_elg,amt,0.0)*10.0)+self._apply_norm(ent,10.0)+self._apply_norm(c_ent,10.0))/4.0
        vpa_bull=np.maximum(np.tanh(vpa*5.0),0.0)
        veto=np.maximum(1.0-self._apply_norm(hm,100.0)*0.9,0.1)*(1.0-vpa_bull*0.5)
        raw=core*amp*veto*(1.0+np.maximum(self._apply_kinematics(df,'profit_pressure_D_scaled',pres/100.0,13),0.0))
        res=np.clip(np.tanh(raw**1.5),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'STATE_PARABOLIC_WARNING_D':para,'distribution_energy_D':dist_e},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_dyn_vs_chip_decay_rise(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name="_calculate_dyn_vs_chip_decay_rise"
        required_signals=['downtrend_strength_D','pressure_trapped_D','absorption_energy_D','chip_kurtosis_D','chip_stability_change_5d_D','reversal_prob_D','intraday_support_test_count_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        pres=self._get_safe_array(df,'pressure_trapped_D',method_name=method_name)
        abs_e=self._get_safe_array(df,'absorption_energy_D',method_name=method_name)
        kurt=self._get_safe_array(df,'chip_kurtosis_D',method_name=method_name)
        stb=self._get_safe_array(df,'chip_stability_change_5d_D',method_name=method_name)
        rev=self._get_safe_array(df,'reversal_prob_D',method_name=method_name)
        sup=self._get_safe_array(df,'intraday_support_test_count_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        down_k=self._apply_kinematics(df,'downtrend_strength_D_scaled',down/100.0,13)
        decay_force=np.abs(np.clip(down_k,None,0.0))
        core=self._apply_norm(down,100.0)*np.tanh(pres*100.0)*decay_force*self._apply_zg(down)*self._apply_zg(decay_force)
        amp=1.0+(np.tanh(abs_e*100.0)+self._apply_norm(kurt,100.0)+np.maximum(np.tanh(self._apply_hab(df,'stb',stb,13)),0.0)+self._apply_norm(rev,100.0)+self._apply_norm(sup,10.0)+self._apply_norm(vac,100.0))/6.0
        raw=core*amp
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'downtrend_strength_D':down,'pressure_trapped_D':pres},calc_nodes={'decay_force':decay_force,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_smart_money_ignition(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V330.0.0 · 内在动能并联版】
        用途：聪明钱与游资协同攻击点火诊断。
        修改要点：消除对外部游资标签的单点故障依赖。当国家队或新晋游资点火未被系统打标时，通过底层特大单净买入(t_net+elg)构建内在点火引擎(intrinsic_ignition)，与外部标签形成双路防爆并联结构，确保真实点火绝不遗漏。
        """
        method_name="_calculate_smart_money_ignition"
        required_signals=['HM_COORDINATED_ATTACK_D','T1_PREMIUM_EXPECTATION_D','IS_MARKET_LEADER_D','flow_acceleration_intraday_D','buy_elg_amount_D','tick_large_order_net_D','amount_D','uptrend_strength_D','STATE_BREAKOUT_CONFIRMED_D','net_energy_flow_D','market_sentiment_score_D','downtrend_strength_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hm=self._get_safe_array(df,'HM_COORDINATED_ATTACK_D',method_name=method_name)
        t1=self._get_safe_array(df,'T1_PREMIUM_EXPECTATION_D',method_name=method_name)
        ldr=self._get_safe_array(df,'IS_MARKET_LEADER_D',method_name=method_name)
        ldr=np.clip(ldr,0.0,1.0)
        f_acc=self._get_safe_array(df,'flow_acceleration_intraday_D',method_name=method_name)
        elg=self._get_safe_array(df,'buy_elg_amount_D',method_name=method_name)
        t_net=self._get_safe_array(df,'tick_large_order_net_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        up=self._get_safe_array(df,'uptrend_strength_D',method_name=method_name)
        brk=self._get_safe_array(df,'STATE_BREAKOUT_CONFIRMED_D',method_name=method_name)
        brk=np.clip(brk,0.0,1.0)
        ne=self._get_safe_array(df,'net_energy_flow_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        down_penalty=np.clip(1.0-self._apply_norm(down,100.0),0.1,1.0)
        # [V330.0.0] 构建物理备用引擎，防范外部游资标签单点故障
        intrinsic_ignition=np.maximum(np.tanh(self._safe_div(t_net+elg,amt,0.0)*10.0),0.0)
        hm_base=np.maximum(self._apply_norm(hm,100.0), intrinsic_ignition*0.8)
        core=hm_base*self._apply_zg(hm_base)
        amp=1.0+(self._apply_norm(up,100.0)+self._apply_norm(t1,100.0)+self._apply_norm(f_acc,100.0)+self._apply_norm(self._safe_div(elg,amt,0.0),0.1)+self._apply_norm(np.maximum(self._safe_div(t_net,amt,0.0),0.0),0.1)+self._apply_norm(ne,100.0)+ldr+self._apply_norm(sent,100.0)+brk)/9.0
        raw=core*amp*down_penalty*(1.0+np.maximum(self._apply_kinematics(df,'HM_COORDINATED_ATTACK_D_scaled',hm/100.0,13),0.0))
        res=np.clip(np.tanh(raw**1.5),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_COORDINATED_ATTACK_D':hm,'buy_elg_amount_D':elg,'tick_large_order_net_D':t_net},calc_nodes={'intrinsic_ignition':intrinsic_ignition,'core':core,'amp':amp,'down_penalty':down_penalty,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_vpa_mf_coherence_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V328.0.0 · 连续非线性防爆版】
        用途：量价与主力均线物理相干共振诊断。
        修改要点：修正 down_penalty 零值死锁。确立最小边界门限 0.1，防御极端下跌带来的物理闭环感染，保护核心共振张量不被意外全灭。
        """
        method_name="_calculate_vpa_mf_coherence_resonance"
        required_signals=['MA_COHERENCE_RESONANCE_D','VPA_MF_ADJUSTED_EFF_D','MA_ACCELERATION_EMA_55_D','VPA_ACCELERATION_13D','chip_convergence_ratio_D','flow_consistency_D','volatility_adjusted_concentration_D','chip_entropy_D','downtrend_strength_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        mc=self._get_safe_array(df,'MA_COHERENCE_RESONANCE_D',method_name=method_name)
        ve=self._get_safe_array(df,'VPA_MF_ADJUSTED_EFF_D',method_name=method_name)
        ma=self._get_safe_array(df,'MA_ACCELERATION_EMA_55_D',method_name=method_name)
        va=self._get_safe_array(df,'VPA_ACCELERATION_13D',method_name=method_name)
        cc=self._get_safe_array(df,'chip_convergence_ratio_D',method_name=method_name)
        fc=self._get_safe_array(df,'flow_consistency_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        down_penalty=np.clip(1.0-self._apply_norm(down,100.0),0.1,1.0)
        core=np.maximum(np.tanh(mc/2.0),0.0)*self._apply_zg(mc)
        amp=1.0+(np.maximum(np.tanh(ve*5.0),0.0)+self._apply_norm(cc,1.0)+self._apply_norm(fc,100.0)+self._apply_norm(vac,100.0)+np.maximum(np.tanh(ma/2.0),0.0)+np.maximum(np.tanh(va/2.0),0.0)+(1.0-self._apply_norm(c_ent,10.0)))/7.0
        raw=core*amp*down_penalty*(1.0+np.maximum(self._apply_kinematics(df,'VPA_MF_ADJUSTED_EFF_D_scaled',ve,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_COHERENCE_RESONANCE_D':mc,'VPA_MF_ADJUSTED_EFF_D':ve},calc_nodes={'core':core,'amp':amp,'down_penalty':down_penalty,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_mtf_fractal_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V329.0.0 · 时空分形防断崖版】
        用途：日/周/月多周期同步对齐识别。
        修改要点：修正 up_penalty 的零值硬截断。若短期 uptrend 剧烈回踩归零，不应摧毁月线/周线耗时数月凝聚的时空共振网络，植入 0.1 最小生存底座，允许系统在长线下跌的末端发出共振火种。
        """
        method_name="_calculate_mtf_fractal_resonance"
        required_signals=['daily_weekly_sync_D','daily_monthly_sync_D','PRICE_FRACTAL_DIM_D','uptrend_continuation_prob_D','mid_long_sync_D','short_mid_sync_D','market_sentiment_score_D','uptrend_strength_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        dw_sync=self._get_safe_array(df,'daily_weekly_sync_D',method_name=method_name)
        dm_sync=self._get_safe_array(df,'daily_monthly_sync_D',method_name=method_name)
        ml_sync=self._get_safe_array(df,'mid_long_sync_D',method_name=method_name)
        sm_sync=self._get_safe_array(df,'short_mid_sync_D',method_name=method_name)
        fractal_dim=self._get_safe_array(df,'PRICE_FRACTAL_DIM_D',method_name=method_name)
        prob=self._get_safe_array(df,'uptrend_continuation_prob_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        up=self._get_safe_array(df,'uptrend_strength_D',method_name=method_name)
        # [V329.0.0] 0.1 的物理底座
        up_penalty=np.clip(self._apply_norm(up,50.0),0.1,1.0)
        sync_sum=dw_sync+dm_sync+ml_sync+sm_sync
        core=self._apply_norm(sync_sum,400.0)*self._apply_zg(sync_sum)*up_penalty
        amp=1.0+((1.0-self._apply_norm(fractal_dim,2.0))+self._apply_norm(prob,100.0)+self._apply_norm(sent,100.0))/3.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'daily_weekly_sync_D_scaled',dw_sync/100.0,13),0.0))
        res=np.clip(np.tanh(raw**1.5),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'daily_weekly_sync_D':dw_sync,'uptrend_strength_D':up},calc_nodes={'core':core,'amp':amp,'up_penalty':up_penalty,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_intraday_siege_exhaustion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name="_calculate_intraday_siege_exhaustion"
        required_signals=['intraday_resistance_test_count_D','intraday_support_test_count_D','CLOSING_STRENGTH_D','vwap_deviation_D','resistance_strength_D','support_strength_D','VPA_EFFICIENCY_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        res_tests=self._get_safe_array(df,'intraday_resistance_test_count_D',method_name=method_name)
        sup_tests=self._get_safe_array(df,'intraday_support_test_count_D',method_name=method_name)
        closing=self._get_safe_array(df,'CLOSING_STRENGTH_D',method_name=method_name)
        vwap_dev=self._get_safe_array(df,'vwap_deviation_D',method_name=method_name)
        res_str=self._get_safe_array(df,'resistance_strength_D',method_name=method_name)
        sup_str=self._get_safe_array(df,'support_strength_D',method_name=method_name)
        vpa=self._get_safe_array(df,'VPA_EFFICIENCY_D',method_name=method_name)
        res_base=self._apply_norm(res_tests,10.0)*self._apply_zg(res_tests)
        sup_base=self._apply_norm(sup_tests,10.0)*self._apply_zg(sup_tests)
        cls_norm=self._apply_norm_adaptive(closing)
        vwap_shock=np.tanh(self._apply_hab(df,'vwap',vwap_dev,21))
        vpa_norm=np.abs(np.tanh(vpa*5.0))
        b_amp=1.0+(self._apply_norm(res_str,100.0)+cls_norm+np.maximum(vwap_shock,0.0)+vpa_norm)/4.0
        d_amp=1.0+(self._apply_norm(sup_str,100.0)+(1.0-cls_norm)+np.abs(np.clip(vwap_shock,None,0.0))+(1.0-vpa_norm))/4.0
        close_vector=cls_norm*2.0-1.0
        total_siege_power=res_base*b_amp+sup_base*d_amp
        core=total_siege_power*close_vector*self._apply_zg(total_siege_power)
        raw=core*(1.0+np.abs(self._apply_kinematics(df,'CLOSING_STRENGTH_D_scaled',cls_norm,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'intraday_resistance_test_count_D':res_tests,'CLOSING_STRENGTH_D':closing},calc_nodes={'total_siege_power':total_siege_power,'close_vector':close_vector,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_overnight_intraday_tearing(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V331.0.0 · 物理量纲硬化版】"""
        method_name="_calculate_overnight_intraday_tearing"
        required_signals=['GAP_MOMENTUM_STRENGTH_D','OCH_ACCELERATION_D','CLOSING_STRENGTH_D','intraday_price_range_ratio_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        gap=self._get_safe_array(df,'GAP_MOMENTUM_STRENGTH_D',method_name=method_name)
        och=self._get_safe_array(df,'OCH_ACCELERATION_D',method_name=method_name)
        closing=self._get_safe_array(df,'CLOSING_STRENGTH_D',method_name=method_name)
        range_ratio=self._get_safe_array(df,'intraday_price_range_ratio_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        gap_base=np.tanh(gap/10.0) 
        # [V331.0.0] 显式物理量纲约束
        cls_norm=self._apply_norm(closing, 1.0)
        och_norm=np.tanh(och/10.0)
        intraday_vector=np.clip((cls_norm-0.5)*2.0+och_norm*0.5,-1.0,1.0)
        tearing_magnifier=1.0+np.abs(gap_base-intraday_vector)
        core=intraday_vector*tearing_magnifier*np.abs(gap_base)*self._apply_zg(gap)
        risk_amp=self._apply_norm(range_ratio,100.0)+np.maximum(np.tanh(self._apply_hab(df,'och',och,13)),0.0)
        opp_amp=self._apply_norm(sent,100.0)
        amp=1.0+np.where(core<0,risk_amp/2.0,opp_amp)
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'GAP_MOMENTUM_STRENGTH_D_scaled',gap/10.0,13)))
        is_res=np.maximum(np.sign(gap)*np.sign(intraday_vector),0.0)
        raw=raw*(1.0-is_res*0.5)
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GAP_MOMENTUM_STRENGTH_D':gap,'CLOSING_STRENGTH_D':closing},calc_nodes={'gap_base':gap_base,'intraday_vector':intraday_vector,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_chip_center_kinematics(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V331.0.0 · 筹码重心解耦死锁版】
        修改要点：解耦 price_dev (价格对成本的偏离度) 在 lock_core 中的零值感染。若股价精准落在主力成本线上(price_to_cost == 1.0)，原始逻辑会导致整个锁仓(lock_core)判断彻底失明为 0。现将 price_dev 改为附加催化剂。
        """
        method_name="_calculate_chip_center_kinematics"
        required_signals=['peak_migration_speed_5d_D','intraday_cost_center_volatility_D','price_to_weight_avg_ratio_D','turnover_rate_f_D','cost_50pct_D','chip_entropy_D','close_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        migration=self._get_safe_array(df,'peak_migration_speed_5d_D',method_name=method_name)
        cost_vol=self._get_safe_array(df,'intraday_cost_center_volatility_D',method_name=method_name)
        price_to_cost=self._get_safe_array(df,'price_to_weight_avg_ratio_D',method_name=method_name)
        price_to_cost=np.where(price_to_cost==0.0,1.0,price_to_cost)
        turnover=self._get_safe_array(df,'turnover_rate_f_D',method_name=method_name)
        cost_50=self._get_safe_array(df,'cost_50pct_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        cls=self._get_safe_array(df,'close_D',method_name=method_name)
        cls=np.where(cls==0.0,1.0,cls)
        mig_base=self._apply_norm(migration,10.0)*self._apply_zg(migration)
        vol_shock=self._apply_norm(cost_vol,100.0)
        to_active=np.tanh(turnover/2.0)
        to_locked=(1.0-to_active)*self._apply_zg(turnover)
        dist_amp=1.0+(vol_shock+to_active+np.maximum(np.tanh(self._apply_hab(df,'cost',cost_50,21)),0.0)+self._apply_norm(c_ent,10.0))/4.0
        distribution=mig_base*dist_amp
        price_dev=np.maximum(price_to_cost-1.0,0.0)
        # [V331.0.0] 解耦 price_dev 零值死锁，降级为附加催化剂
        lock_core=(1.0-mig_base)*to_locked*(1.0 + self._apply_norm(price_dev,0.2))
        lock_amp=1.0+((1.0-vol_shock)+np.maximum(np.tanh(self._apply_hab(df,'ptc',price_to_cost,21)),0.0)+(1.0-self._apply_norm(c_ent,10.0)))/3.0
        lock=lock_core*lock_amp
        cost_norm=self._safe_div(cost_50,cls,1.0)*10.0
        raw=(lock-distribution)*(1.0+np.abs(self._apply_kinematics(df,'cost_50_norm',cost_norm,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'peak_migration_speed_5d_D':migration,'turnover_rate_f_D':turnover},calc_nodes={'lock_core':lock_core,'mig_base':mig_base,'to_locked':to_locked,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_ma_compression_explosion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V328.0.0 · 连续非线性防爆版】
        用途：多周期均线极度粘合与筹码高度密集下的绝对势能奇点引爆。
        修改要点：修正 down_penalty 0值死锁，释放由于深跌结构中末端强力吸筹与均线纠缠产生的爆发势能。
        """
        method_name="_calculate_ma_compression_explosion"
        required_signals=['MA_POTENTIAL_COMPRESSION_RATE_D','chip_convergence_ratio_D','TURNOVER_STABILITY_INDEX_D','MACDh_13_34_8_D','energy_concentration_D','volatility_adjusted_concentration_D','close_D','downtrend_strength_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        ma_comp=self._get_safe_array(df,'MA_POTENTIAL_COMPRESSION_RATE_D',method_name=method_name)
        chip_conv=self._get_safe_array(df,'chip_convergence_ratio_D',method_name=method_name)
        to_stab=self._get_safe_array(df,'TURNOVER_STABILITY_INDEX_D',method_name=method_name)
        macd=self._get_safe_array(df,'MACDh_13_34_8_D',method_name=method_name)
        energy_conc=self._get_safe_array(df,'energy_concentration_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        cls=self._get_safe_array(df,'close_D',method_name=method_name)
        cls=np.where(cls==0.0,1.0,cls)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        macd_ratio=self._safe_div(macd,cls,0.0)*100.0
        ignition=np.tanh(macd_ratio/5.0)*self._apply_zg(np.maximum(macd,0.0))
        down_penalty=np.clip(1.0-self._apply_norm(down,100.0),0.1,1.0)
        core=self._apply_norm(ma_comp,1.0)*ignition*self._apply_zg(ma_comp)*down_penalty
        amp=1.0+(self._apply_norm(chip_conv,1.0)+self._apply_norm(to_stab,1.0)+self._apply_norm(energy_conc,100.0)+self._apply_norm(vac,100.0))/4.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'MA_POTENTIAL_COMPRESSION_RATE_D_scaled',ma_comp,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_POTENTIAL_COMPRESSION_RATE_D':ma_comp,'MACDh_13_34_8_D':macd},calc_nodes={'core':core,'amp':amp,'down_penalty':down_penalty,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_top_tier_hm_harvesting(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name="_calculate_top_tier_hm_harvesting"
        required_signals=['HM_ACTIVE_TOP_TIER_D','CLOSING_STRENGTH_D','tick_large_order_net_D','amount_D','outflow_quality_D','distribution_energy_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        hm_active=self._get_safe_array(df,'HM_ACTIVE_TOP_TIER_D',method_name=method_name)
        closing=self._get_safe_array(df,'CLOSING_STRENGTH_D',method_name=method_name)
        large_net=self._get_safe_array(df,'tick_large_order_net_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        outflow_q=self._get_safe_array(df,'outflow_quality_D',method_name=method_name)
        dist_eng=self._get_safe_array(df,'distribution_energy_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        net_ratio=self._safe_div(large_net,amt,0.0)
        dump_force=np.abs(np.clip(np.tanh(net_ratio*10.0),None,0.0))
        hm_norm=self._apply_norm(hm_active,100.0)
        core=hm_norm*dump_force*self._apply_zg(hm_norm*dump_force)
        cls_norm=self._apply_norm_adaptive(closing)
        amp=1.0+((1.0-cls_norm)+self._apply_norm(outflow_q,100.0)+self._apply_norm(dist_eng,100.0)+(1.0-self._apply_norm(sent,100.0)))/4.0
        raw=core*amp*(1.0+np.abs(np.clip(self._apply_kinematics(df,'net_ratio_scaled',net_ratio*100.0,13),None,0.0)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'HM_ACTIVE_TOP_TIER_D':hm_active,'CLOSING_STRENGTH_D':closing},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_vwap_magnetic_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name="_calculate_vwap_magnetic_divergence"
        required_signals=['vwap_deviation_D','reversal_prob_D','intraday_main_force_activity_D','intraday_cost_center_migration_D','volume_ratio_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        vwap_dev=self._get_safe_array(df,'vwap_deviation_D',method_name=method_name)
        rev_prob=self._get_safe_array(df,'reversal_prob_D',method_name=method_name)
        mf_act=self._get_safe_array(df,'intraday_main_force_activity_D',method_name=method_name)
        cost_mig=self._get_safe_array(df,'intraday_cost_center_migration_D',method_name=method_name)
        vol_ratio=self._get_safe_array(df,'volume_ratio_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        dev_base=np.tanh(vwap_dev/5.0)
        core=-1.0*dev_base*self._apply_zg(dev_base)
        mismatch=(1.0-(self._apply_norm(mf_act,100.0)*2.0-1.0)*np.sign(dev_base))*(1.0-np.tanh(self._apply_hab(df,'mig',cost_mig,21))*np.sign(dev_base))
        amp=1.0+(self._apply_norm(rev_prob,100.0)+self._apply_norm(vol_ratio,10.0)+(1.0-self._apply_norm(c_ent,10.0)))/3.0
        raw=core*np.maximum(mismatch,0.1)*amp*(1.0+np.abs(self._apply_kinematics(df,'vwap_deviation_D_scaled',vwap_dev/5.0,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'vwap_deviation_D':vwap_dev,'reversal_prob_D':rev_prob},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_multi_peak_avalanche_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V327.0.0 · 雪崩燃料并联版】
        修改思路：打破雪崩初期散户套牢盘极小导致的单点乘 0 瘫痪。将“获利盘死压(profit_pressure)”与“套牢盘恐慌(pressure_trapped)”组合成最大值并联燃料网络，真实还原高位断层全周期的能量倾泻。
        """
        method_name="_calculate_multi_peak_avalanche_risk"
        required_signals=['is_multi_peak_D','chip_divergence_ratio_D','intraday_distribution_confidence_D','downtrend_strength_D','chip_entropy_D','chip_stability_change_5d_D','volatility_adjusted_concentration_D','pressure_trapped_D','profit_pressure_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        mp=np.clip(self._get_safe_array(df,'is_multi_peak_D',method_name=method_name),0.0,1.0)
        div=self._get_safe_array(df,'chip_divergence_ratio_D',method_name=method_name)
        dist=self._get_safe_array(df,'intraday_distribution_confidence_D',method_name=method_name)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        stab_chg=self._get_safe_array(df,'chip_stability_change_5d_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        trap=self._get_safe_array(df,'pressure_trapped_D',method_name=method_name)
        prof_p=self._get_safe_array(df,'profit_pressure_D',method_name=method_name)
        avalanche_fuel=np.maximum(np.tanh(trap*100.0),self._apply_norm(prof_p,100.0))
        env_risk=np.maximum(self._apply_norm(dist,100.0),self._apply_norm(down,100.0))
        core=mp*env_risk*avalanche_fuel*self._apply_zg(mp*env_risk)
        stab_diff=np.zeros_like(stab_chg)
        stab_diff[1:]=stab_chg[1:]-stab_chg[:-1]
        amp=1.0+(self._apply_norm(div,100.0)+self._apply_norm(ent,10.0)+np.abs(np.clip(np.tanh(self._apply_hab(df,'stab',stab_diff,13)),None,0.0))+(1.0-self._apply_norm(vac,100.0)))/4.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'chip_divergence_ratio_D_scaled',div/100.0,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'is_multi_peak_D':mp,'downtrend_strength_D':down,'pressure_trapped_D':trap,'profit_pressure_D':prof_p},calc_nodes={'avalanche_fuel':avalanche_fuel,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_sector_lifecycle_tailwind(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name="_calculate_sector_lifecycle_tailwind"
        required_signals=['industry_preheat_score_D','industry_markup_score_D','industry_stagnation_score_D','industry_downtrend_score_D','uptrend_strength_D','industry_rank_accel_D','industry_strength_rank_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        preheat=self._get_safe_array(df,'industry_preheat_score_D',method_name=method_name)
        markup=self._get_safe_array(df,'industry_markup_score_D',method_name=method_name)
        stag=self._get_safe_array(df,'industry_stagnation_score_D',method_name=method_name)
        down=self._get_safe_array(df,'industry_downtrend_score_D',method_name=method_name)
        up=self._get_safe_array(df,'uptrend_strength_D',method_name=method_name)
        rk_acc=self._get_safe_array(df,'industry_rank_accel_D',method_name=method_name)
        rank=self._get_safe_array(df,'industry_strength_rank_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        vector=(self._apply_norm(markup,100.0)*1.0+self._apply_norm(preheat,100.0)*0.5)-(self._apply_norm(stag,100.0)*0.5+self._apply_norm(down,100.0)*1.0)
        core=vector*self._apply_zg(vector)
        amp=1.0+(self._apply_norm(up,100.0)+self._apply_norm(rank,100.0)+np.tanh(rk_acc/10.0)*np.sign(vector)+self._apply_norm(sent,100.0))/4.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'industry_markup_score_D_scaled',markup/100.0,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)*2.0),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'industry_markup_score_D':markup},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_time_asymmetry_trap(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V331.0.0 · 绝对正交极性无界版】
        修改要点：彻底废除简单的 morning - afternoon 造成的“尾盘断头铡刀”误判极性反噬。改用“绝对失衡烈度”正交“收盘多空胜负矢量”。无论早盘拉升还是尾盘砸盘，只要收盘弱(close_vector<0)就是无条件的极恶陷阱。显式约束量纲，防止溢出。
        """
        method_name="_calculate_time_asymmetry_trap"
        required_signals=['morning_flow_ratio_D','afternoon_flow_ratio_D','intraday_peak_valley_ratio_D','profit_pressure_D','CLOSING_STRENGTH_D','high_freq_flow_divergence_D','VPA_EFFICIENCY_D','net_mf_amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        morning=self._get_safe_array(df,'morning_flow_ratio_D',method_name=method_name)
        afternoon=self._get_safe_array(df,'afternoon_flow_ratio_D',method_name=method_name)
        cls_strength=self._get_safe_array(df,'CLOSING_STRENGTH_D',method_name=method_name)
        pv=self._get_safe_array(df,'intraday_peak_valley_ratio_D',method_name=method_name)
        pp=self._get_safe_array(df,'profit_pressure_D',method_name=method_name)
        hd=self._get_safe_array(df,'high_freq_flow_divergence_D',method_name=method_name)
        vpa=self._get_safe_array(df,'VPA_EFFICIENCY_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        # [V331.0.0] 显式物理量纲约束
        morning_norm=self._apply_norm(morning, 100.0)
        afternoon_norm=self._apply_norm(afternoon, 100.0)
        cls_norm=self._apply_norm(cls_strength, 1.0)
        asymmetry_intensity=np.abs(morning_norm-afternoon_norm)
        close_vector=(cls_norm-0.5)*2.0
        asymmetry_dominance=asymmetry_intensity*close_vector*np.where(close_vector<0, 1.5, 1.0).astype(np.float32)
        base_core=np.tanh(asymmetry_dominance*2.0)
        mf_catalyst=np.where((base_core>0)&(mf<0),0.5,np.where((base_core<0)&(mf<0),1.5,1.0)).astype(np.float32)
        core=np.clip(base_core*mf_catalyst, -1.0, 1.0)*self._apply_zg(asymmetry_intensity)
        risk_amp=self._apply_norm(pv,10.0)+self._apply_norm(pp,100.0)+np.maximum(np.tanh(self._apply_hab(df,'hd',hd,21)),0.0)
        opp_amp=np.tanh(np.abs(vpa)*5.0)*3.0
        amp=1.0+np.where(core<0.0,risk_amp/3.0,opp_amp/2.0)
        raw=core*amp*(1.0+np.abs(self._apply_kinematics(df,'morning_flow_ratio_D_scaled',morning_norm,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'morning_flow_ratio_D':morning,'afternoon_flow_ratio_D':afternoon,'net_mf_amount_D':mf,'CLOSING_STRENGTH_D':cls_strength},calc_nodes={'asymmetry_intensity':asymmetry_intensity,'close_vector':close_vector,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_high_pos_liquidity_squeeze(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V331.0.0 · 物理量纲硬化版】"""
        method_name="_calculate_high_pos_liquidity_squeeze"
        required_signals=['price_percentile_position_D','high_position_lock_ratio_90_D','flow_persistence_minutes_D','short_term_chip_ratio_D','uptrend_strength_D','turnover_rate_f_D','chip_entropy_D','net_mf_amount_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        perc=self._get_safe_array(df,'price_percentile_position_D',method_name=method_name)
        lock=self._get_safe_array(df,'high_position_lock_ratio_90_D',method_name=method_name)
        per=self._get_safe_array(df,'flow_persistence_minutes_D',method_name=method_name)
        short=self._get_safe_array(df,'short_term_chip_ratio_D',method_name=method_name)
        up=self._get_safe_array(df,'uptrend_strength_D',method_name=method_name)
        to=self._get_safe_array(df,'turnover_rate_f_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        # [V331.0.0] 显式物理量纲约束
        pos_norm=self._apply_norm(perc, 1.0)
        lock_norm=self._apply_norm(lock, 1.0)
        to_penalty=np.tanh(to/15.0)
        up_penalty=np.clip(self._apply_norm(up,50.0),0.1,1.0)
        mf_gate=np.where(mf<0.0,0.2,1.0).astype(np.float32)
        core=lock_norm*np.maximum(pos_norm,0.0)*(1.0-to_penalty)*self._apply_zg(lock)*up_penalty*mf_gate
        amp=1.0+(self._apply_norm(up,100.0)+self._apply_norm(per,100.0)+(1.0-self._apply_norm(short,1.0))+(1.0-self._apply_norm(c_ent,10.0)))/4.0
        raw=(core*amp)**1.5*(1.0+np.maximum(self._apply_kinematics(df,'high_position_lock_ratio_90_D_scaled',lock_norm,13),0.0))
        res=np.clip(np.tanh(raw*2.0),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'price_percentile_position_D':perc,'high_position_lock_ratio_90_D':lock,'turnover_rate_f_D':to},calc_nodes={'core':core,'amp':amp,'up_penalty':up_penalty,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_institutional_structural_exit(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """【V331.0.0 · 物理量纲硬化版】"""
        method_name="_calculate_institutional_structural_exit"
        required_signals=['sell_elg_amount_D','sell_lg_amount_D','amount_D','distribution_energy_D','downtrend_strength_D','high_position_lock_ratio_90_D','chip_stability_change_5d_D','market_sentiment_score_D','price_percentile_position_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        sell_elg=self._get_safe_array(df,'sell_elg_amount_D',method_name=method_name)
        sell_lg=self._get_safe_array(df,'sell_lg_amount_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        dist=self._get_safe_array(df,'distribution_energy_D',method_name=method_name)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        lock=self._get_safe_array(df,'high_position_lock_ratio_90_D',method_name=method_name)
        stab=self._get_safe_array(df,'chip_stability_change_5d_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        pos=self._get_safe_array(df,'price_percentile_position_D',method_name=method_name)
        sell_ratio=self._safe_div(sell_elg+sell_lg*0.5,amt,0.0)
        sell_energy=self._apply_norm(sell_ratio,0.05)
        # [V331.0.0] 显式物理量纲约束
        pos_norm=self._apply_norm(pos, 1.0)
        lock_norm=self._apply_norm(lock, 1.0)
        high_state=np.maximum(lock_norm,np.maximum(pos_norm,0.0))
        sell_gate=np.clip((sell_ratio-0.01)/0.015,0.0,1.0).astype(np.float32)
        core=high_state*sell_energy*sell_gate*self._apply_zg(high_state*sell_energy)
        lock_diff=np.zeros_like(lock)
        lock_diff[1:]=lock[1:]-lock[:-1]
        amp=1.0+(self._apply_norm(dist,100.0)+self._apply_norm(down,100.0)+np.abs(np.clip(np.tanh(self._apply_hab(df,'lkd',lock_diff,13)),None,0.0))+np.abs(np.clip(np.tanh(self._apply_hab(df,'stb',stab,13)),None,0.0))+(1.0-np.abs(np.tanh(sent/5.0))))/5.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'sell_ratio_scaled',sell_ratio*100.0,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'sell_ratio':sell_ratio,'price_percentile_position_D':pos},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_institutional_sweep(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V329.0.0 · 左侧托底核武版】
        用途：国家队与大公募机构特大单扫货共振识别。
        修改要点：翻转 mf_gate 惩罚逻辑。当全局资金处于恐慌出逃状态 (mf < 0.0) 时，机构特大单主动扫货 (buy_elg) 不应该被当做杂音压制为 0.1，这恰好是“国家队/顶尖主力雪中送炭左侧托底”的极品指纹，因此将此时的 mf_gate 翻转奖励为 1.2。
        """
        method_name="_calculate_institutional_sweep"
        required_signals=['buy_elg_amount_D','buy_lg_amount_D','amount_D','tick_chip_transfer_efficiency_D','flow_consistency_D','net_mf_amount_D','flow_impact_ratio_D','market_sentiment_score_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        buy_elg=self._get_safe_array(df,'buy_elg_amount_D',method_name=method_name)
        buy_lg=self._get_safe_array(df,'buy_lg_amount_D',method_name=method_name)
        amt=self._get_safe_array(df,'amount_D',method_name=method_name)
        amt=np.where(amt==0.0,1.0,amt)
        tr=self._get_safe_array(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        cons=self._get_safe_array(df,'flow_consistency_D',method_name=method_name)
        mf=self._get_safe_array(df,'net_mf_amount_D',method_name=method_name)
        imp=self._get_safe_array(df,'flow_impact_ratio_D',method_name=method_name)
        sent=self._get_safe_array(df,'market_sentiment_score_D',method_name=method_name)
        buy_ratio=self._safe_div(buy_elg+buy_lg*0.5,amt,0.0)
        # [V329.0.0] 将原本的错杀打压(0.1)翻转为左侧英雄奖励(1.2)
        mf_gate=np.where(mf<0.0,1.2,1.0).astype(np.float32)
        sweep_gate=np.clip((buy_ratio-0.01)*50.0,0.0,1.0).astype(np.float32)
        core=self._apply_norm(buy_ratio,0.1)*self._apply_zg(buy_ratio)*mf_gate*sweep_gate
        amp=1.0+(np.maximum(np.tanh(self._apply_hab(df,'mf',mf,55)),0.0)+self._apply_norm(tr,1e6)+self._apply_norm(cons,100.0)+self._apply_norm(imp,10.0)+self._apply_norm(sent,100.0))/5.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'buy_ratio_scaled',buy_ratio*100.0,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'buy_elg_amount_D':buy_elg,'net_mf_amount_D':mf},calc_nodes={'mf_gate':mf_gate,'sweep_gate':sweep_gate,'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_hf_algo_manipulation_risk(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V306.0.0】高频算法诱骗崩塌防线
        修改要点：修复大单异常异动飙升时(斜率为正)，被 np.clip(..., None, 0.0) 错误清零的逻辑漏洞。修正为 np.maximum(..., 0.0) 以确保高频操纵在恶化加速时风控得分能发生非线性急剧爆发。
        """
        method_name="_calculate_hf_algo_manipulation_risk"
        required_signals=['high_freq_flow_skewness_D','high_freq_flow_kurtosis_D','large_order_anomaly_D','price_flow_divergence_D','intraday_price_distribution_skewness_D','tick_abnormal_volume_ratio_D','volatility_adjusted_concentration_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        skew=self._get_safe_array(df,'high_freq_flow_skewness_D',method_name=method_name)
        kurt=self._get_safe_array(df,'high_freq_flow_kurtosis_D',method_name=method_name)
        anom=self._get_safe_array(df,'large_order_anomaly_D',method_name=method_name)
        div=self._get_safe_array(df,'price_flow_divergence_D',method_name=method_name)
        pskew=self._get_safe_array(df,'intraday_price_distribution_skewness_D',method_name=method_name)
        abn=self._get_safe_array(df,'tick_abnormal_volume_ratio_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        skew_base=np.tanh(np.abs(skew)/50.0)
        anom_base=self._apply_norm(anom,1.0)
        core=skew_base*anom_base*self._apply_zg(skew_base*anom_base)
        amp=1.0+(np.maximum(np.tanh(self._apply_hab(df,'div',div,21)),0.0)+np.maximum(np.tanh(self._apply_hab(df,'kurt',kurt,34)),0.0)+np.abs(np.tanh(self._apply_hab(df,'mis',skew-pskew,13)))+np.maximum(np.tanh(self._apply_hab(df,'abn',abn,21)),0.0)+(1.0-self._apply_norm(vac,100.0)))/5.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'large_order_anomaly_D_scaled',anom,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),0.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'high_freq_flow_skewness_D':skew,'large_order_anomaly_D':anom},calc_nodes={'core':core,'amp':amp,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_ma_rubber_band_reversal(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V302.0.0 · 极限算力热修复版】均线张力极值反噬引擎
        修改要点: 移除废弃的 df_index 传参，剥离 Pandas 逻辑。
        """
        method_name="_calculate_ma_rubber_band_reversal"
        required_signals=['MA_RUBBER_BAND_EXTENSION_D','MA_POTENTIAL_TENSION_INDEX_D','ADX_14_D','profit_pressure_D','pressure_trapped_D','BIAS_21_D','reversal_prob_D','chip_entropy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        ext=self._get_safe_array(df,'MA_RUBBER_BAND_EXTENSION_D',method_name=method_name)
        tension=self._get_safe_array(df,'MA_POTENTIAL_TENSION_INDEX_D',method_name=method_name)
        adx=self._get_safe_array(df,'ADX_14_D',method_name=method_name)
        profit_p=self._get_safe_array(df,'profit_pressure_D',method_name=method_name)
        trap_p=self._get_safe_array(df,'pressure_trapped_D',method_name=method_name)
        bias=self._get_safe_array(df,'BIAS_21_D',method_name=method_name)
        rev=self._get_safe_array(df,'reversal_prob_D',method_name=method_name)
        c_ent=self._get_safe_array(df,'chip_entropy_D',method_name=method_name)
        supp=1.0-np.tanh(np.maximum(adx-35.0,0.0)/15.0)
        c_ent_n=self._apply_norm(c_ent,10.0)
        trap_norm=np.tanh(trap_p*100.0)
        # [V302.0.0] 移除 df_index 传参
        t_base=np.maximum(np.tanh(ext/10.0),0.0)*self._apply_zg(np.maximum(ext,0.0))
        t_amp=1.0+(self._apply_norm(tension,100.0)+self._apply_norm(profit_p,100.0)+np.maximum(np.tanh(self._apply_hab(df,'bias',bias,21)),0.0)+self._apply_norm(rev,100.0)+(1.0-c_ent_n))/5.0
        # [V302.0.0] 移除 df_index 传参
        b_base=np.abs(np.clip(np.tanh(ext/10.0),None,0.0))*self._apply_zg(np.clip(ext,None,0.0))
        b_amp=1.0+(self._apply_norm(tension,100.0)+trap_norm+np.abs(np.clip(np.tanh(self._apply_hab(df,'bias',bias,21)),None,0.0))+self._apply_norm(rev,100.0)+(1.0-c_ent_n))/5.0
        raw=(b_base*b_amp-t_base*t_amp)*supp*(1.0+np.abs(self._apply_kinematics(df,'MA_RUBBER_BAND_EXTENSION_D_scaled',ext/100.0,13)))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'MA_RUBBER_BAND_EXTENSION_D':ext},calc_nodes={'top_base':t_base,'bot_base':b_base,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)

    def _calculate_geometric_trend_resonance(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V329.0.0 · 几何流形柔性底座版】
        用途：几何流形回归拟合与二阶导数曲率张力诊断。
        修改要点：修复 trend_context 在横盘或初次起爆时带来的 0 值锁死。当标的从深海中拔地而起展现极强的正向 slope 时，不能因为均线系统滞后导致 uptrend=0 而被一票否决。建立 0.1 的柔性传导底座。
        """
        method_name="_calculate_geometric_trend_resonance"
        required_signals=['GEOM_REG_R2_D','GEOM_REG_SLOPE_D','GEOM_ARC_CURVATURE_D','GEOM_CHANNEL_POS_D','PRICE_FRACTAL_DIM_D','trend_confirmation_score_D','volatility_adjusted_concentration_D','close_D','uptrend_strength_D','downtrend_strength_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        r2=self._get_safe_array(df,'GEOM_REG_R2_D',method_name=method_name)
        slope=self._get_safe_array(df,'GEOM_REG_SLOPE_D',method_name=method_name)
        curv=self._get_safe_array(df,'GEOM_ARC_CURVATURE_D',method_name=method_name)
        pos=self._get_safe_array(df,'GEOM_CHANNEL_POS_D',method_name=method_name)
        frac=self._get_safe_array(df,'PRICE_FRACTAL_DIM_D',method_name=method_name)
        conf=self._get_safe_array(df,'trend_confirmation_score_D',method_name=method_name)
        vac=self._get_safe_array(df,'volatility_adjusted_concentration_D',method_name=method_name)
        cls=self._get_safe_array(df,'close_D',method_name=method_name)
        cls=np.where(cls==0.0,1.0,cls)
        up=self._get_safe_array(df,'uptrend_strength_D',method_name=method_name)
        down=self._get_safe_array(df,'downtrend_strength_D',method_name=method_name)
        slope_ratio=self._safe_div(slope,cls,0.0)*100.0
        # [V329.0.0] 防爆 0.1 阻尼底座
        trend_context=np.where(slope>0,np.clip(self._apply_norm(up,50.0),0.1,1.0),np.clip(self._apply_norm(down,50.0),0.1,1.0)).astype(np.float32)
        core=np.tanh(slope_ratio/10.0)*self._apply_norm(r2,1.0)*self._apply_zg(slope)*self._apply_zg(r2)*trend_context
        dyn=np.tanh(self._apply_hab(df,'curv',curv,21))-(self._apply_norm(pos,1.0)-0.5)*2.0*0.3
        amp=1.0+((1.0-self._apply_norm(frac,2.0))+self._apply_norm(conf,100.0)+np.maximum(dyn,0.0)+self._apply_norm(vac,100.0))/4.0
        raw=core*amp*(1.0+np.maximum(self._apply_kinematics(df,'GEOM_REG_SLOPE_D_scaled',slope_ratio/10.0,13),0.0))
        res=np.clip(np.tanh(np.sign(raw)*(np.abs(raw)**1.5)),-1.0,1.0)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'GEOM_REG_SLOPE_D':slope,'GEOM_REG_R2_D':r2},calc_nodes={'core':core,'amp':amp,'trend_context':trend_context,'raw_score':raw},final_result=res)
        return pd.Series(res,index=df_index,dtype=np.float32)





