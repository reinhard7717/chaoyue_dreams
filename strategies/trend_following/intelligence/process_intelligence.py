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
        """
        【V4.1.0 · MTF动态张量原生版】
        取代统一归一化，使用 Pandas 原生 where 防止矩阵坍缩。
        """
        slope_periods_weights=get_param_value(mtf_weights_config.get('slope_periods'),{"5":0.4,"13":0.3,"21":0.2,"34":0.1})
        accel_periods_weights=get_param_value(mtf_weights_config.get('accel_periods'),{"5":0.6,"13":0.4})
        all_scores_components=[]
        total_combined_weight=0.0
        def _process_kinematic(col_name:str,weight:float,period:int):
            if col_name not in df.columns:
                return 0.0,0.0
            raw_series=df[col_name].astype(np.float32)
            if raw_series.isnull().all():
                return 0.0,0.0
            gated=raw_series.where(raw_series.abs()>=1e-4,0.0)
            hab_window=max(21,period*2)
            shock=self._apply_hab_shock(gated,window=hab_window)
            norm_score=np.tanh(shock)
            if not ascending:
                norm_score=-norm_score
            if not bipolar:
                norm_score=0.5*(1.0+norm_score)
            return norm_score*weight,weight
        for period_str,weight in slope_periods_weights.items():
            period=int(period_str)
            score,w=_process_kinematic(f'SLOPE_{period}_{base_signal_name}',weight,period)
            all_scores_components.append(score)
            total_combined_weight+=w
        for period_str,weight in accel_periods_weights.items():
            period=int(period_str)
            score,w=_process_kinematic(f'ACCEL_{period}_{base_signal_name}',weight,period)
            all_scores_components.append(score)
            total_combined_weight+=w
        if not all_scores_components or total_combined_weight==0:
            return pd.Series(0.0,index=df_index,dtype=np.float32)
        fused_score=sum(all_scores_components)/total_combined_weight
        return fused_score.clip(-1,1).astype(np.float32) if bipolar else fused_score.clip(0,1).astype(np.float32)

    def _get_mtf_slope_score(self, df: pd.DataFrame, base_signal_name: str, mtf_weights: Dict, df_index: pd.Index, method_name: str, bipolar: bool = True) -> pd.Series:
        """
        【V4.1.0 · 纯斜率非线性动能原生版】
        专供单一维度斜率融合，使用 Pandas 的 .where 杜绝索引降维。
        """
        fused_score=pd.Series(0.0,index=df_index,dtype=np.float32)
        total_weight=0.0
        for period_str,weight in mtf_weights.items():
            try:
                period=int(period_str)
            except ValueError:
                continue
            col_name=f'SLOPE_{period}_{base_signal_name}'
            if col_name not in df.columns:
                continue
            raw_series=df[col_name].astype(np.float32)
            if raw_series.isnull().all():
                continue
            gated=raw_series.where(raw_series.abs()>=1e-4,0.0)
            hab_window=max(21,period*2)
            shock=self._apply_hab_shock(gated,window=hab_window)
            score=np.tanh(shock)
            if not bipolar:
                score=0.5*(1.0+score)
            fused_score+=score*weight
            total_weight+=weight
        if total_weight>0:
            fused_score=fused_score/total_weight
        return fused_score.clip(-1,1).astype(np.float32) if bipolar else fused_score.clip(0,1).astype(np.float32)

    def _get_mtf_cohesion_score(self, df: pd.DataFrame, base_signal_names: List[str], mtf_weights_config: Dict, df_index: pd.Index, method_name: str) -> pd.Series:
        """【V4.0.0 · 逆向张量协同探针版】直接基于多维信号动能矩阵计算标准差，利用逆向Tanh将横向极度离散映射为微观协同。"""
        all_fused_mtf_scores={}
        for base_signal_name in base_signal_names:
            fused_score=self._get_mtf_slope_accel_score(df,base_signal_name,mtf_weights_config,df_index,method_name,ascending=True,bipolar=False)
            all_fused_mtf_scores[base_signal_name]=fused_score
        if not all_fused_mtf_scores:
            return pd.Series(0.0,index=df_index,dtype=np.float32)
        fused_scores_df=pd.DataFrame(all_fused_mtf_scores,index=df_index)
        min_periods_std=max(1,int(self.meta_window*0.5))
        instant_std=fused_scores_df.std(axis=1).fillna(0.0)
        smoothed_std=instant_std.rolling(window=self.meta_window,min_periods=min_periods_std).mean().fillna(0.0)
        std_shock=self._apply_hab_shock(smoothed_std,window=34)
        cohesion_score=0.5*(1.0-np.tanh(std_shock))
        return cohesion_score.clip(0,1).astype(np.float32)

    def _normalize_series(self, *args, **kwargs):
        """【V4.0.0 · 核心禁令】统一归一化引擎已彻底废除。禁止调用此方法，各逻辑节点必须内置领域特化的非线性张力映射。"""
        raise NotImplementedError("致命错误：_normalize_series() 已被废弃，请使用局部 HAB 与 Tanh 映射。")

    def _get_atomic_score(self, df: pd.DataFrame, score_name: str, default_value: float = 0.0) -> pd.Series:
        """
        【V1.0 · 原子信号访问器】
        - 核心职责: 提供一个标准的、安全的方法来从 self.strategy.atomic_states 中获取预先计算好的原子信号。
        - 核心逻辑: 尝试从 atomic_states 字典中获取指定的信号 Series。如果不存在，则打印警告并
                     返回一个与 df 索引对齐的、填充了默认值的 Series，以保证数据流的健壮性。
        - 修复: 解决了 'ProcessIntelligence' object has no attribute '_get_atomic_score' 的 AttributeError。
        """
        #  实现了安全的原子信号访问逻辑
        score_series = self.strategy.atomic_states.get(score_name)
        if score_series is None:
            print(f"    -> [过程情报警告] 依赖的原子信号 '{score_name}' 不存在，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return score_series

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
        【V1.0 · 新增】蓝图审查官。解析诊断配置，提取所有信号依赖，并进行统一校验。
        - 核心职责: 作为配置驱动逻辑的安全阀，确保“情报蓝图”中的所有“原料”都真实存在。
        """
        required_signals = []
        # 提取元关系分析中的信号
        if config.get('signal_A'):
            required_signals.append(config['signal_A'])
        if config.get('signal_B'):
            required_signals.append(config['signal_B'])
        # 提取赢家信念中的特殊信号
        if config.get('antidote_signal'):
            required_signals.append(config['antidote_signal'])
        # 提取信号衰减分析中的信号
        if config.get('source_signal'):
            required_signals.append(config['source_signal'])
        # 提取领域反转分析中的公理信号
        if config.get('axioms'):
            for axiom_config in config.get('axioms', []):
                if axiom_config.get('name'):
                    required_signals.append(axiom_config['name'])
        # 如果没有需要校验的信号，则直接通过
        if not required_signals:
            return True
        # 调用通用的校验器进行检查
        return self._validate_required_signals(df, required_signals, method_name)

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
        【V1.2 · 蓝图审查增强版】元分析调度中心
        - 核心升级: 增加对 'PROCESS_META_POWER_TRANSFER' 的特殊处理，豁免其配置信号校验，
                      因为该信号已升级为内置硬编码依赖 L2 底层数据，不再依赖配置文件中的 signal_A/B。
        """
        signal_name = config.get('name', '未知信号')
        # [升级] 豁免 PROCESS_META_POWER_TRANSFER 的配置校验，防止因旧配置导致启动失败
        if signal_name != 'PROCESS_META_POWER_TRANSFER':
            # “蓝图审查”协议，校验配置文件中声明的所有信号
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
        【V2.1 · 校验协议对齐版】元关系诊断分发器。
        - 核心修复: 修正了校验拦截逻辑，确保其与 `_run_meta_analysis` 保持一致，
                      对内置硬编码 L2 依赖的 'PROCESS_META_POWER_TRANSFER' 信号予以校验豁免。
        """
        signal_name = config.get('name', '未知信号')
        # 对内置硬编码 L2 数据依赖的信号实施校验豁免，防止 Blueprint Inspector 误报
        if signal_name != 'PROCESS_META_POWER_TRANSFER':
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
        signal_name = config.get('name')
        df_index = df.index
        meta_score = pd.Series(dtype=np.float32)
        if signal_name == 'PROCESS_META_COST_ADVANTAGE_TREND':
            meta_score = self.calculate_cost_advantage_trend_relationship_processor.calculate(df, config)
        elif signal_name == 'PROCESS_STRATEGY_DYN_VS_CHIP_DECAY':
            relationship_score = self._calculate_dyn_vs_chip_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_PRICE_VS_RETAIL_CAPITULATION':
            relationship_score = self._calculate_price_vs_capitulation_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_PD_DIVERGENCE_CONFIRM':
            relationship_score = self._calculate_pd_divergence_relationship(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_PROFIT_VS_FLOW':
            relationship_score = self._calculate_profit_vs_flow_relationship(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_PF_REL_BULLISH_TURN':
            meta_score = self._calculate_pf_relationship(df, config)
        elif signal_name == 'PROCESS_META_PC_REL_BULLISH_TURN':
            meta_score = self._calculate_pc_relationship(df, config)
        elif signal_name == 'PROCESS_META_PRICE_VOLUME_DYNAMICS':
            meta_score = self.calculate_price_volume_dynamics_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_MAIN_FORCE_RALLY_INTENT':
            meta_score = self.calculate_main_force_rally_intent_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_WINNER_CONVICTION':
            relationship_score = self.calculate_winner_conviction_relationship_processor.calculate(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_LOSER_CAPITULATION':
            meta_score = self._calculate_loser_capitulation(df, config)
        elif signal_name == 'PROCESS_STRATEGY_FF_VS_STRUCTURE_LEAD':
            relationship_score = self._calculate_ff_vs_structure_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_MAIN_FORCE_CONTROL':
            meta_score = self.calculate_main_force_control_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION':
            meta_score = self._calculate_panic_washout_accumulation(df, config)
        elif signal_name == 'PROCESS_META_DECEPTIVE_ACCUMULATION':
            meta_score = self._calculate_deceptive_accumulation(df, config)
        elif signal_name == 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY':
            meta_score = self.calculate_split_order_accumulation_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_UPTHRUST_WASHOUT':
            meta_score = self.calculate_upthrust_washout_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_ACCUMULATION_INFLECTION':
            meta_score = self._calculate_accumulation_inflection(df, config)
        elif signal_name == 'PROCESS_META_BREAKOUT_ACCELERATION':
            meta_score = self._calculate_breakout_acceleration(df, config)
        elif signal_name == 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT':
            meta_score = self._calculate_fund_flow_accumulation_inflection(df, config)
        elif signal_name == 'PROCESS_META_STOCK_SECTOR_SYNC':
            relationship_score = self._calculate_stock_sector_sync(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_HOT_SECTOR_COOLING':
            relationship_score = self._calculate_hot_sector_cooling(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE':
            meta_score = self.calculate_price_momentum_divergence_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_STORM_EYE_CALM':
            meta_score = self.calculate_storm_eye_calm_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_WASH_OUT_REBOUND':
            offensive_absorption_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 0.0)
            meta_score = self._calculate_process_wash_out_rebound(df, offensive_absorption_intent, config)
        elif signal_name == 'PROCESS_META_COVERT_ACCUMULATION':
            meta_score = self.calculate_process_covert_accumulation_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_POWER_TRANSFER':
            # [新增] 显式调用新的权力交接计算逻辑
            meta_score = self._calculate_power_transfer(df, config)
        else:
            relationship_score = self._calculate_instantaneous_relationship(df, config)
            if relationship_score.empty:
                return {}
            self.strategy.atomic_states[f"PROCESS_ATOMIC_REL_SCORE_{signal_name}"] = relationship_score.astype(np.float32)
            diagnosis_mode = config.get('diagnosis_mode', 'direct_confirmation')
            if diagnosis_mode == 'direct_confirmation':
                meta_score = relationship_score
            else:
                meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        if meta_score.empty:
            return {}
        return {signal_name: meta_score}

    def _diagnose_split_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """【V4.0.0 · 裂变张量重构版】摒弃统包归一化，在分裂关系内部直接构建专属的动量和位移 HAB 存量冲击矩阵。"""
        states={}
        output_names=config.get('output_names',{})
        opportunity_signal_name=output_names.get('opportunity')
        risk_signal_name=output_names.get('risk')
        if not opportunity_signal_name or not risk_signal_name:
            return {}
        relationship_score=self._calculate_price_efficiency_relationship(df,config)
        if relationship_score.empty:
            return {}
        df_index=df.index
        relationship_displacement=relationship_score.diff(self.meta_window).fillna(0)
        relationship_momentum=relationship_displacement.diff(1).fillna(0)
        bipolar_displacement=np.tanh(self._apply_hab_shock(relationship_displacement,window=self.meta_window*2))
        bipolar_momentum=np.tanh(self._apply_hab_shock(relationship_momentum,window=13))
        displacement_weight=self.meta_score_weights[0]
        momentum_weight=self.meta_score_weights[1]
        meta_score=(bipolar_displacement*displacement_weight+bipolar_momentum*momentum_weight)
        meta_score=np.sign(meta_score)*(np.abs(meta_score)**1.5)
        meta_score=meta_score.clip(-1,1)
        states[opportunity_signal_name]=meta_score.clip(lower=0).astype(np.float32)
        states[risk_signal_name]=meta_score.clip(upper=0).abs().astype(np.float32)
        return states

    def _apply_hab_shock(self, series: pd.Series, window: int = 21) -> pd.Series:
        """【V2.0.0 · HAB存量冲击系统】将绝对数值转化为相对历史存量的冲击度(Z-Score)"""
        roll_mean = series.rolling(window=window, min_periods=1).mean()
        roll_std = series.rolling(window=window, min_periods=1).std().replace(0, 1e-5).fillna(1e-5)
        return ((series - roll_mean) / roll_std).astype(np.float32)

    def _get_kinematic_tensor(self, df: pd.DataFrame, base_col: str, period: int = 13, method_name: str = "") -> pd.Series:
        """
        【V2.1.0 · 运动学张力处理器 (修复降维Bug)】
        提取导数并利用死区门限与tanh滤除无穷小噪音。
        (修复: 使用 Pandas 原生 where 保留索引结构，避免降维为 ndarray)
        """
        slope=self._get_safe_series(df,f'SLOPE_{period}_{base_col}',0.0,method_name)
        accel=self._get_safe_series(df,f'ACCEL_{period}_{base_col}',0.0,method_name)
        raw_tensor=slope+accel*0.5
        gated_tensor=raw_tensor.where(raw_tensor.abs()>=1e-4,0.0)
        return np.tanh(gated_tensor*20.0).astype(np.float32)

    def _calculate_power_transfer(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0.0 · 权力交接动能透视版】
        计算"主力-散户"微观权力交接评分。
        """
        method_name="_calculate_power_transfer"
        required_signals=['net_mf_amount_D','amount_D','tick_large_order_net_D','tick_chip_transfer_efficiency_D','flow_efficiency_D','intraday_cost_center_migration_D','downtrend_strength_D','chip_concentration_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        net_mf=self._get_safe_series(df,'net_mf_amount_D',method_name=method_name)
        amount=self._get_safe_series(df,'amount_D',method_name=method_name).replace(0,np.nan)
        tick_large_net=self._get_safe_series(df,'tick_large_order_net_D',method_name=method_name)
        transfer_efficiency=self._get_safe_series(df,'tick_chip_transfer_efficiency_D',method_name=method_name)
        flow_efficiency=self._get_safe_series(df,'flow_efficiency_D',method_name=method_name)
        cost_migration=self._get_safe_series(df,'intraday_cost_center_migration_D',method_name=method_name)
        downtrend=self._get_safe_series(df,'downtrend_strength_D',method_name=method_name)
        chip_conc=self._get_safe_series(df,'chip_concentration_ratio_D',method_name=method_name)
        mf_ratio=net_mf/amount
        tick_ratio=tick_large_net/amount
        base_power=(mf_ratio*0.6+tick_ratio*0.4)*10.0
        power_tanh=np.tanh(base_power)
        efficiency_multiplier=1.0+(transfer_efficiency+flow_efficiency)/2.0
        chip_penetration=chip_conc.diff()*efficiency_multiplier
        raw_score=power_tanh*(1.0+chip_penetration.abs())
        trend_discount=pd.Series(1.0,index=df_index).mask(downtrend>0.8,0.6)
        final_score=(raw_score*trend_discount).clip(-1,1).astype(np.float32)
        if hasattr(self.strategy,'atomic_states'):
            self.strategy.atomic_states["PROCESS_DEBUG_power_transfer_spread"]=power_tanh
            self.strategy.atomic_states["PROCESS_DEBUG_power_transfer_penetration"]=chip_penetration
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'net_mf_amount_D':net_mf,'amount_D':amount,'tick_large_order_net_D':tick_large_net,'tick_chip_transfer_efficiency_D':transfer_efficiency,'flow_efficiency_D':flow_efficiency},calc_nodes={'mf_ratio':mf_ratio,'tick_ratio':tick_ratio,'base_power':base_power,'power_tanh':power_tanh,'efficiency_multiplier':efficiency_multiplier,'chip_penetration':chip_penetration,'raw_score':raw_score,'trend_discount':trend_discount},final_result=final_score)
        return final_score

    def _calculate_price_vs_capitulation_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 散户投降背离探针版】
        计算价格与散户割肉的微观博弈关系。
        """
        method_name = "_calculate_price_vs_capitulation_relationship"
        required_signals = [
            'pressure_trapped_D', 'INTRADAY_SUPPORT_INTENT_D', 
            'intraday_low_lock_ratio_D', 'chip_entropy_D',
            'SLOPE_13_pressure_trapped_D', 'ACCEL_13_pressure_trapped_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        pressure = self._get_safe_series(df, 'pressure_trapped_D', method_name=method_name)
        support = self._get_safe_series(df, 'INTRADAY_SUPPORT_INTENT_D', method_name=method_name)
        low_lock = self._get_safe_series(df, 'intraday_low_lock_ratio_D', method_name=method_name)
        entropy = self._get_safe_series(df, 'chip_entropy_D', method_name=method_name)
        kinematics_p = self._get_kinematic_tensor(df, 'pressure_trapped_D', 13, method_name)
        panic_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(pressure, 21)))
        support_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(support, 34)))
        absorption_resonance = support_norm * (1.0 + low_lock.clip(lower=0)) * (1.0 - np.tanh(entropy / 10.0))
        base_divergence = self._calculate_instantaneous_relationship(df, config)
        raw_score = base_divergence * panic_shock * absorption_resonance * (1.0 + kinematics_p.clip(lower=0))
        final_score = np.sign(raw_score) * (np.abs(raw_score) ** 1.5)
        final_score = np.tanh(final_score).clip(-1, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'pressure_trapped_D': pressure, 'INTRADAY_SUPPORT_INTENT_D': support}, calc_nodes={'kinematics_p': kinematics_p, 'panic_shock': panic_shock, 'absorption_resonance': absorption_resonance}, final_result=final_score)
        return final_score

    def _calculate_price_efficiency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 价格效率博弈探针版】
        结合多时间框架效率、主力净额以及换手效率计算定价动能。
        """
        method_name = "_calculate_price_efficiency_relationship"
        required_signals = [
            'VPA_EFFICIENCY_D', 'net_mf_amount_D', 'shakeout_score_D', 
            'tick_chip_transfer_efficiency_D', 'high_freq_flow_skewness_D',
            'SLOPE_13_VPA_EFFICIENCY_D', 'ACCEL_13_VPA_EFFICIENCY_D'
        ]
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
        synergy = eff_shock * mf_conviction * (1.0 + transfer_eff) * (1.0 + kinematics_eff) * (1.0 + np.tanh(flow_skew).clip(lower=0))
        final_score = (synergy * (1.0 - shakeout_penalty ** 1.5)).clip(-1, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'VPA_EFFICIENCY_D': eff, 'net_mf_amount_D': net_mf}, calc_nodes={'kinematics_eff': kinematics_eff, 'eff_shock': eff_shock, 'mf_conviction': mf_conviction, 'synergy': synergy}, final_result=final_score)
        return final_score

    def _calculate_pd_divergence_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 博弈背离深度探针版】
        结合筹码熵、均价成本、高频胜率与游戏激烈度构筑全息张量。
        """
        method_name = "_calculate_pd_divergence_relationship"
        required_signals = [
            'game_intensity_D', 'weight_avg_cost_D', 'close_D', 
            'intraday_chip_game_index_D', 'chip_divergence_ratio_D', 'winner_rate_D',
            'SLOPE_13_game_intensity_D', 'ACCEL_13_game_intensity_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        game = self._get_safe_series(df, 'game_intensity_D', method_name=method_name)
        cost = self._get_safe_series(df, 'weight_avg_cost_D', method_name=method_name).replace(0, np.nan)
        close_p = self._get_safe_series(df, 'close_D', method_name=method_name)
        intra_game = self._get_safe_series(df, 'intraday_chip_game_index_D', method_name=method_name)
        chip_div = self._get_safe_series(df, 'chip_divergence_ratio_D', method_name=method_name)
        winner = self._get_safe_series(df, 'winner_rate_D', method_name=method_name)
        kinematics_game = self._get_kinematic_tensor(df, 'game_intensity_D', 13, method_name)
        game_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(game, 55)))
        price_adv = np.tanh((close_p - cost) / cost * 10.0)
        win_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(winner, 21)))
        base_divergence = self._calculate_instantaneous_relationship(df, config)
        tensor_resonance = game_shock * price_adv * win_norm * (1.0 + intra_game) * (1.0 + chip_div)
        final_score = np.tanh(base_divergence * tensor_resonance * (1.0 + kinematics_game.clip(lower=0))).clip(-1, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'game_intensity_D': game, 'weight_avg_cost_D': cost}, calc_nodes={'kinematics_game': kinematics_game, 'game_shock': game_shock, 'price_adv': price_adv, 'tensor_resonance': tensor_resonance}, final_result=final_score)
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
        【V2.0 · 神谕调度版】通用领域反转诊断调度中心
        - 核心升级: 剥离旧的、有缺陷的计算逻辑，转变为一个纯粹的“调度中心”。
        - 核心职责: 1. 计算各领域的“领域健康度(bipolar_domain_health)”。
                      2. 将健康度数据呈送给新建的 `_judge_domain_reversal` 方法进行最终审判。
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
            # 增加对公理信号是否存在的防御性检查
            if axiom_name not in self.strategy.atomic_states:
                print(f"    -> [过程情报警告] 领域 '{domain_name}' 依赖的公理信号 '{axiom_name}' 不存在，跳过此公理。")
                continue
            axiom_score = self.strategy.atomic_states.get(axiom_name, pd.Series(0.0, index=df_index))
            domain_health_components.append(axiom_score * axiom_weight)
            total_weight += abs(axiom_weight) # 使用绝对值权重总和，以正确处理负权重
        if total_weight == 0:
            print(f"        -> [领域反转诊断] 警告: 领域 '{domain_name}' 的公理权重总和为0，无法计算健康度。")
            return {}
        # 计算该领域的双极性健康度
        bipolar_domain_health = (sum(domain_health_components) / total_weight).clip(-1, 1)
        # 将健康度呈送给新的“神谕审判”方法进行最终裁决
        return self._judge_domain_reversal(bipolar_domain_health, config)

    def _judge_domain_reversal(self, bipolar_domain_health: pd.Series, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.1 · 神谕审判生产版】领域反转信号的核心审判庭
        - 核心模型: 创立基于“情境感知”的“神谕审判”模型，取代旧的、情境盲目的归一化逻辑。
        - 底部反转神谕: `底部反转分 = 健康度正向变化量 × (1 - 昨日健康度)`
        - 顶部反转神谕: `顶部反转分 = 健康度负向变化量(取绝对值) × (1 + 昨日健康度)`
        """
        domain_name = config.get('domain_name', '未知领域')
        output_bottom_name = config.get('output_bottom_reversal_name')
        output_top_name = config.get('output_top_reversal_name')
        # 计算核心变量
        health_yesterday = bipolar_domain_health.shift(1).fillna(0)
        health_change = bipolar_domain_health.diff(1).fillna(0)
        # 底部反转神谕
        bottom_context_factor = (1 - health_yesterday).clip(0, 2) # 情境调节器：昨日越差，反转越有价值
        bottom_reversal_raw = health_change.clip(lower=0) * bottom_context_factor
        bottom_reversal_score = bottom_reversal_raw.clip(0, 1) # 直接裁剪，其值已具意义
        # 顶部反转神谕
        top_context_factor = (1 + health_yesterday).clip(0, 2) # 情境调节器：昨日越好，反转越危险
        top_reversal_raw = health_change.clip(upper=0).abs() * top_context_factor
        top_reversal_score = top_reversal_raw.clip(0, 1) # 直接裁剪
        # [删除] 移除所有“究极探针”调试代码
        return {
            output_bottom_name: bottom_reversal_score.astype(np.float32),
            output_top_name: top_reversal_score.astype(np.float32)
        }

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V6.0.0 · 恐慌洗盘共振探针版】
        计算“恐慌洗盘吸筹”的专属非线性信号。
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
        panic_intensity=np.sqrt(panic_level.abs()*low_lock_ratio.clip(lower=0))
        absorption_intensity=np.sqrt(absorption.abs()*trough_filling.clip(lower=0))
        divergence_bonus=1.0+np.clip(hff_div,0,1)+np.clip(chip_div,0,1)
        base_score=panic_intensity*absorption_intensity*divergence_bonus
        historical_potential_gate=config.get('historical_potential_gate',0.2)
        gate_mask=chip_stab>historical_potential_gate
        final_score=base_score.where(gate_mask,0.0).clip(0,1).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_trapped_D':panic_level,'intraday_low_lock_ratio_D':low_lock_ratio,'absorption_energy_D':absorption,'intraday_trough_filling_degree_D':trough_filling,'high_freq_flow_divergence_D':hff_div,'chip_rsi_divergence_D':chip_div},calc_nodes={'panic_intensity':panic_intensity,'absorption_intensity':absorption_intensity,'divergence_bonus':divergence_bonus,'base_score':base_score,'gate_mask':pd.Series(gate_mask,index=df_index)},final_result=final_score)
        return final_score

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.0.0 · 诡道偏度背离探针版】
        利用资金特征与微观偏移计算“诡道吸筹”信号。
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
        skew_divergence=flow_skew-price_skew
        skew_tension=skew_divergence.clip(lower=0)
        stealth_strength=stealth_flow*tick_clustering*flow_intensity
        camouflage_index=np.clip(price_flow_div,0,1)+skew_tension*0.5
        raw_deception=stealth_strength*camouflage_index
        final_score=np.tanh(raw_deception).clip(0,1).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'stealth_flow_ratio_D':stealth_flow,'tick_clustering_index_D':tick_clustering,'intraday_price_distribution_skewness_D':price_skew,'high_freq_flow_skewness_D':flow_skew,'price_flow_divergence_D':price_flow_div,'chip_flow_intensity_D':flow_intensity},calc_nodes={'skew_divergence':skew_divergence,'skew_tension':skew_tension,'stealth_strength':stealth_strength,'camouflage_index':camouflage_index,'raw_deception':raw_deception},final_result=final_score)
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
        【V4.0.0 · 绝地反击锚定探针版】
        计算底部杀跌势能竭力时的“套牢盘投降”确认信号。
        """
        method_name="_calculate_loser_capitulation"
        required_signals=['pressure_release_index_D','pressure_trapped_D','intraday_low_lock_ratio_D','absorption_energy_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        pressure_release=self._get_safe_series(df,'pressure_release_index_D',method_name=method_name)
        pressure_trapped=self._get_safe_series(df,'pressure_trapped_D',method_name=method_name)
        low_lock=self._get_safe_series(df,'intraday_low_lock_ratio_D',method_name=method_name)
        absorption=self._get_safe_series(df,'absorption_energy_D',method_name=method_name)
        panic_extremum=np.sqrt(pressure_release.clip(lower=0)*pressure_trapped.clip(lower=0))
        absorption_anchor=np.sqrt(low_lock.clip(lower=0)*absorption.clip(lower=0))
        final_score=(panic_extremum*absorption_anchor).clip(0,1).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'pressure_release_index_D':pressure_release,'pressure_trapped_D':pressure_trapped,'intraday_low_lock_ratio_D':low_lock,'absorption_energy_D':absorption},calc_nodes={'panic_extremum':panic_extremum,'absorption_anchor':absorption_anchor},final_result=final_score)
        return final_score

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 突破爆发加速度探针版】
        融入行业强度、资金连续性与异常高频放量，构建非线性突破张量矩阵。
        """
        method_name = "_calculate_breakout_acceleration"
        required_signals = [
            'breakout_confidence_D', 'industry_strength_rank_D', 'net_mf_amount_D', 
            'flow_consistency_D', 'tick_abnormal_volume_ratio_D', 'uptrend_strength_D',
            'SLOPE_13_breakout_confidence_D', 'ACCEL_13_net_mf_amount_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        breakout = self._get_safe_series(df, 'breakout_confidence_D', method_name=method_name)
        industry = self._get_safe_series(df, 'industry_strength_rank_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        consistency = self._get_safe_series(df, 'flow_consistency_D', method_name=method_name)
        abnormal_vol = self._get_safe_series(df, 'tick_abnormal_volume_ratio_D', method_name=method_name)
        uptrend = self._get_safe_series(df, 'uptrend_strength_D', method_name=method_name)
        kinematics_brk = self._get_kinematic_tensor(df, 'breakout_confidence_D', 13, method_name)
        kinematics_mf = self._get_kinematic_tensor(df, 'net_mf_amount_D', 13, method_name)
        mf_shock = np.tanh(self._apply_hab_shock(net_mf, 34))
        mf_power = np.sign(mf_shock) * (np.abs(mf_shock) ** 1.5)
        ind_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(industry, 55)))
        abnorm_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(abnormal_vol, 21)))
        base_tensor = breakout * ind_norm * (1.0 + mf_power.clip(lower=0))
        catalyst = (consistency * abnorm_norm * uptrend) ** 1.5
        raw_score = base_tensor * catalyst * (1.0 + kinematics_brk.clip(lower=0) + kinematics_mf.clip(lower=0))
        final_score = np.tanh(raw_score).clip(0, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'breakout_confidence_D': breakout, 'tick_abnormal_volume_ratio_D': abnormal_vol}, calc_nodes={'kinematics_brk': kinematics_brk, 'mf_power': mf_power, 'catalyst': catalyst, 'raw_score': raw_score}, final_result=final_score)
        return final_score

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 资金流吸筹质变探针版】
        检测吸筹缓冲池溢出并转化为上攻净流的加速度拐点。
        """
        method_name = "_calculate_fund_flow_accumulation_inflection"
        required_signals = [
            'accumulation_signal_score_D', 'net_mf_amount_D', 'flow_efficiency_D', 
            'tick_large_order_net_D', 'intraday_accumulation_confidence_D',
            'SLOPE_21_accumulation_signal_score_D', 'ACCEL_21_net_mf_amount_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        acc_score = self._get_safe_series(df, 'accumulation_signal_score_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        flow_eff = self._get_safe_series(df, 'flow_efficiency_D', method_name=method_name)
        large_net = self._get_safe_series(df, 'tick_large_order_net_D', method_name=method_name)
        intra_acc = self._get_safe_series(df, 'intraday_accumulation_confidence_D', method_name=method_name)
        kinematics_mf = self._get_kinematic_tensor(df, 'net_mf_amount_D', 21, method_name)
        kinematics_acc = self._get_kinematic_tensor(df, 'accumulation_signal_score_D', 21, method_name)
        acc_shock = np.tanh(self._apply_hab_shock(acc_score, 55)).clip(lower=0)
        mf_shock = np.tanh(self._apply_hab_shock(net_mf, 34))
        eff_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(flow_eff, 21)))
        large_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(large_net, 21)))
        base_ignition = acc_shock * eff_norm * large_norm * intra_acc
        synergy_thrust = base_ignition * (1.0 + mf_shock.clip(lower=0) ** 1.5)
        raw_score = synergy_thrust * (1.0 + kinematics_acc.clip(lower=0) + kinematics_mf.clip(lower=0))
        final_score = np.tanh(raw_score * 2.0).clip(0, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'accumulation_signal_score_D': acc_score, 'net_mf_amount_D': net_mf}, calc_nodes={'kinematics_mf': kinematics_mf, 'acc_shock': acc_shock, 'eff_norm': eff_norm, 'synergy_thrust': synergy_thrust}, final_result=final_score)
        return final_score

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 获利压迫与净流对冲张力版】
        利用流向与一致性消化获利派发压力的全息非线性张量对抗。
        """
        method_name = "_calculate_profit_vs_flow_relationship"
        required_signals = [
            'profit_pressure_D', 'net_mf_amount_D', 'profit_ratio_D', 
            'flow_consistency_D', 'winner_rate_D', 'intraday_distribution_confidence_D',
            'SLOPE_13_profit_pressure_D', 'ACCEL_13_profit_pressure_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        pressure = self._get_safe_series(df, 'profit_pressure_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        profit_ratio = self._get_safe_series(df, 'profit_ratio_D', method_name=method_name)
        cons = self._get_safe_series(df, 'flow_consistency_D', method_name=method_name)
        winner = self._get_safe_series(df, 'winner_rate_D', method_name=method_name)
        dist_conf = self._get_safe_series(df, 'intraday_distribution_confidence_D', method_name=method_name)
        kinematics_p = self._get_kinematic_tensor(df, 'profit_pressure_D', 13, method_name)
        pressure_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(pressure, 21)))
        mf_shock = np.tanh(self._apply_hab_shock(net_mf, 34))
        win_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(winner, 55)))
        pressure_tensor = (pressure_shock * profit_ratio * dist_conf) * (1.0 + kinematics_p.clip(lower=0))
        support_tensor = (mf_shock.clip(lower=0) * cons) * (1.0 - win_norm * 0.5)
        raw_score = support_tensor - pressure_tensor * 1.5
        final_score = np.sign(raw_score) * (np.abs(raw_score) ** 1.5)
        final_score = np.tanh(final_score).clip(-1, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'profit_pressure_D': pressure, 'net_mf_amount_D': net_mf, 'profit_ratio_D': profit_ratio}, calc_nodes={'kinematics_p': kinematics_p, 'pressure_shock': pressure_shock, 'mf_shock': mf_shock, 'pressure_tensor': pressure_tensor, 'support_tensor': support_tensor}, final_result=final_score)
        return final_score

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 个股板块协同共振探针版】
        板块势能与个股微观流动性构成的正交矩阵共振。
        """
        method_name = "_calculate_stock_sector_sync"
        required_signals = [
            'pct_change_D', 'industry_strength_rank_D', 'net_mf_amount_D', 
            'flow_consistency_D', 'industry_leader_score_D', 'mid_long_sync_D',
            'SLOPE_5_industry_strength_rank_D', 'ACCEL_5_industry_strength_rank_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        pct = self._get_safe_series(df, 'pct_change_D', method_name=method_name)
        rank = self._get_safe_series(df, 'industry_strength_rank_D', method_name=method_name)
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        cons = self._get_safe_series(df, 'flow_consistency_D', method_name=method_name)
        leader = self._get_safe_series(df, 'industry_leader_score_D', method_name=method_name)
        sync_score = self._get_safe_series(df, 'mid_long_sync_D', method_name=method_name)
        kinematics_rank = self._get_kinematic_tensor(df, 'industry_strength_rank_D', 5, method_name)
        stock_shock = np.tanh(self._apply_hab_shock(pct, 13))
        sector_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(rank, 34)))
        mf_norm = np.tanh(self._apply_hab_shock(net_mf, 21))
        cons_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(cons, 21)))
        sector_tensor = sector_shock * (1.0 + kinematics_rank) * (1.0 + leader)
        flow_tensor = mf_norm * cons_norm * (1.0 + sync_score)
        resonance = stock_shock * sector_tensor * np.abs(flow_tensor)
        final_score = (np.sign(resonance) * (np.abs(resonance) ** 1.5)).clip(-1, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'pct_change_D': pct, 'industry_strength_rank_D': rank}, calc_nodes={'kinematics_rank': kinematics_rank, 'stock_shock': stock_shock, 'sector_shock': sector_shock, 'resonance': resonance}, final_result=final_score)
        return final_score

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 热门板块退潮探针版】
        捕捉高热度动能耗散下的资金连续净流出反转模型。
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
        outflow_tensor = np.abs(mf_shock.clip(upper=0)) * (1.0 + outflow_q)
        stagnation_boost = 1.0 + np.tanh(stagnation * 2.0)
        cooling_resonance = hot_shock * outflow_tensor * stagnation_boost * (1.0 - kinematics_hot.clip(lower=0))
        final_score = np.tanh(cooling_resonance ** 1.5).clip(0, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'THEME_HOTNESS_SCORE_D': hot, 'net_mf_amount_D': net_mf}, calc_nodes={'kinematics_hot': kinematics_hot, 'hot_shock': hot_shock, 'outflow_tensor': outflow_tensor, 'cooling_resonance': cooling_resonance}, final_result=final_score)
        return final_score

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 价资协同非线性元分析版】
        价资双极背离博弈，将绝对动量推演与Z-Tanh映射完全内建。
        """
        method_name = "_calculate_pf_relationship"
        required_signals = [
            'net_mf_amount_D', 'close_D', 'price_vs_ma_13_ratio_D',
            'SLOPE_13_net_mf_amount_D', 'ACCEL_13_net_mf_amount_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        net_mf = self._get_safe_series(df, 'net_mf_amount_D', method_name=method_name)
        close_p = self._get_safe_series(df, 'close_D', method_name=method_name)
        price_ma_ratio = self._get_safe_series(df, 'price_vs_ma_13_ratio_D', method_name=method_name)
        kinematics_mf = self._get_kinematic_tensor(df, 'net_mf_amount_D', 13, method_name)
        mf_shock = np.tanh(self._apply_hab_shock(net_mf, 34))
        momentum_p = np.tanh((close_p.diff(1).fillna(0)) / close_p.rolling(21).std().replace(0, 1e-5))
        thrust_f = mf_shock * (1.0 + np.abs(kinematics_mf)) * (1.0 + price_ma_ratio)
        relationship_score = pd.Series(np.sign(momentum_p + thrust_f) * np.sqrt(np.abs(momentum_p * thrust_f)), index=df_index)
        meta_score = self._perform_meta_analysis_on_score(relationship_score.fillna(0.0), config, df, df_index)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'net_mf_amount_D': net_mf, 'close_D': close_p}, calc_nodes={'kinematics_mf': kinematics_mf, 'mf_shock': mf_shock, 'relationship_score': relationship_score}, final_result=meta_score)
        return meta_score

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 价筹共振非线性元分析版】
        剥离历史包袱，用微积分推演筹码群的绝对协同转移推力。
        """
        method_name = "_calculate_pc_relationship"
        required_signals = [
            'peak_concentration_D', 'close_D', 'chip_convergence_ratio_D',
            'SLOPE_13_peak_concentration_D', 'ACCEL_13_peak_concentration_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        peak_c = self._get_safe_series(df, 'peak_concentration_D', method_name=method_name)
        close_p = self._get_safe_series(df, 'close_D', method_name=method_name)
        convergence = self._get_safe_series(df, 'chip_convergence_ratio_D', method_name=method_name)
        kinematics_pc = self._get_kinematic_tensor(df, 'peak_concentration_D', 13, method_name)
        pc_shock = np.tanh(self._apply_hab_shock(peak_c, 34))
        momentum_p = np.tanh((close_p.diff(1).fillna(0)) / close_p.rolling(21).std().replace(0, 1e-5))
        thrust_c = pc_shock * (1.0 + np.abs(kinematics_pc)) * (1.0 + convergence)
        relationship_score = pd.Series(np.sign(momentum_p + thrust_c) * np.sqrt(np.abs(momentum_p * thrust_c)), index=df_index)
        meta_score = self._perform_meta_analysis_on_score(relationship_score.fillna(0.0), config, df, df_index)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'peak_concentration_D': peak_c, 'close_D': close_p}, calc_nodes={'kinematics_pc': kinematics_pc, 'pc_shock': pc_shock, 'relationship_score': relationship_score}, final_result=meta_score)
        return meta_score

    def _perform_meta_analysis_on_score(self, relationship_score: pd.Series, config: Dict, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """【V4.0.0 · 元动力学非线性整合版】结合 HAB 冲击测算推演信号位移与动量的绝对极值，执行 Power Law 非线性增益放大。"""
        signal_name=config.get('name')
        relationship_displacement=relationship_score.diff(self.meta_window).fillna(0)
        relationship_momentum=relationship_displacement.diff(1).fillna(0)
        bipolar_displacement=np.tanh(self._apply_hab_shock(relationship_displacement,window=self.meta_window*2))
        bipolar_momentum=np.tanh(self._apply_hab_shock(relationship_momentum,window=13))
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
        if scoring_mode=='unipolar':
            meta_score=meta_score.clip(lower=0)
        return meta_score.clip(-1,1).astype(np.float32)

    def _calculate_ff_vs_structure_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 资金结构协同战线版】
        对齐长效结构坚固度与流动资金可信度，防止筹码离散导致的虚假突破。
        """
        method_name = "_calculate_ff_vs_structure_relationship"
        required_signals = [
            'uptrend_strength_D', 'flow_consistency_D', 
            'ma_arrangement_status_D', 'chip_structure_state_D',
            'SLOPE_13_uptrend_strength_D', 'ACCEL_13_uptrend_strength_D'
        ]
        self._validate_required_signals(df, required_signals, method_name)
        df_index = df.index
        struct = self._get_safe_series(df, 'uptrend_strength_D', method_name=method_name)
        cons = self._get_safe_series(df, 'flow_consistency_D', method_name=method_name)
        ma_status = self._get_safe_series(df, 'ma_arrangement_status_D', method_name=method_name)
        chip_struct = self._get_safe_series(df, 'chip_structure_state_D', method_name=method_name)
        kinematics_struct = self._get_kinematic_tensor(df, 'uptrend_strength_D', 13, method_name)
        struct_shock = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(struct, 34)))
        cons_norm = 0.5 * (1.0 + np.tanh(self._apply_hab_shock(cons, 21)))
        base_divergence = self._calculate_instantaneous_relationship(df, config)
        amplifier = 1.0 + (struct_shock * cons_norm * (1.0 + ma_status * 0.5 + chip_struct * 0.5) * (1.0 + np.abs(kinematics_struct)))
        final_score = np.tanh(base_divergence * amplifier).clip(-1, 1).astype(np.float32)
        self._probe_variables(method_name=method_name, df_index=df_index, raw_inputs={'uptrend_strength_D': struct, 'flow_consistency_D': cons}, calc_nodes={'kinematics_struct': kinematics_struct, 'struct_shock': struct_shock, 'amplifier': amplifier}, final_result=final_score)
        return final_score

    def _calculate_dyn_vs_chip_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.1.0 · 动能筹码分歧原生审判版】
        执行派发审查。同步清除了内部 np.where 的张量降级隐患。
        """
        method_name="_calculate_dyn_vs_chip_relationship"
        required_signals=[
            'ROC_13_D','winner_rate_D','profit_ratio_D','chip_mean_D',
            'SLOPE_13_ROC_13_D','ACCEL_13_ROC_13_D'
        ]
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        roc=self._get_safe_series(df,'ROC_13_D',method_name=method_name)
        win=self._get_safe_series(df,'winner_rate_D',method_name=method_name)
        profit=self._get_safe_series(df,'profit_ratio_D',method_name=method_name)
        chip_mean=self._get_safe_series(df,'chip_mean_D',method_name=method_name)
        kinematics_roc=self._get_kinematic_tensor(df,'ROC_13_D',13,method_name)
        base_consensus=self._calculate_instantaneous_relationship(df,config)
        profit_shock=0.5*(1.0+np.tanh(self._apply_hab_shock(profit,55)))
        win_shock=np.tanh(self._apply_hab_shock(win,21))
        distribution_pressure=1.0+(profit_shock*np.abs(kinematics_roc))*(1.0+chip_mean.pct_change().fillna(0).abs())
        final_score=base_consensus.where(base_consensus>=0,base_consensus*distribution_pressure*(1.0-win_shock*0.5))
        final_score=np.sign(final_score)*(np.abs(final_score)**1.5)
        final_score=np.tanh(final_score).clip(-1,1).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'ROC_13_D':roc,'winner_rate_D':win},calc_nodes={'kinematics_roc':kinematics_roc,'profit_shock':profit_shock,'distribution_pressure':distribution_pressure},final_result=final_score)
        return final_score

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 · 瞬时关系共振张量版】
        重构底层A与B特征张量对冲，支持HAB存量意识与Power Law指数级共振。
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
        change_a = signal_a.diff(1).fillna(0) if config.get('change_type_A', 'pct') == 'diff' else ta.percent_return(signal_a, length=1).fillna(0)
        change_b = signal_b.diff(1).fillna(0) if config.get('change_type_B', 'pct') == 'diff' else ta.percent_return(signal_b, length=1).fillna(0)
        momentum_a = np.tanh(self._apply_hab_shock(change_a, 13))
        thrust_b = np.tanh(self._apply_hab_shock(change_b, 13))
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        if relationship_type == 'divergence':
            relationship_score = (signal_b_factor_k * thrust_b - momentum_a) / (signal_b_factor_k + 1.0)
        else:
            force_vector_sum = momentum_a + signal_b_factor_k * thrust_b
            magnitude = (momentum_a.abs() * thrust_b.abs()).pow(0.5)
            relationship_score = np.sign(force_vector_sum) * magnitude
        return (np.sign(relationship_score) * (np.abs(relationship_score) ** 1.5)).clip(-1, 1).fillna(0.0).astype(np.float32)

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, offensive_absorption_intent: pd.Series, config: Dict) -> pd.Series:
        """
        【V4.0.0 · 虚假派发与承接反包探针版】
        识别主力利用极度洗盘环境后进行强力反抽的技术战术。
        """
        method_name="_calculate_process_wash_out_rebound"
        required_signals=['shakeout_score_D','intraday_distribution_confidence_D','intraday_trough_filling_degree_D','intraday_resistance_test_count_D','closing_flow_intensity_D','short_term_chip_ratio_D']
        self._validate_required_signals(df,required_signals,method_name)
        df_index=df.index
        shakeout=self._get_safe_series(df,'shakeout_score_D',method_name=method_name)
        fake_dist=self._get_safe_series(df,'intraday_distribution_confidence_D',method_name=method_name)
        trough_fill=self._get_safe_series(df,'intraday_trough_filling_degree_D',method_name=method_name)
        resist_test=self._get_safe_series(df,'intraday_resistance_test_count_D',method_name=method_name)
        closing_intensity=self._get_safe_series(df,'closing_flow_intensity_D',method_name=method_name)
        short_chip=self._get_safe_series(df,'short_term_chip_ratio_D',method_name=method_name)
        washout_context=shakeout*fake_dist
        rebound_intent=np.sqrt(trough_fill.abs()*(resist_test/10.0).clip(upper=1.0))
        chip_penalty=1.0-np.clip(short_chip,0,0.5)
        raw_score=washout_context*rebound_intent*closing_intensity*chip_penalty
        final_score=np.tanh(raw_score*2.0).clip(0,1).astype(np.float32)
        self._probe_variables(method_name=method_name,df_index=df_index,raw_inputs={'shakeout_score_D':shakeout,'intraday_distribution_confidence_D':fake_dist,'intraday_trough_filling_degree_D':trough_fill,'intraday_resistance_test_count_D':resist_test,'closing_flow_intensity_D':closing_intensity,'short_term_chip_ratio_D':short_chip},calc_nodes={'washout_context':washout_context,'rebound_intent':rebound_intent,'chip_penalty':chip_penalty,'raw_score':raw_score},final_result=final_score)
        return final_score





