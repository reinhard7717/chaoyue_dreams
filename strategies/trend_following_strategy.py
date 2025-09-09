# 文件: strategies/trend_following_strategy.py
# 版本: V100.1 - 初始化顺序与逻辑净化修复版
import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import gc # [代码新增] 导入gc模块以支持内存回收

from .trend_following.intelligence_layer import IntelligenceLayer
from .trend_following.offensive_layer import OffensiveLayer
from .trend_following.warning_layer import WarningLayer
from .trend_following.exit_layer import ExitLayer
from .trend_following.judgment_layer import JudgmentLayer
from .trend_following.simulation_layer import SimulationLayer
from .trend_following.reporting_layer import ReportingLayer
from .trend_following.intelligence.dynamic_mechanics_engine import DynamicMechanicsEngine
from .trend_following.utils import ensure_numeric_types, optimize_df_memory, get_param_value, get_params_block

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    趋势跟踪策略 (V100.1 - 初始化顺序与逻辑净化修复版)
    - 核心修复: 调整了构造函数中共享状态属性的初始化顺序，确保它们在被子模块
                使用前已经被创建，从根本上解决了 AttributeError。
    - 逻辑净化: 移除了对已被重构掉的方法的调用，并清理了未使用的中间属性。
    """
    def __init__(self, config: dict):
        self.unified_config = config
        self.strategy_info = get_params_block(self, 'strategy_info')
        self.params = {}
        self.atomic_states = {}
        self.playbook_states = {}
        self.setup_scores = {}
        self.trigger_events = {}
        self.df_indicators = pd.DataFrame()
        self.df = pd.DataFrame()
        # 初始化所有分层模块，并将主策略实例(self)传递给它们，以便共享配置和状态
        self.intelligence_layer = IntelligenceLayer(self)
        self.offensive_layer = OffensiveLayer(self)
        self.warning_layer = WarningLayer(self)
        self.exit_layer = ExitLayer(self)
        self.judgment_layer = JudgmentLayer(self)
        self.simulation_layer = SimulationLayer(self)
        self.reporting_layer = ReportingLayer(self)
        self.mechanics_engine = DynamicMechanicsEngine(self)

    def apply_strategy(self, df: pd.DataFrame, params: dict, start_date_str: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V401.1 风险接口适配版】策略应用主流程
        - 核心重构: 重构了内部指挥链，以适配 ExitLayer 和 WarningLayer 的新职责。
        - 风险评估统一化: 现在由 WarningLayer 统一分析所有风险，不再区分“致命”和“常规”。
        - 硬性离场分离: ExitLayer 现在独立生成技术性离场信号，并在决策流程最后进行合并，确保其最高优先级。
        - 本次适配: 调用 WarningLayer 的新方法 `run_all_warnings` 并正确处理其四个返回值。
        """
        self.params = params
        if df is None or df.empty:
            # 如果输入数据为空，直接返回空的DataFrame，避免后续错误
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df_for_calculation = df
        if start_date_str:
            try:
                # 根据传入的起始日期过滤数据，用于增量计算或调试
                start_date = pd.to_datetime(start_date_str).date()
                df_filtered = df[df.index.date >= start_date].copy()
                if df_filtered.empty:
                    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                df_for_calculation = df_filtered
            except (ValueError, TypeError) as e:
                logger.error(f"无效的起始日期格式: '{start_date_str}'。错误: {e}。将处理全部历史记录。")
        # 确保数据类型正确，并将其存入实例属性，供所有子模块访问
        self.df_indicators = ensure_numeric_types(df_for_calculation)
        # --- 指挥链 1/7: 基础情报层 ---
        # 运行所有情报诊断模块，生成所有原子信号和触发器
        print("  [指挥链 1/7] 正在运行情报层...")
        self.trigger_events = self.intelligence_layer.run_all_diagnostics()
        # --- 指挥链 2/7: 进攻层 ---
        # 基于情报层的输出，计算总的进攻得分
        print("  [指挥链 2/7] 正在运行进攻层...")
        entry_score, score_details_df = self.offensive_layer.calculate_entry_score(self.trigger_events)
        self.df_indicators['entry_score'] = entry_score
        # --- 指挥链 3/7: 预警层 (统一风险分析中心) ---
        # 调用新的 `run_all_warnings` 方法，它不再计算风险，而是分析由认知层计算好的风险
        print("  [指挥链 3/7] 正在运行预警层...")
        risk_score, risk_details_df, risk_momentum, risk_dynamics = self.warning_layer.run_all_warnings()
        # 将分析结果存入实例属性，供下游模块使用
        self.df_indicators['risk_score'] = risk_score # 融合后的风险总分
        self.risk_score = risk_score # 兼容旧的属性访问
        self.risk_details_df = risk_details_df # 各维度风险分详情
        self.risk_momentum = risk_momentum # 风险动量（斜率、加速度）分析结果
        self.risk_dynamics = risk_dynamics # 风险动态（新增、持续、解决）分析结果
        # --- 指挥链 4/7: 统合判断层 ---
        # JudgmentLayer 接收进攻详情和统一后的风险详情，进行纯数值化决策
        print("  [指挥链 4/7] 正在运行判断层...")
        self.judgment_layer.make_final_decisions(score_details_df, risk_details_df)
        # --- 指挥链 5/7: 离场层 (生成独立的硬性离场信号) ---
        # ExitLayer 现在只生成技术性离场信号，不参与计分
        print("  [指挥链 5/7] 正在运行离场层...")
        hard_exit_triggers_df = self.exit_layer.generate_hard_exit_triggers()
        # 将硬性离场信号合并到主DataFrame中，确保其最终执行力
        is_hard_exit_triggered = hard_exit_triggers_df.any(axis=1)
        self.df_indicators.loc[is_hard_exit_triggered, 'signal_type'] = '卖出信号'
        # 将硬性离场详情也保存起来，用于调试和记录
        self.exit_triggers = hard_exit_triggers_df 
        # --- 指挥链 6/7 & 7/7: 模拟层与报告层 ---
        print("  [指挥链 6/7] 正在运行模拟层...")
        self.simulation_layer.run_position_management_simulation()
        print("  [指挥链 7/7] 正在运行报告层...")
        self.df_indicators = optimize_df_memory(self.df_indicators, verbose=False)
        
        # 手动进行垃圾回收，清理临时变量，优化内存使用
        try:
            del entry_score, risk_score, risk_details_df, risk_momentum, risk_dynamics
            gc.collect()
        except NameError:
            pass
        # 返回主分析结果、进攻得分详情、风险得分详情
        return self.df_indicators, score_details_df, self.risk_details_df

    async def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> Tuple[List, List, List, List, List]:
        """
        【V507.0 全景沙盘版】对外暴露的报告生成接口。
        - 返回值变更: 现在返回一个包含五类对象的元组，新增了 StrategyDailyState 列表。
        """
        return await self.reporting_layer.prepare_db_records(stock_code, result_df, score_details_df, risk_details_df, params, result_timeframe)











