# 文件: strategies/trend_following_strategy.py
# 版本: V101.0 · 终极信号适配版
import logging
import pandas as pd
from typing import Dict, Optional, Tuple, List

from .trend_following.intelligence_layer import IntelligenceLayer
from .trend_following.offensive_layer import OffensiveLayer
from .trend_following.warning_layer import WarningLayer
from .trend_following.structural_defense_layer import StructuralDefenseLayer
from .trend_following.judgment_layer import JudgmentLayer
from .trend_following.simulation_layer import SimulationLayer
from .trend_following.reporting_layer import ReportingLayer
from .trend_following.utils import ensure_numeric_types, optimize_df_memory, get_params_block

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    【V101.0 · 终极信号适配版】
    - 核心重构: 净化了 apply_strategy 的主流程，现在完全依赖 IntelligenceLayer 来完成所有情报的生成与融合。
    """
    def __init__(self, orchestrator_instance, strategy_config: dict):
        """
        【V103.0 · 主权配置协议版】
        - 初始化方法现在接收一份由总指挥分发的、纯净的专属配置 (strategy_config)。
        - 收益: 策略单元不再需要关心配置的来源和净化过程，实现了更高的内聚和独立性。
        """
        self.orchestrator = orchestrator_instance
        # 不再从总指挥处继承，而是使用注入的专属配置
        self.unified_config = strategy_config
        self.params = {}
        self.atomic_states = {}
        self.playbook_states = {}
        self.trigger_events = {}
        self.df_indicators = pd.DataFrame()
        self.intelligence_layer = IntelligenceLayer(self)
        self.offensive_layer = OffensiveLayer(self)
        self.warning_layer = WarningLayer(self)
        self.structural_defense_layer = StructuralDefenseLayer(self)
        self.judgment_layer = JudgmentLayer(self)
        self.simulation_layer = SimulationLayer(self)
        self.reporting_layer = ReportingLayer(self)

    def apply_strategy(self, all_dfs: Dict[str, pd.DataFrame], start_date_str: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V415.0 · 数据上下文校准版】
        - 核心修复: 实现了对 `start_date_str` 参数的正确处理。在所有情报分析开始前，
                      使用此参数对主数据帧 `df_indicators` 进行切片，确保整个策略在正确的、
                      由调用者指定的日期上下文上运行，从根本上解决因数据范围错误导致的索引找不到问题。
        """
        self.params = self.unified_config
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_indicators = self._merge_all_timeframes(all_dfs)
        # [代码修改开始] 根据传入的 start_date_str 对数据帧进行切片，校准分析上下文
        if start_date_str:
            try:
                start_date = pd.to_datetime(start_date_str)
                # 检查并同步时区
                if self.df_indicators.index.tz is not None and start_date.tzinfo is None:
                    start_date = start_date.tz_localize(self.df_indicators.index.tz)
                
                print(f"    -> [上下文校准] 应用起始日期过滤器: {start_date}")
                self.df_indicators = self.df_indicators[self.df_indicators.index >= start_date]
                
                if self.df_indicators.empty:
                    print(f"    -> [上下文校准] 警告: 应用起始日期 '{start_date_str}' 后，数据帧为空。")
                    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            except Exception as e:
                print(f"    -> [上下文校准] 错误: 处理起始日期 '{start_date_str}' 失败: {e}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # [代码修改结束]
        # 步骤1: 情报层完成所有诊断与合成，包括专业层、融合层和认知层。这是唯一的情报生成入口。
        self.intelligence_layer.run_all_diagnostics(self.df_indicators)
        # 步骤2: 基于完整的诊断结果，进行顶层上下文分析
        from .trend_following.utils import calculate_context_scores
        bottom_context_score, top_context_score = calculate_context_scores(self.df_indicators, self.atomic_states)
        self.atomic_states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] = bottom_context_score
        self.atomic_states['SCORE_CONTEXT_TOP_ZONE'] = top_context_score
        # 步骤3: 执行攻防决策与模拟
        entry_score, score_details_df = self.offensive_layer.calculate_entry_score(
            self.trigger_events,
            bottom_context_score,
            top_context_score
        )
        self.df_indicators['entry_score'] = entry_score
        risk_details_df = self.warning_layer.run_all_warnings()
        self.judgment_layer.make_final_decisions(score_details_df, risk_details_df)
        self.simulation_layer.run_position_management_simulation()
        self.df_indicators = optimize_df_memory(self.df_indicators, verbose=False)
        if risk_details_df is None:
            risk_details_df = pd.DataFrame(index=self.df_indicators.index)
        return self.df_indicators, score_details_df, risk_details_df

    def _merge_all_timeframes(self, all_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        将所有时间框架的数据合并到一个DataFrame中。
        """
        if 'D' not in all_dfs:
            return pd.DataFrame()
        merged_df = all_dfs['D'].copy()
        for tf, df in all_dfs.items():
            if tf == 'D' or df.empty:
                continue
            # 对齐索引并向前填充
            aligned_df = df.reindex(merged_df.index, method='ffill')
            # 合并，避免重复列
            merged_df = merged_df.merge(
                aligned_df.drop(columns=[col for col in aligned_df.columns if col in merged_df.columns and col != 'trade_date'], errors='ignore'),
                left_index=True,
                right_index=True,
                how='left'
            )
        return ensure_numeric_types(merged_df)

    async def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> Tuple[List, List, List, List, List]:
        """
        对外暴露的报告生成接口。
        """
        return await self.reporting_layer.prepare_db_records(stock_code, result_df, score_details_df, risk_details_df, params, result_timeframe)
