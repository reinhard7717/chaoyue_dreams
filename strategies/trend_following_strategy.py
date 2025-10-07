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
from .trend_following.utils import ensure_numeric_types, optimize_df_memory

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    【V101.0 · 终极信号适配版】
    - 核心重构: 净化了 apply_strategy 的主流程，现在完全依赖 IntelligenceLayer 来完成所有情报的生成与融合。
    """
    def __init__(self, orchestrator_instance):
        """
        【V102.0 · 指挥链校准协议版】
        - 核心修复: 修正了初始化方法，使其能正确接收总指挥实例(orchestrator_instance)。
        - 核心逻辑: 不再将传入的实例本身当作配置，而是从实例中正确提取 unified_config 字典。
        - 收益: 解决了因依赖注入不匹配导致的 AttributeError，使策略完全融入联邦制架构。
        """
        # [代码修改] 参数名从 config 改为 orchestrator_instance，更符合语义
        self.orchestrator = orchestrator_instance
        # [代码修改] 从总指挥实例中正确获取统一配置字典
        self.unified_config = self.orchestrator.unified_config
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
        【V407.1 · 阿里阿德涅之线协议版】
        - 核心升级: 在接收到 offensive_layer 的分数后，立即部署“观察哨”打印，以确认分数的传递过程。
        """
        self.params = self.unified_config
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_indicators = self._merge_all_timeframes(all_dfs)
        self.intelligence_layer.run_all_diagnostics()
        from .trend_following.utils import calculate_context_scores
        bottom_context_score, top_context_score = calculate_context_scores(self.df_indicators, self.atomic_states)
        entry_score, score_details_df = self.offensive_layer.calculate_entry_score(
            self.trigger_events,
            bottom_context_score,
            top_context_score
        )
        # [代码新增] 部署“阿里阿德涅之线”观察哨
        debug_params = get_params_block(self, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_dates = [pd.to_datetime(d).date() for d in probe_dates_str]
        for date in entry_score.index:
            if date.date() in probe_dates:
                print(f"      -> [阿里阿德涅之线 @ {date.date()}] (Strategy) 接收到 OffensiveLayer 的 entry_score: {entry_score.loc[date]:.0f}")
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
