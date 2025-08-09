# 文件: services/performance_analysis_service.py
# 版本: V1.3 - 在服务入口处强制设定默认日期
import logging
import pandas as pd
from asgiref.sync import sync_to_async
from typing import Tuple, Optional
from datetime import date # 【代码修改】导入date模块以获取当前日期

from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from stock_models.stock_analytics import StrategyDailyScore, StrategyScoreComponent
from strategies.trend_following.performance_analyzer import PerformanceAnalyzer
from strategies.trend_following.utils import get_param_value, get_params_block
from utils.cache_manager import CacheManager
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class PerformanceAnalysisService:
    """
    【V1.3】基于数据库预计算结果的性能分析服务
    - 核心职责: 直接从数据库加载已有的策略分数和行情数据，
                转换成分析器所需的格式，并执行性能回测。
    - 优势: 速度极快，将原本数分钟的计算+分析过程缩短到秒级。
    """
    def __init__(self, cache_manager: CacheManager):
        self.time_trade_dao = StockTimeTradeDAO(cache_manager)
        # 加载策略配置以获取分析参数
        unified_config_path = 'config/trend_follow_strategy.json'
        self.unified_config = load_strategy_config(unified_config_path)
        self.analyzer_params = get_params_block({'unified_config': self.unified_config}, 'performance_analysis_params')
        self.scoring_params = get_params_block({'unified_config': self.unified_config}, 'four_layer_scoring_params')

    async def _fetch_analysis_data_from_db(self, stock_code: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        从数据库中异步获取并构建分析所需的核心DataFrame。
        - 注意: 此方法现在假定传入的 start_date 和 end_date 永远是有效的字符串。
        """
        print(f"    -> [DB Service] 正在为 {stock_code} 从数据库加载 {start_date} 到 {end_date} 的数据...")
        
        # 1. 异步获取日线行情数据 (价格)
        # 此处的调用现在是安全的，因为上层已确保日期不为None
        daily_price_df = await self.time_trade_dao.get_daily_data(stock_code, start_date.replace('-', ''), end_date.replace('-', ''))
        
        if daily_price_df.empty:
            logger.warning(f"[{stock_code}] 在指定日期内未找到日线行情数据。")
            return None, None
        daily_price_df.rename(columns={'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        daily_price_df.index = daily_price_df.index.date

        # 2. 异步获取策略每日分数 (信号)
        # 使用 __range 查询，因为日期现在是确定的
        daily_scores_qs = StrategyDailyScore.objects.filter(
            stock__stock_code=stock_code,
            trade_date__range=(start_date, end_date)
        ).order_by('trade_date')
        daily_scores_list = await sync_to_async(list)(daily_scores_qs.values('trade_date', 'signal_type'))
        if not daily_scores_list:
            logger.warning(f"[{stock_code}] 在指定日期内未找到策略分数数据。")
            return None, None
        daily_scores_df = pd.DataFrame(daily_scores_list)
        daily_scores_df.set_index('trade_date', inplace=True)

        # 3. 合并价格和信号，构建 df_indicators
        df_indicators = daily_price_df.join(daily_scores_df, how='inner')
        if df_indicators.empty:
            logger.warning(f"[{stock_code}] 日线行情与策略分数数据无法合并（日期不匹配）。")
            return None, None
        
        # 4. 异步获取分数构成详情
        score_components_qs = StrategyScoreComponent.objects.filter(
            daily_score__stock__stock_code=stock_code,
            daily_score__trade_date__range=(start_date, end_date)
        ).select_related('daily_score')
        
        components_list = await sync_to_async(list)(
            score_components_qs.values('daily_score__trade_date', 'signal_name', 'score_value')
        )
        if not components_list:
            logger.warning(f"[{stock_code}] 在指定日期内未找到分数构成详情数据。")
            return df_indicators, pd.DataFrame()

        components_df = pd.DataFrame(components_list)
        components_df.rename(columns={'daily_score__trade_date': 'trade_date'}, inplace=True)
        
        score_details_df = components_df.pivot_table(
            index='trade_date',
            columns='signal_name',
            values='score_value',
            fill_value=0
        )
        
        print(f"    -> [DB Service] 数据加载与转换完成[{stock_code}]。行情: {len(df_indicators)}天, 信号详情: {len(score_details_df)}天。")
        return df_indicators, score_details_df

    async def run_analysis_for_stock(self, stock_code: str, start_date: Optional[str], end_date: Optional[str]) -> list:
        """
        为单个股票执行基于数据库的回测分析。
        """
        # 【代码修改】在服务入口处强制处理None或空字符串的日期，赋予其默认值。
        # 这是解决下游DAO组件错误的根本保证。
        if not start_date:
            start_date = '1990-01-01' # 如果起始日期为空，则硬编码为最早日期
        if not end_date:
            end_date = date.today().strftime('%Y-%m-%d') # 如果结束日期为空，则设置为当前日期

        print(f"-> [Service] 开始分析股票 {stock_code}，已解析日期范围: {start_date} 到 {end_date}")

        df_indicators, score_details_df = await self._fetch_analysis_data_from_db(stock_code, start_date, end_date)

        if df_indicators is None or df_indicators.empty or score_details_df is None:
            logger.warning(f"[{stock_code}] 获取并合并数据后，无有效数据可供分析。")
            return []

        # 检查性能分析是否启用
        if not get_param_value(self.analyzer_params.get('enabled'), False):
            logger.info("性能分析模块在配置文件中被禁用。")
            return []
            
        try:
            # 实例化并运行分析器
            analyzer = PerformanceAnalyzer(
                df_indicators=df_indicators,
                score_details_df=score_details_df,
                analysis_params=self.analyzer_params,
                scoring_params=self.scoring_params
            )
            # 返回原始分析结果
            return analyzer.run_analysis()
        except Exception as e:
            logger.error(f"[{stock_code}] 性能分析器在执行过程中发生异常: {e}", exc_info=True)
            return []









