# 文件: services/performance_analysis_service.py
# 新增模块：基于数据库的性能分析服务
import logging
import pandas as pd
from asgiref.sync import sync_to_async
from typing import Tuple, Optional

from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from stock_models.stock_analytics import StrategyDailyScore, StrategyScoreComponent
from strategies.trend_following.performance_analyzer import PerformanceAnalyzer
from strategies.trend_following.utils import get_param_value, get_params_block
from utils.cache_manager import CacheManager
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class PerformanceAnalysisService:
    """
    【V1.0】基于数据库预计算结果的性能分析服务
    - 核心职责: 直接从数据库加载已有的策略分数和行情数据，
                转换成分析器所需的格式，并执行性能回测。
    - 优势: 速度极快，将原本数分钟的计算+分析过程缩短到秒级。
    """
    def __init__(self, cache_manager: CacheManager):
        """
        构造函数
        - 职责: 初始化服务所需的所有依赖和配置。
        """
        # 1. 初始化用于获取行情数据的DAO，并注入缓存管理器
        self.time_trade_dao = StockTimeTradeDAO(cache_manager)
        
        # 2. 定义策略配置文件的路径
        unified_config_path = 'config/trend_follow_strategy.json'
        
        # 3. 使用工具函数加载完整的策略配置，并将其存储为实例属性
        self.unified_config = load_strategy_config(unified_config_path)
        
        # 4. 从总配置中提取出“性能分析”相关的参数块
        #    注意：这里传递的是一个字典，以适配 get_params_block 工具函数
        self.analyzer_params = get_params_block({'unified_config': self.unified_config}, 'performance_analysis_params')
        
        # 5. 从总配置中提取出“四层评分模型”相关的参数块，供分析器使用
        self.scoring_params = get_params_block({'unified_config': self.unified_config}, 'four_layer_scoring_params')


    async def _fetch_analysis_data_from_db(self, stock_code: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        从数据库中异步获取并构建分析所需的核心DataFrame。
        :return: (df_indicators, score_details_df)
        """
        print(f"    -> [DB Service] 正在为 {stock_code} 从数据库加载 {start_date} 到 {end_date} 的数据...")
        
        # 1. 异步获取日线行情数据 (价格)
        daily_price_df = await self.time_trade_dao.get_daily_data(stock_code, start_date.replace('-', ''), end_date.replace('-', ''))
        if daily_price_df.empty:
            logger.warning(f"[{stock_code}] 在指定日期内未找到日线行情数据。")
            return None, None
        # 重命名列以匹配分析器期望的输入
        daily_price_df.rename(columns={'close': 'close_D', 'high': 'high_D', 'low': 'low_D'}, inplace=True)
        daily_price_df.index = daily_price_df.index.date # 将索引从datetime转换为date

        # 2. 异步获取策略每日分数 (信号)
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
        
        # 4. 异步获取分数构成详情，并构建 score_details_df
        score_components_qs = StrategyScoreComponent.objects.filter(
            daily_score__stock__stock_code=stock_code,
            daily_score__trade_date__range=(start_date, end_date)
        ).select_related('daily_score')
        
        components_list = await sync_to_async(list)(
            score_components_qs.values('daily_score__trade_date', 'signal_name', 'score_value')
        )
        if not components_list:
            logger.warning(f"[{stock_code}] 在指定日期内未找到分数构成详情数据。")
            return df_indicators, pd.DataFrame() # 即使没有详情，也可能要分析主信号

        components_df = pd.DataFrame(components_list)
        components_df.rename(columns={'daily_score__trade_date': 'trade_date'}, inplace=True)
        
        # 使用pivot_table将长格式数据转换为宽格式，这是分析器需要的格式
        score_details_df = components_df.pivot_table(
            index='trade_date',
            columns='signal_name',
            values='score_value',
            fill_value=0
        )
        
        print(f"    -> [DB Service] 数据加载与转换完成。行情: {len(df_indicators)}天, 信号详情: {len(score_details_df)}天。")
        return df_indicators, score_details_df

    async def run_analysis_for_stock(self, stock_code: str, start_date: str, end_date: str) -> list:
        """
        为单个股票执行基于数据库的回测分析。
        """
        df_indicators, score_details_df = await self._fetch_analysis_data_from_db(stock_code, start_date, end_date)

        if df_indicators is None or df_indicators.empty or score_details_df is None:
            logger.error(f"[{stock_code}] 无法从数据库获取足够的数据进行分析。")
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
