# 文件: run_daily_scan.py (这是一个新文件，放在项目根目录或一个专门的 `scripts` 目录下)

import asyncio
import datetime
import logging
from typing import Set

from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from services.indicator_services import IndicatorService
from strategies.multi_timeframe_trend_strategy import MultiTimeframeTrendStrategy


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyScanner:
    def __init__(self):
        self.indicator_service = IndicatorService()
        self.strategy_runner = MultiTimeframeTrendStrategy()
        self.stock_dao = StockBasicInfoDao()
        self.industry_dao = IndustryDao()

    async def get_priority_stock_pool(self, trade_date: datetime.date) -> Set[str]:
        """
        【核心】使用行业轮动分析，获取高潜力优先股池。
        """
        # 1. 执行行业轮动分析
        # lookback_days=5, 表示我们关注近一周排名持续上升的板块
        rotation_report = await self.indicator_service.analyze_industry_rotation(
            end_date=trade_date, 
            lookback_days=5
        )

        if rotation_report.empty:
            logger.warning("行业轮动分析未返回结果，将使用全部股票进行分析。")
            # 在没有轮动结果时，可以有一个备用逻辑，比如获取所有股票
            all_stocks = await self.stock_dao.get_stock_list()
            return set(all_stocks)

        # 2. 筛选出“潜力板块”
        # 筛选条件：动量为正（排名在上升）且最新排名在前50%
        potential_industries = rotation_report[
            (rotation_report['rank_momentum'] > 0) &
            (rotation_report['latest_rank'] > 0.5)
        ]
        
        if potential_industries.empty:
            logger.info("未发现明确的轮动上升板块，本次不生成优先股池。")
            return set()

        logger.info(f"发现 {len(potential_industries)} 个潜力轮动板块:")
        print(potential_industries[['industry_name', 'latest_rank', 'rank_momentum']].head())

        # 3. 根据潜力板块，获取所有成分股，构建优先股池
        potential_industry_codes = potential_industries.index.tolist()
        
        # 并行获取所有潜力板块的成分股
        tasks = [self.industry_dao.get_stock_codes_by_industry(code) for code in potential_industry_codes]
        results = await asyncio.gather(*tasks)
        
        # 将所有股票代码合并到一个集合中，自动去重
        priority_pool = set()
        for stock_list in results:
            priority_pool.update(stock_list)
            
        logger.info(f"从潜力板块中共获取 {len(priority_pool)} 只股票，构成优先分析池。")
        return priority_pool

    async def run_full_scan(self, trade_date: datetime.date):
        """
        执行完整的每日扫描流程。
        """
        logger.info(f"--- 开始执行 {trade_date} 的每日全市场扫描 ---")

        # 步骤一：通过行业轮动，确定优先股池
        logger.info("--- 步骤1: 确定优先股池 ---")
        candidate_stocks = await self.get_priority_stock_pool(trade_date)

        if not candidate_stocks:
            logger.info("优先股池为空，今日扫描结束。")
            return

        # 步骤二：对优先股池中的每只股票，执行详细的多时间框架策略分析
        logger.info(f"--- 步骤2: 对 {len(candidate_stocks)} 只优先股执行详细策略分析 ---")
        
        strategy_tasks = []
        for stock_code in candidate_stocks:
            # 为每只股票创建一个策略分析任务
            task = self.strategy_runner.run_for_stock(stock_code, trade_date.strftime('%Y-%m-%d'))
            strategy_tasks.append(task)
            
        # 并行执行所有股票的分析
        # 注意：为了避免同时发起过多请求，可以考虑使用信号量(Semaphore)来限制并发数
        semaphore = asyncio.Semaphore(10) # 例如，最多同时分析10只股票
        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*(run_with_semaphore(t) for t in strategy_tasks))

        # 步骤三：处理并保存结果 (此处省略具体实现)
        logger.info("--- 步骤3: 处理并保存分析结果 ---")
        final_signals = [item for sublist in results if sublist for item in sublist]
        logger.info(f"扫描完成，共产生 {len(final_signals)} 条有效信号。")
        # 在这里添加将 final_signals 保存到数据库的逻辑...

if __name__ == "__main__":
    scanner = DailyScanner()
    # 假设我们分析昨天的交易数据
    analysis_date = datetime.date.today() - datetime.timedelta(days=1) 
    
    # 运行主程序
    asyncio.run(scanner.run_full_scan(analysis_date))

