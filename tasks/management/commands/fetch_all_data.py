#!/usr/bin/env python
"""
从API接口获取全部数据的管理命令
按照接口的依赖顺序执行
"""
import asyncio
import logging
from django.core.management.base import BaseCommand
from django.conf import settings

# 导入相关API和DAO
from api_manager.apis.stock_basic_api import StockBasicAPI
from api_manager.apis.index_api import IndexAPI
from api_manager.apis.datacenter_api import DataCenterAPI
from api_manager.apis.stock_realtime_api import StockRealtimeAPI
from api_manager.apis.fund_flow_api import FundFlowAPI

from api_manager.apis.stock_indicators_api import StockIndicatorsAPI

from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.index_dao import IndexDAO
from dao_manager.daos.datacenter_dao import DataCenterDAO
from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO
from dao_manager.daos.fund_flow_dao import FundFlowDAO, StockPoolDAO

from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO


logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '按照依赖顺序从API接口获取全部数据并计算策略'

    def add_arguments(self, parser):
        # 添加可选参数，用于指定只获取特定类型的数据
        parser.add_argument(
            '--data-type',
            type=str,
            help='指定要获取的数据类型: stock_basic, index, datacenter, market, realtime, fund_flow, stock_pool, indicators, strategy, all',
            default='all'
        )
        parser.add_argument(
            '--stock-codes',
            type=str,
            help='指定要获取数据的股票代码，用逗号分隔，例如: 000001,600000',
            default=''
        )

    def handle(self, *args, **options):
        """命令入口点"""
        data_type = options['data_type']
        stock_codes_str = options['stock_codes']
        stock_codes = stock_codes_str.split(',') if stock_codes_str else None

        self.stdout.write(self.style.SUCCESS(f'开始获取数据，类型: {data_type}'))
        
        try:
            asyncio.run(self.fetch_data(data_type, stock_codes))
            self.stdout.write(self.style.SUCCESS('数据获取完成'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'数据获取失败: {str(e)}'))
            logger.exception(f'数据获取失败: {str(e)}')

    async def fetch_data(self, data_type, stock_codes=None):
        """根据数据类型获取数据"""
        if data_type in ('all', 'stock_basic'):
            await self.fetch_stock_basic(stock_codes)
        
        if data_type in ('all', 'index'):
            await self.fetch_index_data()
        
        if data_type in ('all', 'datacenter'):
            await self.fetch_datacenter_data()
        
        # if data_type in ('all', 'market'):
        #     await self.fetch_market_data()
        
        if data_type in ('all', 'realtime'):
            await self.fetch_realtime_data(stock_codes)
        
        if data_type in ('all', 'fund_flow'):
            await self.fetch_fund_flow_data()
        
        if data_type in ('all', 'stock_pool'):
            await self.fetch_stock_pool_data()
            
        if data_type in ('all', 'indicators'):
            await self.fetch_indicators_data(stock_codes)
            
        if data_type in ('all', 'strategy'):
            await self.calculate_strategy(stock_codes)

    async def fetch_stock_basic(self, stock_codes=None):
        """获取股票基础信息"""
        self.stdout.write('获取股票基础信息...')
        stock_basic_dao = StockBasicDAO()
        
        if stock_codes:
            for stock_code in stock_codes:
                await stock_basic_dao.refresh_stock_info(stock_code)
                self.stdout.write(f'  - 已获取股票 {stock_code} 基础信息')
        else:
            # 获取所有股票
            await stock_basic_dao.refresh_all_stocks()
            self.stdout.write('  - 已获取所有股票基础信息')
            
            # 获取行业信息
            await stock_basic_dao.refresh_stock_industry()
            self.stdout.write('  - 已获取所有股票行业信息')
            
            # 获取概念信息
            await stock_basic_dao.refresh_stock_concept()
            self.stdout.write('  - 已获取所有股票概念信息')

    async def fetch_index_data(self):
        """获取指数数据"""
        self.stdout.write('获取指数数据...')
        index_dao = IndexDAO()
        
        # 获取所有指数基础信息
        await index_dao.refresh_all_indexes()
        self.stdout.write('  - 已获取所有指数基础信息')
        
        # 获取主要指数的实时数据
        await index_dao.refresh_main_indexes_realtime()
        self.stdout.write('  - 已获取主要指数实时数据')
        
        # 获取市场概览
        await index_dao.refresh_market_overview()
        self.stdout.write('  - 已获取市场概览数据')
        
        # 获取各周期K线数据
        periods = ['5', '15', '30', '60', 'Day']
        for period in periods:
            await index_dao.refresh_main_indexes_time_series(period)
            self.stdout.write(f'  - 已获取主要指数 {period} 周期K线数据')
        
        # 获取技术指标
        await index_dao.refresh_main_indexes_technical_indicators('Day')
        self.stdout.write('  - 已获取主要指数日线技术指标')

    async def fetch_datacenter_data(self):
        """获取数据中心数据"""
        self.stdout.write('获取数据中心数据...')
        datacenter_dao = DataCenterDAO()
        
        # 获取财务数据
        await datacenter_dao.refresh_financial_data()
        self.stdout.write('  - 已获取财务数据')
        
        # 获取资金流向数据
        await datacenter_dao.refresh_capital_flow_data()
        self.stdout.write('  - 已获取资金流向数据')
        
        # 获取龙虎榜数据
        await datacenter_dao.refresh_lhb_data()
        self.stdout.write('  - 已获取龙虎榜数据')
        
        # 获取机构持股数据
        await datacenter_dao.refresh_institution_data()
        self.stdout.write('  - 已获取机构持股数据')
        
        # 获取北向南向资金数据
        await datacenter_dao.refresh_north_south_data()
        self.stdout.write('  - 已获取北向南向资金数据')
        
        # 获取统计数据
        await datacenter_dao.refresh_statistics_data()
        self.stdout.write('  - 已获取统计数据')
        
        # 获取市场数据
        await datacenter_dao.refresh_market_data()
        self.stdout.write('  - 已获取市场数据')

    # async def fetch_market_data(self):
    #     """获取市场行情数据"""
    #     self.stdout.write('获取市场行情数据...')
    #     market_dao = MarketDAO()
        
    #     # 这里需要根据实际的市场DAO方法来调用
    #     # 假设有刷新市场概况的方法
    #     await market_dao.refresh_market_overview()
    #     self.stdout.write('  - 已获取市场概况数据')

    async def fetch_realtime_data(self, stock_codes=None):
        """获取实时数据"""
        self.stdout.write('获取股票实时数据...')
        stock_realtime_dao = StockRealtimeDAO()
        
        if stock_codes:
            # 获取指定股票的实时数据
            await stock_realtime_dao.refresh_stocks_realtime(stock_codes)
            self.stdout.write(f'  - 已获取 {len(stock_codes)} 只股票的实时数据')
            
            # 获取指定股票的买卖五档数据
            await stock_realtime_dao.refresh_stocks_level5(stock_codes)
            self.stdout.write(f'  - 已获取 {len(stock_codes)} 只股票的买卖五档数据')
        else:
            # 获取活跃股票的实时数据
            await stock_realtime_dao.refresh_active_stocks_realtime()
            self.stdout.write('  - 已获取活跃股票的实时数据')

    async def fetch_fund_flow_data(self):
        """获取资金流向数据"""
        self.stdout.write('获取资金流向数据...')
        fund_flow_dao = FundFlowDAO()
        
        # 获取热门股票资金流向
        await fund_flow_dao.refresh_popular_stocks_fund_flow()
        self.stdout.write('  - 已获取热门股票资金流向')
        
        # 获取活跃股票分钟资金流向
        await fund_flow_dao.refresh_active_stocks_fund_flow_minute()
        self.stdout.write('  - 已获取活跃股票分钟资金流向')
        
        # 获取行业板块资金流向
        await fund_flow_dao.refresh_sector_fund_flow()
        self.stdout.write('  - 已获取行业板块资金流向')
        
        # 获取主力资金动向阶段数据
        await fund_flow_dao.refresh_market_main_force_phase()
        self.stdout.write('  - 已获取主力资金动向阶段数据')
        
        # 获取历史成交分布
        await fund_flow_dao.refresh_stock_transaction_distribution()
        self.stdout.write('  - 已获取历史成交分布')
        
        # 获取北向南向资金流向
        await fund_flow_dao.refresh_north_south_fund_flow()
        self.stdout.write('  - 已获取北向南向资金流向')

    async def fetch_stock_pool_data(self):
        """获取股票池数据"""
        self.stdout.write('获取股票池数据...')
        stock_pool_dao = StockPoolDAO()
        
        # 获取涨跌停股票池
        await stock_pool_dao.refresh_daily_limit_pools()
        self.stdout.write('  - 已获取涨跌停股票池')
        
        # 获取强势股票池
        await stock_pool_dao.refresh_daily_strong_stocks()
        self.stdout.write('  - 已获取强势股票池')
        
        # 获取炸板股票池
        await stock_pool_dao.refresh_break_limit_pools()
        self.stdout.write('  - 已获取炸板股票池')
        
        # 获取次新股票池
        await stock_pool_dao.refresh_new_stock_pools()
        self.stdout.write('  - 已获取次新股票池')
        
        # 获取概念排行榜前十股票池
        await stock_pool_dao.refresh_concept_top_stocks()
        self.stdout.write('  - 已获取概念排行榜前十股票池')
        
        # 获取行业排行榜前十股票池
        await stock_pool_dao.refresh_industry_top_stocks()
        self.stdout.write('  - 已获取行业排行榜前十股票池')

    async def fetch_indicators_data(self, stock_codes=None):
        """获取股票指标数据"""
        self.stdout.write('获取股票指标数据...')
        stock_indicators_dao = StockIndicatorsDAO()
        
        if not stock_codes:
            from users.models import FavoriteStock
            stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
            if not stock_codes:
                self.stdout.write('  - 没有自选股，使用活跃股票')
                # 这里需要一个获取活跃股票的方法
                # 暂时略过，实际应用中可能需要从实时数据或其他来源获取
        
        if stock_codes:
            # 获取不同周期的K线数据
            periods = ['1', '5', '15', '30', '60', 'Day', 'Week', 'Month']
            for period in periods:
                await stock_indicators_dao.refresh_stocks_time_series(stock_codes, period)
                self.stdout.write(f'  - 已获取 {len(stock_codes)} 只股票的 {period} 周期K线数据')
            
            # 获取技术指标
            await stock_indicators_dao.refresh_stocks_technical_indicators(stock_codes, 'Day')
            self.stdout.write(f'  - 已获取 {len(stock_codes)} 只股票的日线技术指标')

    async def calculate_strategy(self, stock_codes=None):
        """计算策略"""
        self.stdout.write('计算策略...')
        strategy_service = StrategyService()
        calculation_service = CalculationService()
        
        if not stock_codes:
            from users.models import FavoriteStock
            stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
            if not stock_codes:
                self.stdout.write('  - 没有自选股，无需计算策略')
                return
        
        # 计算日内高抛低吸策略
        await strategy_service.calculate_intraday_strategy(stock_codes)
        self.stdout.write('  - 已计算日内高抛低吸策略')
        
        # 计算波段跟踪及高抛低吸策略
        await strategy_service.calculate_wave_tracking_strategy(stock_codes)
        self.stdout.write('  - 已计算波段跟踪及高抛低吸策略')
        
        # 检查股票反转状态
        await strategy_service.check_stock_reversal(stock_codes)
        self.stdout.write('  - 已检查股票反转状态')
        
        # 计算日内信号
        await calculation_service.calculate_intraday_signals(stock_codes)
        self.stdout.write('  - 已计算日内信号')
        
        # 计算日线信号
        await calculation_service.calculate_daily_signals(stock_codes)
        self.stdout.write('  - 已计算日线信号')
        
        # 计算市场整体信号
        await calculation_service.calculate_market_signals()
        self.stdout.write('  - 已计算市场整体信号') 