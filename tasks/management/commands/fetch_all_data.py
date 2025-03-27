#!/usr/bin/env python
"""
从API接口获取全部数据的管理命令

本模块实现了一个Django管理命令，用于按照接口的依赖顺序从API获取各类股票数据，
包括股票基础信息、指数数据、数据中心数据、实时数据、资金流向数据、股票池数据和技术指标数据。

使用方法：
    python manage.py fetch_all_data [--data-type 数据类型] [--stock-codes 股票代码列表]
    
参数说明：
    --data-type: 指定要获取的数据类型，可选值有：
        stock_basic, index, datacenter, market, realtime, fund_flow, stock_pool, indicators, strategy, all
    --stock-codes: 指定要获取数据的股票代码，多个代码用逗号分隔，例如：000001,600000

注意事项：
    1. 各类数据有依赖关系，建议按照默认顺序获取
    2. 如果不指定股票代码，将获取自选股或活跃股票的数据
    3. 该命令使用异步方式调用各个DAO层的方法获取数据
"""
import sys
import asyncio
import logging
from django.core.management.base import BaseCommand
from django.conf import settings
from datetime import datetime

from dao_manager.daos.data_center.discrete_transaction_dao import DiscreteTransactionDao
from dao_manager.daos.data_center.financial_dao import FinancialDao
from dao_manager.daos.data_center.institutional_shareholding_dao import InstitutionalShareholdingDao
from dao_manager.daos.data_center.lhb_dao import LhbDAO
from dao_manager.daos.data_center.stock_statistics_dao import StockStatisticsDao

# 解决Python 3.12上asyncio.coroutines没有_DEBUG属性的问题
if sys.version_info >= (3, 12):
    # 在Python 3.12中，_DEBUG被替换为_is_debug_mode函数
    if not hasattr(asyncio.coroutines, '_DEBUG'):
        # 为了兼容性，添加一个_DEBUG属性，其值由_is_debug_mode()函数确定
        asyncio.coroutines._DEBUG = asyncio.coroutines._is_debug_mode()

# 导入相关API和DAO
from dao_manager.daos.data_center.capital_flow_dao import CapitalFlowDao
from dao_manager.daos.data_center.north_south_dao import NorthSouthDao
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.index_dao import StockIndexDAO
from dao_manager.daos.fund_flow_dao import FundFlowDAO, StockPoolDAO
from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO

# 导入服务层
# from service.strategy_service import StrategyService
# from service.calculation_service import CalculationService

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
        stock_codes = options['stock_codes']
        # stock_codes = stock_codes_str.split(',') if stock_codes_str else None

        self.stdout.write(self.style.SUCCESS(f'开始获取数据，类型: {data_type}'))
        
        try:
            # self.stdout.write(f'handle - stock_codes: {stock_codes}, stock_codes_type: {type(stock_codes)}')
            asyncio.run(self.fetch_data(data_type, stock_codes))
            self.stdout.write(self.style.SUCCESS('数据获取完成'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'数据获取失败: {str(e)}'))
            logger.exception(f'数据获取失败: {str(e)}')

    async def fetch_data(self, data_type, stock_codes=None):
        """根据数据类型获取数据"""
        self.stdout.write(self.style.SUCCESS(f'开始获取数据，类型: {data_type}'))
        
        if data_type in ('all', 'stock_basic'):
            # self.stdout.write(f'fetch_data - stock_codes: {stock_codes}, stock_codes_type: {type(stock_codes)}')
            await self.fetch_stock_basic(stock_codes)
            self.stdout.write(self.style.SUCCESS('完成股票基础信息获取'))
        
        if data_type in ('all', 'index'):
            await self.fetch_index_data()
            self.stdout.write(self.style.SUCCESS('完成指数数据获取'))
        
        if data_type in ('all', 'datacenter'):
            await self.fetch_datacenter_data()
            self.stdout.write(self.style.SUCCESS('完成数据中心数据获取'))
        
        # if data_type in ('all', 'market'):
        #     await self.fetch_market_data()
        
        if data_type in ('all', 'realtime'):
            await self.fetch_realtime_data(stock_codes)
            self.stdout.write(self.style.SUCCESS('完成实时数据获取'))
        
        if data_type in ('all', 'fund_flow'):
            await self.fetch_fund_flow_data()
            self.stdout.write(self.style.SUCCESS('完成资金流向数据获取'))
        
        if data_type in ('all', 'stock_pool'):
            await self.fetch_stock_pool_data()
            self.stdout.write(self.style.SUCCESS('完成股票池数据获取'))
            
        if data_type in ('all', 'indicators'):
            await self.fetch_indicators_data(stock_codes)
            self.stdout.write(self.style.SUCCESS('完成股票指标数据获取'))
            
        if data_type in ('all', 'strategy'):
            await self.calculate_strategy(stock_codes)

    async def fetch_stock_basic(self, stock_codes=None):
        """获取股票基础信息"""
        self.stdout.write('获取股票基础信息...')
        stock_basic_dao = StockBasicDAO()
        stocks = await stock_basic_dao.get_stock_list()
        for stock in stocks:
            await stock_basic_dao.refresh_company_info(stock.stock_code)
            self.stdout.write(f'  - 已获取股票 {stock.stock_code} 公司信息')
        self.stdout.write('  - 已获取所有股票基础信息')        

        await stock_basic_dao.refresh_new_stock_data()
        self.stdout.write('  - 已获取所有新上市股票基础信息')

        await stock_basic_dao.refresh_st_stock_data()
        self.stdout.write('  - 已获取所有ST股票基础信息')

    async def fetch_index_data(self):
        """获取指数数据"""
        self.stdout.write('获取指数数据...')
        index_dao = StockIndexDAO()
        
        # 获取所有指数基础信息
        indexs = await index_dao._refresh_indexes()
        self.stdout.write('  - 已获取所有指数基础信息')
        
        for index in indexs:
            # 获取主要指数的实时数据
            await index_dao._fetch_and_save_realtime_data(index.code)
        self.stdout.write('  - 已获取主要指数实时数据')
            
        # 获取市场概览
        await index_dao._fetch_and_save_market_overview()
        self.stdout.write('  - 已获取市场概览数据')
        
        # 获取各周期K线数据
        periods = ['5', '15', '30', '60', 'Day', 'Week', 'Month']
        for period in periods:
            for index in indexs:
                await index_dao._fetch_and_save_time_series(period)
                self.stdout.write(f'  - 已获取主要指数 {index.code} {period} 周期K线数据')
                await index_dao._fetch_and_save_kdj(index.code)
                self.stdout.write(f'  - 已获取主要指数 {index.code} {period} 周期KDJ数据')
                await index_dao._fetch_and_save_macd(index.code)
                self.stdout.write(f'  - 已获取主要指数 {index.code} {period} 周期MACD数据')
                await index_dao._fetch_and_save_boll(index.code)
                self.stdout.write(f'  - 已获取主要指数 {index.code} {period} 周期BOLL数据')
                await index_dao._fetch_and_save_ma(index.code)
                self.stdout.write(f'  - 已获取主要指数 {index.code} {period} 周期MA数据')
        
        # 获取技术指标
        await index_dao.refresh_main_indexes_technical_indicators('Day')
        self.stdout.write('  - 已获取主要指数日线技术指标')

    async def fetch_datacenter_data(self):
        """获取数据中心数据"""
        self.stdout.write('获取数据中心数据...')
        # north_south_dao = NorthSouthDao()
        capital_flow_dao = CapitalFlowDao()
        institutional_shareholding_dao = InstitutionalShareholdingDao()
        financial_dao = FinancialDao()
        discrete_transaction_dao = DiscreteTransactionDao()
        lhb_dao = LhbDAO()
        stock_statistics_dao = StockStatisticsDao()

        periods = ['5', '10', '30', '60']

        # 获取日级龙虎榜数据
        # await lhb_dao.save_daily_lhb()
        # self.stdout.write('  - 已获取日级龙虎榜数据')
        
        
        # for period in periods:
            # # 获取保存近n日上榜个股
            # await lhb_dao.save_stock_on_list(period)
            # self.stdout.write(f'  - 保存近{period}日上榜个股')

            # # 获取保存近n日上榜营业部
            # await lhb_dao.save_broker_on_list(period)
            # self.stdout.write(f'  - 保存近{period}日上榜营业部')

            # # 获取保存近n日机构交易跟踪
            # await lhb_dao.save_institution_trade_track(period)
            # self.stdout.write(f'  - 保存近{period}日机构交易跟踪')

            # # 获取保存近n日机构交易明细
            # await lhb_dao.save_institution_trade_detail(period)
            # self.stdout.write(f'  - 保存近{period}日机构交易明细')

        # # 获取保存阶段高低榜
        # await stock_statistics_dao.save_stage_high_low()
        # self.stdout.write('  - 保存阶段高低榜')

        # # 保存盘中创新高个股数据
        # await stock_statistics_dao.save_new_high_stocks()
        # self.stdout.write('  - 保存盘中创新高个股数据')

        # #  保存盘中创新低个股数据
        # await stock_statistics_dao.save_new_low_stocks()
        # self.stdout.write('  -  保存盘中创新低个股数据')

        # # 保存成交骤增个股数据
        # await discrete_transaction_dao.save_volume_increase()
        # self.stdout.write('  - 保存成交骤增个股数据')

        # # 获取保存成交骤减个股数据
        # await discrete_transaction_dao.save_volume_decrease()
        # self.stdout.write('  - 保存成交骤减个股数据')

        # # 保存连续放量上涨个股数据
        # await discrete_transaction_dao.save_continuous_volume_increase()
        # self.stdout.write('  - 保存连续放量上涨个股数据')

        # # 保存连续放量下跌个股数据
        # await discrete_transaction_dao.save_continuous_volume_decrease()
        # self.stdout.write('  - 保存连续放量下跌个股数据')

        # # 保存连续上涨个股数据
        # await discrete_transaction_dao.save_continuous_rise()
        # self.stdout.write('  - 保存连续上涨个股数据')

        # # 保存连续下跌个股数据
        # await discrete_transaction_dao.save_continuous_fall()
        # self.stdout.write('  - 保存连续下跌个股数据')

        # # 保存周涨幅榜
        # await financial_dao.save_weekly_rank_change()
        # self.stdout.write('  - 保存周涨幅榜')

        # # 保存月涨幅榜
        # await financial_dao.save_monthly_rank_change()
        # self.stdout.write('  - 保存月涨幅榜')

        # # 保存周强势股
        # await financial_dao.save_weekly_strong_stocks()
        # self.stdout.write('  - 保存周强势股')

        # # 保存月强势股
        # await financial_dao.save_monthly_strong_stocks()
        # self.stdout.write('  - 保存月强势股')

        # # 保存流通市值榜
        # await financial_dao.save_circ_market_value_rank()
        # self.stdout.write('  - 保存流通市值榜')

        # # 保存市盈率榜
        # await financial_dao.save_pe_ratio_rank()
        # self.stdout.write('  - 保存市盈率榜')

        # # 保存市净率榜
        # await financial_dao.save_pb_ratio_rank()
        # self.stdout.write('  - 保存市净率榜')

        # # 保存净资产收益率榜
        # await financial_dao.save_roe_rank()
        # self.stdout.write('  - 保存净资产收益率榜')

        # # 保存机构持仓汇总
        # await institutional_shareholding_dao.save_institution_holding_summary()
        # self.stdout.write('  - 保存机构持仓汇总')

        # # 保存主力持仓
        # await institutional_shareholding_dao.save_fund_heavy_positions()
        # self.stdout.write('  - 保存主力持仓')

        # # 保存社保持仓
        # await institutional_shareholding_dao.save_social_security_heavy_positions()
        # self.stdout.write('  - 保存社保持仓')

        # # 保存QFII持仓
        # await institutional_shareholding_dao.save_qfii_heavy_positions()
        # self.stdout.write('  - 保存QFII持仓')

        # # 保存行业资金流向
        # await capital_flow_dao.save_industry_capital_flow()
        # self.stdout.write('  - 保存行业资金流向')
        
        # 保存概念资金流向
        await capital_flow_dao.save_concept_capital_flow()
        self.stdout.write('  - 保存概念资金流向')
        
        # 保存个股资金流向
        await capital_flow_dao.save_stock_capital_flow()
        self.stdout.write('  - 保存个股资金流向')
        
        # # 保存北向南向资金概况
        # await north_south_dao.save_north_south_fund_overview()
        # self.stdout.write('  - 保存北向南向资金概况')

        # # 保存北向资金趋势
        # await north_south_dao.save_north_fund_trend()
        # self.stdout.write('  - 保存北向资金趋势')

        # # 保存南向资金趋势
        # await north_south_dao.save_south_fund_trend()
        # self.stdout.write('  - 保存南向资金趋势')

        # # 保存北向持股变动
        # await north_south_dao.save_north_stock_holding()
        # self.stdout.write('  - 保存北向持股变动')

    async def fetch_realtime_data(self, stock_codes=None):
        """获取实时数据"""
        self.stdout.write('获取股票实时数据...')
        stock_realtime_dao = StockRealtimeDAO()
        stock_basic_dao = StockBasicDAO()
        
        if stock_codes:
            # 获取指定股票的实时数据
            await stock_realtime_dao.refresh_stocks_realtime(stock_codes)
            self.stdout.write(f'  - 已获取 {len(stock_codes)} 只股票的实时数据')
            
            # 获取指定股票的买卖五档数据
            await stock_realtime_dao.refresh_stocks_level5(stock_codes)
            self.stdout.write(f'  - 已获取 {len(stock_codes)} 只股票的买卖五档数据')
        else:
            stocks = await stock_basic_dao.get_stock_list()
            for stock in stocks:
                await stock_realtime_dao.refresh_stocks_realtime(stock.stock_code)
                self.stdout.write(f'  - 已获取 {stock} 的实时数据')
                await stock_realtime_dao.refresh_stocks_level5(stock.stock_code)
                self.stdout.write(f'  - 已获取 {stock} 的买卖五档数据')
                await stock_realtime_dao.get_trade_details_by_code_and_date(stock.stock_code, datetime.now().strftime('%Y-%m-%d'))
                self.stdout.write(f'  - 已获取 {stock} 的当日成交明细')
                await stock_realtime_dao.get_daily_time_deals(stock.stock_code, datetime.now().strftime('%Y-%m-%d'))
                self.stdout.write(f'  - 已获取 {stock} 的当日分时成交明细')
                

    async def fetch_fund_flow_data(self):
        """获取资金流向数据"""
        self.stdout.write('获取资金流向数据...')
        fund_flow_dao = FundFlowDAO()
        stock_pool_dao = StockPoolDAO()
        stock_basic_dao = StockBasicDAO()

        stocks = await stock_basic_dao.get_stock_list()

        for stock in stocks:
            await fund_flow_dao._fetch_and_save_fund_flow_minute(stock.stock_code)
            self.stdout.write(f'  - 已获取 {stock} 分钟资金流向')

            await fund_flow_dao._fetch_and_save_fund_flow_daily(stock.stock_code)
            self.stdout.write(f'  - 已获取 {stock} 日资金流向')

            await fund_flow_dao._fetch_and_save_last10_fund_flow_daily(stock.stock_code)
            self.stdout.write(f'  - 已获取 {stock} 最近10日资金流向')

            await fund_flow_dao._fetch_and_save_main_force_phase(stock.stock_code)
            self.stdout.write(f'  - 已获取 {stock} 主力资金动向阶段')

            await fund_flow_dao._fetch_and_save_last10_main_force_phase(stock.stock_code)
            self.stdout.write(f'  - 已获取 {stock} 最近10日主力资金动向阶段')

            await fund_flow_dao._fetch_and_save_transaction_distribution(stock.stock_code)
            self.stdout.write(f'  - 已获取 {stock} 主力资金动向分布')

            await fund_flow_dao._fetch_and_save_last10_transaction_distribution(stock.stock_code)
            self.stdout.write(f'  - 已获取 {stock} 最近10日主力资金动向分布')

        await stock_pool_dao._fetch_and_save_limit_up_pool()
        self.stdout.write(f'  - 已获取 涨停股票池')

        await stock_pool_dao._fetch_and_save_limit_down_pool()
        self.stdout.write(f'  - 已获取 跌停股票池')

        await stock_pool_dao._fetch_and_save_strong_stock_pool()
        self.stdout.write(f'  - 已获取 强势股票池')

        await stock_pool_dao._fetch_and_save_new_stock_pool()
        self.stdout.write(f'  - 已获取 次新股票池')
        
        await stock_pool_dao._fetch_and_save_break_limit_pool()
        self.stdout.write(f'  - 已获取 炸板股票池')
        
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

    async def fetch_indicators_data(self, stock_codes=None):
        """获取股票指标数据"""
        self.stdout.write('获取股票指标数据...')
        stock_indicators_dao = StockIndicatorsDAO()
        stock_basic_dao = StockBasicDAO()

        # 获取不同周期的K线数据
        periods = ['1', '5', '15', '30', '60', 'Day', 'Week', 'Month']
        
        if not stock_codes:
            from users.models import FavoriteStock
            stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
            if not stock_codes:
                self.stdout.write('  - 没有自选股，使用活跃股票')
                # 这里需要一个获取活跃股票的方法
                # 暂时略过，实际应用中可能需要从实时数据或其他来源获取
            stocks = await stock_basic_dao.get_stock_list()
            for stock in stocks:
                for period in periods:
                    await stock_indicators_dao.refresh_time_trade(stock.stock_code, period)
                    self.stdout.write(f'  - 已获取 {stock} {period} 周期K线数据')
                    await stock_indicators_dao.refresh_kdj(stock.stock_code, period)
                    self.stdout.write(f'  - 已获取 {stock} {period} 周期KDJ数据')
                    await stock_indicators_dao.refresh_macd(stock, period)
                    self.stdout.write(f'  - 已获取 {stock} {period} 周期MACD数据')
                    await stock_indicators_dao.refresh_boll(stock, period)
                    self.stdout.write(f'  - 已获取 {stock} {period} 周期BOLL数据')


    # async def calculate_strategy(self, stock_codes=None):
    #     """计算策略"""
    #     self.stdout.write('计算策略...')
    #     strategy_service = StrategyService()
    #     calculation_service = CalculationService()
        
    #     if not stock_codes:
    #         from users.models import FavoriteStock
    #         stock_codes = list(FavoriteStock.objects.values_list('stock_code', flat=True).distinct())
    #         if not stock_codes:
    #             self.stdout.write('  - 没有自选股，无需计算策略')
    #             return
        
    #     # 计算日内高抛低吸策略
    #     await strategy_service.calculate_intraday_strategy(stock_codes)
    #     self.stdout.write('  - 已计算日内高抛低吸策略')
        
    #     # 计算波段跟踪及高抛低吸策略
    #     await strategy_service.calculate_wave_tracking_strategy(stock_codes)
    #     self.stdout.write('  - 已计算波段跟踪及高抛低吸策略')
        
    #     # 检查股票反转状态
    #     await strategy_service.check_stock_reversal(stock_codes)
    #     self.stdout.write('  - 已检查股票反转状态')
        
    #     # 计算日内信号
    #     await calculation_service.calculate_intraday_signals(stock_codes)
    #     self.stdout.write('  - 已计算日内信号')
        
    #     # 计算日线信号
    #     await calculation_service.calculate_daily_signals(stock_codes)
    #     self.stdout.write('  - 已计算日线信号')
        
    #     # 计算市场整体信号
    #     await calculation_service.calculate_market_signals()
    #     self.stdout.write('  - 已计算市场整体信号') 

