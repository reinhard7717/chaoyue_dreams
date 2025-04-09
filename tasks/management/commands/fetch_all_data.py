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
import concurrent.futures
import logging
from typing import List, Dict, Tuple, Optional, Any # 引入类型提示
from django.core.management.base import BaseCommand
from django.conf import settings
from datetime import datetime
from asgiref.sync import sync_to_async

from dao_manager.daos.data_center.discrete_transaction_dao import DiscreteTransactionDao
from dao_manager.daos.data_center.financial_dao import FinancialDao
from dao_manager.daos.data_center.institutional_shareholding_dao import InstitutionalShareholdingDao
from dao_manager.daos.data_center.lhb_dao import LhbDAO
from dao_manager.daos.data_center.stock_statistics_dao import StockStatisticsDao
from services.indicator_services import IndicatorService
from stock_models.stock_basic import StockInfo, StockTimeTrade

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

logger = logging.getLogger("dao")

# --- 新的异步工作函数 ---
async def _process_single_stock_period_async(
    stock: 'StockInfo',
    period: str,
    stock_indicators_dao: 'StockIndicatorsDAO',
    semaphore: asyncio.Semaphore, # 传入 Semaphore
    stdout_writer: callable
) -> Tuple[str, str, str, Optional[str]]:
    """
    处理单个股票和周期的逻辑（异步执行）。
    """
    stock_code = stock.stock_code
    status = "Skipped"
    message = None
    result_details = None # 用于存储 fetch_and_save 的结果

    try:
        # 1. 执行同步的 count 检查 (仍然需要 sync_to_async)
        get_data_count_sync = sync_to_async(
            lambda: StockTimeTrade.objects.filter(stock=stock, time_level=period).count(),
            thread_sensitive=True
        )
        data_count = await get_data_count_sync()
        # logger.info(f"[AsyncWorker] 数据库检查: 股票 {stock_code}, 周期 {period}, 数据量: {data_count}")

        if data_count < 699:
            # logger.info(f"[AsyncWorker] 数据量 ({data_count}) < 699 for {stock_code} {period}, 准备获取...")
            # 2. 使用 Semaphore 控制并发
            async with semaphore:
                logger.info(f"[AsyncWorker] 获取信号量，开始处理: {stock_code} {period}")
                try:
                    # 3. *** 直接 await 异步的 fetch_and_save 方法 ***
                    result_details = await stock_indicators_dao.fetch_and_save_history_time_trade(stock_code, period)
                    status = "Fetched"
                    message = f'已获取 {stock_code} {period} 历史时间序列数据'
                    logger.info(f"[AsyncWorker] 处理完成: {stock_code} {period}, 结果: {result_details}")
                    # stdout_writer(f'  - [Async] {message}') # 考虑并发写入问题
                except Exception as fetch_exc:
                    logger.error(f"[AsyncWorker] fetch_and_save 处理 {stock_code} {period} 时出错: {fetch_exc}", exc_info=True)
                    status = "Error"
                    message = str(fetch_exc)
                    # stdout_writer(f'  - [Async] 处理 {stock_code} {period} 时出错: {fetch_exc}')
        else:
            pass
            # message = f'数据量 ({data_count}) >= 699, 跳过'
            # logger.info(f"[AsyncWorker] {stock_code} {period}: {message}")

        # 返回包含 fetch 结果的元组（如果执行了）
        return stock_code, period, status, message, result_details

    except Exception as e:
        # 捕获 count 检查或其他意外错误
        logger.error(f"[AsyncWorker] 处理 {stock_code} {period} 的外层逻辑时出错: {e}", exc_info=True)
        status = "Error"
        message = f"外层错误: {e}"
        # stdout_writer(f'  - [Async] 处理 {stock_code} {period} 时外层出错: {e}')
        return stock_code, period, status, message, None
    
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

        if data_type in ('all', 'stock_trade'):
            await self.fetch_indicators_data()
            self.stdout.write(self.style.SUCCESS('完成股票交易数据获取'))
            
        if data_type in ('all', 'stock_trade_fetch_redis'):
            await self.fetch_stock_trade_data_from_db()
            self.stdout.write(self.style.SUCCESS('完成股票交易数据缓存刷新'))

        if data_type in ('all', 'calculate_stock'):
            await self.calculate_stock_indicators()
            self.stdout.write(self.style.SUCCESS('完成股票指标数据计算'))

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
        
        # # 获取所有指数基础信息
        indexs = await index_dao.get_all_indexes()
        self.stdout.write('  - 已获取所有指数基础信息')

        # await index_dao.fetch_and_save_all_realtime_data()
        # self.stdout.write('  - 已获取所有指数实时数据')

        await index_dao.fetch_and_save_all_latest_time_series()
        self.stdout.write('  - 已获取所有指数最新时间序列数据')
            
        # # 获取市场概览
        # await index_dao.fetch_and_save_market_overview()
        # self.stdout.write('  - 已获取市场概览数据')

        # # 获取指数技术指标数据
        # await index_dao.fetch_and_save_all_history_boll()
        # self.stdout.write('  - 已获取所有指数历史BOLL指标数据')

        # await index_dao.fetch_and_save_all_history_ma()
        # self.stdout.write('  - 已获取所有指数历史MA指标数据')

        # await index_dao.fetch_and_save_all_history_macd()
        # self.stdout.write('  - 已获取所有指数历史MACD指标数据')

        # await index_dao.fetch_and_save_all_history_kdj()
        # self.stdout.write('  - 已获取所有指数历史KDJ指标数据')

        # await index_dao.fetch_and_save_all_history_time_series()
        # self.stdout.write('  - 已获取所有指数历史时间序列数据')

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
        await lhb_dao.save_daily_lhb()
        self.stdout.write('  - 已获取日级龙虎榜数据')
        
        # for period in periods:
        #     # 获取保存近n日上榜个股
        #     await lhb_dao.save_stock_on_list(period)
        #     self.stdout.write(f'  - 保存近{period}日上榜个股')

        #     # 获取保存近n日上榜营业部
        #     await lhb_dao.save_broker_on_list(period)
        #     self.stdout.write(f'  - 保存近{period}日上榜营业部')

        #     # 获取保存近n日机构交易跟踪
        #     await lhb_dao.save_institution_trade_track(period)
        #     self.stdout.write(f'  - 保存近{period}日机构交易跟踪')

        #     # 获取保存近n日机构交易明细
        #     await lhb_dao.save_institution_trade_detail(period)
        #     self.stdout.write(f'  - 保存近{period}日机构交易明细')

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
        
        # # 保存概念资金流向
        # await capital_flow_dao.save_concept_capital_flow()
        # self.stdout.write('  - 保存概念资金流向')
        
        # # 保存行业资金流向数据
        # await capital_flow_dao.save_industry_capital_flow()
        # self.stdout.write('  - 保存行业资金流向数据')

        # # 保存概念资金流向数据
        # await capital_flow_dao.save_concept_capital_flow()
        # self.stdout.write('  - 保存概念资金流向数据')

        # # 保存个股阶段统计总览数据
        # await capital_flow_dao.save_stock_period_statistics_overview()
        # self.stdout.write('  - 保存个股阶段统计总览数据')

        # # 保存净流入额排名数据
        # await capital_flow_dao.save_net_inflow_amount_rank()
        # self.stdout.write('  - 保存净流入额排名数据')

        # # 保存净流入率排名数据
        # await capital_flow_dao.save_net_inflow_rate_rank()
        # self.stdout.write('  - 保存净流入率排名数据')

        # # 保存主力净流入额排名数据
        # await capital_flow_dao.save_main_net_inflow_amount_rank()
        # self.stdout.write('  - 保存主力净流入额排名数据')

        # # 保存主力净流入率排名数据
        # await capital_flow_dao.save_main_net_inflow_rate_rank()
        # self.stdout.write('  - 保存主力净流入率排名数据')

        # # 保存散户净流入额排名数据
        # await capital_flow_dao.save_retail_net_inflow_amount_rank()
        # self.stdout.write('  - 保存散户净流入额排名数据')

        # # 保存散户净流入率排名数据
        # await capital_flow_dao.save_retail_net_inflow_rate_rank()
        # self.stdout.write('  - 保存散户净流入率排名数据')
        
        


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

    async def fetch_indicators_data(self, stock_codes_filter=None):
        """获取股票指标数据 (使用 asyncio 并发处理)"""
        self.stdout.write('获取股票指标数据 (使用 asyncio 并发)...')
        stock_indicators_dao = StockIndicatorsDAO()
        stock_basic_dao = StockBasicDAO()

        periods = ['5','15','30','60','Day','Day_qfq','Day_hfq','Week','Week_qfq','Week_hfq','Month','Month_qfq','Month_hfq','Year','Year_qfq','Year_hfq']

        try:
            all_stocks = await stock_basic_dao.get_stock_list()
            # ... (股票列表获取和过滤逻辑保持不变) ...
            if not all_stocks:
                self.stdout.write("未能获取到股票列表，操作终止。")
                return
            if stock_codes_filter:
                target_stocks = [s for s in all_stocks if s.stock_code in stock_codes_filter]
            else:
                target_stocks = all_stocks
            if not target_stocks:
                self.stdout.write("没有符合条件的股票需要处理。")
                return
            self.stdout.write(f"目标股票数量: {len(target_stocks)}")

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}", exc_info=True)
            self.stdout.write(f"错误：获取股票列表失败: {e}")
            return

        # 创建 Semaphore 来限制并发数量
        # 这个值需要根据 API 限制、数据库连接池大小、内存等因素调整
        concurrency_limit = 5 # 示例值：同时最多运行 20 个 fetch_and_save 任务
        semaphore = asyncio.Semaphore(concurrency_limit)
        self.stdout.write(f"使用 asyncio.Semaphore 限制并发数为: {concurrency_limit}")

        tasks = []
        self.stdout.write(f"创建 {len(target_stocks) * len(periods)} 个处理任务...")

        for stock in target_stocks:
            for period in periods:
                # 创建异步任务，调用新的异步工作函数
                task = asyncio.create_task(
                    _process_single_stock_period_async(
                        stock,
                        period,
                        stock_indicators_dao,
                        semaphore, # 传递 semaphore
                        self.stdout.write
                    )
                )
                tasks.append(task)

        self.stdout.write(f"已创建 {len(tasks)} 个任务，使用 asyncio.gather 等待完成...")

        # 使用 asyncio.gather 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        self.stdout.write("所有任务处理完成。")

        # 处理结果 (需要调整以适应新的返回值结构)
        success_count = 0
        skipped_count = 0
        fetched_count = 0
        error_count = 0
        error_details = []
        total_created = 0
        total_updated = 0

        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                logger.error(f"任务执行期间发生意外错误: {result}", exc_info=result)
                error_details.append(f"未知任务错误: {result}")
            else:
                # 解包新的返回值结构
                stock_code, period, status, message, result_details = result
                if status == "Fetched":
                    fetched_count += 1
                    success_count += 1
                    if isinstance(result_details, dict): # 累加创建和更新的数量
                        total_created += result_details.get('创建', 0)
                        total_updated += result_details.get('更新', 0)
                    # self.stdout.write(f"  - 成功: {message}")
                elif status == "Skipped":
                    skipped_count += 1
                    success_count += 1
                elif status == "Error":
                    error_count += 1
                    error_details.append(f"{stock_code} ({period}): {message}")
                    # self.stdout.write(f"  - 失败: {stock_code} ({period}): {message}")

        self.stdout.write("--- 处理结果统计 ---")
        self.stdout.write(f"总任务数: {len(tasks)}")
        self.stdout.write(f"成功任务数: {success_count} (其中获取: {fetched_count}, 跳过: {skipped_count})")
        self.stdout.write(f"失败任务数: {error_count}")
        if fetched_count > 0:
             self.stdout.write(f"总计创建记录: {total_created}")
             self.stdout.write(f"总计更新记录: {total_updated}")
        if error_details:
            self.stdout.write("错误详情:")
            for detail in error_details[:10]:
                self.stdout.write(f"  - {detail}")
            if len(error_details) > 10:
                self.stdout.write(f"  ... (还有 {len(error_details) - 10} 条错误未显示)")

    async def fetch_stock_trade_data_from_db(self):
        """从数据库获取股票交易数据"""
        self.stdout.write('从数据库获取股票交易数据...')
        stock_indicators_dao = StockIndicatorsDAO()
        stock_basic_dao = StockBasicDAO()
        cache_limit = 233 * 3
        TIME_LEVELS = ['5','15','30','60','Day','Week','Month','Year']

        stocks = await stock_basic_dao.get_stock_list()
        logger.info(f"重新缓存{len(stocks)}只股票历史分时成交数据")
        for stock in stocks:
            for time_level in TIME_LEVELS:
                get_data_sync = sync_to_async(
                    lambda: list( # <-- 将 QuerySet 转换为列表
                        StockTimeTrade.objects.filter(stock=stock, time_level=time_level
                        ).order_by('-trade_time')[:cache_limit] # <-- 使用切片语法替代 limit()
                    ),
                    thread_sensitive=True # 对于 ORM 操作，建议设置为 True
                )
                datas = await get_data_sync()
                logger.info(f"重新缓存{stock.stock_code}股票{time_level}级别历史分时成交数据, length: {len(datas)}")
                if datas:
                    for item in datas:
                       cache_data = stock_indicators_dao.data_format_process.set_time_trade_data(stock, time_level, item)
                    #    logger.info(f"缓存{stock.stock_code}股票{time_level}级别历史分时成交数据, cache_data: {cache_data}")
                       await stock_indicators_dao.cache_set.history_time_trade(stock.stock_code, time_level, cache_data)
            
    async def calculate_stock_indicators(self):
        """计算股票指标数据"""
        self.stdout.write('计算股票指标数据...')
        indicator_services = IndicatorService()
        # stock_basic_dao = StockBasicDAO()
        # all_stocks = await stock_basic_dao.get_stock_list()
        # for stock in all_stocks:
        #     await indicator_services.calculate_and_save_all_indicators(stock.stock_code, 'Day')
        await indicator_services.calculate_and_save_all_indicators('000001', 'Day')
        self.stdout.write(self.style.SUCCESS('完成股票指标数据计算'))
            
        

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

