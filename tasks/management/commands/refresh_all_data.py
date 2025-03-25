#!/usr/bin/env python
"""
一键刷新所有数据的管理命令
按照最佳顺序执行所有任务
"""
import logging
import time
from django.core.management.base import BaseCommand
from django.core import management

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '一键按照最佳顺序刷新所有数据'

    def add_arguments(self, parser):
        parser.add_argument(
            '--with-strategy',
            action='store_true',
            help='是否同时计算策略',
        )
        parser.add_argument(
            '--stock-codes',
            type=str,
            help='指定要获取数据的股票代码，用逗号分隔，例如: 000001,600000',
            default=''
        )

    def handle(self, *args, **options):
        """命令入口点"""
        with_strategy = options['with_strategy']
        stock_codes = options['stock_codes']
        
        self.stdout.write(self.style.SUCCESS('开始一键刷新所有数据...'))
        start_time = time.time()
        
        # 按照依赖顺序执行数据获取
        self._refresh_stock_basic(stock_codes)
        self._refresh_index_data()
        self._refresh_datacenter_data()
        self._refresh_market_data()
        self._refresh_realtime_data(stock_codes)
        self._refresh_fund_flow_data()
        self._refresh_stock_pool_data()
        self._refresh_indicators_data(stock_codes)
        
        if with_strategy:
            self._calculate_strategy(stock_codes)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        self.stdout.write(self.style.SUCCESS(f'所有数据刷新完成，耗时: {elapsed_time:.2f}秒'))

    def _refresh_stock_basic(self, stock_codes):
        """刷新股票基础信息"""
        self.stdout.write(self.style.NOTICE('正在刷新股票基础信息...'))
        if stock_codes:
            management.call_command('fetch_all_data', data_type='stock_basic', stock_codes=stock_codes)
        else:
            management.call_command('fetch_all_data', data_type='stock_basic')

    def _refresh_index_data(self):
        """刷新指数数据"""
        self.stdout.write(self.style.NOTICE('正在刷新指数数据...'))
        management.call_command('fetch_all_data', data_type='index')

    def _refresh_datacenter_data(self):
        """刷新数据中心数据"""
        self.stdout.write(self.style.NOTICE('正在刷新数据中心数据...'))
        management.call_command('fetch_all_data', data_type='datacenter')

    def _refresh_market_data(self):
        """刷新市场行情数据"""
        self.stdout.write(self.style.NOTICE('正在刷新市场行情数据...'))
        management.call_command('fetch_all_data', data_type='market')

    def _refresh_realtime_data(self, stock_codes):
        """刷新实时数据"""
        self.stdout.write(self.style.NOTICE('正在刷新实时数据...'))
        if stock_codes:
            management.call_command('fetch_all_data', data_type='realtime', stock_codes=stock_codes)
        else:
            management.call_command('fetch_all_data', data_type='realtime')

    def _refresh_fund_flow_data(self):
        """刷新资金流向数据"""
        self.stdout.write(self.style.NOTICE('正在刷新资金流向数据...'))
        management.call_command('fetch_all_data', data_type='fund_flow')

    def _refresh_stock_pool_data(self):
        """刷新股票池数据"""
        self.stdout.write(self.style.NOTICE('正在刷新股票池数据...'))
        management.call_command('fetch_all_data', data_type='stock_pool')

    def _refresh_indicators_data(self, stock_codes):
        """刷新指标数据"""
        self.stdout.write(self.style.NOTICE('正在刷新指标数据...'))
        if stock_codes:
            management.call_command('fetch_all_data', data_type='indicators', stock_codes=stock_codes)
        else:
            management.call_command('fetch_all_data', data_type='indicators')

    def _calculate_strategy(self, stock_codes):
        """计算策略"""
        self.stdout.write(self.style.NOTICE('正在计算策略...'))
        if stock_codes:
            management.call_command('fetch_all_data', data_type='strategy', stock_codes=stock_codes)
        else:
            management.call_command('fetch_all_data', data_type='strategy') 