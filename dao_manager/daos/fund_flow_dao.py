# dao/fund_flow_dao.py

import logging
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Union
from django.db import transaction
from django.core.cache import cache
from django.utils import timezone
from asgiref.sync import sync_to_async

from api_manager.apis.fund_flow_api import FundFlowAPI, StockPoolAPI
from api_manager.mappings.fund_flow_mapping import BREAK_LIMIT_POOL_MAPPING, DAILY_FUND_FLOW_MAPPING, FUND_FLOW_TREND_MAPPING, LIMIT_DOWN_POOL_MAPPING, LIMIT_UP_POOL_MAPPING, MAIN_FORCE_PHASE_MAPPING, NEW_STOCK_POOL_MAPPING, STRONG_STOCK_POOL_MAPPING, TRANSACTION_DISTRIBUTION_MAPPING
from models.fund_flow import BreakLimitPool, FundFlowDaily, FundFlowMinute, LimitDownPool, LimitUpPool, MainForcePhase, NewStockPool, StrongStockPool, TransactionDistribution
from models.stock_basic import StockBasic


logger = logging.getLogger(__name__)

class FundFlowDAO:
    """
    资金流向数据访问对象
    
    负责资金流向相关数据的读写操作，实现三层数据访问结构：API、Redis缓存、MySQL持久化
    """
    
    def __init__(self):
        """初始化DAO对象，创建API实例"""
        self.api = FundFlowAPI()
        # 设置缓存过期时间（秒）
        self.cache_timeout = {
            'stock': 86400,          # 股票基本信息缓存1天
            'fund_flow_minute': 300, # 分钟级资金流向缓存5分钟
            'fund_flow_daily': 3600, # 日级资金流向缓存1小时
            'main_force': 3600,      # 主力动向数据缓存1小时
            'transaction': 3600,     # 成交分布数据缓存1小时
        }
    
    # ================ 读取方法 ================
    
    async def get_stock_by_code(self, code: str) -> Optional[StockBasic]:
        """
        根据股票代码获取股票信息
        
        如果数据库中不存在该股票，则创建基本的股票记录
        
        Args:
            code: 股票代码
            
        Returns:
            Optional[Stock]: 股票对象，如不存在返回None
        """
        cache_key = f'stock_{code}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取
        try:
            stock = await sync_to_async(StockBasic.objects.get)(code=code)
            cache.set(cache_key, stock, self.cache_timeout['stock'])
            return stock
        except StockBasic.DoesNotExist:
            # 如果股票不存在，创建一个基本记录
            exchange = 'sh' if code.startswith(('60', '68')) else 'sz'
            name = f"临时名称_{code}"
            
            @transaction.atomic
            def create_stock():
                stock = StockBasic(code=code, name=name, exchange=exchange)
                stock.save()
                return stock
            
            stock = await sync_to_async(create_stock)()
            logger.info(f"为股票代码[{code}]创建基本记录")
            cache.set(cache_key, stock, self.cache_timeout['stock'])
            return stock
    
    async def get_fund_flow_minute(self, stock_code: str, limit: int = 100) -> List[FundFlowMinute]:
        """
        获取股票的分钟级资金流向数据
        
        Args:
            stock_code: 股票代码
            limit: 返回数据条数限制
            
        Returns:
            List[FundFlowMinute]: 分钟级资金流向数据列表
        """
        cache_key = f'fund_flow_minute_{stock_code}_{limit}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                FundFlowMinute.objects.filter(stock=stock).order_by('-trade_time')[:limit]
            )
            
            # 如果数据库中数据不足，从API获取
            if len(data) < 10:
                logger.info(f"股票[{stock_code}]的分钟级资金流向数据不足，从API获取")
                await self._fetch_and_save_fund_flow_minute(stock_code)
                data = await sync_to_async(list)(
                    FundFlowMinute.objects.filter(stock=stock).order_by('-trade_time')[:limit]
                )
            
            cache.set(cache_key, data, self.cache_timeout['fund_flow_minute'])
            return data
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的分钟级资金流向数据失败: {str(e)}")
            return []
    
    async def get_fund_flow_daily(self, stock_code: str, start_date: Optional[datetime.date] = None, 
                                 end_date: Optional[datetime.date] = None, limit: int = 100) -> List[FundFlowDaily]:
        """
        获取股票的日级资金流向数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回数据条数限制
            
        Returns:
            List[FundFlowDaily]: 日级资金流向数据列表
        """
        cache_key = f'fund_flow_daily_{stock_code}_{start_date}_{end_date}_{limit}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 从数据库获取
        try:
            query = {'stock': stock}
            if start_date:
                query['trade_date__gte'] = start_date
            if end_date:
                query['trade_date__lte'] = end_date
            
            data = await sync_to_async(list)(
                FundFlowDaily.objects.filter(**query).order_by('-trade_date')[:limit]
            )
            
            # 如果数据库中数据不足，从API获取
            if len(data) < 10:
                logger.info(f"股票[{stock_code}]的日级资金流向数据不足，从API获取")
                await self._fetch_and_save_fund_flow_daily(stock_code)
                data = await sync_to_async(list)(
                    FundFlowDaily.objects.filter(**query).order_by('-trade_date')[:limit]
                )
            
            cache.set(cache_key, data, self.cache_timeout['fund_flow_daily'])
            return data
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的日级资金流向数据失败: {str(e)}")
            return []
    
    async def get_last10_fund_flow_daily(self, stock_code: str) -> List[FundFlowDaily]:
        """
        获取股票最近10天的日级资金流向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[FundFlowDaily]: 最近10天的日级资金流向数据列表
        """
        cache_key = f'fund_flow_daily_last10_{stock_code}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                FundFlowDaily.objects.filter(stock=stock).order_by('-trade_date')[:10]
            )
            
            # 如果数据库中数据不足，从API获取
            if len(data) < 10:
                logger.info(f"股票[{stock_code}]的最近10天日级资金流向数据不足，从API获取")
                await self._fetch_and_save_last10_fund_flow_daily(stock_code)
                data = await sync_to_async(list)(
                    FundFlowDaily.objects.filter(stock=stock).order_by('-trade_date')[:10]
                )
            
            cache.set(cache_key, data, self.cache_timeout['fund_flow_daily'])
            return data
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的最近10天日级资金流向数据失败: {str(e)}")
            return []
    
    async def get_main_force_phase(self, stock_code: str, limit: int = 100) -> List[MainForcePhase]:
        """
        获取股票的阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            limit: 返回数据条数限制
            
        Returns:
            List[MainForcePhase]: 阶段主力动向数据列表
        """
        cache_key = f'main_force_phase_{stock_code}_{limit}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                MainForcePhase.objects.filter(stock=stock).order_by('-trade_date')[:limit]
            )
            
            # 如果数据库中数据不足，从API获取
            if len(data) < 10:
                logger.info(f"股票[{stock_code}]的阶段主力动向数据不足，从API获取")
                await self._fetch_and_save_main_force_phase(stock_code)
                data = await sync_to_async(list)(
                    MainForcePhase.objects.filter(stock=stock).order_by('-trade_date')[:limit]
                )
            
            cache.set(cache_key, data, self.cache_timeout['main_force'])
            return data
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的阶段主力动向数据失败: {str(e)}")
            return []
    
    async def get_last10_main_force_phase(self, stock_code: str) -> List[MainForcePhase]:
        """
        获取股票最近10天的阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[MainForcePhase]: 最近10天的阶段主力动向数据列表
        """
        cache_key = f'main_force_phase_last10_{stock_code}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                MainForcePhase.objects.filter(stock=stock).order_by('-trade_date')[:10]
            )
            
            # 如果数据库中数据不足，从API获取
            if len(data) < 10:
                logger.info(f"股票[{stock_code}]的最近10天阶段主力动向数据不足，从API获取")
                await self._fetch_and_save_last10_main_force_phase(stock_code)
                data = await sync_to_async(list)(
                    MainForcePhase.objects.filter(stock=stock).order_by('-trade_date')[:10]
                )
            
            cache.set(cache_key, data, self.cache_timeout['main_force'])
            return data
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的最近10天阶段主力动向数据失败: {str(e)}")
            return []
    
    async def get_transaction_distribution(self, stock_code: str, limit: int = 100) -> List[TransactionDistribution]:
        """
        获取股票的历史成交分布数据
        
        Args:
            stock_code: 股票代码
            limit: 返回数据条数限制
            
        Returns:
            List[TransactionDistribution]: 历史成交分布数据列表
        """
        cache_key = f'transaction_distribution_{stock_code}_{limit}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                TransactionDistribution.objects.filter(stock=stock).order_by('-trade_date')[:limit]
            )
            
            # 如果数据库中数据不足，从API获取
            if len(data) < 10:
                logger.info(f"股票[{stock_code}]的历史成交分布数据不足，从API获取")
                await self._fetch_and_save_transaction_distribution(stock_code)
                data = await sync_to_async(list)(
                    TransactionDistribution.objects.filter(stock=stock).order_by('-trade_date')[:limit]
                )
            
            cache.set(cache_key, data, self.cache_timeout['transaction'])
            return data
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的历史成交分布数据失败: {str(e)}")
            return []
    
    async def get_last10_transaction_distribution(self, stock_code: str) -> List[TransactionDistribution]:
        """
        获取股票最近10天的历史成交分布数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[TransactionDistribution]: 最近10天的历史成交分布数据列表
        """
        cache_key = f'transaction_distribution_last10_{stock_code}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                TransactionDistribution.objects.filter(stock=stock).order_by('-trade_date')[:10]
            )
            
            # 如果数据库中数据不足，从API获取
            if len(data) < 10:
                logger.info(f"股票[{stock_code}]的最近10天历史成交分布数据不足，从API获取")
                await self._fetch_and_save_last10_transaction_distribution(stock_code)
                data = await sync_to_async(list)(
                    TransactionDistribution.objects.filter(stock=stock).order_by('-trade_date')[:10]
                )
            
            cache.set(cache_key, data, self.cache_timeout['transaction'])
            return data
        except Exception as e:
            logger.error(f"获取股票[{stock_code}]的最近10天历史成交分布数据失败: {str(e)}")
            return []
    
    # ================ 写入方法 ================
    
    async def _fetch_and_save_fund_flow_minute(self, stock_code: str) -> List[FundFlowMinute]:
        """
        获取并保存股票的分钟级资金流向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[FundFlowMinute]: 保存的分钟级资金流向数据列表
        """
        try:
            # 获取股票对象
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                logger.warning(f"股票[{stock_code}]不存在，无法获取资金流向数据")
                return []
            
            # 从API获取数据
            api_data = await self.api.get_fund_flow_trend(stock_code)
            if not api_data:
                logger.warning(f"获取股票[{stock_code}]的分钟级资金流向数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'stock': stock}
                    
                    for api_field, model_field in FUND_FLOW_TREND_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = FundFlowMinute.objects.update_or_create(
                        stock=stock,
                        trade_time=model_data['trade_time'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存股票[{stock_code}]的分钟级资金流向数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'fund_flow_minute_{stock_code}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的分钟级资金流向数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_fund_flow_daily(self, stock_code: str) -> List[FundFlowDaily]:
        """
        获取并保存股票的日级资金流向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[FundFlowDaily]: 保存的日级资金流向数据列表
        """
        try:
            # 获取股票对象
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                logger.warning(f"股票[{stock_code}]不存在，无法获取资金流向数据")
                return []
            
            # 从API获取数据
            api_data = await self.api.get_capital_flow_trend(stock_code)
            if not api_data:
                logger.warning(f"获取股票[{stock_code}]的日级资金流向数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'stock': stock}
                    
                    for api_field, model_field in DAILY_FUND_FLOW_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = FundFlowDaily.objects.update_or_create(
                        stock=stock,
                        trade_date=model_data['trade_date'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存股票[{stock_code}]的日级资金流向数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'fund_flow_daily_{stock_code}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的日级资金流向数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_last10_fund_flow_daily(self, stock_code: str) -> List[FundFlowDaily]:
        """
        获取并保存股票最近10天的日级资金流向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[FundFlowDaily]: 保存的最近10天日级资金流向数据列表
        """
        try:
            # 获取股票对象
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                logger.warning(f"股票[{stock_code}]不存在，无法获取资金流向数据")
                return []
            
            # 从API获取数据
            api_data = await self.api.get_last10_capital_flow(stock_code)
            if not api_data:
                logger.warning(f"获取股票[{stock_code}]的最近10天日级资金流向数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'stock': stock}
                    
                    for api_field, model_field in DAILY_FUND_FLOW_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = FundFlowDaily.objects.update_or_create(
                        stock=stock,
                        trade_date=model_data['trade_date'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存股票[{stock_code}]的最近10天日级资金流向数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache.delete(f'fund_flow_daily_last10_{stock_code}')
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的最近10天日级资金流向数据失败: {str(e)}")
            return []
    
    # 其余的_fetch_and_save方法也类似，此处省略
    
    # ================ 公共方法 ================
    
    async def refresh_stock_info(self, stock_code: str, name: str, exchange: str) -> Optional[StockBasic]:
        """
        刷新股票基本信息
        
        Args:
            stock_code: 股票代码
            name: 股票名称
            exchange: 交易所代码
            
        Returns:
            Optional[Stock]: 更新后的股票对象
        """
        try:
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                @transaction.atomic
                def create_stock():
                    new_stock = StockBasic(code=stock_code, name=name, exchange=exchange)
                    new_stock.save()
                    return new_stock
                
                stock = await sync_to_async(create_stock)()
                logger.info(f"创建股票[{stock_code}]基本信息")
            else:
                @transaction.atomic
                def update_stock():
                    stock.name = name
                    stock.exchange = exchange
                    stock.save()
                    return stock
                
                stock = await sync_to_async(update_stock)()
                logger.info(f"更新股票[{stock_code}]基本信息")
            
            # 更新缓存
            cache.set(f'stock_{stock_code}', stock, self.cache_timeout['stock'])
            
            return stock
        except Exception as e:
            logger.error(f"刷新股票[{stock_code}]基本信息失败: {str(e)}")
            return None
    
    async def refresh_fund_flow_minute(self, stock_code: str) -> List[FundFlowMinute]:
        """
        刷新股票的分钟级资金流向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[FundFlowMinute]: 更新后的分钟级资金流向数据
        """
        try:
            return await self._fetch_and_save_fund_flow_minute(stock_code)
        except Exception as e:
            logger.error(f"刷新股票[{stock_code}]的分钟级资金流向数据失败: {str(e)}")
            return []
    
    async def refresh_fund_flow_daily(self, stock_code: str) -> List[FundFlowDaily]:
        """
        刷新股票的日级资金流向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[FundFlowDaily]: 更新后的日级资金流向数据
        """
        try:
            return await self._fetch_and_save_fund_flow_daily(stock_code)
        except Exception as e:
            logger.error(f"刷新股票[{stock_code}]的日级资金流向数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_main_force_phase(self, stock_code: str) -> List[MainForcePhase]:
        """
        获取并保存股票的阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[MainForcePhase]: 保存的阶段主力动向数据列表
        """
        try:
            # 获取股票对象
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                logger.warning(f"股票[{stock_code}]不存在，无法获取阶段主力动向数据")
                return []
            
            # 从API获取数据
            api_data = await self.api.get_main_force_direction(stock_code)
            if not api_data:
                logger.warning(f"获取股票[{stock_code}]的阶段主力动向数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'stock': stock}
                    
                    for api_field, model_field in MAIN_FORCE_PHASE_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = MainForcePhase.objects.update_or_create(
                        stock=stock,
                        trade_date=model_data['trade_date'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存股票[{stock_code}]的阶段主力动向数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'main_force_phase_{stock_code}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的阶段主力动向数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_last10_main_force_phase(self, stock_code: str) -> List[MainForcePhase]:
        """
        获取并保存股票最近10天的阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[MainForcePhase]: 保存的最近10天阶段主力动向数据列表
        """
        try:
            # 获取股票对象
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                logger.warning(f"股票[{stock_code}]不存在，无法获取阶段主力动向数据")
                return []
            
            # 从API获取数据
            api_data = await self.api.get_last10_main_force_direction(stock_code)
            if not api_data:
                logger.warning(f"获取股票[{stock_code}]的最近10天阶段主力动向数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'stock': stock}
                    
                    for api_field, model_field in MAIN_FORCE_PHASE_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = MainForcePhase.objects.update_or_create(
                        stock=stock,
                        trade_date=model_data['trade_date'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存股票[{stock_code}]的最近10天阶段主力动向数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache.delete(f'main_force_phase_last10_{stock_code}')
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的最近10天阶段主力动向数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_transaction_distribution(self, stock_code: str) -> List[TransactionDistribution]:
        """
        获取并保存股票的历史成交分布数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[TransactionDistribution]: 保存的历史成交分布数据列表
        """
        try:
            # 获取股票对象
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                logger.warning(f"股票[{stock_code}]不存在，无法获取历史成交分布数据")
                return []
            
            # 从API获取数据
            api_data = await self.api.get_trading_distribution(stock_code)
            if not api_data:
                logger.warning(f"获取股票[{stock_code}]的历史成交分布数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'stock': stock}
                    
                    for api_field, model_field in TRANSACTION_DISTRIBUTION_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = TransactionDistribution.objects.update_or_create(
                        stock=stock,
                        trade_date=model_data['trade_date'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存股票[{stock_code}]的历史成交分布数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache_keys = [key for key in cache._cache.keys() if f'transaction_distribution_{stock_code}' in key]
            for key in cache_keys:
                cache.delete(key)
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的历史成交分布数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_last10_transaction_distribution(self, stock_code: str) -> List[TransactionDistribution]:
        """
        获取并保存股票最近10天的历史成交分布数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[TransactionDistribution]: 保存的最近10天历史成交分布数据列表
        """
        try:
            # 获取股票对象
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                logger.warning(f"股票[{stock_code}]不存在，无法获取历史成交分布数据")
                return []
            
            # 从API获取数据
            api_data = await self.api.get_last10_trading_distribution(stock_code)
            if not api_data:
                logger.warning(f"获取股票[{stock_code}]的最近10天历史成交分布数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'stock': stock}
                    
                    for api_field, model_field in TRANSACTION_DISTRIBUTION_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = TransactionDistribution.objects.update_or_create(
                        stock=stock,
                        trade_date=model_data['trade_date'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存股票[{stock_code}]的最近10天历史成交分布数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache.delete(f'transaction_distribution_last10_{stock_code}')
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的最近10天历史成交分布数据失败: {str(e)}")
            return []
    
    async def refresh_main_force_phase(self, stock_code: str) -> List[MainForcePhase]:
        """
        刷新股票的阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[MainForcePhase]: 更新后的阶段主力动向数据
        """
        try:
            return await self._fetch_and_save_main_force_phase(stock_code)
        except Exception as e:
            logger.error(f"刷新股票[{stock_code}]的阶段主力动向数据失败: {str(e)}")
            return []
    
    async def refresh_last10_main_force_phase(self, stock_code: str) -> List[MainForcePhase]:
        """
        刷新股票最近10天的阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[MainForcePhase]: 更新后的最近10天阶段主力动向数据
        """
        try:
            return await self._fetch_and_save_last10_main_force_phase(stock_code)
        except Exception as e:
            logger.error(f"刷新股票[{stock_code}]的最近10天阶段主力动向数据失败: {str(e)}")
            return []
    
    async def refresh_transaction_distribution(self, stock_code: str) -> List[TransactionDistribution]:
        """
        刷新股票的历史成交分布数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[TransactionDistribution]: 更新后的历史成交分布数据
        """
        try:
            return await self._fetch_and_save_transaction_distribution(stock_code)
        except Exception as e:
            logger.error(f"刷新股票[{stock_code}]的历史成交分布数据失败: {str(e)}")
            return []
    
    async def refresh_last10_transaction_distribution(self, stock_code: str) -> List[TransactionDistribution]:
        """
        刷新股票最近10天的历史成交分布数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[TransactionDistribution]: 更新后的最近10天历史成交分布数据
        """
        try:
            return await self._fetch_and_save_last10_transaction_distribution(stock_code)
        except Exception as e:
            logger.error(f"刷新股票[{stock_code}]的最近10天历史成交分布数据失败: {str(e)}")
            return []

class StockPoolDAO:
    """
    股票池数据访问对象
    
    负责股票池相关数据的读写操作，实现三层数据访问结构：API、Redis缓存、MySQL持久化
    """
    
    def __init__(self):
        """初始化DAO对象，创建API实例"""
        self.api = StockPoolAPI()
        # 设置缓存过期时间（秒）
        self.cache_timeout = {
            'limit_up_pool': 600,     # 涨停股池缓存10分钟
            'limit_down_pool': 600,   # 跌停股池缓存10分钟
            'strong_stock_pool': 600, # 强势股池缓存10分钟
            'new_stock_pool': 3600,   # 次新股池缓存1小时
            'break_limit_pool': 600,  # 炸板股池缓存10分钟
        }
    
    # ================ 读取方法 ================
    
    async def get_limit_up_pool(self, date: str) -> List[LimitUpPool]:
        """
        获取某日的涨停股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[LimitUpPool]: 涨停股池数据列表
        """
        cache_key = f'limit_up_pool_{date}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                LimitUpPool.objects.filter(date=date).order_by('first_limit_time')
            )
            
            # 如果数据库中没有数据，从API获取
            if not data:
                logger.info(f"日期[{date}]的涨停股池数据不存在，从API获取")
                await self._fetch_and_save_limit_up_pool(date)
                data = await sync_to_async(list)(
                    LimitUpPool.objects.filter(date=date).order_by('first_limit_time')
                )
            
            cache.set(cache_key, data, self.cache_timeout['limit_up_pool'])
            return data
        except Exception as e:
            logger.error(f"获取日期[{date}]的涨停股池数据失败: {str(e)}")
            return []
    
    async def get_limit_down_pool(self, date: str) -> List[LimitDownPool]:
        """
        获取某日的跌停股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[LimitDownPool]: 跌停股池数据列表
        """
        cache_key = f'limit_down_pool_{date}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                LimitDownPool.objects.filter(date=date).order_by('limit_funds')
            )
            
            # 如果数据库中没有数据，从API获取
            if not data:
                logger.info(f"日期[{date}]的跌停股池数据不存在，从API获取")
                await self._fetch_and_save_limit_down_pool(date)
                data = await sync_to_async(list)(
                    LimitDownPool.objects.filter(date=date).order_by('limit_funds')
                )
            
            cache.set(cache_key, data, self.cache_timeout['limit_down_pool'])
            return data
        except Exception as e:
            logger.error(f"获取日期[{date}]的跌停股池数据失败: {str(e)}")
            return []
    
    async def get_strong_stock_pool(self, date: str) -> List[StrongStockPool]:
        """
        获取某日的强势股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[StrongStockPool]: 强势股池数据列表
        """
        cache_key = f'strong_stock_pool_{date}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                StrongStockPool.objects.filter(date=date).order_by('-change_percent')
            )
            
            # 如果数据库中没有数据，从API获取
            if not data:
                logger.info(f"日期[{date}]的强势股池数据不存在，从API获取")
                await self._fetch_and_save_strong_stock_pool(date)
                data = await sync_to_async(list)(
                    StrongStockPool.objects.filter(date=date).order_by('-change_percent')
                )
            
            cache.set(cache_key, data, self.cache_timeout['strong_stock_pool'])
            return data
        except Exception as e:
            logger.error(f"获取日期[{date}]的强势股池数据失败: {str(e)}")
            return []
    
    async def get_new_stock_pool(self, date: str) -> List[NewStockPool]:
        """
        获取某日的次新股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[NewStockPool]: 次新股池数据列表
        """
        cache_key = f'new_stock_pool_{date}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                NewStockPool.objects.filter(date=date).order_by('days_after_open')
            )
            
            # 如果数据库中没有数据，从API获取
            if not data:
                logger.info(f"日期[{date}]的次新股池数据不存在，从API获取")
                await self._fetch_and_save_new_stock_pool(date)
                data = await sync_to_async(list)(
                    NewStockPool.objects.filter(date=date).order_by('days_after_open')
                )
            
            cache.set(cache_key, data, self.cache_timeout['new_stock_pool'])
            return data
        except Exception as e:
            logger.error(f"获取日期[{date}]的次新股池数据失败: {str(e)}")
            return []
    
    async def get_break_limit_pool(self, date: str) -> List[BreakLimitPool]:
        """
        获取某日的炸板股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[BreakLimitPool]: 炸板股池数据列表
        """
        cache_key = f'break_limit_pool_{date}'
        
        # 尝试从缓存获取
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取
        try:
            data = await sync_to_async(list)(
                BreakLimitPool.objects.filter(date=date).order_by('first_limit_time')
            )
            
            # 如果数据库中没有数据，从API获取
            if not data:
                logger.info(f"日期[{date}]的炸板股池数据不存在，从API获取")
                await self._fetch_and_save_break_limit_pool(date)
                data = await sync_to_async(list)(
                    BreakLimitPool.objects.filter(date=date).order_by('first_limit_time')
                )
            
            cache.set(cache_key, data, self.cache_timeout['break_limit_pool'])
            return data
        except Exception as e:
            logger.error(f"获取日期[{date}]的炸板股池数据失败: {str(e)}")
            return []
    
    # ================ 写入方法 ================
    
    async def _fetch_and_save_limit_up_pool(self, date: str) -> List[LimitUpPool]:
        """
        获取并保存某日的涨停股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[LimitUpPool]: 保存的涨停股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_limit_up_pool(date)
            if not api_data:
                logger.warning(f"获取日期[{date}]的涨停股池数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'date': date}
                    
                    for api_field, model_field in LIMIT_UP_POOL_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = LimitUpPool.objects.update_or_create(
                        date=date,
                        code=model_data['code'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存日期[{date}]的涨停股池数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache.delete(f'limit_up_pool_{date}')
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存日期[{date}]的涨停股池数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_limit_down_pool(self, date: str) -> List[LimitDownPool]:
        """
        获取并保存某日的跌停股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[LimitDownPool]: 保存的跌停股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_limit_down_pool(date)
            if not api_data:
                logger.warning(f"获取日期[{date}]的跌停股池数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'date': date}
                    
                    for api_field, model_field in LIMIT_DOWN_POOL_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = LimitDownPool.objects.update_or_create(
                        date=date,
                        code=model_data['code'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存日期[{date}]的跌停股池数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache.delete(f'limit_down_pool_{date}')
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存日期[{date}]的跌停股池数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_strong_stock_pool(self, date: str) -> List[StrongStockPool]:
        """
        获取并保存某日的强势股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[StrongStockPool]: 保存的强势股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_strong_stock_pool(date)
            if not api_data:
                logger.warning(f"获取日期[{date}]的强势股池数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'date': date}
                    
                    for api_field, model_field in STRONG_STOCK_POOL_MAPPING.items():
                        if api_field in api_item:
                            # 布尔型字段特殊处理
                            if model_field == 'is_new_high':
                                model_data[model_field] = bool(int(api_item[api_field]))
                            else:
                                model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = StrongStockPool.objects.update_or_create(
                        date=date,
                        code=model_data['code'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存日期[{date}]的强势股池数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache.delete(f'strong_stock_pool_{date}')
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存日期[{date}]的强势股池数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_new_stock_pool(self, date: str) -> List[NewStockPool]:
        """
        获取并保存某日的次新股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[NewStockPool]: 保存的次新股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_new_stock_pool(date)
            if not api_data:
                logger.warning(f"获取日期[{date}]的次新股池数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'date': date}
                    
                    for api_field, model_field in NEW_STOCK_POOL_MAPPING.items():
                        if api_field in api_item:
                            # 布尔型字段特殊处理
                            if model_field == 'is_new_high':
                                model_data[model_field] = bool(int(api_item[api_field]))
                            # 日期字段特殊处理
                            elif model_field in ['open_date', 'ipo_date']:
                                date_str = api_item[api_field]
                                model_data[model_field] = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                            else:
                                model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = NewStockPool.objects.update_or_create(
                        date=date,
                        code=model_data['code'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存日期[{date}]的次新股池数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache.delete(f'new_stock_pool_{date}')
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存日期[{date}]的次新股池数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_break_limit_pool(self, date: str) -> List[BreakLimitPool]:
        """
        获取并保存某日的炸板股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[BreakLimitPool]: 保存的炸板股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_break_limit_pool(date)
            if not api_data:
                logger.warning(f"获取日期[{date}]的炸板股池数据失败")
                return []
            
            # 将API数据转换为模型数据并保存
            saved_data = []
            
            @transaction.atomic
            def save_data():
                saved_items = []
                for api_item in api_data:
                    model_data = {'date': date}
                    
                    for api_field, model_field in BREAK_LIMIT_POOL_MAPPING.items():
                        if api_field in api_item:
                            model_data[model_field] = api_item[api_field]
                    
                    # 使用update_or_create避免重复数据
                    data, created = BreakLimitPool.objects.update_or_create(
                        date=date,
                        code=model_data['code'],
                        defaults=model_data
                    )
                    saved_items.append(data)
                
                return saved_items
            
            saved_data = await sync_to_async(save_data)()
            logger.info(f"成功保存日期[{date}]的炸板股池数据，共{len(saved_data)}条")
            
            # 更新缓存
            cache.delete(f'break_limit_pool_{date}')
            
            return saved_data
        except Exception as e:
            logger.error(f"获取并保存日期[{date}]的炸板股池数据失败: {str(e)}")
            return []
    
    # ================ 公共方法 ================
    
    async def refresh_limit_up_pool(self, date: str) -> List[LimitUpPool]:
        """
        刷新某日的涨停股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[LimitUpPool]: 更新后的涨停股池数据
        """
        try:
            return await self._fetch_and_save_limit_up_pool(date)
        except Exception as e:
            logger.error(f"刷新日期[{date}]的涨停股池数据失败: {str(e)}")
            return []
    
    async def refresh_limit_down_pool(self, date: str) -> List[LimitDownPool]:
        """
        刷新某日的跌停股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[LimitDownPool]: 更新后的跌停股池数据
        """
        try:
            return await self._fetch_and_save_limit_down_pool(date)
        except Exception as e:
            logger.error(f"刷新日期[{date}]的跌停股池数据失败: {str(e)}")
            return []
    
    async def refresh_strong_stock_pool(self, date: str) -> List[StrongStockPool]:
        """
        刷新某日的强势股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[StrongStockPool]: 更新后的强势股池数据
        """
        try:
            return await self._fetch_and_save_strong_stock_pool(date)
        except Exception as e:
            logger.error(f"刷新日期[{date}]的强势股池数据失败: {str(e)}")
            return []
    
    async def refresh_new_stock_pool(self, date: str) -> List[NewStockPool]:
        """
        刷新某日的次新股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[NewStockPool]: 更新后的次新股池数据
        """
        try:
            return await self._fetch_and_save_new_stock_pool(date)
        except Exception as e:
            logger.error(f"刷新日期[{date}]的次新股池数据失败: {str(e)}")
            return []
    
    async def refresh_break_limit_pool(self, date: str) -> List[BreakLimitPool]:
        """
        刷新某日的炸板股池数据
        
        Args:
            date: 日期，格式yyyy-MM-dd
            
        Returns:
            List[BreakLimitPool]: 更新后的炸板股池数据
        """
        try:
            return await self._fetch_and_save_break_limit_pool(date)
        except Exception as e:
            logger.error(f"刷新日期[{date}]的炸板股池数据失败: {str(e)}")
            return []
