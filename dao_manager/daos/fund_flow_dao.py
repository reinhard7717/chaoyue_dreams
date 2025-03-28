# dao/fund_flow_dao.py

import logging
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Union, Type, Tuple
from django.db import transaction
from django.core.cache import cache
from django.utils import timezone
from django.db.models import Model, Q
from asgiref.sync import sync_to_async

from api_manager.apis.fund_flow_api import FundFlowAPI, StockPoolAPI
from api_manager.mappings.fund_flow_mapping import (
    BREAK_LIMIT_POOL_MAPPING, DAILY_FUND_FLOW_MAPPING, FUND_FLOW_TREND_MAPPING, 
    LIMIT_DOWN_POOL_MAPPING, LIMIT_UP_POOL_MAPPING, MAIN_FORCE_PHASE_MAPPING, 
    NEW_STOCK_POOL_MAPPING, STRONG_STOCK_POOL_MAPPING, TRANSACTION_DISTRIBUTION_MAPPING
)
from stock_models.fund_flow import (
    BreakLimitPool, FundFlowDaily, FundFlowMinute, LimitDownPool, LimitUpPool, 
    MainForcePhase, NewStockPool, StrongStockPool, TransactionDistribution
)
from stock_models.stock_basic import StockInfo
from dao_manager.base_dao import BaseDAO


logger = logging.getLogger('dao')

class FundFlowDAO(BaseDAO):
    """
    资金流向数据访问对象
    
    负责资金流向相关数据的读写操作，实现三层数据访问结构：API、Redis缓存、MySQL持久化
    """
    
    # 缓存超时设置（秒）
    CACHE_TIMEOUT = {
        'stock': 86400,          # 股票基本信息缓存1天
        'fund_flow_minute': 300, # 分钟级资金流向缓存5分钟
        'fund_flow_daily': 3600, # 日级资金流向缓存1小时
        'main_force': 3600,      # 主力动向数据缓存1小时
        'transaction': 3600,     # 成交分布数据缓存1小时
    }
    
    def __init__(self):
        """初始化DAO对象，创建API实例"""
        self.api = FundFlowAPI()
    
    # ================ 通用方法 ================
    
    @staticmethod
    def _build_cache_key(prefix: str, *args) -> str:
        """
        构建缓存键
        
        Args:
            prefix: 缓存前缀
            *args: 缓存参数
            
        Returns:
            str: 缓存键
        """
        return f"{prefix}_{'_'.join([str(arg) for arg in args if arg is not None])}"
    
    @staticmethod
    async def _get_from_cache(cache_key: str) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的数据，如果不存在则返回None
        """
        return await asyncio.to_thread(lambda: cache.get(cache_key))
    
    @staticmethod
    async def _set_to_cache(cache_key: str, data: Any, timeout: int) -> None:
        """
        将数据存入缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            timeout: 过期时间（秒）
        """
        await asyncio.to_thread(lambda: cache.set(cache_key, data, timeout))
    
    @staticmethod
    async def _delete_cache_pattern(pattern: str) -> None:
        """
        删除匹配模式的缓存
        
        Args:
            pattern: 匹配模式
        """
        keys = [key for key in cache._cache.keys() if pattern in key]
        for key in keys:
            await asyncio.to_thread(lambda: cache.delete(key))
    
    async def _process_query_with_cache(
        self, 
        model_class: Type[Model], 
        query_kwargs: Dict[str, Any], 
        cache_key: str, 
        timeout: int,
        order_by: str = '-id',
        limit: Optional[int] = None,
        fetch_func = None,
        fetch_args = None
    ) -> List[Model]:
        """
        通用的带缓存的数据查询处理方法
        
        实现顺序：
        1. 先从缓存获取
        2. 若缓存未命中，从数据库查询
        3. 若数据库数据不足，从API获取
        4. 更新缓存
        
        Args:
            model_class: 模型类
            query_kwargs: 查询参数
            cache_key: 缓存键
            timeout: 缓存超时时间
            order_by: 排序字段
            limit: 查询数量限制
            fetch_func: API数据获取函数
            fetch_args: API数据获取参数
            
        Returns:
            List[Model]: 查询结果列表
        """
        # 1. 尝试从缓存获取
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"缓存命中: {cache_key}")
            return cached_data
        
        logger.debug(f"缓存未命中: {cache_key}")
        
        # 2. 从数据库获取
        try:
            query = model_class.objects.filter(**query_kwargs).order_by(order_by)
            if limit:
                query = query[:limit]
            
            data = await sync_to_async(list)(query)
            
            # 3. 如果数据库中数据不足且提供了获取函数，从API获取
            if len(data) < 10 and fetch_func and fetch_args:
                logger.info(f"数据不足，从API获取")
                await fetch_func(*fetch_args)
                
                # 重新从数据库获取
                query = model_class.objects.filter(**query_kwargs).order_by(order_by)
                if limit:
                    query = query[:limit]
                
                data = await sync_to_async(list)(query)
            
            # 4. 将数据转换为字典格式并更新缓存
            cache_data = []
            for item in data:
                item_dict = {}
                for field in item._meta.fields:
                    value = getattr(item, field.name)
                    # 日期字段处理
                    if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                        item_dict[field.name] = self._parse_datetime(value)
                    # 数值字段处理
                    elif (isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    )) and 'code' not in field.name.lower():
                        item_dict[field.name] = self._parse_number(value)
                    else:
                        item_dict[field.name] = value
                cache_data.append(item_dict)
            
            await self._set_to_cache(cache_key, cache_data, timeout)
            return data
        except Exception as e:
            logger.error(f"查询数据失败: {str(e)}")
            return []
    
    async def _process_save_data(
        self, 
        model_class: Type[Model], 
        stock: StockInfo, 
        api_data: List[Dict[str, Any]], 
        mapping: Dict[str, str],
        unique_fields: List[str],
        cache_prefix: str = None,
        stock_code: str = None
    ) -> List[Model]:
        """
        通用的数据保存处理方法
        
        1. 将API数据转换为模型数据
        2. 批量保存到数据库
        3. 清除相关缓存
        
        Args:
            model_class: 模型类
            stock: 股票对象
            api_data: API数据列表
            mapping: 字段映射
            unique_fields: 唯一字段列表
            cache_prefix: 缓存前缀
            stock_code: 股票代码
            
        Returns:
            List[Model]: 保存的数据列表
        """
        if not api_data:
            logger.warning("未提供API数据")
            return []
        
        # 将API数据转换为模型数据并保存
        saved_data = []
        
        # 定义事务处理函数
        @transaction.atomic
        def save_data():
            saved_items = []
            for api_item in api_data:
                # 创建标准字典格式的数据
                model_data = {'stock': stock} if stock else {}
                
                # 映射字段并进行格式转换
                for api_field, model_field in mapping.items():
                    if api_field in api_item:
                        value = api_item[api_field]
                        # 日期字段处理
                        if model_field.endswith('_date') or model_field.endswith('_time') or model_field == 't':
                            model_data[model_field] = self._parse_datetime(value)
                        # 数值字段处理
                        elif (isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        )) and 'code' not in model_field.lower():
                            model_data[model_field] = self._parse_number(value)
                        else:
                            model_data[model_field] = value
                
                # 构建唯一条件
                unique_kwargs = {
                    field: model_data[field] for field in unique_fields if field in model_data
                }
                if stock and 'stock' not in unique_kwargs and hasattr(model_class, 'stock'):
                    unique_kwargs['stock'] = stock
                
                defaults = {k: v for k, v in model_data.items() if k not in unique_kwargs}
                
                # 使用update_or_create避免重复数据
                obj, created = model_class.objects.update_or_create(
                    **unique_kwargs,
                    defaults=defaults
                )
                saved_items.append(obj)
            
            return saved_items
        
        try:
            # 执行保存操作
            saved_data = await sync_to_async(save_data)()
            
            # 清除相关缓存
            if cache_prefix and stock_code:
                await self._delete_cache_pattern(f"{cache_prefix}_{stock_code}")
            
            logger.info(f"成功保存数据，共{len(saved_data)}条")
            return saved_data
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            return []
    
    # ================ 股票基本信息方法 ================
    
    async def get_stock_by_code(self, code: str) -> Optional[StockInfo]:
        """
        根据股票代码获取股票信息
        
        如果数据库中不存在该股票，则创建基本的股票记录
        
        Args:
            code: 股票代码
            
        Returns:
            Optional[StockInfo]: 股票对象，如不存在返回None
        """
        cache_key = self._build_cache_key('stock', code)
        
        # 尝试从缓存获取
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 从数据库获取
        try:
            stock = await sync_to_async(lambda: StockInfo.objects.filter(code=code).first())()
            
            if stock:
                # 更新缓存
                await self._set_to_cache(cache_key, stock, self.CACHE_TIMEOUT['stock'])
                return stock
            else:
                # 如果股票不存在，创建一个基本记录
                exchange = 'sh' if code.startswith(('60', '68')) else 'sz'
                name = f"临时名称_{code}"
                
                @transaction.atomic
                def create_stock():
                    stock = StockInfo(code=code, name=name, exchange=exchange)
                    stock.save()
                    return stock
                
                stock = await sync_to_async(create_stock)()
                logger.info(f"为股票代码[{code}]创建基本记录")
                
                # 更新缓存
                await self._set_to_cache(cache_key, stock, self.CACHE_TIMEOUT['stock'])
                return stock
                
        except Exception as e:
            logger.error(f"获取股票[{code}]信息失败: {str(e)}")
            return None
    
    async def get_fund_flow_minute(self, stock_code: str, limit: int = 100) -> List[FundFlowMinute]:
        """
        获取股票的分钟级资金流向数据
        
        Args:
            stock_code: 股票代码
            limit: 返回数据条数限制
            
        Returns:
            List[FundFlowMinute]: 分钟级资金流向数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('fund_flow_minute', stock_code, limit)
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 使用通用查询方法
        return await self._process_query_with_cache(
            model_class=FundFlowMinute,
            query_kwargs={'stock': stock},
            cache_key=cache_key,
            timeout=self.CACHE_TIMEOUT['fund_flow_minute'],
            order_by='-trade_time',
            limit=limit,
            fetch_func=self._fetch_and_save_fund_flow_minute,
            fetch_args=(stock_code,)
        )
    
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
        # 构建缓存键
        cache_key = self._build_cache_key('fund_flow_daily', stock_code, start_date, end_date, limit)
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 构建查询条件
        query_kwargs = {'stock': stock}
        if start_date:
            query_kwargs['trade_date__gte'] = start_date
        if end_date:
            query_kwargs['trade_date__lte'] = end_date
        
        # 使用通用查询方法
        return await self._process_query_with_cache(
            model_class=FundFlowDaily,
            query_kwargs=query_kwargs,
            cache_key=cache_key,
            timeout=self.CACHE_TIMEOUT['fund_flow_daily'],
            order_by='-trade_date',
            limit=limit,
            fetch_func=self._fetch_and_save_fund_flow_daily,
            fetch_args=(stock_code,)
        )
    
    async def get_last10_fund_flow_daily(self, stock_code: str) -> List[FundFlowDaily]:
        """
        获取股票最近10天的日级资金流向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[FundFlowDaily]: 最近10天的日级资金流向数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('fund_flow_daily_last10', stock_code)
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 使用通用查询方法
        return await self._process_query_with_cache(
            model_class=FundFlowDaily,
            query_kwargs={'stock': stock},
            cache_key=cache_key,
            timeout=self.CACHE_TIMEOUT['fund_flow_daily'],
            order_by='-trade_date',
            limit=10,
            fetch_func=self._fetch_and_save_last10_fund_flow_daily,
            fetch_args=(stock_code,)
        )
    
    async def get_main_force_phase(self, stock_code: str, limit: int = 100) -> List[MainForcePhase]:
        """
        获取股票的阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            limit: 返回数据条数限制
            
        Returns:
            List[MainForcePhase]: 阶段主力动向数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('main_force_phase', stock_code, limit)
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 使用通用查询方法
        return await self._process_query_with_cache(
            model_class=MainForcePhase,
            query_kwargs={'stock': stock},
            cache_key=cache_key,
            timeout=self.CACHE_TIMEOUT['main_force'],
            order_by='-trade_date',
            limit=limit,
            fetch_func=self._fetch_and_save_main_force_phase,
            fetch_args=(stock_code,)
        )
    
    async def get_last10_main_force_phase(self, stock_code: str) -> List[MainForcePhase]:
        """
        获取股票最近10天的阶段主力动向数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[MainForcePhase]: 最近10天的阶段主力动向数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('main_force_phase_last10', stock_code)
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 使用通用查询方法
        return await self._process_query_with_cache(
            model_class=MainForcePhase,
            query_kwargs={'stock': stock},
            cache_key=cache_key,
            timeout=self.CACHE_TIMEOUT['main_force'],
            order_by='-trade_date',
            limit=10,
            fetch_func=self._fetch_and_save_last10_main_force_phase,
            fetch_args=(stock_code,)
        )
    
    async def get_transaction_distribution(self, stock_code: str, limit: int = 100) -> List[TransactionDistribution]:
        """
        获取股票的历史成交分布数据
        
        Args:
            stock_code: 股票代码
            limit: 返回数据条数限制
            
        Returns:
            List[TransactionDistribution]: 历史成交分布数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('transaction_distribution', stock_code, limit)
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 使用通用查询方法
        return await self._process_query_with_cache(
            model_class=TransactionDistribution,
            query_kwargs={'stock': stock},
            cache_key=cache_key,
            timeout=self.CACHE_TIMEOUT['transaction'],
            order_by='-trade_date',
            limit=limit,
            fetch_func=self._fetch_and_save_transaction_distribution,
            fetch_args=(stock_code,)
        )
    
    async def get_last10_transaction_distribution(self, stock_code: str) -> List[TransactionDistribution]:
        """
        获取股票最近10天的历史成交分布数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[TransactionDistribution]: 最近10天的历史成交分布数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('transaction_distribution_last10', stock_code)
        
        # 获取股票对象
        stock = await self.get_stock_by_code(stock_code)
        if not stock:
            return []
        
        # 使用通用查询方法
        return await self._process_query_with_cache(
            model_class=TransactionDistribution,
            query_kwargs={'stock': stock},
            cache_key=cache_key,
            timeout=self.CACHE_TIMEOUT['transaction'],
            order_by='-trade_date',
            limit=10,
            fetch_func=self._fetch_and_save_last10_transaction_distribution,
            fetch_args=(stock_code,)
        )
    
    # ================ 数据获取和保存方法 ================
    
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
            
            # 使用通用保存方法
            return await self._process_save_data(
                model_class=FundFlowMinute,
                stock=stock,
                api_data=api_data,
                mapping=FUND_FLOW_TREND_MAPPING,
                unique_fields=['trade_time'],
                cache_prefix='fund_flow_minute',
                stock_code=stock_code
            )
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
            
            # 使用通用保存方法
            return await self._process_save_data(
                model_class=FundFlowDaily,
                stock=stock,
                api_data=api_data,
                mapping=DAILY_FUND_FLOW_MAPPING,
                unique_fields=['trade_date'],
                cache_prefix='fund_flow_daily',
                stock_code=stock_code
            )
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
            
            # 使用通用保存方法
            return await self._process_save_data(
                model_class=FundFlowDaily,
                stock=stock,
                api_data=api_data,
                mapping=DAILY_FUND_FLOW_MAPPING,
                unique_fields=['trade_date'],
                cache_prefix='fund_flow_daily_last10',
                stock_code=stock_code
            )
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的最近10天日级资金流向数据失败: {str(e)}")
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
            
            # 使用通用保存方法
            return await self._process_save_data(
                model_class=MainForcePhase,
                stock=stock,
                api_data=api_data,
                mapping=MAIN_FORCE_PHASE_MAPPING,
                unique_fields=['trade_date'],
                cache_prefix='main_force_phase',
                stock_code=stock_code
            )
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
            
            # 使用通用保存方法
            return await self._process_save_data(
                model_class=MainForcePhase,
                stock=stock,
                api_data=api_data,
                mapping=MAIN_FORCE_PHASE_MAPPING,
                unique_fields=['trade_date'],
                cache_prefix='main_force_phase_last10',
                stock_code=stock_code
            )
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
            
            # 使用通用保存方法
            return await self._process_save_data(
                model_class=TransactionDistribution,
                stock=stock,
                api_data=api_data,
                mapping=TRANSACTION_DISTRIBUTION_MAPPING,
                unique_fields=['trade_date'],
                cache_prefix='transaction_distribution',
                stock_code=stock_code
            )
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
            
            # 使用通用保存方法
            return await self._process_save_data(
                model_class=TransactionDistribution,
                stock=stock,
                api_data=api_data,
                mapping=TRANSACTION_DISTRIBUTION_MAPPING,
                unique_fields=['trade_date'],
                cache_prefix='transaction_distribution_last10',
                stock_code=stock_code
            )
        except Exception as e:
            logger.error(f"获取并保存股票[{stock_code}]的最近10天历史成交分布数据失败: {str(e)}")
            return []
    
    # ================ 公共方法 ================
    
    async def refresh_stock_info(self, stock_code: str, name: str, exchange: str) -> Optional[StockInfo]:
        """
        刷新股票基本信息
        
        Args:
            stock_code: 股票代码
            name: 股票名称
            exchange: 交易所代码
            
        Returns:
            Optional[StockInfo]: 更新后的股票对象
        """
        try:
            stock = await self.get_stock_by_code(stock_code)
            
            if not stock:
                # 创建新股票
                @transaction.atomic
                def create_stock():
                    new_stock = StockInfo(code=stock_code, name=name, exchange=exchange)
                    new_stock.save()
                    return new_stock
                
                stock = await sync_to_async(create_stock)()
                logger.info(f"创建股票[{stock_code}]基本信息")
            else:
                # 更新股票信息
                @transaction.atomic
                def update_stock():
                    stock.name = name
                    stock.exchange = exchange
                    stock.save()
                    return stock
                
                stock = await sync_to_async(update_stock)()
                logger.info(f"更新股票[{stock_code}]基本信息")
            
            # 更新缓存
            cache_key = self._build_cache_key('stock', stock_code)
            await self._set_to_cache(cache_key, stock, self.CACHE_TIMEOUT['stock'])
            
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

class StockPoolDAO(BaseDAO):
    """
    股票池数据访问对象
    
    负责股票池相关数据的读写操作，实现三层数据访问结构：API、Redis缓存、MySQL持久化
    """
    
    # 缓存超时设置（秒）
    CACHE_TIMEOUT = {
        'pool': 3600,  # 股票池数据缓存1小时
    }
    
    def __init__(self):
        """初始化DAO对象，创建API实例"""
        self.api = StockPoolAPI()
    
    @staticmethod
    def _build_cache_key(prefix: str, *args) -> str:
        """
        构建缓存键
        
        Args:
            prefix: 缓存前缀
            *args: 缓存参数
            
        Returns:
            str: 缓存键
        """
        return f"{prefix}_{'_'.join([str(arg) for arg in args if arg is not None])}"
    
    @staticmethod
    async def _get_from_cache(cache_key: str) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的数据，如果不存在则返回None
        """
        return await asyncio.to_thread(lambda: cache.get(cache_key))
    
    @staticmethod
    async def _set_to_cache(cache_key: str, data: Any, timeout: int) -> None:
        """
        将数据存入缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            timeout: 过期时间（秒）
        """
        await asyncio.to_thread(lambda: cache.set(cache_key, data, timeout))
    
    async def _process_pool_query_with_cache(
        self, 
        model_class: Type[Model], 
        date: str,
        cache_key: str,
        fetch_func
    ) -> List[Model]:
        """
        通用的股票池数据查询处理方法
        
        Args:
            model_class: 模型类
            date: 日期字符串
            cache_key: 缓存键
            fetch_func: API数据获取函数
            
        Returns:
            List[Model]: 查询结果列表
        """
        # 1. 尝试从缓存获取
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"缓存命中: {cache_key}")
            return cached_data
        
        logger.debug(f"缓存未命中: {cache_key}")
        
        # 2. 从数据库获取
        try:
            data = await sync_to_async(list)(model_class.objects.filter(date=date))
            
            # 3. 如果数据库中没有数据，从API获取
            if not data:
                logger.info(f"数据不存在，从API获取")
                await fetch_func(date)
                data = await sync_to_async(list)(model_class.objects.filter(date=date))
            
            # 4. 将数据转换为字典格式并更新缓存
            cache_data = []
            for item in data:
                item_dict = {}
                for field in item._meta.fields:
                    value = getattr(item, field.name)
                    # 日期字段处理
                    if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                        item_dict[field.name] = self._parse_datetime(value)
                    # 数值字段处理
                    elif (isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    )) and 'code' not in field.name.lower():
                        item_dict[field.name] = self._parse_number(value)
                    else:
                        item_dict[field.name] = value
                cache_data.append(item_dict)
            
            await self._set_to_cache(cache_key, cache_data, self.CACHE_TIMEOUT['pool'])
            return data
        except Exception as e:
            logger.error(f"查询股票池数据失败: {str(e)}")
            return []
    
    async def _process_pool_save_data(
        self, 
        model_class: Type[Model], 
        api_data: List[Dict[str, Any]], 
        mapping: Dict[str, str],
        date: str
    ) -> List[Model]:
        """
        通用的股票池数据保存处理方法
        
        1. 将API数据转换为模型数据
        2. 批量保存到数据库
        3. 清除相关缓存
        
        Args:
            model_class: 模型类
            api_data: API数据列表
            mapping: 字段映射
            date: 日期字符串
            
        Returns:
            List[Model]: 保存的数据列表
        """
        if not api_data:
            logger.warning("未提供API数据")
            return []
        
        # 将API数据转换为模型数据并保存
        saved_data = []
        
        # 定义事务处理函数
        @transaction.atomic
        def save_data():
            # 首先删除该日期的旧数据
            model_class.objects.filter(date=date).delete()
            
            saved_items = []
            for api_item in api_data:
                # 创建标准字典格式的数据
                model_data = {'date': date}
                
                # 映射字段并进行格式转换
                for api_field, model_field in mapping.items():
                    if api_field in api_item:
                        value = api_item[api_field]
                        # 日期字段处理
                        if model_field.endswith('_date') or model_field.endswith('_time') or model_field == 't':
                            model_data[model_field] = self._parse_datetime(value)
                        # 数值字段处理
                        elif (isinstance(value, (int, float)) or (
                            isinstance(value, str) and value.replace('.', '', 1).isdigit()
                        )) and 'code' not in model_field.lower():
                            model_data[model_field] = self._parse_number(value)
                        else:
                            model_data[model_field] = value
                
                # 创建新记录
                obj = model_class.objects.create(**model_data)
                saved_items.append(obj)
            
            return saved_items
        
        try:
            # 执行保存操作
            saved_data = await sync_to_async(save_data)()
            
            # 清除相关缓存
            cache_key = self._build_cache_key(model_class.__name__.lower(), date)
            await self._delete_cache_pattern(cache_key)
            
            return saved_data
        except Exception as e:
            logger.error(f"保存股票池数据失败: {str(e)}")
            return []

    # ================ 股票池查询方法 ================
    
    async def get_limit_up_pool(self, date: str) -> List[LimitUpPool]:
        """
        获取涨停股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[LimitUpPool]: 涨停股池数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('limituppool', date)
        
        # 使用通用股票池查询方法
        return await self._process_pool_query_with_cache(
            model_class=LimitUpPool,
            date=date,
            cache_key=cache_key,
            fetch_func=self._fetch_and_save_limit_up_pool
        )
    
    async def get_limit_down_pool(self, date: str) -> List[LimitDownPool]:
        """
        获取跌停股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[LimitDownPool]: 跌停股池数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('limitdownpool', date)
        
        # 使用通用股票池查询方法
        return await self._process_pool_query_with_cache(
            model_class=LimitDownPool,
            date=date,
            cache_key=cache_key,
            fetch_func=self._fetch_and_save_limit_down_pool
        )
    
    async def get_strong_stock_pool(self, date: str) -> List[StrongStockPool]:
        """
        获取强势股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[StrongStockPool]: 强势股池数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('strongstockpool', date)
        
        # 使用通用股票池查询方法
        return await self._process_pool_query_with_cache(
            model_class=StrongStockPool,
            date=date,
            cache_key=cache_key,
            fetch_func=self._fetch_and_save_strong_stock_pool
        )
    
    async def get_new_stock_pool(self, date: str) -> List[NewStockPool]:
        """
        获取次新股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[NewStockPool]: 次新股池数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('newstockpool', date)
        
        # 使用通用股票池查询方法
        return await self._process_pool_query_with_cache(
            model_class=NewStockPool,
            date=date,
            cache_key=cache_key,
            fetch_func=self._fetch_and_save_new_stock_pool
        )
    
    async def get_break_limit_pool(self, date: str) -> List[BreakLimitPool]:
        """
        获取炸板股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[BreakLimitPool]: 炸板股池数据列表
        """
        # 构建缓存键
        cache_key = self._build_cache_key('breaklimitpool', date)
        
        # 使用通用股票池查询方法
        return await self._process_pool_query_with_cache(
            model_class=BreakLimitPool,
            date=date,
            cache_key=cache_key,
            fetch_func=self._fetch_and_save_break_limit_pool
        )
    
    # ================ 股票池数据获取和保存方法 ================
    
    async def _fetch_and_save_limit_up_pool(self, date: str) -> List[LimitUpPool]:
        """
        获取并保存涨停股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[LimitUpPool]: 保存的涨停股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_limit_up_pool(date)
            if not api_data:
                logger.warning(f"获取{date}的涨停股池数据失败")
                return []
            
            # 使用通用保存方法
            return await self._process_pool_save_data(
                model_class=LimitUpPool,
                api_data=api_data,
                mapping=LIMIT_UP_POOL_MAPPING,
                date=date
            )
        except Exception as e:
            logger.error(f"获取并保存{date}的涨停股池数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_limit_down_pool(self, date: str) -> List[LimitDownPool]:
        """
        获取并保存跌停股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[LimitDownPool]: 保存的跌停股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_limit_down_pool(date)
            if not api_data:
                logger.warning(f"获取{date}的跌停股池数据失败")
                return []
            
            # 使用通用保存方法
            return await self._process_pool_save_data(
                model_class=LimitDownPool,
                api_data=api_data,
                mapping=LIMIT_DOWN_POOL_MAPPING,
                date=date
            )
        except Exception as e:
            logger.error(f"获取并保存{date}的跌停股池数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_strong_stock_pool(self, date: str) -> List[StrongStockPool]:
        """
        获取并保存强势股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[StrongStockPool]: 保存的强势股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_strong_stock_pool(date)
            if not api_data:
                logger.warning(f"获取{date}的强势股池数据失败")
                return []
            
            # 使用通用保存方法
            return await self._process_pool_save_data(
                model_class=StrongStockPool,
                api_data=api_data,
                mapping=STRONG_STOCK_POOL_MAPPING,
                date=date
            )
        except Exception as e:
            logger.error(f"获取并保存{date}的强势股池数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_new_stock_pool(self, date: str) -> List[NewStockPool]:
        """
        获取并保存次新股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[NewStockPool]: 保存的次新股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_new_stock_pool(date)
            if not api_data:
                logger.warning(f"获取{date}的次新股池数据失败")
                return []
            
            # 使用通用保存方法
            return await self._process_pool_save_data(
                model_class=NewStockPool,
                api_data=api_data,
                mapping=NEW_STOCK_POOL_MAPPING,
                date=date
            )
        except Exception as e:
            logger.error(f"获取并保存{date}的次新股池数据失败: {str(e)}")
            return []
    
    async def _fetch_and_save_break_limit_pool(self, date: str) -> List[BreakLimitPool]:
        """
        获取并保存炸板股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[BreakLimitPool]: 保存的炸板股池数据列表
        """
        try:
            # 从API获取数据
            api_data = await self.api.get_break_limit_pool(date)
            if not api_data:
                logger.warning(f"获取{date}的炸板股池数据失败")
                return []
            
            # 使用通用保存方法
            return await self._process_pool_save_data(
                model_class=BreakLimitPool,
                api_data=api_data,
                mapping=BREAK_LIMIT_POOL_MAPPING,
                date=date
            )
        except Exception as e:
            logger.error(f"获取并保存{date}的炸板股池数据失败: {str(e)}")
            return []
    
    # ================ 刷新方法 ================
    
    async def refresh_limit_up_pool(self, date: str) -> List[LimitUpPool]:
        """
        刷新涨停股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[LimitUpPool]: 刷新后的涨停股池数据
        """
        try:
            return await self._fetch_and_save_limit_up_pool(date)
        except Exception as e:
            logger.error(f"刷新{date}的涨停股池数据失败: {str(e)}")
            return []
    
    async def refresh_limit_down_pool(self, date: str) -> List[LimitDownPool]:
        """
        刷新跌停股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[LimitDownPool]: 刷新后的跌停股池数据
        """
        try:
            return await self._fetch_and_save_limit_down_pool(date)
        except Exception as e:
            logger.error(f"刷新{date}的跌停股池数据失败: {str(e)}")
            return []
    
    async def refresh_strong_stock_pool(self, date: str) -> List[StrongStockPool]:
        """
        刷新强势股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[StrongStockPool]: 刷新后的强势股池数据
        """
        try:
            return await self._fetch_and_save_strong_stock_pool(date)
        except Exception as e:
            logger.error(f"刷新{date}的强势股池数据失败: {str(e)}")
            return []
    
    async def refresh_new_stock_pool(self, date: str) -> List[NewStockPool]:
        """
        刷新次新股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[NewStockPool]: 刷新后的次新股池数据
        """
        try:
            return await self._fetch_and_save_new_stock_pool(date)
        except Exception as e:
            logger.error(f"刷新{date}的次新股池数据失败: {str(e)}")
            return []
    
    async def refresh_break_limit_pool(self, date: str) -> List[BreakLimitPool]:
        """
        刷新炸板股池数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'
            
        Returns:
            List[BreakLimitPool]: 刷新后的炸板股池数据
        """
        try:
            return await self._fetch_and_save_break_limit_pool(date)
        except Exception as e:
            logger.error(f"刷新{date}的炸板股池数据失败: {str(e)}")
            return []
