import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Type
from datetime import datetime, date
from decimal import Decimal

from django.db import transaction
from django.core.cache import cache
from django.db.models import Q
from django.db.models.base import Model

from api_manager.apis.stock_indicators_api import StockIndicatorsAPI, TimeLevel
from api_manager.mappings.stock_indicators import BOLL_INDICATOR_MAPPING, KDJ_INDICATOR_MAPPING, MA_INDICATOR_MAPPING, MACD_INDICATOR_MAPPING, TIME_TRADE_MAPPING
from dao_manager.base_dao import BaseDAO
from models.stock_indicators import BOLLIndicator, KDJIndicator, MACDIndicator, MAIndicator, TimeTrade


logger = logging.getLogger(__name__)

class BaseIndicatorDAO(BaseDAO):
    """
    技术指标DAO的基础类，提供通用的指标数据处理方法
    """
    
    def __init__(self, model_class: Type[Model], cache_prefix: str, mapping: Dict[str, str]):
        """
        初始化BaseIndicatorDAO
        
        Args:
            model_class: 技术指标模型类
            cache_prefix: 缓存前缀
            mapping: 字段映射表
        """
        super().__init__(model_class, cache_prefix)
        self.api = StockIndicatorsAPI()
        self.mapping = mapping
        logger.info(f"初始化{self.__class__.__name__}")
    
    def _map_api_to_model(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        mapped_data = {}
        for api_field, model_field in self.mapping.items():
            if api_field in api_data and api_data[api_field] is not None:
                # 处理日期/时间字段的特殊情况
                if model_field == 'trade_time' and isinstance(api_data[api_field], str):
                    mapped_data[model_field] = self._parse_datetime(api_data[api_field])
                else:
                    mapped_data[model_field] = api_data[api_field]
        return mapped_data
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """
        解析日期时间字符串
        
        Args:
            datetime_str: 日期时间字符串，格式可能是"yyyy-MM-dd HH:mm:ss"或"yyyy-MM-dd"
            
        Returns:
            datetime: 解析后的日期时间对象
        """
        try:
            if len(datetime_str) > 10:  # 包含时间部分
                return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            else:  # 仅包含日期部分
                return datetime.strptime(datetime_str, '%Y-%m-%d')
        except (ValueError, TypeError) as e:
            logger.warning(f"日期时间格式解析失败: {datetime_str}, 错误: {e}")
            return datetime.now()  # 返回当前时间作为默认值
    
    async def _check_if_exists(self, stock_code: str, time_level: str, trade_time: datetime) -> Optional[Model]:
        """
        检查指定条件的记录是否存在
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            trade_time: 交易时间
            
        Returns:
            Optional[Model]: 如果存在返回记录对象，否则返回None
        """
        try:
            return await self.model_class.objects.filter(
                stock_code=stock_code,
                time_level=time_level,
                trade_time=trade_time
            ).afirst()
        except Exception as e:
            logger.error(f"检查记录是否存在出错: {e}")
            return None
    
    async def _check_if_data_identical(self, existing: Model, new_data: Dict[str, Any]) -> bool:
        """
        检查现有记录与新数据是否相同
        
        Args:
            existing: 现有记录对象
            new_data: 新数据字典
            
        Returns:
            bool: 如果数据相同返回True，否则返回False
        """
        for field, value in new_data.items():
            if field not in ('stock_code', 'time_level', 'trade_time', 'created_at', 'updated_at'):
                if hasattr(existing, field):
                    existing_value = getattr(existing, field)
                    
                    # 处理Decimal类型的比较
                    if isinstance(existing_value, Decimal) and value is not None:
                        if existing_value != Decimal(str(value)):
                            return False
                    # 常规比较
                    elif existing_value != value:
                        return False
        
        return True
    
    @transaction.atomic
    async def _save_or_update(self, data: Dict[str, Any]) -> Model:
        """
        保存或更新单条记录
        
        Args:
            data: 模型数据
            
        Returns:
            Model: 保存或更新后的记录对象
        """
        stock_code = data['stock_code']
        time_level = data['time_level']
        trade_time = data['trade_time']
        
        # 查找是否已存在记录
        existing = await self._check_if_exists(stock_code, time_level, trade_time)
        
        if existing:
            # 检查数据是否有变化
            if not await self._check_if_data_identical(existing, data):
                # 数据不同，更新记录
                for field, value in data.items():
                    if field not in ('created_at',):
                        setattr(existing, field, value)
                
                await existing.asave()
                logger.debug(f"更新记录: {stock_code}-{time_level}-{trade_time}")
            else:
                logger.debug(f"记录数据相同，跳过更新: {stock_code}-{time_level}-{trade_time}")
            
            return existing
        else:
            # 创建新记录
            instance = self.model_class(**data)
            await instance.asave()
            logger.debug(f"创建新记录: {stock_code}-{time_level}-{trade_time}")
            return instance
    
    async def _batch_save_or_update(self, data_list: List[Dict[str, Any]]) -> List[Model]:
        """
        批量保存或更新记录
        
        Args:
            data_list: 模型数据列表
            
        Returns:
            List[Model]: 保存或更新后的记录对象列表
        """
        if not data_list:
            return []
        
        # 提取所有记录的关键属性
        stock_codes = set()
        time_levels = set()
        trade_times = set()
        
        for data in data_list:
            stock_codes.add(data['stock_code'])
            time_levels.add(data['time_level'])
            trade_times.add(data['trade_time'])
        
        # 查询已存在的记录
        existing_records = {}
        try:
            queryset = self.model_class.objects.filter(
                stock_code__in=list(stock_codes),
                time_level__in=list(time_levels),
                trade_time__in=list(trade_times)
            )
            
            async for record in queryset:
                key = (record.stock_code, record.time_level, record.trade_time)
                existing_records[key] = record
        except Exception as e:
            logger.error(f"批量查询已存在记录出错: {e}")
        
        # 分类处理数据
        to_create = []
        to_update = []
        unchanged = []
        
        for data in data_list:
            key = (data['stock_code'], data['time_level'], data['trade_time'])
            
            if key in existing_records:
                existing = existing_records[key]
                if not await self._check_if_data_identical(existing, data):
                    # 需要更新的记录
                    for field, value in data.items():
                        if field not in ('created_at',):
                            setattr(existing, field, value)
                    to_update.append(existing)
                else:
                    # 无需更新的记录
                    unchanged.append(existing)
            else:
                # 需要创建的记录
                to_create.append(self.model_class(**data))
        
        # 批量创建新记录
        created_records = []
        if to_create:
            try:
                created_records = await self.model_class.objects.abulk_create(to_create)
                logger.info(f"批量创建记录: {len(created_records)}条")
            except Exception as e:
                logger.error(f"批量创建记录出错: {e}")
                # 回退到单条创建
                for item in to_create:
                    try:
                        await item.asave()
                        created_records.append(item)
                    except Exception as sub_e:
                        logger.error(f"单条创建记录出错: {sub_e}")
        
        # 批量更新记录
        updated_records = []
        if to_update:
            try:
                for record in to_update:
                    await record.asave()
                    updated_records.append(record)
                logger.info(f"批量更新记录: {len(updated_records)}条")
            except Exception as e:
                logger.error(f"批量更新记录出错: {e}")
        
        # 返回所有处理的记录
        return created_records + updated_records + unchanged


class TimeTradeDAO(BaseIndicatorDAO):
    """
    分时交易数据DAO
    """
    
    def __init__(self):
        """初始化TimeTradeDAO"""
        super().__init__(TimeTrade, "time_trade", TIME_TRADE_MAPPING)
    
    async def get_latest_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[TimeTrade]:
        """
        获取最新的分时交易数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            Optional[TimeTrade]: 最新的分时交易数据
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"time_trade:{stock_code}:{time_level}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从API获取
        try:
            api_data = await self.api.get_time_trade(stock_code, time_level)
            if api_data:
                # 处理数据
                mapped_data = self._map_api_to_model(api_data)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level
                
                # 保存到数据库
                time_trade = await self._save_or_update(mapped_data)
                
                # 更新缓存
                await self.set_to_cache(cache_key, time_trade, timeout=300)  # 5分钟
                return time_trade
        except Exception as e:
            logger.error(f"获取最新分时交易数据出错: {e}")
        
        # 3. 从数据库获取最新记录
        try:
            latest_data = await TimeTrade.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time').afirst()
            
            if latest_data:
                # 更新缓存
                await self.set_to_cache(cache_key, latest_data, timeout=300)
                return latest_data
        except Exception as e:
            logger.error(f"从数据库获取最新分时交易数据出错: {e}")
        
        return None
    
    async def get_history_time_trades(self, stock_code: str, time_level: Union[TimeLevel, str], 
                                      limit: int = 1000) -> List[TimeTrade]:
        """
        获取历史分时交易数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            limit: 返回记录数量限制
            
        Returns:
            List[TimeTrade]: 历史分时交易数据列表
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"time_trade:{stock_code}:{time_level}:history:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 先尝试从数据库获取
        try:
            db_data = await TimeTrade.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if db_data and len(db_data) >= limit * 0.8:  # 如果数据库中有足够的数据（80%以上）
                # 更新缓存
                await self.set_to_cache(cache_key, list(db_data), timeout=3600)  # 1小时
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取历史分时交易数据出错: {e}")
        
        # 3. 从API获取完整历史数据
        try:
            api_data_list = await self.api.get_history_trade(stock_code, time_level)
            if api_data_list:
                # 处理数据
                processed_data = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level
                    processed_data.append(mapped_data)
                
                # 批量保存到数据库
                saved_trades = await self._batch_save_or_update(processed_data)
                
                # 返回有限数量的记录
                result = sorted(saved_trades, key=lambda x: x.trade_time, reverse=True)[:limit]
                
                # 更新缓存
                await self.set_to_cache(cache_key, result, timeout=3600)  # 1小时
                return result
        except Exception as e:
            logger.error(f"获取历史分时交易数据出错: {e}")
        
        # 如果API获取失败，再尝试从数据库返回已有数据（即使不够完整）
        try:
            db_data = await TimeTrade.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if db_data:
                await self.set_to_cache(cache_key, list(db_data), timeout=1800)  # 30分钟
                return db_data
        except Exception as e:
            logger.error(f"重新从数据库获取历史分时交易数据出错: {e}")
        
        return []
    
    async def refresh_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[TimeTrade]:
        """
        强制刷新分时交易数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            Optional[TimeTrade]: 刷新后的分时交易数据
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 清除缓存
        cache_key = f"time_trade:{stock_code}:{time_level}:latest"
        await self.delete_from_cache(cache_key)
        
        # 从API获取最新数据
        return await self.get_latest_time_trade(stock_code, time_level)


class KDJIndicatorDAO(BaseIndicatorDAO):
    """
    KDJ指标数据DAO
    """
    
    def __init__(self):
        """初始化KDJIndicatorDAO"""
        super().__init__(KDJIndicator, "kdj_indicator", KDJ_INDICATOR_MAPPING)
    
    async def get_latest_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[KDJIndicator]:
        """
        获取最新的KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            Optional[KDJIndicator]: 最新的KDJ指标数据
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"kdj:{stock_code}:{time_level}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从API获取
        try:
            api_data = await self.api.get_kdj(stock_code, time_level)
            if api_data:
                # 处理数据
                mapped_data = self._map_api_to_model(api_data)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level
                
                # 保存到数据库
                kdj = await self._save_or_update(mapped_data)
                
                # 更新缓存
                await self.set_to_cache(cache_key, kdj, timeout=300)  # 5分钟
                return kdj
        except Exception as e:
            logger.error(f"获取最新KDJ指标数据出错: {e}")
        
        # 3. 从数据库获取最新记录
        try:
            latest_data = await KDJIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time').afirst()
            
            if latest_data:
                # 更新缓存
                await self.set_to_cache(cache_key, latest_data, timeout=300)
                return latest_data
        except Exception as e:
            logger.error(f"从数据库获取最新KDJ指标数据出错: {e}")
        
        return None
    
    async def get_history_kdj(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 1000) -> List[KDJIndicator]:
        """
        获取历史KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            limit: 返回记录数量限制
            
        Returns:
            List[KDJIndicator]: 历史KDJ指标数据列表
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"kdj:{stock_code}:{time_level}:history:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 先尝试从数据库获取
        try:
            db_data = await KDJIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if db_data and len(db_data) >= limit * 0.8:  # 如果数据库中有足够的数据
                # 更新缓存
                await self.set_to_cache(cache_key, list(db_data), timeout=3600)  # 1小时
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取历史KDJ指标数据出错: {e}")
        
        # 3. 从API获取完整历史数据
        try:
            api_data_list = await self.api.get_history_kdj(stock_code, time_level)
            if api_data_list:
                # 处理数据
                processed_data = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level
                    processed_data.append(mapped_data)
                
                # 批量保存到数据库
                saved_kdjs = await self._batch_save_or_update(processed_data)
                
                # 返回有限数量的记录
                result = sorted(saved_kdjs, key=lambda x: x.trade_time, reverse=True)[:limit]
                
                # 更新缓存
                await self.set_to_cache(cache_key, result, timeout=3600)  # 1小时
                return result
        except Exception as e:
            logger.error(f"获取历史KDJ指标数据出错: {e}")
        
        # 如果API获取失败，再尝试从数据库返回已有数据
        try:
            db_data = await KDJIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if db_data:
                await self.set_to_cache(cache_key, list(db_data), timeout=1800)  # 30分钟
                return db_data
        except Exception as e:
            logger.error(f"重新从数据库获取历史KDJ指标数据出错: {e}")
        
        return []


class MACDIndicatorDAO(BaseIndicatorDAO):
    """
    MACD指标数据DAO
    """
    
    def __init__(self):
        """初始化MACDIndicatorDAO"""
        super().__init__(MACDIndicator, "macd_indicator", MACD_INDICATOR_MAPPING)
    
    async def get_latest_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[MACDIndicator]:
        """
        获取最新的MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            Optional[MACDIndicator]: 最新的MACD指标数据
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"macd:{stock_code}:{time_level}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从API获取
        try:
            api_data = await self.api.get_macd(stock_code, time_level)
            if api_data:
                # 处理数据
                mapped_data = self._map_api_to_model(api_data)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level
                
                # 保存到数据库
                macd = await self._save_or_update(mapped_data)
                
                # 更新缓存
                await self.set_to_cache(cache_key, macd, timeout=300)  # 5分钟
                return macd
        except Exception as e:
            logger.error(f"获取最新MACD指标数据出错: {e}")
        
        # 3. 从数据库获取最新记录
        try:
            latest_data = await MACDIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time').afirst()
            
            if latest_data:
                # 更新缓存
                await self.set_to_cache(cache_key, latest_data, timeout=300)
                return latest_data
        except Exception as e:
            logger.error(f"从数据库获取最新MACD指标数据出错: {e}")
        
        return None
    
    async def get_history_macd(self, stock_code: str, time_level: Union[TimeLevel, str], 
                              limit: int = 1000) -> List[MACDIndicator]:
        """
        获取历史MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            limit: 返回记录数量限制
            
        Returns:
            List[MACDIndicator]: 历史MACD指标数据列表
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"macd:{stock_code}:{time_level}:history:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 先尝试从数据库获取
        try:
            db_data = await MACDIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if db_data and len(db_data) >= limit * 0.8:  # 如果数据库中有足够的数据
                # 更新缓存
                await self.set_to_cache(cache_key, list(db_data), timeout=3600)  # 1小时
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取历史MACD指标数据出错: {e}")
        
        # 3. 从API获取完整历史数据
        try:
            api_data_list = await self.api.get_history_macd(stock_code, time_level)
            if api_data_list:
                # 处理数据
                processed_data = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level
                    processed_data.append(mapped_data)
                
                # 批量保存到数据库
                saved_macds = await self._batch_save_or_update(processed_data)
                
                # 返回有限数量的记录
                result = sorted(saved_macds, key=lambda x: x.trade_time, reverse=True)[:limit]
                
                # 更新缓存
                await self.set_to_cache(cache_key, result, timeout=3600)  # 1小时
                return result
        except Exception as e:
            logger.error(f"获取历史MACD指标数据出错: {e}")
        
        # 如果API获取失败，再尝试从数据库返回已有数据
        try:
            db_data = await MACDIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if db_data:
                await self.set_to_cache(cache_key, list(db_data), timeout=1800)  # 30分钟
                return db_data
        except Exception as e:
            logger.error(f"重新从数据库获取历史MACD指标数据出错: {e}")
        
        return []


class MAIndicatorDAO(BaseIndicatorDAO):
    """
    MA指标数据DAO
    """
    
    def __init__(self):
        """初始化MAIndicatorDAO"""
        super().__init__(MAIndicator, "ma_indicator", MA_INDICATOR_MAPPING)
    
    async def get_latest_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[MAIndicator]:
        """
        获取最新的MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            Optional[MAIndicator]: 最新的MA指标数据
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"ma:{stock_code}:{time_level}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从API获取
        try:
            api_data = await self.api.get_ma(stock_code, time_level)
            if api_data:
                # 处理数据
                mapped_data = self._map_api_to_model(api_data)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level
                
                # 保存到数据库
                ma = await self._save_or_update(mapped_data)
                
                # 更新缓存
                await self.set_to_cache(cache_key, ma, timeout=300)  # 5分钟
                return ma
        except Exception as e:
            logger.error(f"获取最新MA指标数据出错: {e}")
        
        # 3. 从数据库获取最新记录
        try:
            latest_data = await MAIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time').afirst()
            
            if latest_data:
                # 更新缓存
                await self.set_to_cache(cache_key, latest_data, timeout=300)
                return latest_data
        except Exception as e:
            logger.error(f"从数据库获取最新MA指标数据出错: {e}")
        
        return None
    
    async def get_history_ma(self, stock_code: str, time_level: Union[TimeLevel, str], 
                            limit: int = 1000) -> List[MAIndicator]:
        """
        获取历史MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            limit: 返回记录数量限制
            
        Returns:
            List[MAIndicator]: 历史MA指标数据列表
        """
        # 规范化time_level参数
        if isinstance(time_level, TimeLevel):
            time_level = time_level.value
        
        # 1. 从缓存获取
        cache_key = f"ma:{stock_code}:{time_level}:history:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 先尝试从数据库获取
        try:
            db_data = await MAIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if db_data and len(db_data) >= limit * 0.8:  # 如果数据库中有足够的数据
                # 更新缓存
                await self.set_to_cache(cache_key, list(db_data), timeout=3600)  # 1小时
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取历史MA指标数据出错: {e}")
        
        # 3. 从API获取完整历史数据
        try:
            api_data_list = await self.api.get_history_ma(stock_code, time_level)
            if api_data_list:
                # 处理数据
                processed_data = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level
                    processed_data.append(mapped_data)
                
                # 批量保存到数据库
                saved_mas = await self._batch_save_or_update(processed_data)
                
                # 返回有限数量的记录
                result = sorted(saved_mas, key=lambda x: x.trade_time, reverse=True)[:limit]
                
                # 更新缓存
                await self.set_to_cache(cache_key, result, timeout=3600)  # 1小时
                return result
        except Exception as e:
            logger.error(f"获取历史MA指标数据出错: {e}")
        
        # 如果API获取失败，再尝试从数据库返回已有数据
        try:
            db_data = await MAIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level
            ).order_by('-trade_time')[:limit].all()
            
            if db_data:
                await self.set_to_cache(cache_key, list(db_data), timeout=1800)  # 30分钟
                return db_data
        except Exception as e:
            logger.error(f"重新从数据库获取历史MA指标数据出错: {e}")
        
        return []


class BOLLIndicatorDAO(BaseIndicatorDAO[BOLLIndicator]):
    """
    BOLL指标数据DAO
    """
    
    def __init__(self):
        """初始化BOLLIndicatorDAO"""
        super().__init__(BOLLIndicator, "boll_indicator", BOLL_INDICATOR_MAPPING)
    
    async def get_latest_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[BOLLIndicator]:
        """
        获取最新的BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            Optional[BOLLIndicator]: 最新的BOLL指标数据，如不存在则返回None
        """
        time_level_str = str(time_level)
        
        # 1. 从缓存获取
        cache_key = f"{stock_code}:{time_level_str}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从API获取
        try:
            api_data = await self.api.get_boll(stock_code, time_level)
            if api_data:
                # 映射数据
                mapped_data = self._map_api_to_model(api_data, self.mapping)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level_str
                
                # 保存到数据库
                records = await self._get_or_create_many(
                    [mapped_data], 
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                if records:
                    record = records[0]
                    # 更新缓存
                    await self.set_to_cache(cache_key, record, timeout=300)  # 5分钟
                    return record
        except Exception as e:
            logger.error(f"获取最新BOLL指标数据出错: {e}")
        
        # 3. 从数据库获取最新记录
        try:
            record = await BOLLIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level_str
            ).order_by('-trade_time').afirst()
            
            if record:
                # 更新缓存
                await self.set_to_cache(cache_key, record, timeout=300)
                return record
        except Exception as e:
            logger.error(f"从数据库获取最新BOLL指标数据出错: {e}")
        
        return None
    
    async def get_history_boll(self, stock_code: str, time_level: Union[TimeLevel, str], 
                              limit: int = 100) -> List[BOLLIndicator]:
        """
        获取历史BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            limit: 返回的最大记录数
            
        Returns:
            List[BOLLIndicator]: 历史BOLL指标数据列表
        """
        time_level_str = str(time_level)
        
        # 1. 从缓存获取
        cache_key = f"{stock_code}:{time_level_str}:history:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库查询
        try:
            db_records = await BOLLIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level_str
            ).order_by('-trade_time')[:limit].all()
            
            if len(db_records) == limit:
                # 如果数据库中的记录数已满足要求，直接返回
                await self.set_to_cache(cache_key, list(db_records), timeout=3600)  # 1小时
                return db_records
        except Exception as e:
            logger.error(f"从数据库获取历史BOLL指标数据出错: {e}")
            db_records = []
        
        # 3. 从API获取历史数据
        try:
            api_data_list = await self.api.get_history_boll(stock_code, time_level)
            if api_data_list:
                # 映射并保存历史数据
                mapped_items = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data, self.mapping)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level_str
                    mapped_items.append(mapped_data)
                
                saved_records = await self._get_or_create_many(
                    mapped_items,
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                # 对记录进行排序和限制数量
                sorted_records = sorted(
                    saved_records, 
                    key=lambda x: x.trade_time, 
                    reverse=True
                )[:limit]
                
                # 更新缓存
                await self.set_to_cache(cache_key, sorted_records, timeout=3600)
                return sorted_records
        except Exception as e:
            logger.error(f"从API获取历史BOLL指标数据出错: {e}")
        
        # 如果从API获取失败，返回从数据库获取的记录
        if db_records:
            await self.set_to_cache(cache_key, list(db_records), timeout=3600)
        return db_records
    
    async def batch_update_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[BOLLIndicator]:
        """
        批量更新BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            List[BOLLIndicator]: 更新后的BOLL指标数据列表
        """
        time_level_str = str(time_level)
        
        # 清除相关缓存
        await self.clear_stock_cache(stock_code, time_level_str)
        
        # 从API获取历史数据
        try:
            api_data_list = await self.api.get_history_boll(stock_code, time_level)
            if api_data_list:
                # 映射数据
                mapped_items = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data, self.mapping)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level_str
                    mapped_items.append(mapped_data)
                
                # 批量保存
                saved_records = await self._get_or_create_many(
                    mapped_items,
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                logger.info(f"批量更新BOLL指标数据: {stock_code}, {time_level_str}, {len(saved_records)}条")
                return saved_records
        except Exception as e:
            logger.error(f"批量更新BOLL指标数据出错: {e}")
        
        return []


class MACDIndicatorDAO(BaseIndicatorDAO[MACDIndicator]):
    """
    MACD指标数据DAO
    """
    
    def __init__(self):
        """初始化MACDIndicatorDAO"""
        super().__init__(MACDIndicator, "macd_indicator", MACD_INDICATOR_MAPPING)
    
    async def get_latest_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[MACDIndicator]:
        """
        获取最新的MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            Optional[MACDIndicator]: 最新的MACD指标数据，如不存在则返回None
        """
        time_level_str = str(time_level)
        
        # 1. 从缓存获取
        cache_key = f"{stock_code}:{time_level_str}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从API获取
        try:
            api_data = await self.api.get_macd(stock_code, time_level)
            if api_data:
                # 映射数据
                mapped_data = self._map_api_to_model(api_data, self.mapping)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level_str
                
                # 保存到数据库
                records = await self._get_or_create_many(
                    [mapped_data], 
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                if records:
                    record = records[0]
                    # 更新缓存
                    await self.set_to_cache(cache_key, record, timeout=300)  # 5分钟
                    return record
        except Exception as e:
            logger.error(f"获取最新MACD指标数据出错: {e}")
        
        # 3. 从数据库获取最新记录
        try:
            record = await MACDIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level_str
            ).order_by('-trade_time').afirst()
            
            if record:
                # 更新缓存
                await self.set_to_cache(cache_key, record, timeout=300)
                return record
        except Exception as e:
            logger.error(f"从数据库获取最新MACD指标数据出错: {e}")
        
        return None
    
    async def get_history_macd(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 100) -> List[MACDIndicator]:
        """
        获取历史MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            limit: 返回的最大记录数
            
        Returns:
            List[MACDIndicator]: 历史MACD指标数据列表
        """
        time_level_str = str(time_level)
        
        # 1. 从缓存获取
        cache_key = f"{stock_code}:{time_level_str}:history:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库查询
        try:
            db_records = await MACDIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level_str
            ).order_by('-trade_time')[:limit].all()
            
            if len(db_records) == limit:
                # 如果数据库中的记录数已满足要求，直接返回
                await self.set_to_cache(cache_key, list(db_records), timeout=3600)  # 1小时
                return db_records
        except Exception as e:
            logger.error(f"从数据库获取历史MACD指标数据出错: {e}")
            db_records = []
        
        # 3. 从API获取历史数据
        try:
            api_data_list = await self.api.get_history_macd(stock_code, time_level)
            if api_data_list:
                # 映射并保存历史数据
                mapped_items = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data, self.mapping)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level_str
                    mapped_items.append(mapped_data)
                
                saved_records = await self._get_or_create_many(
                    mapped_items,
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                # 对记录进行排序和限制数量
                sorted_records = sorted(
                    saved_records, 
                    key=lambda x: x.trade_time, 
                    reverse=True
                )[:limit]
                
                # 更新缓存
                await self.set_to_cache(cache_key, sorted_records, timeout=3600)
                return sorted_records
        except Exception as e:
            logger.error(f"从API获取历史MACD指标数据出错: {e}")
        
        # 如果从API获取失败，返回从数据库获取的记录
        if db_records:
            await self.set_to_cache(cache_key, list(db_records), timeout=3600)
        return db_records
    
    async def batch_update_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[MACDIndicator]:
        """
        批量更新MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            List[MACDIndicator]: 更新后的MACD指标数据列表
        """
        time_level_str = str(time_level)
        
        # 清除相关缓存
        await self.clear_stock_cache(stock_code, time_level_str)
        
        # 从API获取历史数据
        try:
            api_data_list = await self.api.get_history_macd(stock_code, time_level)
            if api_data_list:
                # 映射数据
                mapped_items = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data, self.mapping)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level_str
                    mapped_items.append(mapped_data)
                
                # 批量保存
                saved_records = await self._get_or_create_many(
                    mapped_items,
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                logger.info(f"批量更新MACD指标数据: {stock_code}, {time_level_str}, {len(saved_records)}条")
                return saved_records
        except Exception as e:
            logger.error(f"批量更新MACD指标数据出错: {e}")
        
        return []


class MAIndicatorDAO(BaseIndicatorDAO[MAIndicator]):
    """
    MA指标数据DAO
    """
    
    def __init__(self):
        """初始化MAIndicatorDAO"""
        super().__init__(MAIndicator, "ma_indicator", MA_INDICATOR_MAPPING)
    
    async def get_latest_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[MAIndicator]:
        """
        获取最新的MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            Optional[MAIndicator]: 最新的MA指标数据，如不存在则返回None
        """
        time_level_str = str(time_level)
        
        # 1. 从缓存获取
        cache_key = f"{stock_code}:{time_level_str}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从API获取
        try:
            api_data = await self.api.get_ma(stock_code, time_level)
            if api_data:
                # 映射数据
                mapped_data = self._map_api_to_model(api_data, self.mapping)
                mapped_data['stock_code'] = stock_code
                mapped_data['time_level'] = time_level_str
                
                # 保存到数据库
                records = await self._get_or_create_many(
                    [mapped_data], 
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                if records:
                    record = records[0]
                    # 更新缓存
                    await self.set_to_cache(cache_key, record, timeout=300)  # 5分钟
                    return record
        except Exception as e:
            logger.error(f"获取最新MA指标数据出错: {e}")
        
        # 3. 从数据库获取最新记录
        try:
            record = await MAIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level_str
            ).order_by('-trade_time').afirst()
            
            if record:
                # 更新缓存
                await self.set_to_cache(cache_key, record, timeout=300)
                return record
        except Exception as e:
            logger.error(f"从数据库获取最新MA指标数据出错: {e}")
        
        return None
    
    async def get_history_ma(self, stock_code: str, time_level: Union[TimeLevel, str], 
                           limit: int = 100) -> List[MAIndicator]:
        """
        获取历史MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            limit: 返回的最大记录数
            
        Returns:
            List[MAIndicator]: 历史MA指标数据列表
        """
        time_level_str = str(time_level)
        
        # 1. 从缓存获取
        cache_key = f"{stock_code}:{time_level_str}:history:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库查询
        try:
            db_records = await MAIndicator.objects.filter(
                stock_code=stock_code,
                time_level=time_level_str
            ).order_by('-trade_time')[:limit].all()
            
            if len(db_records) == limit:
                # 如果数据库中的记录数已满足要求，直接返回
                await self.set_to_cache(cache_key, list(db_records), timeout=3600)  # 1小时
                return db_records
        except Exception as e:
            logger.error(f"从数据库获取历史MA指标数据出错: {e}")
            db_records = []
        
        # 3. 从API获取历史数据
        try:
            api_data_list = await self.api.get_history_ma(stock_code, time_level)
            if api_data_list:
                # 映射并保存历史数据
                mapped_items = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data, self.mapping)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level_str
                    mapped_items.append(mapped_data)
                
                saved_records = await self._get_or_create_many(
                    mapped_items,
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                # 对记录进行排序和限制数量
                sorted_records = sorted(
                    saved_records, 
                    key=lambda x: x.trade_time, 
                    reverse=True
                )[:limit]
                
                # 更新缓存
                await self.set_to_cache(cache_key, sorted_records, timeout=3600)
                return sorted_records
        except Exception as e:
            logger.error(f"从API获取历史MA指标数据出错: {e}")
        
        # 如果从API获取失败，返回从数据库获取的记录
        if db_records:
            await self.set_to_cache(cache_key, list(db_records), timeout=3600)
        return db_records
    
    async def batch_update_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[MAIndicator]:
        """
        批量更新MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 分时级别
            
        Returns:
            List[MAIndicator]: 更新后的MA指标数据列表
        """
        time_level_str = str(time_level)
        
        # 清除相关缓存
        await self.clear_stock_cache(stock_code, time_level_str)
        
        # 从API获取历史数据
        try:
            api_data_list = await self.api.get_history_ma(stock_code, time_level)
            if api_data_list:
                # 映射数据
                mapped_items = []
                for api_data in api_data_list:
                    mapped_data = self._map_api_to_model(api_data, self.mapping)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['time_level'] = time_level_str
                    mapped_items.append(mapped_data)
                
                # 批量保存
                saved_records = await self._get_or_create_many(
                    mapped_items,
                    ['stock_code', 'time_level', 'trade_time']
                )
                
                logger.info(f"批量更新MA指标数据: {stock_code}, {time_level_str}, {len(saved_records)}条")
                return saved_records
        except Exception as e:
            logger.error(f"批量更新MA指标数据出错: {e}")
        
        return []
