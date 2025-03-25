from decimal import Decimal
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, date, time
import json

from django.db import transaction
from django.core.cache import cache
from django.db.models import Q, F

from api_manager.apis.stock_realtime_api import StockRealtimeAPI
from api_manager.mappings.stock_realtime import ABNORMAL_MOVEMENT_MAPPING, BIG_DEAL_MAPPING, LEVEL5_DATA_MAPPING, PRICE_PERCENT_MAPPING, REALTIME_DATA_MAPPING, TIME_DEAL_MAPPING, TRADE_DETAIL_MAPPING
from dao_manager.base_dao import BaseDAO
from models.stock_realtime import AbnormalMovement, BigDeal, Level5Data, PricePercent, RealtimeData, TimeDeal, TradeDetail


logger = logging.getLogger(__name__)

class RealtimeDataDAO(BaseDAO[RealtimeData]):
    """
    实时交易数据DAO
    """
    
    def __init__(self):
        """初始化RealtimeDataDAO"""
        super().__init__(RealtimeData, "realtime_data")
        self.api = StockRealtimeAPI()
        logger.info("初始化RealtimeDataDAO")
    
    async def get_latest_by_code(self, stock_code: str) -> Optional[RealtimeData]:
        """
        获取股票最新的实时交易数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[RealtimeData]: 最新的实时交易数据，如不存在则返回None
        """
        # 1. 首先从缓存获取
        cache_key = f"realtime:{stock_code}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            realtime_data = await RealtimeData.objects.filter(
                stock_code=stock_code
            ).order_by('-update_time').afirst()
            
            if realtime_data:
                # 存入缓存并返回
                await self.set_to_cache(cache_key, realtime_data, 60)  # 实时数据缓存时间较短，60秒
                return realtime_data
        except Exception as e:
            logger.error(f"从数据库获取实时交易数据出错: {e}")
        
        # 3. 数据库未找到，从API获取
        try:
            api_data = await self.api.get_realtime_data(stock_code)
            
            if api_data:
                # 转换日期时间格式
                if 't' in api_data and api_data['t']:
                    try:
                        api_data['t'] = datetime.strptime(api_data['t'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        logger.warning(f"日期时间格式转换失败: {api_data['t']}")
                
                # 映射并保存到数据库
                mapped_data = self._map_api_to_model(api_data, REALTIME_DATA_MAPPING)
                mapped_data['stock_code'] = stock_code
                
                realtime_data = await self._save_to_db(mapped_data)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, realtime_data, 60)
                return realtime_data
        except Exception as e:
            logger.error(f"从API获取实时交易数据出错: {e}")
        
        return None
    
    async def get_realtime_data_batch(self, stock_codes: List[str]) -> Dict[str, RealtimeData]:
        """
        批量获取多只股票的实时交易数据
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict[str, RealtimeData]: 股票代码到实时数据的映射
        """
        if not stock_codes:
            return {}
        
        result = {}
        missing_codes = []
        
        # 1. 批量从缓存获取
        cache_keys = {code: f"realtime:{code}:latest" for code in stock_codes}
        
        # 尝试通过Redis pipeline批量获取缓存
        try:
            from django_redis import get_redis_connection
            redis_conn = get_redis_connection("default")
            pipe = redis_conn.pipeline()
            for code in stock_codes:
                pipe.get(f"realtime:{code}:latest")
            cache_results = pipe.execute()
            
            # 处理缓存结果
            for i, code in enumerate(stock_codes):
                cached_item = cache_results[i]
                if cached_item:
                    try:
                        result[code] = json.loads(cached_item)
                    except:
                        missing_codes.append(code)
                else:
                    missing_codes.append(code)
        except Exception as e:
            logger.warning(f"批量获取缓存失败: {e}")
            missing_codes = stock_codes
        
        if not missing_codes:
            return result
        
        # 2. 从数据库批量获取缺失的股票数据
        try:
            latest_data_queryset = RealtimeData.objects.filter(
                stock_code__in=missing_codes
            ).order_by('stock_code', '-update_time').distinct('stock_code')
            
            latest_data = await latest_data_queryset
            
            for data in latest_data:
                result[data.stock_code] = data
                missing_codes.remove(data.stock_code)
                
                # 更新缓存
                await self.set_to_cache(f"realtime:{data.stock_code}:latest", data, 60)
        except Exception as e:
            logger.error(f"从数据库批量获取实时数据出错: {e}")
        
        if not missing_codes:
            return result
        
        # 3. 从API获取剩余缺失的股票数据
        for code in missing_codes:
            try:
                data = await self.get_latest_by_code(code)
                if data:
                    result[code] = data
            except Exception as e:
                logger.error(f"获取股票{code}实时数据出错: {e}")
        
        return result
    
    async def refresh_realtime_data(self, stock_code: str) -> Optional[RealtimeData]:
        """
        强制从API刷新股票实时数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[RealtimeData]: 最新的实时交易数据
        """
        # 清除缓存
        cache_key = f"realtime:{stock_code}:latest"
        await self.delete_from_cache(cache_key)
        
        # 从API获取最新数据
        try:
            api_data = await self.api.get_realtime_data(stock_code)
            
            if api_data:
                # 转换日期时间格式
                if 't' in api_data and api_data['t']:
                    try:
                        api_data['t'] = datetime.strptime(api_data['t'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        logger.warning(f"日期时间格式转换失败: {api_data['t']}")
                
                # 映射并保存到数据库
                mapped_data = self._map_api_to_model(api_data, REALTIME_DATA_MAPPING)
                mapped_data['stock_code'] = stock_code
                
                # 检查是否存在相同数据
                exists = await self._check_if_exists(stock_code, mapped_data.get('update_time'))
                
                if exists:
                    # 检查数据是否有变化
                    if not await self._check_if_data_identical(exists, mapped_data):
                        # 数据有变化，更新
                        realtime_data = await self._update_db(exists.id, mapped_data)
                    else:
                        # 数据相同，直接返回
                        realtime_data = exists
                else:
                    # 不存在，创建新记录
                    realtime_data = await self._save_to_db(mapped_data)
                
                # 更新缓存
                await self.set_to_cache(cache_key, realtime_data, 60)
                return realtime_data
        except Exception as e:
            logger.error(f"刷新实时交易数据出错: {e}")
        
        return None
    
    async def _check_if_exists(self, stock_code: str, update_time: datetime) -> Optional[RealtimeData]:
        """
        检查指定时间的实时数据是否已存在
        
        Args:
            stock_code: 股票代码
            update_time: 更新时间
            
        Returns:
            Optional[RealtimeData]: 如存在则返回对象，否则返回None
        """
        if not update_time:
            return None
            
        try:
            return await RealtimeData.objects.filter(
                stock_code=stock_code,
                update_time=update_time
            ).afirst()
        except Exception as e:
            logger.error(f"检查实时数据是否存在出错: {e}")
            return None
    
    async def _check_if_data_identical(self, existing: RealtimeData, new_data: Dict[str, Any]) -> bool:
        """
        检查新数据与已存在数据是否相同
        
        Args:
            existing: 已存在的数据对象
            new_data: 新数据字典
            
        Returns:
            bool: 是否完全相同
        """
        # 检查关键字段是否相同
        key_fields = ['current_price', 'turnover_value', 'volume', 'price_change_percent']
        
        for field in key_fields:
            if field in new_data and getattr(existing, field) != new_data[field]:
                return False
        
        return True
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
        }
    
    @transaction.atomic
    async def _save_to_db(self, data: Dict[str, Any]) -> RealtimeData:
        """
        保存数据到数据库
        
        Args:
            data: 模型数据
            
        Returns:
            RealtimeData: 保存后的模型实例
        """
        try:
            # 使用Django的ORM创建记录
            instance = RealtimeData(**data)
            await instance.asave()
            logger.debug(f"创建实时数据记录: {data['stock_code']}-{data.get('update_time')}")
            return instance
        except Exception as e:
            logger.error(f"保存实时数据出错: {e}")
            raise
    
    @transaction.atomic
    async def _update_db(self, record_id: int, data: Dict[str, Any]) -> Optional[RealtimeData]:
        """
        更新数据库中的记录
        
        Args:
            record_id: 记录ID
            data: 更新的数据
            
        Returns:
            Optional[RealtimeData]: 更新后的记录
        """
        try:
            instance = await RealtimeData.objects.filter(id=record_id).afirst()
            if instance:
                for field, value in data.items():
                    setattr(instance, field, value)
                await instance.asave()
                logger.debug(f"更新实时数据记录: {instance.stock_code}-{instance.update_time}")
                return instance
            return None
        except Exception as e:
            logger.error(f"更新实时数据出错: {e}")
            return None


class Level5DataDAO(BaseDAO[Level5Data]):
    """
    买卖五档盘口数据DAO
    """
    
    def __init__(self):
        """初始化Level5DataDAO"""
        super().__init__(Level5Data, "level5_data")
        self.api = StockRealtimeAPI()
        logger.info("初始化Level5DataDAO")
    
    async def get_latest_by_code(self, stock_code: str) -> Optional[Level5Data]:
        """
        获取股票最新的买卖五档盘口数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Level5Data]: 最新的买卖五档盘口数据，如不存在则返回None
        """
        # 1. 首先从缓存获取
        cache_key = f"level5:{stock_code}:latest"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，从数据库查询
        try:
            level5_data = await Level5Data.objects.filter(
                stock_code=stock_code
            ).order_by('-update_time').afirst()
            
            if level5_data:
                # 存入缓存并返回
                await self.set_to_cache(cache_key, level5_data, 60)  # 实时数据缓存时间较短，60秒
                return level5_data
        except Exception as e:
            logger.error(f"从数据库获取买卖五档盘口数据出错: {e}")
        
        # 3. 数据库未找到，从API获取
        try:
            api_data = await self.api.get_level5_data(stock_code)
            
            if api_data:
                # 转换日期时间格式
                if 't' in api_data and api_data['t']:
                    try:
                        api_data['t'] = datetime.strptime(api_data['t'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        logger.warning(f"日期时间格式转换失败: {api_data['t']}")
                
                # 映射并保存到数据库
                mapped_data = self._map_api_to_model(api_data, LEVEL5_DATA_MAPPING)
                mapped_data['stock_code'] = stock_code
                
                level5_data = await self._save_to_db(mapped_data)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, level5_data, 60)
                return level5_data
        except Exception as e:
            logger.error(f"从API获取买卖五档盘口数据出错: {e}")
        
        return None
    
    async def refresh_level5_data(self, stock_code: str) -> Optional[Level5Data]:
        """
        强制从API刷新买卖五档盘口数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Level5Data]: 最新的买卖五档盘口数据
        """
        # 清除缓存
        cache_key = f"level5:{stock_code}:latest"
        await self.delete_from_cache(cache_key)
        
        # 从API获取最新数据
        try:
            api_data = await self.api.get_level5_data(stock_code)
            
            if api_data:
                # 转换日期时间格式
                if 't' in api_data and api_data['t']:
                    try:
                        api_data['t'] = datetime.strptime(api_data['t'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        logger.warning(f"日期时间格式转换失败: {api_data['t']}")
                
                # 映射并保存到数据库
                mapped_data = self._map_api_to_model(api_data, LEVEL5_DATA_MAPPING)
                mapped_data['stock_code'] = stock_code
                
                # 检查是否存在相同数据
                exists = await self._check_if_exists(stock_code, mapped_data.get('update_time'))
                
                if exists:
                    # 检查数据是否有变化
                    if not await self._check_if_data_identical(exists, mapped_data):
                        # 数据有变化，更新
                        level5_data = await self._update_db(exists.id, mapped_data)
                    else:
                        # 数据相同，直接返回
                        level5_data = exists
                else:
                    # 不存在，创建新记录
                    level5_data = await self._save_to_db(mapped_data)
                
                # 更新缓存
                await self.set_to_cache(cache_key, level5_data, 60)
                return level5_data
        except Exception as e:
            logger.error(f"刷新买卖五档盘口数据出错: {e}")
        
        return None
    
    async def _check_if_exists(self, stock_code: str, update_time: datetime) -> Optional[Level5Data]:
        """
        检查指定时间的盘口数据是否已存在
        
        Args:
            stock_code: 股票代码
            update_time: 更新时间
            
        Returns:
            Optional[Level5Data]: 如存在则返回对象，否则返回None
        """
        if not update_time:
            return None
            
        try:
            return await Level5Data.objects.filter(
                stock_code=stock_code,
                update_time=update_time
            ).afirst()
        except Exception as e:
            logger.error(f"检查盘口数据是否存在出错: {e}")
            return None
    
    async def _check_if_data_identical(self, existing: Level5Data, new_data: Dict[str, Any]) -> bool:
        """
        检查新数据与已存在数据是否相同
        
        Args:
            existing: 已存在的数据对象
            new_data: 新数据字典
            
        Returns:
            bool: 是否完全相同
        """
        # 检查关键字段是否相同（买一卖一价格和量）
        key_fields = ['buy_price1', 'buy_volume1', 'sell_price1', 'sell_volume1']
        
        for field in key_fields:
            if field in new_data and getattr(existing, field) != new_data[field]:
                return False
        
        return True
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
        }
    
    @transaction.atomic
    async def _save_to_db(self, data: Dict[str, Any]) -> Level5Data:
        """
        保存数据到数据库
        
        Args:
            data: 模型数据
            
        Returns:
            Level5Data: 保存后的模型实例
        """
        try:
            # 使用Django的ORM创建记录
            instance = Level5Data(**data)
            await instance.asave()
            logger.debug(f"创建盘口数据记录: {data['stock_code']}-{data.get('update_time')}")
            return instance
        except Exception as e:
            logger.error(f"保存盘口数据出错: {e}")
            raise
    
    @transaction.atomic
    async def _update_db(self, record_id: int, data: Dict[str, Any]) -> Optional[Level5Data]:
        """
        更新数据库中的记录
        
        Args:
            record_id: 记录ID
            data: 更新的数据
            
        Returns:
            Optional[Level5Data]: 更新后的记录
        """
        try:
            instance = await Level5Data.objects.filter(id=record_id).afirst()
            if instance:
                for field, value in data.items():
                    setattr(instance, field, value)
                await instance.asave()
                logger.debug(f"更新盘口数据记录: {instance.stock_code}-{instance.update_time}")
                return instance
            return None
        except Exception as e:
            logger.error(f"更新盘口数据出错: {e}")
            return None


class TradeDetailDAO(BaseDAO[TradeDetail]):
    """
    逐笔交易数据DAO
    """
    
    def __init__(self):
        """初始化TradeDetailDAO"""
        super().__init__(TradeDetail, "trade_detail")
        self.api = StockRealtimeAPI()
        logger.info("初始化TradeDetailDAO")
    
    async def get_by_code_and_date(self, stock_code: str, trade_date: Optional[date] = None) -> List[TradeDetail]:
        """
        获取指定股票和日期的逐笔交易数据
        
        Args:
            stock_code: 股票代码
            trade_date: 交易日期，如果不指定则使用最新交易日
            
        Returns:
            List[TradeDetail]: 逐笔交易数据列表
        """
        # 1. 首先从缓存获取
        if trade_date:
            date_str = trade_date.isoformat()
        else:
            date_str = 'latest'
        
        cache_key = f"trade_detail:{stock_code}:{date_str}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，确定查询日期
        if not trade_date:
            # 获取最新的交易日期
            try:
                latest_record = await TradeDetail.objects.filter(
                    stock_code=stock_code
                ).order_by('-trade_date').afirst()
                
                if latest_record:
                    trade_date = latest_record.trade_date
                else:
                    # 没有记录，默认使用今天
                    trade_date = date.today()
            except Exception as e:
                logger.error(f"获取最新交易日期出错: {e}")
                trade_date = date.today()
        
        # 3. 从数据库查询
        try:
            trades = await TradeDetail.objects.filter(
                stock_code=stock_code,
                trade_date=trade_date
            ).order_by('-trade_time')
            
            if trades:
                # 存入缓存并返回
                await self.set_to_cache(cache_key, list(trades), 3600)  # 历史数据可以缓存更长时间
                return trades
        except Exception as e:
            logger.error(f"从数据库获取逐笔交易数据出错: {e}")
        
        # 4. 数据库未找到，从API获取
        try:
            api_data_list = await self.api.get_onebyone_trades(stock_code)
            
            if api_data_list:
                # 处理并批量保存数据
                trades = await self._process_and_save_trades(stock_code, api_data_list)
                
                # 过滤出指定日期的交易
                if trade_date:
                    trades = [t for t in trades if t.trade_date == trade_date]
                
                # 保存到缓存
                await self.set_to_cache(cache_key, trades, 3600)
                return trades
        except Exception as e:
            logger.error(f"从API获取逐笔交易数据出错: {e}")
        
        return []
    
    async def _process_and_save_trades(self, stock_code: str, api_data_list: List[Dict[str, Any]]) -> List[TradeDetail]:
        """
        处理并批量保存交易数据
        
        Args:
            stock_code: 股票代码
            api_data_list: API返回的数据列表
            
        Returns:
            List[TradeDetail]: 保存后的交易数据列表
        """
        if not api_data_list:
            return []
        
        # 处理日期和时间
        processed_data = []
        for item in api_data_list:
            if 'd' in item and 't' in item:
                try:
                    trade_date = date.fromisoformat(item['d'])
                    trade_time = datetime.strptime(item['t'], '%H:%M:%S').time()
                    
                    # 准备映射数据
                    mapped_data = self._map_api_to_model(item, TRADE_DETAIL_MAPPING)
                    mapped_data['stock_code'] = stock_code
                    mapped_data['trade_date'] = trade_date
                    mapped_data['trade_time'] = trade_time
                    
                    processed_data.append(mapped_data)
                except Exception as e:
                    logger.warning(f"处理交易数据日期时间出错: {e}, 数据: {item}")
        
        # 查询数据库中已存在的记录
        existing_records = set()
        try:
            # 获取日期列表
            dates = {data['trade_date'] for data in processed_data}
            
            for trade_date in dates:
                records = await TradeDetail.objects.filter(
                    stock_code=stock_code,
                    trade_date=trade_date
                ).values_list('trade_time', 'volume', 'price')
                
                for record in records:
                    # 使用时间、成交量和价格作为唯一标识
                    key = (record[0], record[1], record[2])
                    existing_records.add(key)
        except Exception as e:
            logger.error(f"查询已存在交易记录出错: {e}")
        
        # 过滤出需要插入的新记录
        new_records = []
        for data in processed_data:
            key = (data['trade_time'], data['volume'], data['price'])
            if key not in existing_records:
                new_records.append(data)
        
        # 批量创建新记录
        created_records = []
        if new_records:
            try:
                # 使用bulk_create批量创建记录
                instances = [TradeDetail(**data) for data in new_records]
                created = await TradeDetail.objects.abulk_create(instances)
                created_records.extend(created)
                logger.info(f"批量创建交易记录成功: {len(created)}条")
            except Exception as e:
                logger.error(f"批量创建交易记录出错: {e}")
        
        # 返回所有记录（包括已存在的和新创建的）
        try:
            all_trades = await TradeDetail.objects.filter(
                stock_code=stock_code,
                trade_date__in=[data['trade_date'] for data in processed_data]
            ).order_by('-trade_time')
            
            return all_trades
        except Exception as e:
            logger.error(f"查询所有交易记录出错: {e}")
            return created_records
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        # 注意：日期和时间字段会在调用方法中特殊处理
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
            and api_field not in ['d', 't']  # 排除日期和时间字段
        }


class TimeDealDAO(BaseDAO[TimeDeal]):
    """
    分时成交数据DAO
    """
    
    def __init__(self):
        """初始化TimeDealDAO"""
        super().__init__(TimeDeal, "time_deal")
        self.api = StockRealtimeAPI()
        logger.info("初始化TimeDealDAO")
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        # 注意：日期和时间字段会在外部方法中特殊处理
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
            and api_field not in ['d', 't']  # 排除日期和时间字段，这些需要特殊处理
        }
    
    async def get_daily_time_deals(self, stock_code: str, trade_date: Optional[date] = None) -> List[TimeDeal]:
        """
        获取指定股票某日的所有分时成交数据
        
        Args:
            stock_code: 股票代码
            trade_date: 交易日期，默认为当天
            
        Returns:
            List[TimeDeal]: 分时成交数据列表
        """
        if trade_date is None:
            trade_date = date.today()
        
        # 1. 从缓存获取
        cache_key = f"time_deal:{stock_code}:{trade_date.isoformat()}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库查询
        try:
            db_data = await TimeDeal.objects.filter(
                stock_code=stock_code,
                trade_date=trade_date
            ).order_by('-trade_time').all()
            
            if db_data:
                # 更新缓存，当天数据缓存时间较长
                await self.set_to_cache(cache_key, list(db_data), timeout=3600)  # 1小时
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取分时成交数据出错: {e}")
        
        # 3. 从API获取
        try:
            api_data_list = await self.api.get_time_deal(stock_code)
            if api_data_list:
                # 筛选指定日期的数据
                filtered_data = [
                    data for data in api_data_list 
                    if 'd' in data and data['d'] and self._parse_date(data['d']) == trade_date
                ]
                
                if filtered_data:
                    # 批量保存到数据库
                    saved_deals = await self._batch_save_deals(stock_code, filtered_data)
                    
                    # 更新缓存
                    await self.set_to_cache(cache_key, saved_deals, timeout=3600)
                    return saved_deals
        except Exception as e:
            logger.error(f"从API获取分时成交数据出错: {e}")
        
        return []
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """
        解析日期字符串
        
        Args:
            date_str: 日期字符串 (格式: YYYY-MM-DD)
            
        Returns:
            Optional[date]: 解析后的日期对象，解析失败则返回None
        """
        try:
            return date.fromisoformat(date_str)
        except (ValueError, TypeError):
            logger.warning(f"日期解析失败: {date_str}")
            return None
    
    def _parse_time(self, time_str: str) -> Optional[time]:
        """
        解析时间字符串
        
        Args:
            time_str: 时间字符串 (格式: HH:MM:SS)
            
        Returns:
            Optional[time]: 解析后的时间对象，解析失败则返回None
        """
        try:
            return datetime.strptime(time_str, '%H:%M:%S').time()
        except (ValueError, TypeError):
            logger.warning(f"时间解析失败: {time_str}")
            return None
    
    async def _batch_save_deals(self, stock_code: str, api_data_list: List[Dict[str, Any]]) -> List[TimeDeal]:
        """
        批量保存分时成交数据
        
        Args:
            stock_code: 股票代码
            api_data_list: API返回的数据列表
            
        Returns:
            List[TimeDeal]: 保存后的分时成交数据列表
        """
        if not api_data_list:
            return []
        
        # 1. 从数据库获取已存在的记录
        dates = set()
        for item in api_data_list:
            if 'd' in item and item['d']:
                parsed_date = self._parse_date(item['d'])
                if parsed_date:
                    dates.add(parsed_date)
        
        existing_records = {}
        for d in dates:
            records = await TimeDeal.objects.filter(
                stock_code=stock_code,
                trade_date=d
            ).values_list('trade_date', 'trade_time')
            
            for record in records:
                key = (record[0], record[1])
                existing_records[key] = True
        
        # 2. 准备新的数据记录
        to_create = []
        for item in api_data_list:
            if 'd' in item and 't' in item:
                parsed_date = self._parse_date(item['d'])
                parsed_time = self._parse_time(item['t'])
                
                if parsed_date and parsed_time:
                    key = (parsed_date, parsed_time)
                    
                    # 检查是否已存在
                    if key not in existing_records:
                        mapped_data = self._map_api_to_model(item, TIME_DEAL_MAPPING)
                        mapped_data['stock_code'] = stock_code
                        mapped_data['trade_date'] = parsed_date
                        mapped_data['trade_time'] = parsed_time
                        
                        to_create.append(TimeDeal(**mapped_data))
        
        # 3. 批量创建新记录
        created_records = []
        if to_create:
            try:
                created_records = await TimeDeal.objects.abulk_create(to_create)
                logger.info(f"批量创建分时成交记录: {len(created_records)}条")
            except Exception as e:
                logger.error(f"批量创建分时成交记录出错: {e}")
        
        # 4. 返回所有记录（包括新创建的和已存在的）
        try:
            all_deals = await TimeDeal.objects.filter(
                stock_code=stock_code,
                trade_date__in=list(dates)
            ).order_by('-trade_date', '-trade_time').all()
            
            return all_deals
        except Exception as e:
            logger.error(f"获取所有分时成交记录出错: {e}")
            return created_records


class PricePercentDAO(BaseDAO[PricePercent]):
    """
    分价成交占比数据DAO
    """
    
    def __init__(self):
        """初始化PricePercentDAO"""
        super().__init__(PricePercent, "price_percent")
        self.api = StockRealtimeAPI()
        logger.info("初始化PricePercentDAO")
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        # 注意：日期字段会在外部方法中特殊处理
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
            and api_field not in ['d']  # 排除日期字段，这需要特殊处理
        }
    
    async def get_daily_price_percents(self, stock_code: str, trade_date: Optional[date] = None) -> List[PricePercent]:
        """
        获取指定股票某日的所有分价成交占比数据
        
        Args:
            stock_code: 股票代码
            trade_date: 交易日期，默认为当天
            
        Returns:
            List[PricePercent]: 分价成交占比数据列表
        """
        if trade_date is None:
            trade_date = date.today()
        
        # 1. 从缓存获取
        cache_key = f"price_percent:{stock_code}:{trade_date.isoformat()}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库查询
        try:
            db_data = await PricePercent.objects.filter(
                stock_code=stock_code,
                trade_date=trade_date
            ).order_by('price').all()
            
            if db_data:
                # 更新缓存
                await self.set_to_cache(cache_key, list(db_data), timeout=3600)  # 1小时
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取分价成交占比数据出错: {e}")
        
        # 3. 从API获取
        try:
            api_data_list = await self.api.get_real_percent(stock_code)
            if api_data_list:
                # 筛选指定日期的数据
                filtered_data = [
                    data for data in api_data_list 
                    if 'd' in data and data['d'] and self._parse_date(data['d']) == trade_date
                ]
                
                if filtered_data:
                    # 批量保存到数据库
                    saved_percents = await self._batch_save_percents(stock_code, filtered_data)
                    
                    # 更新缓存
                    await self.set_to_cache(cache_key, saved_percents, timeout=3600)
                    return saved_percents
        except Exception as e:
            logger.error(f"从API获取分价成交占比数据出错: {e}")
        
        return []
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """
        解析日期字符串
        
        Args:
            date_str: 日期字符串 (格式: YYYY-MM-DD)
            
        Returns:
            Optional[date]: 解析后的日期对象，解析失败则返回None
        """
        try:
            return date.fromisoformat(date_str)
        except (ValueError, TypeError):
            logger.warning(f"日期解析失败: {date_str}")
            return None
    
    async def _batch_save_percents(self, stock_code: str, api_data_list: List[Dict[str, Any]]) -> List[PricePercent]:
        """
        批量保存分价成交占比数据
        
        Args:
            stock_code: 股票代码
            api_data_list: API返回的数据列表
            
        Returns:
            List[PricePercent]: 保存后的分价成交占比数据列表
        """
        if not api_data_list:
            return []
        
        # 1. 从数据库获取已存在的记录
        dates = set()
        for item in api_data_list:
            if 'd' in item and item['d']:
                parsed_date = self._parse_date(item['d'])
                if parsed_date:
                    dates.add(parsed_date)
        
        existing_records = {}
        for d in dates:
            records = await PricePercent.objects.filter(
                stock_code=stock_code,
                trade_date=d
            ).values_list('trade_date', 'price')
            
            for record in records:
                key = (record[0], record[1])
                existing_records[key] = True
        
        # 2. 准备新的数据记录
        to_create = []
        for item in api_data_list:
            if 'd' in item and 'p' in item:
                parsed_date = self._parse_date(item['d'])
                price = item['p']
                
                if parsed_date and price is not None:
                    key = (parsed_date, Decimal(str(price)))
                    
                    # 检查是否已存在
                    if key not in existing_records:
                        mapped_data = self._map_api_to_model(item, PRICE_PERCENT_MAPPING)
                        mapped_data['stock_code'] = stock_code
                        mapped_data['trade_date'] = parsed_date
                        
                        to_create.append(PricePercent(**mapped_data))
        
        # 3. 批量创建新记录
        created_records = []
        if to_create:
            try:
                created_records = await PricePercent.objects.abulk_create(to_create)
                logger.info(f"批量创建分价成交占比记录: {len(created_records)}条")
            except Exception as e:
                logger.error(f"批量创建分价成交占比记录出错: {e}")
        
        # 4. 返回所有记录（包括新创建的和已存在的）
        try:
            all_percents = await PricePercent.objects.filter(
                stock_code=stock_code,
                trade_date__in=list(dates)
            ).order_by('-trade_date', 'price').all()
            
            return all_percents
        except Exception as e:
            logger.error(f"获取所有分价成交占比记录出错: {e}")
            return created_records


class BigDealDAO(BaseDAO[BigDeal]):
    """
    逐笔大单交易数据DAO
    """
    
    def __init__(self):
        """初始化BigDealDAO"""
        super().__init__(BigDeal, "big_deal")
        self.api = StockRealtimeAPI()
        logger.info("初始化BigDealDAO")
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        # 注意：日期和时间字段会在外部方法中特殊处理
        return {
            model_field: api_data.get(api_field)
            for api_field, model_field in mapping.items()
            if api_field in api_data and api_data.get(api_field) is not None
            and api_field not in ['d', 't']  # 排除日期和时间字段，这些需要特殊处理
        }
    
    async def get_daily_big_deals(self, stock_code: str, trade_date: Optional[date] = None) -> List[BigDeal]:
        """
        获取指定股票某日的所有逐笔大单交易数据
        
        Args:
            stock_code: 股票代码
            trade_date: 交易日期，默认为当天
            
        Returns:
            List[BigDeal]: 逐笔大单交易数据列表
        """
        if trade_date is None:
            trade_date = date.today()
        
        # 1. 从缓存获取
        cache_key = f"big_deal:{stock_code}:{trade_date.isoformat()}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库查询
        try:
            db_data = await BigDeal.objects.filter(
                stock_code=stock_code,
                trade_date=trade_date
            ).order_by('-trade_time').all()
            
            if db_data:
                # 更新缓存，当天数据缓存时间较长
                await self.set_to_cache(cache_key, list(db_data), timeout=3600)  # 1小时
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取逐笔大单交易数据出错: {e}")
        
        # 3. 从API获取
        try:
            api_data_list = await self.api.get_big_deal(stock_code)
            if api_data_list:
                # 筛选指定日期的数据
                filtered_data = [
                    data for data in api_data_list 
                    if 'd' in data and data['d'] and self._parse_date(data['d']) == trade_date
                ]
                
                if filtered_data:
                    # 批量保存到数据库
                    saved_deals = await self._batch_save_big_deals(stock_code, filtered_data)
                    
                    # 更新缓存
                    await self.set_to_cache(cache_key, saved_deals, timeout=3600)
                    return saved_deals
        except Exception as e:
            logger.error(f"从API获取逐笔大单交易数据出错: {e}")
        
        return []
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """
        解析日期字符串
        
        Args:
            date_str: 日期字符串 (格式: YYYY-MM-DD)
            
        Returns:
            Optional[date]: 解析后的日期对象，解析失败则返回None
        """
        try:
            return date.fromisoformat(date_str)
        except (ValueError, TypeError):
            logger.warning(f"日期解析失败: {date_str}")
            return None
    
    def _parse_time(self, time_str: str) -> Optional[time]:
        """
        解析时间字符串
        
        Args:
            time_str: 时间字符串 (格式: HH:MM:SS)
            
        Returns:
            Optional[time]: 解析后的时间对象，解析失败则返回None
        """
        try:
            return datetime.strptime(time_str, '%H:%M:%S').time()
        except (ValueError, TypeError):
            logger.warning(f"时间解析失败: {time_str}")
            return None
    
    async def _batch_save_big_deals(self, stock_code: str, api_data_list: List[Dict[str, Any]]) -> List[BigDeal]:
        """
        批量保存逐笔大单交易数据
        
        Args:
            stock_code: 股票代码
            api_data_list: API返回的数据列表
            
        Returns:
            List[BigDeal]: 保存后的逐笔大单交易数据列表
        """
        if not api_data_list:
            return []
        
        # 1. 从数据库获取已存在的记录
        dates = set()
        for item in api_data_list:
            if 'd' in item and item['d']:
                parsed_date = self._parse_date(item['d'])
                if parsed_date:
                    dates.add(parsed_date)
        
        existing_records = {}
        for d in dates:
            records = await BigDeal.objects.filter(
                stock_code=stock_code,
                trade_date=d
            ).values_list('trade_date', 'trade_time', 'volume', 'price')
            
            for record in records:
                key = (record[0], record[1], record[2], record[3])
                existing_records[key] = True
        
        # 2. 准备新的数据记录
        to_create = []
        for item in api_data_list:
            if all(k in item for k in ['d', 't', 'v', 'p']):
                parsed_date = self._parse_date(item['d'])
                parsed_time = self._parse_time(item['t'])
                volume = item['v']
                price = item['p']
                
                if parsed_date and parsed_time and volume is not None and price is not None:
                    key = (parsed_date, parsed_time, volume, Decimal(str(price)))
                    
                    # 检查是否已存在
                    if key not in existing_records:
                        mapped_data = self._map_api_to_model(item, BIG_DEAL_MAPPING)
                        mapped_data['stock_code'] = stock_code
                        mapped_data['trade_date'] = parsed_date
                        mapped_data['trade_time'] = parsed_time
                        
                        to_create.append(BigDeal(**mapped_data))
        
        # 3. 批量创建新记录
        created_records = []
        if to_create:
            try:
                created_records = await BigDeal.objects.abulk_create(to_create)
                logger.info(f"批量创建逐笔大单交易记录: {len(created_records)}条")
            except Exception as e:
                logger.error(f"批量创建逐笔大单交易记录出错: {e}")
        
        # 4. 返回所有记录（包括新创建的和已存在的）
        try:
            all_deals = await BigDeal.objects.filter(
                stock_code=stock_code,
                trade_date__in=list(dates)
            ).order_by('-trade_date', '-trade_time').all()
            
            return all_deals
        except Exception as e:
            logger.error(f"获取所有逐笔大单交易记录出错: {e}")
            return created_records


class AbnormalMovementDAO(BaseDAO[AbnormalMovement]):
    """
    盘中异动数据DAO
    """
    
    def __init__(self):
        """初始化AbnormalMovementDAO"""
        super().__init__(AbnormalMovement, "abnormal_movement")
        self.api = StockRealtimeAPI()
        logger.info("初始化AbnormalMovementDAO")
    
    def _map_api_to_model(self, api_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        将API返回的数据映射到模型字段
        
        Args:
            api_data: API返回的数据
            mapping: 字段映射关系
            
        Returns:
            Dict[str, Any]: 映射后的数据
        """
        # 盘中异动数据的时间字段需要特殊处理
        result = {}
        for api_field, model_field in mapping.items():
            if api_field in api_data and api_data.get(api_field) is not None:
                if api_field == 't':  # 处理时间字段
                    try:
                        result[model_field] = datetime.strptime(api_data[api_field], '%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        logger.warning(f"时间解析失败: {api_data[api_field]}")
                        result[model_field] = datetime.now()
                else:
                    result[model_field] = api_data[api_field]
        return result
    
    async def get_latest_movements(self, limit: int = 1000) -> List[AbnormalMovement]:
        """
        获取最新的盘中异动数据
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            List[AbnormalMovement]: 盘中异动数据列表
        """
        # 1. 从缓存获取
        cache_key = f"abnormal_movement:latest:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从API获取
        try:
            api_data_list = await self.api.get_abnormal_movements()
            if api_data_list:
                # 最多取limit条记录
                api_data_list = api_data_list[:limit]
                
                # 批量保存到数据库
                saved_movements = await self._batch_save_movements(api_data_list)
                
                # 更新缓存，异动数据缓存时间较短
                await self.set_to_cache(cache_key, saved_movements, timeout=300)  # 5分钟
                return saved_movements
        except Exception as e:
            logger.error(f"从API获取盘中异动数据出错: {e}")
        
        # 3. 从数据库获取最新记录
        try:
            db_data = await AbnormalMovement.objects.order_by('-movement_time')[:limit].all()
            
            if db_data:
                # 更新缓存
                await self.set_to_cache(cache_key, list(db_data), timeout=300)
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取盘中异动数据出错: {e}")
        
        return []
    
    async def get_stock_movements(self, stock_code: str, limit: int = 100) -> List[AbnormalMovement]:
        """
        获取指定股票的盘中异动数据
        
        Args:
            stock_code: 股票代码
            limit: 返回记录数量限制
            
        Returns:
            List[AbnormalMovement]: 盘中异动数据列表
        """
        # 1. 从缓存获取
        cache_key = f"abnormal_movement:{stock_code}:{limit}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 2. 从数据库查询
        try:
            db_data = await AbnormalMovement.objects.filter(
                stock_code=stock_code
            ).order_by('-movement_time')[:limit].all()
            
            if db_data:
                # 更新缓存
                await self.set_to_cache(cache_key, list(db_data), timeout=300)  # 5分钟
                return db_data
        except Exception as e:
            logger.error(f"从数据库获取股票盘中异动数据出错: {e}")
        
        # 3. 从API获取所有异动数据，然后筛选
        try:
            # 先获取最新的所有异动数据
            all_movements = await self.get_latest_movements(1000)
            
            # 筛选指定股票的异动
            stock_movements = [m for m in all_movements if m.stock_code == stock_code]
            
            # 限制返回数量
            stock_movements = stock_movements[:limit]
            
            # 更新缓存
            await self.set_to_cache(cache_key, stock_movements, timeout=300)
            return stock_movements
        except Exception as e:
            logger.error(f"筛选股票盘中异动数据出错: {e}")
        
        return []
    
    async def _batch_save_movements(self, api_data_list: List[Dict[str, Any]]) -> List[AbnormalMovement]:
        """
        批量保存盘中异动数据
        
        Args:
            api_data_list: API返回的数据列表
            
        Returns:
            List[AbnormalMovement]: 保存后的盘中异动数据列表
        """
        if not api_data_list:
            return []
        
        # 1. 映射API数据到模型字段
        mapped_data_list = []
        for item in api_data_list:
            mapped_data = self._map_api_to_model(item, ABNORMAL_MOVEMENT_MAPPING)
            mapped_data_list.append(mapped_data)
        
        # 2. 检查数据库中是否已存在相同记录
        existing_records = {}
        
        # 获取所有的股票代码和异动时间
        stock_codes = [data['stock_code'] for data in mapped_data_list if 'stock_code' in data]
        movement_times = [data['movement_time'] for data in mapped_data_list if 'movement_time' in data]
        
        if stock_codes and movement_times:
            try:
                records = await AbnormalMovement.objects.filter(
                    stock_code__in=stock_codes,
                    movement_time__in=movement_times
                ).values_list('stock_code', 'movement_time', 'movement_type')
                
                for record in records:
                    key = (record[0], record[1], record[2])
                    existing_records[key] = True
            except Exception as e:
                logger.error(f"查询已存在盘中异动记录出错: {e}")
        
        # 3. 准备新的数据记录
        to_create = []
        for mapped_data in mapped_data_list:
            if all(field in mapped_data for field in ['stock_code', 'movement_time', 'movement_type']):
                key = (mapped_data['stock_code'], mapped_data['movement_time'], mapped_data['movement_type'])
                
                # 检查是否已存在
                if key not in existing_records:
                    to_create.append(AbnormalMovement(**mapped_data))
        
        # 4. 批量创建新记录
        created_records = []
        if to_create:
            try:
                created_records = await AbnormalMovement.objects.abulk_create(to_create)
                logger.info(f"批量创建盘中异动记录: {len(created_records)}条")
            except Exception as e:
                logger.error(f"批量创建盘中异动记录出错: {e}")
        
        # 5. 返回最新的记录（按时间倒序）
        try:
            # 获取所有相关记录
            stock_codes = [data['stock_code'] for data in mapped_data_list if 'stock_code' in data]
            
            all_movements = await AbnormalMovement.objects.filter(
                stock_code__in=stock_codes
            ).order_by('-movement_time').all()
            
            return all_movements
        except Exception as e:
            logger.error(f"获取所有盘中异动记录出错: {e}")
            return created_records
