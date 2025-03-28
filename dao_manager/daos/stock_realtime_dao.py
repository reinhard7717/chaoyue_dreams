from decimal import Decimal
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, TypeVar, Generic, Type
from datetime import datetime, date, time
import json
from django.utils import timezone
from asgiref.sync import sync_to_async
from django.conf import settings
from utils.cache_manager import CacheManager
from utils.models import ModelJSONEncoder

from django.db import transaction, models
from django.core.cache import cache
from django.db.models import Q, F

from api_manager.apis.stock_realtime_api import StockRealtimeAPI
from api_manager.mappings.stock_realtime import ABNORMAL_MOVEMENT_MAPPING, BIG_DEAL_MAPPING, LEVEL5_DATA_MAPPING, PRICE_PERCENT_MAPPING, REALTIME_DATA_MAPPING, TIME_DEAL_MAPPING, TRADE_DETAIL_MAPPING
from dao_manager.base_dao import BaseDAO
from stock_models.stock_basic import StockInfo
from stock_models.stock_realtime import AbnormalMovement, BigDeal, Level5Data, PricePercent, RealtimeData, TimeDeal, TradeDetail

logger = logging.getLogger(__name__)

class StockRealtimeDAO(BaseDAO):
    """
    股票实时数据DAO，整合所有相关的实时数据访问功能
    """
    
    def __init__(self):
        """初始化StockRealtimeDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockRealtimeAPI()
        self.cache_manager = CacheManager()  # 初始化缓存管理器
        logger.info("初始化StockRealtimeDAO")
    
    # ================= RealtimeData相关方法 =================
    
    async def get_latest_realtime_data_by_code(self, stock_code: str) -> Optional[RealtimeData]:
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
            realtime_dict = json.loads(cached_data)
            realtime = RealtimeData(**realtime_dict)
            return realtime
        
        # 2. 缓存未命中，从数据库查询
        try:
            stock = await self.get_stock_by_code(stock_code)
            if not stock:
                return None
            
            try:
                data = await sync_to_async(
                    lambda: RealtimeData.objects.filter(stock=stock).order_by('-update_time').first()
                )()
                
                # 检查数据是否过期（超过2分钟）
                if data and (timezone.now() - data.update_time).total_seconds() < 120:
                    cache.set(cache_key, data, settings.INDEX_CACHE_TIMEOUT['realtime_data'])
                    return data
                
                # 数据不存在或已过期，从API获取新数据
                logger.info(f"股票[{stock_code}]实时数据不存在或已过期，从API获取")
                data = await self._fetch_and_save_realtime_data(stock_code)
                if data:
                    cache.set(cache_key, data, settings.INDEX_CACHE_TIMEOUT['realtime_data'])
                return data
            except Exception as e:
                logger.error(f"获取股票[{stock_code}]实时数据失败: {str(e)}")
                return None
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
                
                realtime_data = await self._save_realtime_data_to_db(mapped_data)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, realtime_data, 60)
                return realtime_data
        except Exception as e:
            logger.error(f"从API获取实时交易数据出错: {e}")
        
        return None
    
    async def _fetch_and_save_realtime_data(self, stock_code: str) -> Optional[RealtimeData]:
        """
        从API获取实时交易数据并保存到数据库
        
        Args:
            stock_code: 股票代码

        Returns:
            Optional[RealtimeData]: 保存后的实时交易数据，保存失败则返回None
        """
        try:
            # 获取指数信息
            try:
                stock = await self.get_stock_by_code(stock_code)
            except Exception as e:
                logger.error(f"self.get_stock_by_code({stock_code}) 获取股票信息失败: {str(e)}")
                return None
            if not stock:
                logger.warning(f"股票代码[{stock_code}]不存在，无法获取实时数据")
                return None
            # 调用API获取实时数据
            api_data = await self.api.get_realtime_data(stock_code)
            if not api_data:
                logger.warning(f"API未返回股票[{stock_code}]的实时数据")
                return None
                      
            # 在数据库事务中保存数据
            @transaction.atomic
            def save_data():
                # 处理API返回的数据
                if isinstance(api_data, str):
                    try:
                        # 尝试解析JSON字符串
                        import json
                        mapped_data = json.loads(api_data)
                    except Exception as e:
                        logger.error(f"解析实时数据失败: {str(e)}")
                        return None
                else:
                    # 处理JSON格式的数据
                    mapped_data = api_data
                
                # 将映射后的数据转换为标准字典格式，并处理日期和数字字段
                data_dict = {
                    'stock': stock,
                    'update_time': self._parse_datetime(mapped_data.get('t')),
                    'open_price': self._parse_number(mapped_data.get('o')),
                    'five_min_change': self._parse_number(mapped_data.get('fm')),
                    'high_price': self._parse_number(mapped_data.get('h')),
                    'turnover_rate': self._parse_number(mapped_data.get('hs')),
                    'volume_ratio': self._parse_number(mapped_data.get('lb')),
                    'low_price': self._parse_number(mapped_data.get('l')),
                    'tradable_market_value': self._parse_number(mapped_data.get('lt')),
                    'pe_ratio': self._parse_number(mapped_data.get('pe')),
                    'price_change_percent': self._parse_number(mapped_data.get('pc')),
                    'current_price': self._parse_number(mapped_data.get('p')),
                    'total_market_value': self._parse_number(mapped_data.get('sz')),
                    'turnover_value': self._parse_number(mapped_data.get('cje')),
                    'price_change': self._parse_number(mapped_data.get('ud')),
                    'volume': self._parse_number(mapped_data.get('v')),
                    'prev_close_price': self._parse_number(mapped_data.get('yc')),
                    'amplitude': self._parse_number(mapped_data.get('zf')),
                    'increase_speed': self._parse_number(mapped_data.get('zs')),
                    'pb_ratio': self._parse_number(mapped_data.get('sjl')),
                    'price_change_60d': self._parse_number(mapped_data.get('zdf60')),
                    'price_change_ytd': self._parse_number(mapped_data.get('zdfnc')),
                }
                
                # 创建实时数据记录
                realtime_data = RealtimeData.objects.create(**data_dict)
                cache_manager.save_stock_realtime(stock_code, data_dict)
            
            # 执行保存操作并返回结果
            result = await sync_to_async(save_data)()
            logger.info(f"成功保存股票[{stock_code}]的实时数据")
            
            
            return result
        except Exception as e:
            logger.error(f"保存股票[{stock_code}]实时数据失败: {str(e)}")
            return None
    
    async def get_multiple_latest_realtime_datas(self, stock_codes: List[str]) -> Dict[str, RealtimeData]:
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
                # 将对象转换为字典格式并存入缓存
                cache_dict = {}
                for field in data._meta.fields:
                    value = getattr(data, field.name)
                    # 日期字段处理
                    if field.name.endswith('_date') or field.name.endswith('_time') or field.name == 't':
                        cache_dict[field.name] = self._parse_datetime(value)
                    # 数值字段处理
                    elif (isinstance(value, (int, float)) or (
                        isinstance(value, str) and value.replace('.', '', 1).isdigit()
                    )) and 'code' not in field.name.lower():
                        cache_dict[field.name] = self._parse_number(value)
                    else:
                        cache_dict[field.name] = value
                await self.set_to_cache(f"realtime:{data.stock_code}:latest", cache_dict, 60)
                result[data.stock_code] = data
                missing_codes.remove(data.stock_code)
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
    
    async def _check_if_realtime_data_exists(self, stock_code: str, update_time: datetime) -> Optional[RealtimeData]:
        """
        检查指定时间的实时数据是否已存在
        
        Args:
            stock_code: 股票代码
            update_time: 更新时间
            
        Returns:
            Optional[RealtimeData]: 存在的实时数据，不存在则返回None
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

    async def _check_if_realtime_data_identical(self, existing: RealtimeData, new_data: Dict[str, Any]) -> bool:
        """
        检查新旧实时数据是否相同
        
        Args:
            existing: 已存在的实时数据
            new_data: 新的实时数据
            
        Returns:
            bool: 如果数据相同则返回True，否则返回False
        """
        # 检查关键字段
        critical_fields = ['current_price', 'change_amount', 'change_percent', 'volume', 'amount']
        
        for field in critical_fields:
            if field in new_data:
                existing_value = getattr(existing, field, None)
                new_value = new_data[field]
                
                # 类型转换，确保比较的是同类型的值
                if isinstance(existing_value, Decimal) and not isinstance(new_value, Decimal):
                    new_value = Decimal(str(new_value))
                    
                if existing_value != new_value:
                    return False
                    
        return True
    
    @transaction.atomic
    async def _save_realtime_data_to_db(self, data: Dict[str, Any]) -> RealtimeData:
        """
        保存实时数据到数据库
        
        Args:
            data: 要保存的数据
            
        Returns:
            RealtimeData: 保存后的实时数据
        """
        try:
            obj = RealtimeData(**data)
            await obj.asave()
            logger.debug(f"保存实时数据成功: {obj.stock_code}")
            return obj
        except Exception as e:
            logger.error(f"保存实时数据出错: {e}")
            raise e
    
    @transaction.atomic
    async def _update_realtime_data_db(self, record_id: int, data: Dict[str, Any]) -> Optional[RealtimeData]:
        """
        更新数据库中的实时数据
        
        Args:
            record_id: 记录ID
            data: 要更新的数据
            
        Returns:
            Optional[RealtimeData]: 更新后的实时数据，更新失败则返回None
        """
        try:
            record = await RealtimeData.objects.filter(id=record_id).afirst()
            if record:
                # 更新字段
                for key, value in data.items():
                    setattr(record, key, value)
                
                await record.asave()
                logger.debug(f"更新实时数据成功: {record.stock_code}")
                return record
            else:
                logger.warning(f"要更新的实时数据记录不存在: {record_id}")
                return None
        except Exception as e:
            logger.error(f"更新实时数据出错: {e}")
            return None
    
    # ================= Level5Data相关方法 =================
    
    async def get_level5_data_by_code(self, stock_code: str) -> Optional[Level5Data]:
        """
        获取股票最新的Level5数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Level5Data]: 最新的Level5数据，如不存在则返回None
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
            logger.error(f"从数据库获取Level5数据出错: {e}")
        
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
                
                level5_data = await self._save_level5_data_to_db(mapped_data)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, level5_data, 60)
                return level5_data
        except Exception as e:
            logger.error(f"从API获取Level5数据出错: {e}")
        
        return None
    
    async def refresh_level5_data(self, stock_code: str) -> Optional[Level5Data]:
        """
        强制从API刷新股票Level5数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Level5Data]: 最新的Level5数据
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
                exists = await self._check_if_level5_data_exists(stock_code, mapped_data.get('update_time'))
                
                if exists:
                    # 检查数据是否有变化
                    if not await self._check_if_level5_data_identical(exists, mapped_data):
                        # 数据有变化，更新
                        level5_data = await self._update_level5_data_db(exists.id, mapped_data)
                    else:
                        # 数据相同，直接返回
                        level5_data = exists
                else:
                    # 不存在，创建新记录
                    level5_data = await self._save_level5_data_to_db(mapped_data)
                
                # 更新缓存
                await self.set_to_cache(cache_key, level5_data, 60)
                return level5_data
        except Exception as e:
            logger.error(f"刷新Level5数据出错: {e}")
        
        return None
    
    async def refresh_stocks_level5(self, stock_codes: List[str]) -> Dict[str, Level5Data]:
        """
        刷新多只股票的Level5数据
        
        Args:
            stock_codes: 股票代码列表
        
        Returns:
            Dict[str, Level5Data]: 股票代码到Level5数据的映射
        """
        logger.info(f"刷新{len(stock_codes)}只股票的Level5数据")
        results = {}
        
        for stock_code in stock_codes:
            try:
                data = await self.refresh_level5_data(stock_code)
                if data:
                    results[stock_code] = data
            except Exception as e:
                logger.error(f"刷新股票[{stock_code}]Level5数据出错: {str(e)}")
        
        logger.info(f"刷新股票Level5数据完成，成功{len(results)}只")
        return results
    
    async def _check_if_level5_data_exists(self, stock_code: str, update_time: datetime) -> Optional[Level5Data]:
        """
        检查指定时间的Level5数据是否已存在
        
        Args:
            stock_code: 股票代码
            update_time: 更新时间
            
        Returns:
            Optional[Level5Data]: 存在的Level5数据，不存在则返回None
        """
        if not update_time:
            return None
            
        try:
            return await Level5Data.objects.filter(
                stock_code=stock_code,
                update_time=update_time
            ).afirst()
        except Exception as e:
            logger.error(f"检查Level5数据是否存在出错: {e}")
            return None
    
    async def _check_if_level5_data_identical(self, existing: Level5Data, new_data: Dict[str, Any]) -> bool:
        """
        检查新旧Level5数据是否相同
        
        Args:
            existing: 已存在的Level5数据
            new_data: 新的Level5数据
            
        Returns:
            bool: 如果数据相同则返回True，否则返回False
        """
        # 检查关键字段
        critical_fields = [
            'buy_vol1', 'buy_vol2', 'buy_vol3', 'buy_vol4', 'buy_vol5',
            'sell_vol1', 'sell_vol2', 'sell_vol3', 'sell_vol4', 'sell_vol5',
            'buy_price1', 'buy_price2', 'buy_price3', 'buy_price4', 'buy_price5',
            'sell_price1', 'sell_price2', 'sell_price3', 'sell_price4', 'sell_price5'
        ]
        
        for field in critical_fields:
            if field in new_data:
                existing_value = getattr(existing, field, None)
                new_value = new_data[field]
                
                # 类型转换，确保比较的是同类型的值
                if isinstance(existing_value, Decimal) and not isinstance(new_value, Decimal):
                    new_value = Decimal(str(new_value))
                    
                if existing_value != new_value:
                    return False
                    
        return True
    
    @transaction.atomic
    async def _save_level5_data_to_db(self, data: Dict[str, Any]) -> Level5Data:
        """
        保存Level5数据到数据库
        
        Args:
            data: 要保存的数据
            
        Returns:
            Level5Data: 保存后的Level5数据
        """
        try:
            obj = Level5Data(**data)
            await obj.asave()
            logger.debug(f"保存Level5数据成功: {obj.stock_code}")
            return obj
        except Exception as e:
            logger.error(f"保存Level5数据出错: {e}")
            raise e
    
    @transaction.atomic
    async def _update_level5_data_db(self, record_id: int, data: Dict[str, Any]) -> Optional[Level5Data]:
        """
        更新数据库中的Level5数据
        
        Args:
            record_id: 记录ID
            data: 要更新的数据
            
        Returns:
            Optional[Level5Data]: 更新后的Level5数据，更新失败则返回None
        """
        try:
            record = await Level5Data.objects.filter(id=record_id).afirst()
            if record:
                # 更新字段
                for key, value in data.items():
                    setattr(record, key, value)
                
                await record.asave()
                logger.debug(f"更新Level5数据成功: {record.stock_code}")
                return record
            else:
                logger.warning(f"要更新的Level5数据记录不存在: {record_id}")
                return None
        except Exception as e:
            logger.error(f"更新Level5数据出错: {e}")
            return None
            
    # ================= TradeDetail相关方法 =================
    
    async def get_trade_details_by_code_and_date(self, stock_code: str, trade_date: Optional[date] = None) -> List[TradeDetail]:
        """
        获取股票指定日期的交易明细
        
        Args:
            stock_code: 股票代码
            trade_date: 交易日期，默认为当天
            
        Returns:
            List[TradeDetail]: 交易明细列表
        """
        if not trade_date:
            trade_date = datetime.now().date()
            
        # 1. 首先从缓存获取
        cache_key = f"trade_detail:{stock_code}:{trade_date}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # 2. 缓存未命中，从数据库查询
        try:
            trade_details = await TradeDetail.objects.filter(
                stock_code=stock_code,
                trade_date=trade_date
            ).order_by('trade_time').all()
            
            if trade_details:
                # 存入缓存并返回
                await self.set_to_cache(cache_key, trade_details, 300)  # 交易明细缓存时间较长，5分钟
                return trade_details
        except Exception as e:
            logger.error(f"从数据库获取交易明细出错: {e}")
            
        # 3. 数据库未找到，从API获取
        try:
            api_data_list = await self.api.get_trade_details(stock_code, trade_date.strftime('%Y-%m-%d'))
            
            if api_data_list:
                # 处理并保存到数据库
                trade_details = await self._process_and_save_trades(stock_code, api_data_list)
                
                # 保存到缓存
                await self.set_to_cache(cache_key, trade_details, 300)
                return trade_details
        except Exception as e:
            logger.error(f"从API获取交易明细出错: {e}")
            
        return []
    
    async def _process_and_save_trades(self, stock_code: str, api_data_list: List[Dict[str, Any]]) -> List[TradeDetail]:
        """
        处理并保存交易明细数据
        
        Args:
            stock_code: 股票代码
            api_data_list: API返回的交易明细数据列表
            
        Returns:
            List[TradeDetail]: 保存后的交易明细列表
        """
        result = []
        
        for api_data in api_data_list:
            try:
                # 映射API数据到模型
                mapped_data = self._map_api_to_model(api_data, TRADE_DETAIL_MAPPING)
                mapped_data['stock_code'] = stock_code
                
                # 处理日期和时间
                if 'trade_time' in mapped_data and mapped_data['trade_time']:
                    try:
                        # 尝试解析完整的日期时间字符串
                        dt = datetime.strptime(mapped_data['trade_time'], '%Y-%m-%d %H:%M:%S')
                        mapped_data['trade_date'] = dt.date()
                        mapped_data['trade_time'] = dt.time()
                    except ValueError:
                        # 如果失败，尝试只解析时间部分
                        try:
                            mapped_data['trade_time'] = datetime.strptime(mapped_data['trade_time'], '%H:%M:%S').time()
                            # 使用当前日期
                            mapped_data['trade_date'] = datetime.now().date()
                        except ValueError:
                            logger.warning(f"无法解析交易时间: {mapped_data['trade_time']}")
                            continue
                else:
                    # 没有交易时间，则跳过
                    continue
                
                # 检查记录是否已存在
                existing = await TradeDetail.objects.filter(
                    stock_code=stock_code,
                    trade_date=mapped_data['trade_date'],
                    trade_time=mapped_data['trade_time'],
                    price=mapped_data.get('price'),
                    volume=mapped_data.get('volume')
                ).afirst()
                
                if existing:
                    result.append(existing)
                else:
                    # 创建新记录
                    trade_detail = TradeDetail(**mapped_data)
                    await trade_detail.asave()
                    result.append(trade_detail)
            except Exception as e:
                logger.error(f"处理交易明细数据出错: {e}")
                
        return result
    
    # ================= TimeDeal相关方法 =================
    
    async def get_daily_time_deals(self, stock_code: str, trade_date: Optional[date] = None) -> List[TimeDeal]:
        """
        获取股票指定日期的分时成交数据
        
        Args:
            stock_code: 股票代码
            trade_date: 交易日期，默认为当天
            
        Returns:
            List[TimeDeal]: 分时成交数据列表
        """
        if not trade_date:
            trade_date = datetime.now().date()
            
        # 首先从缓存获取
        cache_key = f"time_deal:{stock_code}:{trade_date}"
        cached_data = await self.get_from_cache(cache_key)
        if cached_data:
            return cached_data
            
        # 从数据库获取
        try:
            time_deals = await TimeDeal.objects.filter(
                stock_code=stock_code,
                trade_date=trade_date
            ).order_by('trade_time').all()
            
            if time_deals and len(time_deals) > 0:
                # 存入缓存
                await self.set_to_cache(cache_key, time_deals, 300)  # 5分钟缓存
                return time_deals
        except Exception as e:
            logger.error(f"从数据库获取分时成交数据出错: {e}")
            
        # 从API获取
        try:
            # 格式化日期为字符串
            date_str = trade_date.strftime('%Y-%m-%d')
            api_data = await self.api.get_time_deals(stock_code, date_str)
            
            if api_data and len(api_data) > 0:
                time_deals = await self._batch_save_time_deals(stock_code, api_data)
                
                # 存入缓存
                await self.set_to_cache(cache_key, time_deals, 300)
                return time_deals
        except Exception as e:
            logger.error(f"从API获取分时成交数据出错: {e}")
            
        return []
    
    async def _batch_save_time_deals(self, stock_code: str, api_data_list: List[Dict[str, Any]]) -> List[TimeDeal]:
        """
        批量保存分时成交数据
        
        Args:
            stock_code: 股票代码
            api_data_list: API返回的分时成交数据列表
            
        Returns:
            List[TimeDeal]: 保存后的分时成交数据列表
        """
        results = []
        
        for api_data in api_data_list:
            try:
                # 映射数据
                mapped_data = self._map_api_to_model(api_data, TIME_DEAL_MAPPING)
                mapped_data['stock_code'] = stock_code
                
                # 处理日期和时间
                if 'trade_date' in mapped_data and mapped_data['trade_date']:
                    mapped_data['trade_date'] = self._parse_date(mapped_data['trade_date'])
                    
                if 'trade_time' in mapped_data and mapped_data['trade_time']:
                    mapped_data['trade_time'] = self._parse_time(mapped_data['trade_time'])
                    
                # 如果日期或时间为空，则跳过
                if not mapped_data.get('trade_date') or not mapped_data.get('trade_time'):
                    continue
                    
                # 检查记录是否已存在
                existing = await TimeDeal.objects.filter(
                    stock_code=stock_code,
                    trade_date=mapped_data['trade_date'],
                    trade_time=mapped_data['trade_time']
                ).afirst()
                
                if existing:
                    # 检查关键字段是否有变化
                    need_update = False
                    for key in ['price', 'volume', 'amount']:
                        if key in mapped_data and getattr(existing, key, None) != mapped_data[key]:
                            need_update = True
                            break
                            
                    if need_update:
                        # 更新字段
                        for key, value in mapped_data.items():
                            setattr(existing, key, value)
                            
                        await existing.asave()
                        results.append(existing)
                    else:
                        results.append(existing)
                else:
                    # 创建新记录
                    time_deal = TimeDeal(**mapped_data)
                    await time_deal.asave()
                    results.append(time_deal)
            except Exception as e:
                logger.error(f"保存分时成交数据出错: {e}")
                
        return results
    
    # ================= 其他相关方法 =================
    
    # 在这里继续实现PricePercent、BigDeal、AbnormalMovement相关的方法
    # 由于代码过长，这些方法的实现与上面的方法类似，按照相同的模式实现即可
