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
from api_manager.mappings.stock_indicators_mapping import BOLL_INDICATOR_MAPPING, KDJ_INDICATOR_MAPPING, MA_INDICATOR_MAPPING, MACD_INDICATOR_MAPPING, TIME_TRADE_MAPPING
from dao_manager.base_dao import BaseDAO
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from users.models import FavoriteStock
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class UserDAO(BaseDAO):
    """
    用户DAO，用于管理用户相关操作
    """
    def __init__(self):
        from utils.data_format_process import UserDataFormatProcess
        from utils.cache_set import UserCacheSet
        from utils.cache_get import UserCacheGet
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.cache_set = UserCacheSet()
        self.cache_get = UserCacheGet()
        self.data_format_process = UserDataFormatProcess()

    async def get_user_favorites(self, user_id: int) -> List[FavoriteStock]: # 方法名改为 get_user_favorites 以匹配视图调用
        """
        异步获取用户自选股列表
        """
        try:
            # 尝试从缓存获取
            cached_favorites = await self.cache_get.user_favorites(user_id)
            if cached_favorites is not None: # 检查是否为 None，因为空列表也可能被缓存
                logger.debug(f"从缓存命中用户 {user_id} 的自选股列表")
                return cached_favorites
            logger.debug(f"缓存未命中，从数据库查询用户 {user_id} 的自选股列表")
            # --- 使用异步 ORM 获取数据 ---
            # 使用列表推导式和 async for 将 QuerySet 转换为列表
            # 使用 select_related 优化查询
            favorite_stocks_list = [
                fav async for fav in FavoriteStock.objects.select_related('stock').filter(user_id=user_id).order_by('added_at') # 可选排序
            ]
            # --- 异步 ORM 结束 ---
            # 将查询结果（列表）存入缓存
            data_to_cache_list = [self.data_format_process.set_user_favorites(user_id, fav) for fav in favorite_stocks_list]  # 生成列表
            await self.cache_set.user_favorites(user_id, data_to_cache_list)  # 一次性传递列表
            logger.debug(f"已缓存用户 {user_id} 的自选股列表")
            return favorite_stocks_list
        except Exception as e:
            # 记录详细错误，包括 traceback
            logger.error(f"获取用户 {user_id} 自选股列表失败: {str(e)}", exc_info=True)
            return [] # 出错时返回空列表

    async def get_all_favorite_stocks(self) -> List[FavoriteStock]:
        """
        获取所有用户自选股列表
        """
        try:
            # 从缓存中获取所有自选股列表
            cache_key = "all_favorite_stocks"
            all_favorite_stocks = cache.get(cache_key)
            if all_favorite_stocks:
                return all_favorite_stocks
            
            # 从数据库中获取所有自选股列表
            all_favorite_stocks = await FavoriteStock.objects.all()
            
            # 将所有自选股列表缓存
            cache.set(cache_key, all_favorite_stocks, self.cache_ttl)
            return all_favorite_stocks
        except Exception as e:
            logger.error(f"获取所有用户自选股列表失败: {str(e)}")
            return []

    
