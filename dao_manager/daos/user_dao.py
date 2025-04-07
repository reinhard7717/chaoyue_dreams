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
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型

    async def get_favorite_stocks(self, user_id: int) -> List[FavoriteStock]:
        """
        获取用户自选股列表
        """
        try:
            # 从缓存中获取自选股列表
            cache_key = f"favorite_stocks_{user_id}"
            favorite_stocks = cache.get(cache_key)
            if favorite_stocks:
                return favorite_stocks
            
            # 从数据库中获取自选股列表
            favorite_stocks = await FavoriteStock.objects.filter(user_id=user_id).all()
            
            # 将自选股列表缓存
            cache.set(cache_key, favorite_stocks, self.cache_ttl)
            return favorite_stocks
        except Exception as e:
            logger.error(f"获取用户自选股列表失败: {str(e)}")
            return []

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
            
