import logging
from typing import List
from django.core.cache import cache
from dao_manager.base_dao import BaseDAO
from users.models import FavoriteStock
from utils.cache_set import UserCacheSet
from utils.cache_get import UserCacheGet
from utils.data_format_process import UserDataFormatProcess

logger = logging.getLogger(__name__)

class UserDAO(BaseDAO):
    """
    用户DAO，用于管理用户相关操作
    """
    def __init__(self):
        super().__init__(None, None, 3600)
        self.cache_set = None
        self.cache_get = None
        self.data_format_process = UserDataFormatProcess()

    async def initialize_cache_objects(self):
        self.cache_set = UserCacheSet()
        self.cache_get = UserCacheGet()

    async def get_user_favorites(self, user_id: int) -> List[FavoriteStock]:
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            cached_favorites = await self.cache_get.user_favorites(user_id)
            if cached_favorites is not None:
                return cached_favorites
            favorite_stocks_list = [fav async for fav in FavoriteStock.objects.select_related('stock').filter(user_id=user_id).order_by('added_at')]
            data_to_cache_list = [self.data_format_process.set_user_favorites(user_id, fav) for fav in favorite_stocks_list]
            if self.cache_set is None:
                await self.initialize_cache_objects()
            await self.cache_set.user_favorites(user_id, data_to_cache_list)
            return favorite_stocks_list
        except Exception as e:
            logger.error(f"获取用户 {user_id} 自选股列表失败: {str(e)}", exc_info=True)
            return []

    async def get_all_favorite_stocks(self) -> List[FavoriteStock]:
        try:
            if self.cache_get is None:
                await self.initialize_cache_objects()
            cached_favorites = await self.cache_get.all_favorites()
            if cached_favorites:
                return cached_favorites
            all_favorite_stocks = [fav async for fav in FavoriteStock.objects.all()]
            if self.cache_set is None:
                await self.initialize_cache_objects()
            await self.cache_set.all_favorites(all_favorite_stocks)
            return all_favorite_stocks
        except Exception as e:
            logger.error(f"获取所有用户自选股列表失败: {str(e)}", exc_info=True)
            return []
    
