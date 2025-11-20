import logging
from typing import List
from django.core.cache import cache
from dao_manager.base_dao import BaseDAO
from users.models import FavoriteStock
from utils.cache_manager import CacheManager
from utils.cache_set import UserCacheSet
from utils.cache_get import UserCacheGet
from utils.data_format_process import UserDataFormatProcess

logger = logging.getLogger(__name__)

class UserDAO(BaseDAO):
    """
    用户DAO，用于管理用户相关操作
    """
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.cache_set = UserCacheSet(self.cache_manager)
        self.cache_get = UserCacheGet(self.cache_manager)
        self.data_format_process = UserDataFormatProcess(cache_manager_instance)
    async def get_user_favorites(self, user_id: int) -> List[FavoriteStock]:
        """
        仅从数据库获取用户自选股列表，不使用缓存
        """
        try:
            # 直接异步查询数据库，获取自选股列表
            favorite_stocks_list = [
                fav async for fav in FavoriteStock.objects.select_related('stock').filter(user_id=user_id).order_by('added_at')
            ]
            print(f"从数据库获取用户{user_id}自选股数量: {len(favorite_stocks_list)}")  # 调试信息
            return favorite_stocks_list
        except Exception as e:
            logger.error(f"获取用户 {user_id} 自选股列表失败: {str(e)}", exc_info=True)
            print(f"获取用户 {user_id} 自选股列表失败: {str(e)}")  # 调试信息
            return []
    
