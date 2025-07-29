# services/cache_service.py
from django.apps import AppConfig

class CacheService:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            from utils.cache_manager import CacheManager
            cls._instance = cache_manager
        return cls._instance

# 在应用启动时初始化
class StockAppConfig(AppConfig):
    name = 'stock_app'
    
    def ready(self):
        # 预热缓存管理器
        CacheService.get_instance()
