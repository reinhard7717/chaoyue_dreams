# apps/utils/rate_limiter.py

import os
import time
import asyncio
import uuid
import functools
import threading # 导入threading
from redis.asyncio.client import Pipeline
from django.conf import settings
from utils.cache_manager import CacheManager
import logging

logger = logging.getLogger(__name__)

class DistributedRateLimiter:
    """
    使用Redis ZSET实现一个分布式、跨进程的滑动窗口速率限制器。
    这个类保持不变，它是一个可重用的核心组件。
    """
    def __init__(self, key: str, max_calls: int, period: int, cache_manager: CacheManager):
        self.redis_key = f"rate_limiter:{key}"
        self.max_calls = max_calls
        self.period = period
        self.cache_manager = cache_manager
        print(f"DEBUG: 分布式速率限制器 '{self.redis_key}' 已创建，限制为 {max_calls}次/{period}秒。")
    async def acquire(self) -> bool:
        try:
            redis_client = await self.cache_manager._ensure_client()
            async with redis_client.pipeline(transaction=True) as pipe:
                current_timestamp_ms = int(time.time() * 1000)
                window_start_ms = current_timestamp_ms - self.period * 1000
                pipe.zremrangebyscore(self.redis_key, 0, window_start_ms)
                pipe.zcard(self.redis_key)
                results = await pipe.execute()
                current_count = results[1]
                # print(f"DEBUG: [Limiter: {self.redis_key}] 当前窗口请求数: {current_count}/{self.max_calls}")
                if current_count < self.max_calls:
                    async with redis_client.pipeline(transaction=True) as write_pipe:
                        member = f"{current_timestamp_ms}:{uuid.uuid4()}"
                        write_pipe.zadd(self.redis_key, {member: current_timestamp_ms})
                        write_pipe.expire(self.redis_key, self.period + 5)
                        await write_pipe.execute()
                    # print(f"DEBUG: [Limiter: {self.redis_key}] 请求允许。")
                    return True
                else:
                    # print(f"DEBUG: [Limiter: {self.redis_key}] 速率限制触发！请求被拒绝。")
                    return False
        except Exception as e:
            logger.error(f"速率限制器 '{self.redis_key}' 执行出错: {e}", exc_info=True)
            return True

# 速率限制器工厂类
class RateLimiterFactory:
    """
    一个单例工厂，用于创建和管理多个具名的分布式速率限制器。
    这确保了不同的API可以使用各自独立的速率限制。
    """
    _instance = None
    _lock = threading.Lock() # 用于保护实例创建和字典操作的线程锁
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._limiters = {} # 缓存已创建的限流器实例
                    cls._instance._cache_manager = CacheManager() # 持有CacheManager单例
                    # print("DEBUG: RateLimiterFactory 单例已初始化。")
        return cls._instance
    def get_limiter(self, name: str) -> DistributedRateLimiter:
        """
        获取一个具名的速率限制器。
        它会自动从 settings.API_RATE_LIMITS 读取配置。
        :param name: 限流器的唯一名称 (例如 'api_cyq_chips')，必须与settings中的key对应。
        :return: 一个 DistributedRateLimiter 实例。
        """
        with self._lock:
            if name not in self._limiters:
                # print(f"DEBUG: 工厂正在为 '{name}' 查找配置并创建新的限流器实例...")
                # 从 settings.py 获取所有速率限制配置
                all_configs = settings.API_RATE_LIMITS
                # 获取特定名称的配置，如果找不到，则使用 'DEFAULT' 配置
                config = all_configs.get(name, all_configs['DEFAULT'])
                max_calls = config['MAX_CALLS']
                period = config['PERIOD']
                self._limiters[name] = DistributedRateLimiter(
                    key=name,
                    max_calls=max_calls,
                    period=period,
                    cache_manager=self._cache_manager
                )
            return self._limiters[name]

# 导出一个工厂的单例，而不是单个限流器实例
rate_limiter_factory = RateLimiterFactory()


# ==============================================================================
# 新增速率限制装饰器 (依赖注入模式)
# ==============================================================================
def with_rate_limit(name: str):
    """
    一个装饰器工厂，用于为异步DAO方法注入一个配置好的分布式速率限制器。
    它会从 settings.API_RATE_LIMITS 中查找名为 `name` 的配置，
    创建或获取对应的限流器实例，并将其作为关键字参数 `limiter` 注入到
    被装饰的异步方法中。
    使用方法:
    @with_rate_limit(name='api_cyq_chips')
    async def my_dao_method(self, some_arg, *, limiter):
        # 在这里使用注入的 limiter 对象
        while not await limiter.acquire():
            await asyncio.sleep(1)
        # ... 调用API ...
    :param name: 在 settings.API_RATE_LIMITS 中定义的API限流配置的键名。
    """
    def decorator(func):
        # 使用 @functools.wraps 来保留原始函数的元信息（如名称、文档字符串）
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 从工厂获取指定的限流器实例
            limiter_instance = rate_limiter_factory.get_limiter(name=name)
            # 将获取到的限流器实例通过关键字参数 'limiter' 注入
            # 如果调用者已经手动传入了 limiter，则不会覆盖
            if 'limiter' not in kwargs:
                kwargs['limiter'] = limiter_instance
            # 执行原始的异步函数，并传入包含 limiter 的参数
            return await func(*args, **kwargs)
        return wrapper
    return decorator











