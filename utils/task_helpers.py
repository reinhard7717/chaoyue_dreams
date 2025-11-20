# utils/task_helpers.py

import logging
from functools import wraps
from asgiref.sync import async_to_sync
from .cache_manager import CacheManager # 从同级目录导入 CacheManager

logger = logging.getLogger(__name__)

def with_cache_manager(task_function):
    """
    一个装饰器，用于自动管理Celery任务中的CacheManager生命周期。
    功能:
    1. 在任务开始前，自动创建一个 CacheManager 实例。
    2. 将创建的实例作为关键字参数 `cache_manager` 注入到被装饰的任务函数中。
    3. 无论任务成功还是失败，在任务结束后，自动调用 cache_manager.close() 来关闭和清理Redis连接。
    """
    @wraps(task_function)
    def wrapper(*args, **kwargs):
        # 从Celery任务的参数中识别出 'self' (如果 bind=True)
        task_instance = args[0] if args and hasattr(args[0], 'request') else None
        task_id = task_instance.request.id if task_instance else 'N/A'
        cache_manager_instance = None
        try:
            # 1. 创建 CacheManager 实例
            # print(f"任务 {task_id}: [装饰器] 正在创建 CacheManager...")
            cache_manager_instance = CacheManager()
            # 2. 将实例注入到任务函数的关键字参数中
            kwargs['cache_manager'] = cache_manager_instance
            # 3. 执行原始的任务函数
            return task_function(*args, **kwargs)
        except Exception as e:
            logger.error(f"任务 {task_id} 在执行过程中发生未捕获的异常: {e}", exc_info=True)
            # 重新抛出异常，以便Celery能正确记录任务失败状态
            raise
        finally:
            # 4. 在任务结束时，无论如何都关闭连接
            if cache_manager_instance:
                # print(f"任务 {task_id}: [装饰器] 任务完成，正在关闭Redis连接...")
                try:
                    # 因为 close 是异步方法，需要使用 async_to_sync
                    async def close_main():
                        await cache_manager_instance.close()
                    async_to_sync(close_main)()
                    # print(f"任务 {task_id}: [装饰器] Redis连接已成功关闭。")
                except Exception as e:
                    logger.error(f"任务 {task_id}: [装饰器] 在关闭Redis连接时发生错误: {e}", exc_info=True)
    return wrapper

def with_cache_manager_for_views(view_func):
    """
    一个专为 Django 视图设计的装饰器，用于自动管理 CacheManager 的生命周期。
    它适用于函数视图(FBV)和类视图(CBV)的方法。
    功能:
    1. 在视图函数执行前，自动创建一个 CacheManager 实例。
    2. 将创建的实例作为关键字参数 `cache_manager` 注入到视图函数中。
    3. 无论视图成功返回响应还是引发异常，在执行后都自动调用 cache_manager.close() 来关闭和清理Redis连接。
    """
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        view_name = view_func.__name__
        cache_manager_instance = None
        try:
            # 1. 创建 CacheManager 实例
            # print(f"视图 {view_name}: [装饰器] 正在创建 CacheManager...")
            cache_manager_instance = CacheManager()
            # 2. 将实例注入到视图函数的关键字参数中
            kwargs['cache_manager'] = cache_manager_instance
            # 3. 执行原始的视图函数
            response = view_func(*args, **kwargs)
            return response
        except Exception as e:
            logger.error(f"视图 {view_name} 执行时发生未捕获的异常: {e}", exc_info=True)
            # 重新抛出异常，让Django的错误处理中间件接管
            raise
        finally:
            # 4. 在任务结束时，无论如何都关闭连接
            if cache_manager_instance:
                # print(f"视图 {view_name}: [装饰器] 视图处理完成，正在关闭Redis连接...")
                try:
                    # 因为 close 是异步方法，需要使用 async_to_sync
                    async def close_main():
                        await cache_manager_instance.close()
                    async_to_sync(close_main)()
                    # print(f"视图 {view_name}: [装饰器] Redis连接已成功关闭。")
                except Exception as e:
                    logger.error(f"视图 {view_name}: [装饰器] 关闭Redis连接时发生错误: {e}", exc_info=True)
    return wrapper
