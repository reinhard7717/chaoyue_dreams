# Tasks包初始化文件
"""
任务模块
包含所有的定时任务和手动触发任务
"""

# 导入必要的模块
import sys
import logging
logger = logging.getLogger(__name__)

# 解决Python 3.12上asyncio.coroutines没有_DEBUG属性的问题
if sys.version_info >= (3, 12):
    import asyncio.coroutines
    if not hasattr(asyncio.coroutines, '_DEBUG'):
        # 为了兼容性，添加一个_DEBUG属性，其值由_is_debug_mode()函数确定
        asyncio.coroutines._DEBUG = asyncio.coroutines._is_debug_mode()

# 避免在模块加载时导入任务，防止循环导入和AppRegistryNotReady错误
# 使用惰性导入机制，仅在需要时导入各模块

def get_add_task():
    """获取add测试任务"""
    from .test_tasks import add
    return add

def get_long_task():
    """获取long_task测试任务"""
    from .test_tasks import long_task
    return long_task

def get_refresh_all_data_task():
    """获取刷新所有数据的综合任务"""
    from celery import shared_task
    
    @shared_task
    def refresh_all_data():
        """
        手动触发刷新所有数据的任务
        包括所有指数、股票、资金流向、数据中心、股票池数据和所有策略计算
        这是一个综合性任务，会消耗大量API调用配额，请谨慎使用
        """
        logger.info("开始全面刷新所有数据")
        
        try:
            # 延迟导入，避免循环导入问题
            from .index_tasks import manual_refresh_all_index_data
            from .fund_flow_tasks import manual_refresh_all_fund_flow_data
            from .stock_tasks import manual_refresh_all_favorites_data
            from .datacenter_tasks import manual_refresh_all_datacenter_data
            from .stock_pool_tasks import manual_refresh_all_stock_pools
            from .strategy_tasks import manual_calculate_all_favorites_strategy
            
            # 刷新所有指数数据
            logger.info("开始刷新所有指数数据")
            manual_refresh_all_index_data.delay()
            logger.info("刷新所有指数数据任务已提交")
            
            # 刷新所有资金流向数据
            logger.info("开始刷新所有资金流向数据")
            manual_refresh_all_fund_flow_data.delay()
            logger.info("刷新所有资金流向数据任务已提交")
            
            # 刷新所有自选股数据
            logger.info("开始刷新所有自选股数据")
            manual_refresh_all_favorites_data.delay()
            logger.info("刷新所有自选股数据任务已提交")
            
            # 刷新所有数据中心数据
            logger.info("开始刷新所有数据中心数据")
            manual_refresh_all_datacenter_data.delay()
            logger.info("刷新所有数据中心数据任务已提交")
            
            # 刷新所有股票池
            logger.info("开始刷新所有股票池")
            manual_refresh_all_stock_pools.delay()
            logger.info("刷新所有股票池任务已提交")
            
            # 计算所有自选股策略
            logger.info("开始计算所有自选股策略")
            manual_calculate_all_favorites_strategy.delay()
            logger.info("计算所有自选股策略任务已提交")
            
            logger.info("所有刷新任务已提交")
            return "所有刷新任务已提交"
        except Exception as e:
            logger.error(f"刷新所有数据失败: {str(e)}")
            return f"刷新所有数据失败: {str(e)}"
    
    return refresh_all_data

# 为了保持向后兼容性，提供refresh_all_data任务
refresh_all_data = get_refresh_all_data_task() 