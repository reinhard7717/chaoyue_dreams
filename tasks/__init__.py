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

# 原始导入代码暂时保留为注释
"""
# 添加手动刷新所有数据的全局任务
import asyncio
import logging
from celery import shared_task

# 导入所有手动刷新任务
from .index_tasks import manual_refresh_all_index_data
from .fund_flow_tasks import manual_refresh_all_fund_flow_data
from .stock_tasks import manual_refresh_all_favorites_data
from .datacenter_tasks import manual_refresh_all_datacenter_data
from .stock_pool_tasks import manual_refresh_all_stock_pools
from .strategy_tasks import manual_calculate_all_favorites_strategy

logger = logging.getLogger(__name__)

@shared_task
def refresh_all_data():
    # 手动触发刷新所有数据的任务
    # 包括所有指数、股票、资金流向、数据中心、股票池数据和所有策略计算
    # 这是一个综合性任务，会消耗大量API调用配额，请谨慎使用
    logger.info("开始全面刷新所有数据")
    
    # 刷新所有指数数据
    logger.info("开始刷新所有指数数据")
    manual_refresh_all_index_data.delay().get()
    logger.info("刷新所有指数数据完成")
    
    # 刷新所有资金流向数据
    logger.info("开始刷新所有资金流向数据")
    manual_refresh_all_fund_flow_data.delay().get()
    logger.info("刷新所有资金流向数据完成")
    
    # 刷新所有自选股数据
    logger.info("开始刷新所有自选股数据")
    manual_refresh_all_favorites_data.delay().get()
    logger.info("刷新所有自选股数据完成")
    
    # 刷新所有数据中心数据
    logger.info("开始刷新所有数据中心数据")
    manual_refresh_all_datacenter_data.delay().get()
    logger.info("刷新所有数据中心数据完成")
    
    # 刷新所有股票池
    logger.info("开始刷新所有股票池")
    manual_refresh_all_stock_pools.delay().get()
    logger.info("刷新所有股票池完成")
    
    # 计算所有自选股策略
    logger.info("开始计算所有自选股策略")
    manual_calculate_all_favorites_strategy.delay().get()
    logger.info("计算所有自选股策略完成")
    
    logger.info("全面刷新所有数据完成")
    return "全面刷新所有数据完成" 
""" 