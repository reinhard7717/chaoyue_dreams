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

# 显式导入所有任务模块
# from . import index_tasks
# from . import stock_indicator_tasks
# from . import indicator_tasks
# from . import stock_tasks
# from . import fund_flow_tasks
# from . import datacenter_tasks
# from . import stock_pool_tasks
# from . import strategy_tasks
