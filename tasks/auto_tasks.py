"""
自动任务模块
定义所有自动定时执行的任务
这个模块不应该依赖其他业务模块来避免循环导入
"""
import logging
from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)

@shared_task
def update_market_data():
    """
    更新市场数据的自动任务
    从API获取最新的市场数据并存储到数据库
    """
    logger.info(f"开始自动更新市场数据 - {timezone.now()}")
    
    try:
        # 推迟导入，避免循环导入问题
        from tasks.index_tasks import refresh_index_list, refresh_market_overview
        
        # 更新指数列表
        refresh_index_list.delay()
        
        # 更新市场概览
        refresh_market_overview.delay()
        
        logger.info("市场数据更新任务已提交")
        return "市场数据更新任务已提交"
    except Exception as e:
        logger.error(f"更新市场数据失败: {str(e)}")
        return f"更新市场数据失败: {str(e)}"

@shared_task
def update_stock_data():
    """
    更新股票数据的自动任务
    从API获取最新的股票数据并存储到数据库
    """
    logger.info(f"开始自动更新股票数据 - {timezone.now()}")
    
    try:
        # 推迟导入，避免循环导入问题
        from tasks.stock_tasks import refresh_all_stock_basic, refresh_all_favorites_realtime
        
        # 更新所有股票基础信息
        refresh_all_stock_basic.delay()
        
        # 更新所有自选股实时数据
        refresh_all_favorites_realtime.delay()
        
        logger.info("股票数据更新任务已提交")
        return "股票数据更新任务已提交"
    except Exception as e:
        logger.error(f"更新股票数据失败: {str(e)}")
        return f"更新股票数据失败: {str(e)}"

@shared_task
def calculate_strategy():
    """
    计算策略信号的自动任务
    根据最新数据计算各种策略的信号
    """
    logger.info(f"开始自动计算策略信号 - {timezone.now()}")
    
    try:
        # 推迟导入，避免循环导入问题
        from tasks.strategy_tasks import calculate_all_favorites_strategy
        
        # 计算所有自选股策略
        calculate_all_favorites_strategy.delay()
        
        logger.info("策略计算任务已提交")
        return "策略计算任务已提交"
    except Exception as e:
        logger.error(f"计算策略信号失败: {str(e)}")
        return f"计算策略信号失败: {str(e)}" 