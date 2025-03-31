"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task

from api_manager.apis.index_api import StockIndexAPI
from dao_manager.daos.index_dao import StockIndexDAO

logger = logging.getLogger(__name__)

# API和DAO实例
index_api = StockIndexAPI()
index_dao = StockIndexDAO()

@shared_task
def save_index_all_latest_realtime_data():
    """
    保存指数最新实时数据
    """
    logger.info("开始保存指数最新实时数据")
    asyncio.run(index_dao.fetch_and_save_all_realtime_data())
    logger.info("保存指数最新实时数据完成")
    return "保存指数最新实时数据完成"

@shared_task
def save_index_all_latest_market_overview():
    """
    保存指数最新市场概览数据
    """
    logger.info("开始保存指数最新市场概览数据")
    asyncio.run(index_dao.fetch_and_save_market_overview())
    logger.info("保存指数最新市场概览数据完成")
    return "保存指数最新市场概览数据完成"

@shared_task
def save_index_all_latest_time_series():
    """
    保存指数时间序列数据
    """
    logger.info("开始保存指数时间序列数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_time_series())
    logger.info("保存指数时间序列数据完成")
    return "保存指数时间序列数据完成"

@shared_task
def save_index_all_history_time_series():
    """
    保存指数历史时间序列数据
    """
    logger.info("开始保存指数历史时间序列数据")
    asyncio.run(index_dao.fetch_and_save_all_history_time_series())
    logger.info("保存指数历史时间序列数据完成")
    return "保存指数历史时间序列数据完成"

@shared_task
def save_index_all_latest_kdj():
    """
    保存指数最新KDJ数据
    """
    logger.info("开始保存指数最新KDJ数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_kdj())
    logger.info("保存指数最新KDJ数据完成")
    return "保存指数最新KDJ数据完成"

@shared_task
def save_index_all_history_kdj():
    """
    保存指数历史KDJ数据
    """
    logger.info("开始保存指数历史KDJ数据")
    asyncio.run(index_dao.fetch_and_save_all_history_kdj())
    logger.info("保存指数历史KDJ数据完成")
    return "保存指数历史KDJ数据完成"

@shared_task
def save_index_all_latest_macd():
    """
    保存指数最新MACD数据
    """
    logger.info("开始保存指数最新MACD数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_macd())
    logger.info("保存指数最新MACD数据完成")
    return "保存指数最新MACD数据完成"

@shared_task
def save_index_all_history_macd():
    """
    保存指数历史MACD数据
    """
    logger.info("开始保存指数历史MACD数据")
    asyncio.run(index_dao.fetch_and_save_all_history_macd())
    logger.info("保存指数历史MACD数据完成")
    return "保存指数历史MACD数据完成"

@shared_task
def save_index_all_latest_boll():
    """
    保存指数最新BOLL数据
    """
    logger.info("开始保存指数最新BOLL数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_boll())
    logger.info("保存指数最新BOLL数据完成")
    return "保存指数最新BOLL数据完成"

@shared_task
def save_index_all_history_boll():
    """
    保存指数历史BOLL数据
    """
    logger.info("开始保存指数历史BOLL数据")
    asyncio.run(index_dao.fetch_and_save_all_history_boll())
    logger.info("保存指数历史BOLL数据完成")
    return "保存指数历史BOLL数据完成"

@shared_task
def save_index_all_latest_ma():
    """
    保存指数最新MA数据
    """
    logger.info("开始保存指数最新MA数据")
    asyncio.run(index_dao.fetch_and_save_all_latest_ma())
    logger.info("保存指数最新MA数据完成")
    return "保存指数最新MA数据完成"

@shared_task
def save_index_all_history_ma():
    """
    保存指数历史MA数据
    """
    logger.info("开始保存指数历史MA数据")
    asyncio.run(index_dao.fetch_and_save_all_history_ma())
    logger.info("保存指数历史MA数据完成")
    return "保存指数历史MA数据完成"

@shared_task
def refresh_indexes():
    """
    刷新所有指数基础信息
    每天早上8点执行
    """
    logger.info("开始刷新所有指数基础信息")
    asyncio.run(index_dao.refresh_all_indexes())
    logger.info("刷新所有指数基础信息完成")
    return "刷新所有指数基础信息完成"

@shared_task
def refresh_main_indexes_realtime_data():
    """
    刷新主要指数的实时数据
    交易时间段每分钟执行
    """
    logger.info("开始刷新主要指数的实时数据")
    asyncio.run(index_dao.refresh_main_indexes_realtime())
    logger.info("刷新主要指数的实时数据完成")
    return "刷新主要指数的实时数据完成"

@shared_task
def refresh_market_overview():
    """
    刷新市场概览信息
    交易时间段每2分钟执行
    """
    logger.info("开始刷新市场概览信息")
    asyncio.run(index_dao.refresh_market_overview())
    logger.info("刷新市场概览信息完成")
    return "刷新市场概览信息完成"

@shared_task
def refresh_main_indexes_time_series(period):
    """
    刷新主要指数的K线数据
    根据不同周期定时执行
    
    Args:
        period: K线周期 (5, 15, 30, 60, Day)
    """
    logger.info(f"开始刷新主要指数的{period}分钟K线数据")
    asyncio.run(index_dao.refresh_main_indexes_time_series(period))
    logger.info(f"刷新主要指数的{period}分钟K线数据完成")
    return f"刷新主要指数的{period}分钟K线数据完成"

@shared_task
def refresh_main_indexes_technical_indicators(period):
    """
    刷新主要指数的技术指标
    日线数据每个交易日收盘后执行
    
    Args:
        period: K线周期 (Day)
    """
    logger.info(f"开始刷新主要指数的{period}技术指标")
    asyncio.run(index_dao.refresh_main_indexes_technical_indicators(period))
    logger.info(f"刷新主要指数的{period}技术指标完成")
    return f"刷新主要指数的{period}技术指标完成"

@shared_task
def manual_refresh_all_index_data():
    """
    手动触发刷新所有指数数据的任务
    包括基础信息、实时数据、K线数据和技术指标
    """
    logger.info("手动开始刷新所有指数数据")
    
    # 刷新基础信息
    asyncio.run(index_dao.refresh_all_indexes())
    
    # 刷新实时数据
    asyncio.run(index_dao.refresh_main_indexes_realtime())
    
    # 刷新市场概览
    asyncio.run(index_dao.refresh_market_overview())
    
    # 刷新不同周期的K线数据
    periods = ['5', '15', '30', '60', 'Day']
    for period in periods:
        asyncio.run(index_dao.refresh_main_indexes_time_series(period))
    
    # 刷新技术指标
    asyncio.run(index_dao.refresh_main_indexes_technical_indicators('Day'))
    
    logger.info("手动刷新所有指数数据完成")
    return "手动刷新所有指数数据完成"

@shared_task
def update_market_data():
    """
    定时任务：更新市场数据
    每60秒执行一次
    包括指数行情、市场概览等数据
    """
    logger.info("开始执行市场数据更新任务")
    try:
        # TODO: 实现具体的市场数据更新逻辑
        # 例如：从API获取最新指数数据，更新到数据库和缓存
        logger.info("市场数据更新成功")
        return "市场数据更新成功"
    except Exception as e:
        logger.error(f"市场数据更新失败: {str(e)}")
        return f"市场数据更新失败: {str(e)}" 