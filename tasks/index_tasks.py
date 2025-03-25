import logging
import asyncio
import datetime
from celery import shared_task

from django.utils import timezone

from dao_manager.daos.index_dao import StockIndexDAO

logger = logging.getLogger(__name__)

# 创建异步事件循环运行器
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@shared_task
def refresh_indexes():
    """
    刷新所有指数列表
    """
    logger.info("开始刷新所有指数列表")
    dao = StockIndexDAO()
    result = run_async(dao.refresh_all_indexes())
    logger.info(f"刷新所有指数列表完成，结果: {result}")
    return result

@shared_task
def refresh_main_indexes_realtime_data():
    """
    刷新主要指数的实时数据
    """
    logger.info("开始刷新主要指数实时数据")
    dao = StockIndexDAO()
    main_indexes = ["sh000001", "sz399001", "sz399006"]  # 上证指数、深证成指、创业板指
    results = []
    
    for index_code in main_indexes:
        try:
            result = run_async(dao.refresh_index_realtime_data(index_code))
            results.append({index_code: "success" if result else "failed"})
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]实时数据失败: {str(e)}")
            results.append({index_code: f"error: {str(e)}"})
    
    logger.info(f"刷新主要指数实时数据完成，结果: {results}")
    return results

@shared_task
def refresh_index_realtime_data(index_code):
    """
    刷新指定指数的实时数据
    
    Args:
        index_code: 指数代码
    """
    logger.info(f"开始刷新指数[{index_code}]实时数据")
    dao = StockIndexDAO()
    result = run_async(dao.refresh_index_realtime_data(index_code))
    logger.info(f"刷新指数[{index_code}]实时数据完成，结果: {result is not None}")
    return result is not None

@shared_task
def refresh_market_overview():
    """
    刷新市场概览数据
    """
    logger.info("开始刷新市场概览数据")
    dao = StockIndexDAO()
    result = run_async(dao.refresh_market_overview())
    logger.info(f"刷新市场概览数据完成，结果: {result is not None}")
    return result is not None

@shared_task
def refresh_main_indexes_time_series(time_level):
    """
    刷新主要指数的分时数据
    
    Args:
        time_level: 时间级别 (5, 15, 30, 60, Day, Week, Month, Year)
    """
    logger.info(f"开始刷新主要指数的{time_level}级别分时数据")
    dao = StockIndexDAO()
    main_indexes = ["sh000001", "sz399001", "sz399006"]  # 上证指数、深证成指、创业板指
    results = []
    
    for index_code in main_indexes:
        try:
            result = run_async(dao.refresh_time_series_data(index_code, time_level))
            results.append({index_code: "success" if result else "failed"})
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别分时数据失败: {str(e)}")
            results.append({index_code: f"error: {str(e)}"})
    
    logger.info(f"刷新主要指数的{time_level}级别分时数据完成，结果: {results}")
    return results

@shared_task
def refresh_main_indexes_technical_indicators(time_level):
    """
    刷新主要指数的技术指标数据
    
    Args:
        time_level: 时间级别 (5, 15, 30, 60, Day, Week, Month, Year)
    """
    logger.info(f"开始刷新主要指数的{time_level}级别技术指标数据")
    dao = StockIndexDAO()
    main_indexes = ["sh000001", "sz399001", "sz399006"]  # 上证指数、深证成指、创业板指
    results = []
    
    for index_code in main_indexes:
        try:
            # 刷新KDJ指标
            kdj_result = run_async(dao.refresh_kdj_data(index_code, time_level))
            # 刷新MACD指标
            macd_result = run_async(dao.refresh_macd_data(index_code, time_level))
            # 刷新MA指标
            ma_result = run_async(dao.refresh_ma_data(index_code, time_level))
            # 刷新BOLL指标
            boll_result = run_async(dao.refresh_boll_data(index_code, time_level))
            
            results.append({
                index_code: {
                    "kdj": len(kdj_result),
                    "macd": len(macd_result),
                    "ma": len(ma_result),
                    "boll": len(boll_result)
                }
            })
        except Exception as e:
            logger.error(f"刷新指数[{index_code}]的{time_level}级别技术指标数据失败: {str(e)}")
            results.append({index_code: f"error: {str(e)}"})
    
    logger.info(f"刷新主要指数的{time_level}级别技术指标数据完成，结果: {results}")
    return results

@shared_task
def historical_data_import(index_code, start_date=None, end_date=None):
    """
    导入指定指数的历史数据
    
    Args:
        index_code: 指数代码
        start_date: 开始日期，格式YYYY-MM-DD，默认为None（从最早数据开始）
        end_date: 结束日期，格式YYYY-MM-DD，默认为None（到最新数据结束）
    """
    logger.info(f"开始导入指数[{index_code}]的历史数据，时间范围: {start_date} 至 {end_date}")
    dao = StockIndexDAO()
    
    # 导入日线数据
    day_result = run_async(dao.refresh_time_series_data(index_code, 'Day'))
    
    # 导入技术指标数据
    kdj_result = run_async(dao.refresh_kdj_data(index_code, 'Day'))
    macd_result = run_async(dao.refresh_macd_data(index_code, 'Day'))
    ma_result = run_async(dao.refresh_ma_data(index_code, 'Day'))
    boll_result = run_async(dao.refresh_boll_data(index_code, 'Day'))
    
    results = {
        "time_series": len(day_result),
        "kdj": len(kdj_result),
        "macd": len(macd_result),
        "ma": len(ma_result),
        "boll": len(boll_result)
    }
    
    logger.info(f"导入指数[{index_code}]的历史数据完成，结果: {results}")
    return results

@shared_task
def bulk_historical_data_import():
    """
    批量导入所有主要指数的历史数据
    """
    logger.info("开始批量导入所有主要指数的历史数据")
    dao = StockIndexDAO()
    
    # 刷新指数列表
    run_async(dao.refresh_all_indexes())
    
    # 获取所有指数
    all_indexes = run_async(dao.get_all_indexes())
    index_codes = [index.code for index in all_indexes]
    
    # 批量导入每个指数的历史数据
    for index_code in index_codes:
        historical_data_import.delay(index_code)
    
    logger.info(f"已安排{len(index_codes)}个指数的历史数据导入任务")
    return len(index_codes)
