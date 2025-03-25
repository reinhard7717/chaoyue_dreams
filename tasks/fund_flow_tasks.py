import logging
import asyncio
import datetime
from celery import shared_task

from django.utils import timezone

from dao_manager.daos.fund_flow_dao import FundFlowDAO, StockPoolDAO

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
def refresh_stock_fund_flow(stock_code):
    """
    刷新指定股票的资金流向数据
    
    Args:
        stock_code: 股票代码
    """
    logger.info(f"开始刷新股票[{stock_code}]的资金流向数据")
    dao = FundFlowDAO()
    
    try:
        # 刷新分钟级资金流向
        minute_result = run_async(dao.refresh_fund_flow_minute(stock_code))
        # 刷新日级资金流向
        daily_result = run_async(dao.refresh_fund_flow_daily(stock_code))
        # 刷新阶段主力动向
        main_force_result = run_async(dao.refresh_main_force_phase(stock_code))
        # 刷新成交分布
        transaction_result = run_async(dao.refresh_transaction_distribution(stock_code))
        
        results = {
            "minute": len(minute_result),
            "daily": len(daily_result),
            "main_force": len(main_force_result),
            "transaction": len(transaction_result)
        }
        
        logger.info(f"刷新股票[{stock_code}]的资金流向数据完成，结果: {results}")
        return results
    except Exception as e:
        logger.error(f"刷新股票[{stock_code}]的资金流向数据失败: {str(e)}")
        return {"error": str(e)}

@shared_task
def refresh_last10_stock_fund_flow(stock_code):
    """
    刷新指定股票的最近10天资金流向数据
    
    Args:
        stock_code: 股票代码
    """
    logger.info(f"开始刷新股票[{stock_code}]的最近10天资金流向数据")
    dao = FundFlowDAO()
    
    try:
        # 刷新最近10天日级资金流向
        daily_result = run_async(dao.refresh_last10_fund_flow_daily(stock_code))
        # 刷新最近10天阶段主力动向
        main_force_result = run_async(dao.refresh_last10_main_force_phase(stock_code))
        # 刷新最近10天成交分布
        transaction_result = run_async(dao.refresh_last10_transaction_distribution(stock_code))
        
        results = {
            "daily": len(daily_result),
            "main_force": len(main_force_result),
            "transaction": len(transaction_result)
        }
        
        logger.info(f"刷新股票[{stock_code}]的最近10天资金流向数据完成，结果: {results}")
        return results
    except Exception as e:
        logger.error(f"刷新股票[{stock_code}]的最近10天资金流向数据失败: {str(e)}")
        return {"error": str(e)}

@shared_task
def refresh_popular_stocks_fund_flow():
    """
    刷新热门股票的资金流向数据
    """
    logger.info("开始刷新热门股票的资金流向数据")
    
    # 热门股票代码列表 (示例，实际应从数据库或配置中获取)
    popular_stocks = [
        "000001", "600000", "600036", "601318", "600519", "300750", "000858",
        "002415", "300059", "600276", "601166", "600887", "000333", "603288"
    ]
    
    for stock_code in popular_stocks:
        refresh_stock_fund_flow.delay(stock_code)
    
    logger.info(f"已安排{len(popular_stocks)}个热门股票的资金流向数据刷新任务")
    return len(popular_stocks)

@shared_task
def refresh_active_stocks_fund_flow_minute():
    """
    刷新活跃股票的分钟级资金流向数据
    """
    logger.info("开始刷新活跃股票的分钟级资金流向数据")
    dao = FundFlowDAO()
    
    # 从当日成交量最大的股票中获取活跃股票 (示例，实际逻辑应更复杂)
    active_stocks = [
        "000001", "600000", "600036", "601318", "600519", "300750", "000858",
        "002415", "300059", "600276", "601166", "600887", "000333", "603288"
    ]
    
    results = []
    for stock_code in active_stocks:
        try:
            result = run_async(dao.refresh_fund_flow_minute(stock_code))
            results.append({stock_code: len(result)})
        except Exception as e:
            logger.error(f"刷新股票[{stock_code}]的分钟级资金流向数据失败: {str(e)}")
            results.append({stock_code: f"error: {str(e)}"})
    
    logger.info(f"刷新活跃股票的分钟级资金流向数据完成，结果: {results}")
    return results

@shared_task
def historical_fund_flow_import(stock_code):
    """
    导入指定股票的历史资金流向数据
    
    Args:
        stock_code: 股票代码
    """
    logger.info(f"开始导入股票[{stock_code}]的历史资金流向数据")
    dao = FundFlowDAO()
    
    try:
        # 刷新日级资金流向
        daily_result = run_async(dao.refresh_fund_flow_daily(stock_code))
        # 刷新阶段主力动向
        main_force_result = run_async(dao.refresh_main_force_phase(stock_code))
        # 刷新成交分布
        transaction_result = run_async(dao.refresh_transaction_distribution(stock_code))
        
        results = {
            "daily": len(daily_result),
            "main_force": len(main_force_result),
            "transaction": len(transaction_result)
        }
        
        logger.info(f"导入股票[{stock_code}]的历史资金流向数据完成，结果: {results}")
        return results
    except Exception as e:
        logger.error(f"导入股票[{stock_code}]的历史资金流向数据失败: {str(e)}")
        return {"error": str(e)}

@shared_task
def bulk_historical_fund_flow_import(stock_codes=None):
    """
    批量导入多只股票的历史资金流向数据
    
    Args:
        stock_codes: 股票代码列表，如果为None则使用默认热门股票列表
    """
    if stock_codes is None:
        # 默认热门股票列表
        stock_codes = [
            "000001", "600000", "600036", "601318", "600519", "300750", "000858",
            "002415", "300059", "600276", "601166", "600887", "000333", "603288"
        ]
    
    logger.info(f"开始批量导入{len(stock_codes)}只股票的历史资金流向数据")
    
    for stock_code in stock_codes:
        historical_fund_flow_import.delay(stock_code)
    
    logger.info(f"已安排{len(stock_codes)}只股票的历史资金流向数据导入任务")
    return len(stock_codes)


@shared_task
def refresh_limit_up_pool(date=None):
    """
    刷新指定日期的涨停股池数据
    
    Args:
        date: 日期，格式YYYY-MM-DD，默认为当天
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始刷新日期[{date}]的涨停股池数据")
    dao = StockPoolDAO()
    
    try:
        result = run_async(dao.refresh_limit_up_pool(date))
        logger.info(f"刷新日期[{date}]的涨停股池数据完成，获取到{len(result)}条记录")
        return len(result)
    except Exception as e:
        logger.error(f"刷新日期[{date}]的涨停股池数据失败: {str(e)}")
        return {"error": str(e)}

@shared_task
def refresh_limit_down_pool(date=None):
    """
    刷新指定日期的跌停股池数据
    
    Args:
        date: 日期，格式YYYY-MM-DD，默认为当天
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始刷新日期[{date}]的跌停股池数据")
    dao = StockPoolDAO()
    
    try:
        result = run_async(dao.refresh_limit_down_pool(date))
        logger.info(f"刷新日期[{date}]的跌停股池数据完成，获取到{len(result)}条记录")
        return len(result)
    except Exception as e:
        logger.error(f"刷新日期[{date}]的跌停股池数据失败: {str(e)}")
        return {"error": str(e)}

@shared_task
def refresh_strong_stock_pool(date=None):
    """
    刷新指定日期的强势股池数据
    
    Args:
        date: 日期，格式YYYY-MM-DD，默认为当天
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始刷新日期[{date}]的强势股池数据")
    dao = StockPoolDAO()
    
    try:
        result = run_async(dao.refresh_strong_stock_pool(date))
        logger.info(f"刷新日期[{date}]的强势股池数据完成，获取到{len(result)}条记录")
        return len(result)
    except Exception as e:
        logger.error(f"刷新日期[{date}]的强势股池数据失败: {str(e)}")
        return {"error": str(e)}

@shared_task
def refresh_new_stock_pool(date=None):
    """
    刷新指定日期的次新股池数据
    
    Args:
        date: 日期，格式YYYY-MM-DD，默认为当天
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始刷新日期[{date}]的次新股池数据")
    dao = StockPoolDAO()
    
    try:
        result = run_async(dao.refresh_new_stock_pool(date))
        logger.info(f"刷新日期[{date}]的次新股池数据完成，获取到{len(result)}条记录")
        return len(result)
    except Exception as e:
        logger.error(f"刷新日期[{date}]的次新股池数据失败: {str(e)}")
        return {"error": str(e)}

@shared_task
def refresh_break_limit_pool(date=None):
    """
    刷新指定日期的炸板股池数据
    
    Args:
        date: 日期，格式YYYY-MM-DD，默认为当天
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始刷新日期[{date}]的炸板股池数据")
    dao = StockPoolDAO()
    
    try:
        result = run_async(dao.refresh_break_limit_pool(date))
        logger.info(f"刷新日期[{date}]的炸板股池数据完成，获取到{len(result)}条记录")
        return len(result)
    except Exception as e:
        logger.error(f"刷新日期[{date}]的炸板股池数据失败: {str(e)}")
        return {"error": str(e)}

@shared_task
def refresh_daily_limit_pools(date=None):
    """
    刷新指定日期的涨停和跌停股池数据
    
    Args:
        date: 日期，格式YYYY-MM-DD，默认为当天
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始刷新日期[{date}]的涨停和跌停股池数据")
    
    # 启动子任务
    refresh_limit_up_pool.delay(date)
    refresh_limit_down_pool.delay(date)
    refresh_break_limit_pool.delay(date)
    
    logger.info(f"已安排日期[{date}]的涨停和跌停股池数据刷新任务")
    return True

@shared_task
def refresh_daily_strong_stocks(date=None):
    """
    刷新指定日期的强势股和次新股池数据
    
    Args:
        date: 日期，格式YYYY-MM-DD，默认为当天
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始刷新日期[{date}]的强势股和次新股池数据")
    
    # 启动子任务
    refresh_strong_stock_pool.delay(date)
    refresh_new_stock_pool.delay(date)
    
    logger.info(f"已安排日期[{date}]的强势股和次新股池数据刷新任务")
    return True

@shared_task
def historical_stock_pools_import(start_date, end_date=None):
    """
    导入历史股票池数据
    
    Args:
        start_date: 开始日期，格式YYYY-MM-DD
        end_date: 结束日期，格式YYYY-MM-DD，默认为当天
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始导入从{start_date}到{end_date}的历史股票池数据")
    
    # 计算日期范围
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # 生成日期列表，只包含工作日（周一至周五）
    date_list = []
    current = start
    while current <= end:
        # 判断是否为工作日（周一到周五）
        if current.weekday() < 5:
            date_list.append(current.strftime('%Y-%m-%d'))
        current += datetime.timedelta(days=1)
    
    # 为每个日期安排任务
    for date in date_list:
        refresh_limit_up_pool.delay(date)
        refresh_limit_down_pool.delay(date)
        refresh_strong_stock_pool.delay(date)
        refresh_new_stock_pool.delay(date)
        refresh_break_limit_pool.delay(date)
    
    logger.info(f"已安排{len(date_list)}天的历史股票池数据导入任务")
    return len(date_list)
