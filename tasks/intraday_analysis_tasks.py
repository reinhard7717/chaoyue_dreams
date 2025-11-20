# 文件: tasks/intraday_analysis_tasks.py

import logging
from datetime import date, datetime, time, timedelta
from celery import shared_task
from django.db import transaction
from asgiref.sync import async_to_sync

from config.celery_app import celery_app
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockDailyBasic
from stock_models.time_trade import IntradayChipDynamics, DailyTurnoverDistribution
from services.calculators.intraday_dynamics_calculator import IntradayDynamicsCalculator
from utils.task_helpers import with_cache_manager # 引入装饰器

logger = logging.getLogger('celery_tasks')

@celery_app.task(bind=True, name='tasks.intraday_analysis.compute_dynamics_for_stock_and_date')
@with_cache_manager
def compute_dynamics_for_stock_and_date(self, stock_code: str, trade_date_str: str, *, cache_manager):
    """
    【核心执行器】为单只股票、单个交易日计算并存储日内动态指标。
    这是一个原子任务，可以被上层调度器大规模调用。
    Args:
        stock_code (str): 股票代码, e.g., '000001.SZ'
        trade_date_str (str): 交易日期字符串, 格式 'YYYY-MM-DD'
    """
    # --- 1. 初始化DAO和解析参数 ---
    time_trade_dao = StockTimeTradeDAO(cache_manager)
    basic_dao = StockBasicInfoDao(cache_manager)
    trade_date = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
    logger.info(f"[{stock_code} @ {trade_date_str}] 开始执行日内动态指标计算任务...")
    # --- 2. 异步数据准备 ---
    # 使用 async_to_sync 在同步的Celery任务中调用异步DAO方法
    @async_to_sync
    async def get_required_data():
        # 2.1 获取1分钟K线数据 (最高精度)
        # 注意：get_minute_kline_by_daterange 需要带时区的datetime对象
        start_dt = datetime.combine(trade_date, time.min)
        end_dt = datetime.combine(trade_date, time.max)
        # Django ORM查询时不需要时区，但如果DAO内部有处理，需注意
        minute_df = await time_trade_dao.get_minute_kline_by_daterange(
            stock_code=stock_code,
            time_level='1T', # 使用 '1T' 作为频率字符串
            start_dt=start_dt,
            end_dt=end_dt
        )
        # 2.2 获取股票基础信息
        stock_info = await basic_dao.get_stock_by_code(stock_code)
        # 2.3 获取当日的日线基本面信息 (用于获取流通股本)
        # 注意：这里假设有一个 get_daily_basic_by_date 的DAO方法
        # 如果没有，需要在这里实现它
        @transaction.atomic # 确保数据库读取在事务中
        def get_daily_basic_sync(stock, t_date):
            try:
                # 使用同步ORM查询
                return StockDailyBasic.objects.get(stock=stock, trade_time=t_date)
            except StockDailyBasic.DoesNotExist:
                return None
        daily_basic_info = get_daily_basic_sync(stock_info, trade_date)
        return minute_df, stock_info, daily_basic_info
    try:
        minute_df, stock_info, daily_basic_info = get_required_data()
        # --- 3. 数据校验 ---
        if stock_info is None:
            logger.error(f"[{stock_code}] 无法在数据库中找到股票基础信息，任务终止。")
            return {'status': 'failed', 'reason': 'StockInfo not found'}
        if minute_df is None or minute_df.empty:
            logger.warning(f"[{stock_code} @ {trade_date_str}] 未找到当日的1分钟K线数据，任务跳过。")
            return {'status': 'skipped', 'reason': 'No 1-minute data'}
        if daily_basic_info is None:
            logger.warning(f"[{stock_code} @ {trade_date_str}] 未找到当日的日线基本面数据(StockDailyBasic)，部分指标可能无法计算。")
            # 即使没有，我们也可以继续，计算器内部会处理None的情况
        # --- 4. 实例化计算器并执行计算 ---
        calculator = IntradayDynamicsCalculator(minute_df, stock_info, daily_basic_info)
        dynamics_data, distribution_data = calculator.calculate_all()
        # --- 5. 数据持久化 ---
        # 使用事务确保数据一致性，要么都成功，要么都失败
        with transaction.atomic():
            # 5.1 创建或更新 IntradayChipDynamics 记录
            dynamics_obj, created = IntradayChipDynamics.objects.update_or_create(
                stock=stock_info,
                trade_date=trade_date,
                defaults=dynamics_data
            )
            action = "创建" if created else "更新"
            logger.info(f"[{stock_code} @ {trade_date_str}] 成功 {action} IntradayChipDynamics 记录。")
            # 5.2 创建或更新 DailyTurnoverDistribution 记录
            DailyTurnoverDistribution.objects.update_or_create(
                intraday_dynamics=dynamics_obj,
                defaults=distribution_data
            )
            logger.info(f"[{stock_code} @ {trade_date_str}] 成功关联并保存 DailyTurnoverDistribution 记录。")
        return {'status': 'success', 'stock_code': stock_code, 'trade_date': trade_date_str}
    except Exception as e:
        logger.error(f"[{stock_code} @ {trade_date_str}] 计算日内动态指标时发生未知异常: {e}", exc_info=True)
        # 重新抛出异常，以便Celery可以将任务标记为失败
        raise self.retry(exc=e, countdown=60, max_retries=3)

