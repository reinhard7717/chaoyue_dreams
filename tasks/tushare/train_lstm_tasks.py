import os
import logging
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
import pandas as pd
import asyncio
from chaoyue_dreams.celery import app as celery_app
from django.core.management.base import CommandError
from services.indicator_services import IndicatorService
from stock_models.stock_basic import StockInfo
from strategies.trend_following_strategy import TrendFollowingStrategy

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='tasks.tushare.train_lstm_tasks.batch_train_following_strategy_lstm')
def batch_train_following_strategy_lstm(self, stock: StockInfo, params_file: str = "strategies/indicator_parameters.json", model_dir="models"):
    """
    批量训练LSTM模型并自动保存模型和Scaler
    :param stock_list: 股票代码列表
    :param data_dir: 训练数据目录
    :param params_file: 策略参数文件
    :param model_dir: 模型保存目录
    """
    indicator_service = IndicatorService()
    try:
        # 传递参数文件路径，服务内部应解析所有需要的指标参数
        print(f"开始执行 {stock} 的深度学习任务")
        data_df = asyncio.run(indicator_service.prepare_strategy_dataframe(
            stock_code=stock.stock_code,
            params_file=params_file # 传递参数文件路径
        ))
    except Exception as prep_err:
        print(f"[{stock}] 调用 prepare_strategy_dataframe 准备数据时出错: {prep_err}")
        return
    if data_df is None or data_df.empty:
        print(f"[{stock}] 未能准备足够的数据 (prepare_strategy_dataframe 返回空)。")
        return
    print(f"开始训练: {stock}")
    strategy = TrendFollowingStrategy(params_file, model_dir=model_dir)
    try:
        strategy.train_and_save_lstm_model(data_df, stock.stock_code)
    except Exception as e:
        print(f"训练{stock}时出错: {e}")


@celery_app.task(bind=True, name='tasks.tushare.train_lstm_tasks.train_lstm_trend_following_strategy_task')
def train_lstm_trend_following_strategy_task(self): # 最大循环10万个，每310个一组循环一次是99510个
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: train_lstm_trend_following_strategy_task (调度器模式) - 获取股票列表并分派批量任务")
    try:
        total_dispatched_batches = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        for stock in all_stocks:
            logger.info(f"创建 深度学习 批次任务...")
            # 使用新的批量任务，并指定队列
            batch_train_following_strategy_lstm.s(stock=stock).set(queue="Train_LSTM").apply_async()
            total_dispatched_batches += 1
        logger.info(f"任务结束: train_lstm_trend_following_strategy_task (调度器模式) - 共分派 {total_dispatched_batches} 个任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 train_lstm_trend_following_strategy_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}
















