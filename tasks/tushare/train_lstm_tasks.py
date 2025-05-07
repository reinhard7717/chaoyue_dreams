import os
import logging
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
import pandas as pd
import asyncio
from chaoyue_dreams.celery import app as celery_app
from services.indicator_services import IndicatorService
from strategies.trend_following_strategy import TrendFollowingStrategy
# 导入 prepare_data_for_lstm 函数，因为新的准备数据任务需要直接调用它
from strategies.utils.deep_learning_utils import prepare_data_for_lstm

logger = logging.getLogger("tasks")

# 任务：准备 LSTM 训练数据并保存
@celery_app.task(bind=True, name='tasks.tushare.train_lstm_tasks.batch_prepare_lstm_data')
def batch_prepare_lstm_data(self, stock_code: str, params_file: str = "strategies/indicator_parameters.json", model_dir="models", base_bars: int = 10000):
    """
    为特定股票准备 LSTM 训练数据（特征、目标、Scaler）并保存到文件。
    这个任务只负责数据准备和保存，不进行模型训练。
    :param stock_code: 股票代码
    :param params_file: 策略参数文件
    :param model_dir: 模型和数据保存目录
    :param base_bars: 需要的基础 K 线数据量
    """
    logger.info(f"开始执行 {stock_code} 的数据准备任务...")
    indicator_service = IndicatorService()
    strategy = TrendFollowingStrategy(params_file, base_model_dir=model_dir)
    try:
        # 1. 获取原始数据
        logger.info(f"[{stock_code}] 获取原始数据...")
        # prepare_strategy_dataframe 应该获取并计算基础指标（如MACD, RSI, DMI等），但不生成最终交易信号
        data_df = asyncio.run(indicator_service.prepare_strategy_dataframe(
            stock_code=stock_code,
            params_file=params_file,
            base_needed_bars=base_bars
        ))
    except Exception as prep_err:
        logger.error(f"[{stock_code}] 调用 prepare_strategy_dataframe 获取原始数据时出错: {prep_err}", exc_info=True)
        # 任务失败，可以抛出异常让 Celery 标记为失败
        raise prep_err

    if data_df is None or data_df.empty:
        logger.warning(f"[{stock_code}] 未能获取原始数据 (prepare_strategy_dataframe 返回空)。")
        # 任务成功完成，但没有数据可准备
        return {"status": "warning", "message": "未能获取原始数据"}

    logger.info(f"[{stock_code}] 原始数据获取完成，共 {len(data_df)} 条。")

    # --- 关键修改：调用策略方法生成交易信号，包括 'final_signal' ---
    try:
        logger.info(f"[{stock_code}] 调用策略生成信号...")
        # 调用策略的 generate_signals 方法，它会计算出 final_signal 等列并添加到 DataFrame 中
        # 确保 generate_signals 方法返回修改后的 DataFrame 或者在原地修改 data_df
        # 稳妥起见，假设它返回修改后的 DataFrame
        data_df = strategy.generate_signals(data_df)
        if data_df is None or data_df.empty or 'final_signal' not in data_df.columns:
            logger.error(f"[{stock_code}] 策略生成信号失败或未生成 'final_signal' 列。")
            # 任务失败
            raise RuntimeError("策略生成信号失败或未生成 'final_signal' 列。")
        logger.info(f"[{stock_code}] 信号生成完成，DataFrame 现在包含 'final_signal' 列。")
    except Exception as signal_err:
        logger.error(f"[{stock_code}] 调用策略生成信号时出错: {signal_err}", exc_info=True)
        # 任务失败
        raise signal_err
    # --- 结束关键修改 ---


    # 2. 准备 LSTM 训练数据 (调用 deep_learning_utils 中的函数)
    try:
        logger.info(f"[{stock_code}] 开始准备 LSTM 数据...")
        # 从策略参数中获取数据准备相关的配置
        tf_params = strategy.params.get('trend_following_params', {})
        features_scaled_train, targets_scaled_train, \
        features_scaled_val, targets_scaled_val, \
        features_scaled_test, targets_scaled_test, \
        feature_scaler, target_scaler = prepare_data_for_lstm(
            data=data_df, # 使用经过 generate_signals 处理后的 data_df
            required_columns=strategy.get_required_columns(), # 使用策略定义的所需列
            target_column='final_signal', # 目标列是规则生成的 final_signal
            scaler_type=tf_params.get('lstm_scaler_type', 'minmax'),
            train_split=tf_params.get('lstm_train_split', 0.7),
            val_split=tf_params.get('lstm_val_split', 0.15),
            apply_variance_threshold=tf_params.get('lstm_apply_variance_threshold', False), # 添加方差过滤参数
            variance_threshold_value=tf_params.get('lstm_variance_threshold_value', 0.01), # 添加方差阈值参数
            use_feature_selection=tf_params.get('lstm_use_feature_selection', False),
            feature_selector_model=tf_params.get('lstm_feature_selector_model', 'rf'),
            max_features_fs=tf_params.get('lstm_max_features_fs', None),
            feature_selection_threshold=tf_params.get('lstm_feature_selection_threshold', 'median'),
            use_pca=tf_params.get('lstm_use_pca', False),
            n_components=tf_params.get('lstm_pca_n_components', 0.99),
            target_scaler_type=tf_params.get('lstm_target_scaler_type', 'minmax')
        )
        # 检查数据是否有效 (确保训练集有数据且 scaler 成功拟合)
        # prepare_data_for_lstm 在数据无效时会返回 None Scaler 或空数组
        if features_scaled_train.shape[0] == 0 or targets_scaled_train.shape[0] == 0 or feature_scaler is None or target_scaler is None:
             logger.error(f"[{stock_code}] 数据准备失败，训练集为空或 Scaler 未成功拟合。")
             # 任务失败
             # 明确抛出异常，让 Celery 标记为失败
             raise ValueError("数据准备失败，训练集为空或 Scaler 未成功拟合。")

        logger.info(f"[{stock_code}] 数据准备完成。训练集 shape: {features_scaled_train.shape}, 验证集 shape: {features_scaled_val.shape}, 测试集 shape: {features_scaled_test.shape}")
    except Exception as data_prep_err:
        logger.error(f"[{stock_code}] 准备 LSTM 数据时出错: {data_prep_err}", exc_info=True)
        # 任务失败，重新抛出异常
        raise data_prep_err

    # 3. 保存准备好的数据和 Scaler (调用策略类的方法)
    try:
        # set_model_paths 在 save_prepared_data 内部调用或在 train_lstm_model_from_prepared_data 中调用
        # 这里保存时调用一次确保目录存在
        strategy.set_model_paths(stock_code)
        strategy.save_prepared_data(
            stock_code,
            features_scaled_train, targets_scaled_train,
            features_scaled_val, targets_scaled_val,
            features_scaled_test, targets_scaled_test,
            feature_scaler, target_scaler
        )
        logger.info(f"[{stock_code}] 准备好的数据和 Scaler 已成功保存。")
        return {"status": "success", "stock_code": stock_code, "train_samples": features_scaled_train.shape[0]}

    except Exception as save_err:
        logger.error(f"[{stock_code}] 保存准备好的数据或 Scaler 时出错: {save_err}", exc_info=True)
        # 任务失败，重新抛出异常
        raise save_err
    
# 调度器任务：仅调度数据准备任务
@celery_app.task(bind=True, name='tasks.tushare.train_lstm_tasks.schedule_lstm_data_preparation')
def schedule_lstm_data_preparation(self, params_file: str = "strategies/indicator_parameters.json", model_dir="models", base_bars_to_request: int = 10000):
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 为每个股票创建并分派一个数据准备任务 (batch_prepare_lstm_data) 到指定队列。
    这个任务由 Celery Beat 调度，用于触发数据准备的多进程处理。
    :param params_file: 策略参数文件路径
    :param model_dir: 模型和数据保存目录
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: schedule_lstm_data_preparation (调度器模式) - 获取股票列表并分派数据准备任务")
    total_dispatched_tasks = 0
    try:
        stock_basic_dao = StockBasicInfoDao()
        # 假设 get_stock_list 返回的是一个列表或迭代器，每个元素有 stock_code 属性
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())

        if not all_stocks:
            logger.warning("未获取到股票列表，跳过数据准备任务分派。")
            return {"status": "warning", "message": "未获取到股票列表", "dispatched_tasks": 0}

        for stock in all_stocks:
            stock_code = stock.stock_code
            logger.info(f"分派 {stock_code} 的数据准备任务到 'Prepare_Data' 队列...")

            # 定义数据准备任务签名
            prepare_task_signature = batch_prepare_lstm_data.s(
                stock_code=stock_code,
                params_file=params_file,
                model_dir=model_dir,
                base_bars=base_bars_to_request
            ).set(queue="Prepare_Data") # 将任务分配到 Prepare_Data 队列

            # 分派单个任务异步执行
            prepare_task_signature.apply_async()

            total_dispatched_tasks += 1

        logger.info(f"任务结束: schedule_lstm_data_preparation (调度器模式) - 共分派 {total_dispatched_tasks} 个数据准备任务")
        return {"status": "success", "dispatched_tasks": total_dispatched_tasks}

    except Exception as e:
        logger.error(f"执行 schedule_lstm_data_preparation (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_tasks": total_dispatched_tasks}


# 修改任务：训练 LSTM 模型 (从已准备数据加载)
# 移除 base_bars 参数，因为它不再负责数据获取
@celery_app.task(bind=True, name='tasks.tushare.train_lstm_tasks.batch_train_following_strategy_lstm')
def batch_train_following_strategy_lstm(self, stock_code: str, params_file: str = "strategies/indicator_parameters.json", model_dir="models"):
    """
    为特定股票加载已准备好的数据，构建并训练 LSTM 模型。
    这个任务假设数据已经通过 prepare_lstm_data_task 准备好并保存。
    :param stock_code: 股票代码
    :param params_file: 策略参数文件
    :param model_dir: 模型和数据保存目录
    """
    logger.info(f"开始执行 {stock_code} 的模型训练任务 (从已准备数据加载)...")
    strategy = TrendFollowingStrategy(params_file, base_model_dir=model_dir)

    try:
        # 调用策略类中修改后的训练方法，它内部会加载数据
        strategy.train_lstm_model_from_prepared_data(stock_code)
        # train_lstm_model_from_prepared_data 内部会处理加载失败的情况并记录错误
        # 如果该方法成功执行到最后，则认为训练任务成功
        if strategy.lstm_model is not None:
             logger.info(f"[{stock_code}] 模型训练任务完成。")
             return {"status": "success", "stock_code": stock_code}
        else:
             # 如果 strategy.lstm_model 为 None，说明训练方法内部加载数据失败或构建/训练模型失败
             logger.error(f"[{stock_code}] 模型训练任务失败，请检查日志获取详细信息。")
             # 任务失败
             raise RuntimeError("模型训练任务失败，请检查日志。")

    except Exception as e:
        logger.error(f"执行股票 {stock_code} 的模型训练任务时发生意外错误: {e}", exc_info=True)
        # 任务失败
        raise e


# 修改调度器任务，创建任务链
@celery_app.task(bind=True, name='tasks.tushare.train_lstm_tasks.train_lstm_trend_following_strategy_task')
def train_lstm_trend_following_strategy_task(self, base_bars_to_request: int = 8000):
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 为每个股票创建一个任务链：先准备数据，然后训练模型。
    3. 将任务链分派到指定队列。
    这个任务由 Celery Beat 调度。
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: train_lstm_trend_following_strategy_task (调度器模式) - 获取股票列表并分派任务链")
    try:
        total_dispatched_chains = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        for stock in all_stocks:
            stock_code = stock.stock_code
            logger.info(f"创建 {stock_code} 的数据准备和模型训练任务链...")

            # 定义数据准备任务签名
            prepare_task = prepare_lstm_data_task.s(
                stock_code=stock_code,
                params_file="strategies/indicator_parameters.json", # 可以从参数获取
                model_dir="models", # 可以从参数获取
                base_bars=base_bars_to_request
            ).set(queue="Prepare_Data") # 可以指定数据准备队列

            # 定义模型训练任务签名
            train_task = batch_train_following_strategy_lstm.s(
                stock_code=stock_code,
                params_file="strategies/indicator_parameters.json", # 可以从参数获取
                model_dir="models" # 可以从参数获取
            ).set(queue="Train_LSTM") # 可以指定模型训练队列

            # 创建任务链：先准备数据，然后训练
            task_chain = prepare_task | train_task

            # 分派任务链
            task_chain.apply_async()

            total_dispatched_chains += 1
        logger.info(f"任务结束: train_lstm_trend_following_strategy_task (调度器模式) - 共分派 {total_dispatched_chains} 个任务链")
        return {"status": "success", "dispatched_chains": total_dispatched_chains}
    except Exception as e:
        logger.error(f"执行 train_lstm_trend_following_strategy_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_chains": 0}

