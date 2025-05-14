# tasks/tushare/train_transformer_tasks.py
import json
import os
import logging
import asyncio # 导入 asyncio

from django.conf import settings
# 假设 StockBasicInfoDao 存在且可用
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
import pandas as pd
# 假设 celery 实例存在且可用
from chaoyue_dreams.celery import app as celery_app
# 假设 IndicatorService 存在且可用
from services.indicator_services import IndicatorService
# 导入修改后的 TrendFollowingStrategy 类
from strategies.trend_following_strategy import TrendFollowingStrategy
# 导入 prepare_data_for_transformer 函数
from strategies.utils.deep_learning_utils import prepare_data_for_transformer

logger = logging.getLogger("tasks")

# 任务：准备 Transformer 训练数据并保存
# 修改任务名称以更准确地反映其功能：处理股票数据以进行 Transformer 训练
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.process_stock_data_for_transformer_training') # 修改行：修改任务名称
def process_stock_data_for_transformer_training(self, stock_code: str, params_file: str = None, model_dir: str = None, base_bars: int = 10000):
    task_id_str = f"任务 {self.request.id if self.request else 'UnknownID'}"
    logger.info(f"{task_id_str}：开始为 {stock_code} 执行 Transformer 数据处理流程...")

    # IndicatorService 实例化可以在任务内部，也可以在策略内部，取决于设计。
    # 策略内部实例化 IndicatorService 更符合面向对象的设计，任务只负责调用策略。
    # indicator_service = IndicatorService() # 移除行：IndicatorService 在策略内部实例化

    logger.info(f"{task_id_str} [{stock_code}]：实例化 TrendFollowingStrategy...")
    # 策略初始化时会加载参数文件并实例化 IndicatorService
    strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_dir) # 修改行：实例化策略
    logger.info(f"{task_id_str} [{stock_code}]：TrendFollowingStrategy 实例化完毕，策略名: '{strategy.strategy_name}'。")

    # --- 关键检查点 ---
    if not strategy.params: # 检查 self.params 是否为空
        logger.error(f"{task_id_str} [{stock_code}]：CRITICAL TASK HALT: strategy.params 为空。参数文件可能未加载或无效。任务无法继续。")
        return {"status": "error", "message": "策略参数 strategy.params 为空，任务终止。"}

    if 'trend_following_params' not in strategy.params or not strategy.tf_params: # 检查 tf_params
        logger.error(f"{task_id_str} [{stock_code}]：CRITICAL TASK HALT: 'trend_following_params' 在 strategy.params 中缺失或 strategy.tf_params 为空。任务无法继续。")
        return {"status": "error", "message": "'trend_following_params' 缺失或为空，任务终止。"}

    logger.info(f"{task_id_str} [{stock_code}]：策略参数检查通过 (params 和 tf_params 非空)。")
    logger.info(f"{task_id_str} [{stock_code}]：确认使用的 transformer_window_size: {strategy.transformer_window_size}") # 打印一个关键参数

    # --- 阶段 1: 使用 IndicatorService 准备数据 ---
    try:
        logger.info(f"{task_id_str} [{stock_code}]：开始使用 IndicatorService 准备数据...")
        # 调用策略的 prepare_data 方法，它内部会调用 IndicatorService
        # prepare_data 返回包含所有 OHLCV, 指标, 外部特征, 衍生特征的 DataFrame
        data_df = asyncio.run(strategy.prepare_data(stock_code=stock_code)) # 修改行：调用策略的 prepare_data 方法

        if data_df is None or data_df.empty:
            logger.warning(f"{task_id_str} [{stock_code}]：未能获取准备好的数据 (data_df为空)。")
            return {"status": "warning", "message": "未能获取准备好的数据"}
        logger.info(f"{task_id_str} [{stock_code}]：准备好的数据获取完成，{len(data_df)}行，{len(data_df.columns)}列。")
        # logger.debug(f"{task_id_str} [{stock_code}]：data_df.head():\n{data_df.head()}")

        # IndicatorService.prepare_strategy_dataframe 返回的第二个元素 indicator_configs
        # 在这里不需要直接使用，因为 strategy.generate_signals 内部会处理。
        # 如果 strategy.generate_signals 需要这个参数，它应该在 prepare_data 内部获取并存储，或者由 prepare_data 返回。
        # 根据 strategy.prepare_data 的定义，它只返回 data_df，所以这里不需要 indicator_configs。

    except Exception as prep_err:
        logger.error(f"{task_id_str} [{stock_code}]：使用 IndicatorService 准备数据时出错: {prep_err}", exc_info=True)
        raise prep_err # 重新抛出，让 Celery 标记任务失败

    # --- 阶段 2: 使用策略生成规则信号 (包括 Transformer 目标列) ---
    try:
        logger.info(f"{task_id_str} [{stock_code}]：调用策略 generate_signals 方法生成规则信号及中间数据...")
        # generate_signals 需要传入 IndicatorService 返回的 data_df
        # generate_signals 内部会计算 final_rule_signal 等列并添加到 DataFrame 中
        # generate_signals 返回的 DataFrame 包含了所有用于 Transformer 的潜在特征和目标列
        data_with_all_signals = strategy.generate_signals(data=data_df, stock_code=stock_code) # 修改行：调用策略的 generate_signals 方法

        if data_with_all_signals is None or data_with_all_signals.empty:
            logger.error(f"{task_id_str} [{stock_code}]：策略 generate_signals 方法返回空或None DataFrame。")
            raise RuntimeError("策略 generate_signals 方法执行失败或未返回数据。")
        logger.info(f"{task_id_str} [{stock_code}]：策略 generate_signals 方法执行完毕，返回DataFrame {data_with_all_signals.shape}。")
        # logger.debug(f"{task_id_str} [{stock_code}]：data_with_all_signals.columns: {data_with_all_signals.columns.tolist()}")

        # 检查 Transformer 的目标列是否存在于返回的 DataFrame 中
        transformer_target_column = strategy.transformer_target_column # 从策略实例获取目标列名
        if transformer_target_column not in data_with_all_signals.columns:
            logger.error(f"{task_id_str} [{stock_code}]：返回的 DataFrame 中缺少 Transformer 目标列 '{transformer_target_column}'。")
            raise RuntimeError(f"策略未生成 Transformer 目标列 '{transformer_target_column}'。")
        logger.info(f"{task_id_str} [{stock_code}]：Transformer 目标列 '{transformer_target_column}' 存在。")
        # logger.debug(f"{task_id_str} [{stock_code}]：'{transformer_target_column}' (tail):\n{data_with_all_signals[transformer_target_column].tail()}")

    except Exception as signal_err:
        logger.error(f"{task_id_str} [{stock_code}]：策略 generate_signals 或后续检查时出错: {signal_err}", exc_info=True)
        raise signal_err

    # --- 阶段 3: 准备 Transformer 训练数据 (特征选择、标准化、窗口化、分割) ---
    data_for_prep = data_with_all_signals # 使用包含所有信号的 DataFrame 作为 prepare_data_for_transformer 的输入

    try:
        logger.info(f"{task_id_str} [{stock_code}]：开始调用 prepare_data_for_transformer...")
        tf_params = strategy.tf_params
        data_prep_config = tf_params.get('transformer_data_prep_config', {})

        # prepare_data_for_transformer 需要知道哪些列是潜在特征，哪些是目标
        # 它应该从 data_for_prep 中选择特征列，并使用 target_column 作为目标
        # strategy.get_required_columns 返回的是 IndicatorService 准备的原始列名列表
        # prepare_data_for_transformer 应该根据 data_for_prep.columns 和 data_prep_config 来决定最终特征
        # 传递 data_for_prep.columns 作为 available_columns 可能有用，或者 prepare_data_for_transformer 自己处理
        # 假设 prepare_data_for_transformer 能够智能地从 data_for_prep 中选择特征列
        # 传递 strategy.get_required_columns(stock_code) 列表给 prepare_data_for_transformer 可能不再必要，
        # 因为 data_for_prep 已经包含了所有这些列以及策略内部生成的列。
        # prepare_data_for_transformer 应该根据 data_prep_config 中的特征选择逻辑来工作。

        # 调用 prepare_data_for_transformer
        features_scaled_train, targets_scaled_train, \
        features_scaled_val, targets_scaled_val, \
        features_scaled_test, targets_scaled_test, \
        feature_scaler, target_scaler, \
        selected_feature_names = prepare_data_for_transformer(
            data=data_for_prep, # 修改行：传入包含所有信号的 DataFrame
            # required_columns=strategy.get_required_columns(stock_code), # 移除行：不再需要传递原始请求列
            target_column=transformer_target_column, # 修改行：传递 Transformer 目标列名
            window_size=strategy.transformer_window_size, # 修改行：传递 window_size
            data_prep_config=data_prep_config, # 修改行：传递数据准备配置
            # prepare_data_for_transformer 内部应该处理保存逻辑，或者我们在这里调用 strategy.save_prepared_data
            # 根据原代码结构，prepare_data_for_transformer 返回数据，然后任务调用 strategy.save_prepared_data
            # 所以这里不传递 save_paths 给 prepare_data_for_transformer
            # ... (其他 prepare_data_for_transformer 参数从 data_prep_config 获取)
            # scaler_type=data_prep_config.get('scaler_type', 'standard'), # 这些参数应该在 prepare_data_for_transformer 内部处理
            # train_split=data_prep_config.get('train_split', 0.7),
            # val_split=data_prep_config.get('val_split', 0.15),
            # apply_variance_threshold=data_prep_config.get('apply_variance_threshold', False),
            # variance_threshold_value=data_prep_config.get('variance_threshold_value', 0.01),
            # use_pca=data_prep_config.get('use_pca', False),
            # pca_n_components=data_prep_config.get('pca_n_components', 0.99),
            # pca_solver=data_prep_config.get('pca_solver', 'auto'),
            # use_feature_selection=data_prep_config.get('use_feature_selection', True),
            # feature_selector_model_type=data_prep_config.get('feature_selector_model_type', 'rf'),
            # fs_model_n_estimators=data_prep_config.get('fs_model_n_estimators', 100),
            # fs_model_max_depth=data_prep_config.get('fs_model_max_depth', None),
            # fs_max_features=data_prep_config.get('fs_max_features', 50),
            # fs_selection_threshold=data_prep_config.get('fs_selection_threshold', 'median'),
            # target_scaler_type=data_prep_config.get('target_scaler_type', 'minmax')
        )
        logger.info(f"{task_id_str} [{stock_code}]：prepare_data_for_transformer 调用成功。")

        if features_scaled_train is None or features_scaled_train.shape[0] == 0 or \
           targets_scaled_train is None or targets_scaled_train.shape[0] == 0 or \
           feature_scaler is None or target_scaler is None or not selected_feature_names:
             logger.error(f"{task_id_str} [{stock_code}]：Transformer 数据准备后，训练集为空或 Scaler/特征列表未成功生成。")
             raise ValueError("Transformer 数据准备后，训练集为空或 Scaler/特征列表未成功生成。")

        logger.info(f"{task_id_str} [{stock_code}]：Transformer 数据准备完成。训练集 shape: {features_scaled_train.shape}, 最终特征数: {len(selected_feature_names)}")

    except Exception as data_prep_err:
        logger.error(f"{task_id_str} [{stock_code}]：准备 Transformer 数据时出错: {data_prep_err}", exc_info=True)
        raise data_prep_err

    # --- 阶段 4: 保存准备好的数据和 Scaler ---
    try:
        logger.info(f"{task_id_str} [{stock_code}]：开始保存准备好的数据和 Scaler...")
        # strategy.save_prepared_data 内部会使用 strategy 实例的 model_path 等属性
        strategy.save_prepared_data(
            stock_code,
            features_scaled_train, targets_scaled_train,
            features_scaled_val, targets_scaled_val,
            features_scaled_test, targets_scaled_test,
            feature_scaler, target_scaler,
            selected_feature_names
        )
        logger.info(f"{task_id_str} [{stock_code}]：准备好的 Transformer 数据和 Scaler 已成功保存。")
        return {"status": "success", "stock_code": stock_code, "train_samples": features_scaled_train.shape[0], "final_features": len(selected_feature_names)}

    except Exception as save_err:
        logger.error(f"{task_id_str} [{stock_code}]：保存准备好的数据或 Scaler 时出错: {save_err}", exc_info=True)
        raise save_err

# 调度器任务：仅调度数据准备任务
# 修改任务名称以匹配新的 prepare 任务名称
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_transformer_data_processing') # 修改行：修改任务名称
def schedule_transformer_data_processing(self, params_file: str = None, base_data_dir: str = None, base_bars_to_request: int = 10000):
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 为每个股票创建并分派一个 Transformer 数据处理任务到指定队列。
    这个任务由 Celery Beat 调度，用于触发数据处理的多进程处理。
    :param params_file: 策略参数文件路径
    :param model_dir: 模型和数据保存目录
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: schedule_transformer_data_processing (调度器模式) - 获取股票列表并分派数据处理任务") # 修改行：日志信息
    if params_file is None:
        params_file = settings.INDICATOR_PARAMETERS_CONFIG_PATH
    if base_data_dir is None:
        base_data_dir = settings.STRATEGY_DATA_DIR
    total_dispatched_tasks = 0
    try:
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())

        if not all_stocks:
            logger.warning("未获取到股票列表，跳过数据处理任务分派。") # 修改行：日志信息
            return {"status": "warning", "message": "未获取到股票列表", "dispatched_tasks": 0}

        for stock in all_stocks:
            stock_code = stock.stock_code
            logger.info(f"分派 {stock_code} 的 Transformer 数据处理任务到 'Train_Transformer_Prepare_Data' 队列，params_file:{params_file}...") # 修改行：日志信息

            # 调用修改后的任务名称
            prepare_task_signature = process_stock_data_for_transformer_training.s( # 修改行：调用新的任务函数名
                stock_code=stock_code,
                params_file=params_file,
                model_dir=base_data_dir,
                base_bars=base_bars_to_request
            ).set(queue="Train_Transformer_Prepare_Data")

            prepare_task_signature.apply_async()
            total_dispatched_tasks += 1

        logger.info(f"任务结束: schedule_transformer_data_processing (调度器模式) - 共分派 {total_dispatched_tasks} 个数据处理任务") # 修改行：日志信息
        return {"status": "success", "dispatched_tasks": total_dispatched_tasks}

    except Exception as e:
        logger.error(f"执行 schedule_transformer_data_processing (调度器模式) 时出错: {e}", exc_info=True) # 修改行：日志信息
        return {"status": "error", "message": str(e), "dispatched_tasks": total_dispatched_tasks}


# 任务：训练 Transformer 模型 (从已准备数据加载)
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.batch_train_following_strategy_transformer')
def batch_train_following_strategy_transformer(self, stock_code: str, params_file: str = "strategies/indicator_parameters.json", model_dir="models"):
    logger.info(f"开始执行 {stock_code} 的 Transformer 模型训练任务...")
    # 实例化策略，传递参数文件和模型目录
    strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_dir)
    try:
        # strategy.train_transformer_model_from_prepared_data 内部会加载数据并训练
        strategy.train_transformer_model_from_prepared_data(stock_code)
        if strategy.transformer_model and strategy.feature_scaler and \
           strategy.target_scaler and strategy.selected_feature_names_for_transformer:
            logger.info(f"[{stock_code}] Transformer 模型训练任务完成。")
            return {"status": "success", "stock_code": stock_code}
        else:
            logger.error(f"[{stock_code}] Transformer 模型训练任务失败，模型或相关组件未成功加载/训练。")
            # 训练失败时，抛出异常以便 Celery 标记任务失败
            raise RuntimeError("Transformer 模型训练任务失败，模型或相关组件未成功加载/训练。")
    except Exception as e:
        logger.error(f"执行股票 {stock_code} 的 Transformer 模型训练任务时发生意外错误: {e}", exc_info=True)
        raise e # 重新抛出异常

# 修改调度器任务，创建任务链 (prepare -> train)
# 任务名和函数名都反映改为 Transformer
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_transformer_training_chain') # 修改行：修改任务名称
def schedule_transformer_training_chain(self, params_file: str = None, base_data_dir: str = None, base_bars_to_request: int = 10000): # 参数名一致性
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 为每个股票创建一个任务链：先准备数据，然后训练模型。
    3. 将任务链分派到指定队列。
    这个任务由 Celery Beat 调度。
    :param params_file: 策略参数文件路径
    :param model_dir: 模型和数据保存目录
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: schedule_transformer_training_chain (调度器模式) - 获取股票列表并分派任务链") # 修改行：日志信息
    try:
        total_dispatched_chains = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())

        if not all_stocks:
             logger.warning("未获取到股票列表，跳过任务链分派。")
             return {"status": "warning", "message": "未获取到股票列表", "dispatched_chains": 0}

        # 从参数获取或固定
        if params_file is None:
            params_file = settings.INDICATOR_PARAMETERS_CONFIG_PATH
        if base_data_dir is None:
            base_data_dir = settings.STRATEGY_DATA_DIR


        for stock in all_stocks:
            stock_code = stock.stock_code
            logger.info(f"创建 {stock_code} 的 Transformer 数据处理和模型训练任务链...") # 修改行：日志信息

            # 定义数据处理任务签名 (调用 process_stock_data_for_transformer_training)
            prepare_task_signature = process_stock_data_for_transformer_training.s( # 修改行：调用新的任务函数名
                stock_code=stock_code,
                params_file=params_file,
                model_dir=base_data_dir,
                base_bars=base_bars_to_request
            ).set(queue="Train_Transformer_Prepare_Data") # 指定数据准备队列

            # 定义模型训练任务签名 (调用 batch_train_following_strategy_transformer)
            train_task_signature = batch_train_following_strategy_transformer.s(
                stock_code=stock_code,
                params_file=params_file,
                model_dir=base_data_dir # 修改行：传递 base_data_dir 作为 model_dir
            ).set(queue="Train_Transformer_Model") # 指定模型训练队列 (新的队列名建议)

            # 创建任务链: prepare_task_signature | train_task_signature
            # Chain 的第一个任务通常不指定队列，让 Celery 自动处理。或者为 chain 指定一个默认队列。
            # 更好的做法是每个子任务指定自己的队列。
            task_chain = prepare_task_signature | train_task_signature

            # 分派任务链异步执行
            task_chain.apply_async()

            total_dispatched_chains += 1

        logger.info(f"任务结束: schedule_transformer_training_chain (调度器模式) - 共分派 {total_dispatched_chains} 个任务链") # 修改行：日志信息
        return {"status": "success", "dispatched_chains": total_dispatched_chains}

    except Exception as e:
        logger.error(f"执行 schedule_transformer_training_chain (调度器模式) 时出错: {e}", exc_info=True) # 修改行：日志信息
        return {"status": "error", "message": str(e), "dispatched_chains": 0}

