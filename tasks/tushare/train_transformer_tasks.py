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
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.process_stock_data_for_transformer_training')
def process_stock_data_for_transformer_training(self, stock_code: str, params_file: str = None, model_dir: str = None, base_bars: int = 10000):
    # 构建任务ID字符串用于日志记录
    task_id_str = f"任务 {self.request.id if self.request else 'UnknownID'}"
    # 记录任务开始信息
    logger.info(f"{task_id_str}：开始为 {stock_code} 执行 Transformer 数据处理流程...")
    # 记录策略实例化信息
    logger.info(f"{task_id_str} [{stock_code}]：实例化 TrendFollowingStrategy...")
    # 实例化策略，传递参数文件路径和模型目录
    strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_dir)
    # 记录策略实例化完成信息
    logger.info(f"{task_id_str} [{stock_code}]：TrendFollowingStrategy 实例化完毕，策略名: '{strategy.strategy_name}'。")
    # 检查策略参数是否为空
    if not strategy.params:
        # 记录严重错误并终止任务
        logger.error(f"{task_id_str} [{stock_code}]：CRITICAL TASK HALT: strategy.params 为空。参数文件可能未加载或无效。任务无法继续。")
        return {"status": "error", "message": "策略参数 strategy.params 为空，任务终止。"}
    # 检查 Transformer 相关参数是否存在且非空
    if 'trend_following_params' not in strategy.params or not strategy.tf_params:
        # 记录严重错误并终止任务
        logger.error(f"{task_id_str} [{stock_code}]：CRITICAL TASK HALT: 'trend_following_params' 在 strategy.params 中缺失或 strategy.tf_params 为空。任务无法继续。")
        return {"status": "error", "message": "'trend_following_params' 缺失或为空，任务终止。"}
    # 记录参数检查通过信息
    logger.info(f"{task_id_str} [{stock_code}]：策略参数检查通过 (params 和 tf_params 非空)。")
    # 打印 Transformer 窗口大小关键参数 (注意：window_size 用于 TimeSeriesDataset，不直接用于 prepare_data_for_transformer)
    logger.info(f"{task_id_str} [{stock_code}]：确认 Transformer 窗口大小 (用于后续的TimeSeriesDataset): {strategy.transformer_window_size}")
    # 阶段 1: 使用 IndicatorService 准备数据
    try:
        # 记录数据准备开始信息
        logger.info(f"{task_id_str} [{stock_code}]：开始使用 IndicatorService 准备数据...")
        # 调用策略的 prepare_data 方法，该方法内部调用 IndicatorService
        # prepare_data 返回包含所有 OHLCV, 指标, 外部特征, 衍生特征的 DataFrame 和实际使用的指标配置列表
        data_df, actual_indicator_configs = asyncio.run(strategy.prepare_data(stock_code=stock_code, base_needed_count=base_bars))
        # 检查返回的数据是否为空
        if data_df is None or data_df.empty:
            # 记录警告信息
            logger.warning(f"{task_id_str} [{stock_code}]：未能获取准备好的数据 (data_df为空)。")
            return {"status": "warning", "message": "未能获取准备好的数据"}
        # 记录数据准备完成信息
        logger.info(f"{task_id_str} [{stock_code}]：准备好的数据获取完成，{len(data_df)}行，{len(data_df.columns)}列。")
        # 记录获取到的指标配置数量
        logger.info(f"{task_id_str} [{stock_code}]：获取到 {len(actual_indicator_configs)} 个实际使用的指标配置。")
    except Exception as prep_err:
        # 记录数据准备错误信息并重新抛出异常
        logger.error(f"{task_id_str} [{stock_code}]：使用 IndicatorService 准备数据时出错: {prep_err}", exc_info=True)
        raise prep_err
    # 阶段 2: 使用策略生成规则信号 (包括 Transformer 目标列)
    try:
        # 记录信号生成开始信息
        logger.info(f"{task_id_str} [{stock_code}]：调用策略 generate_signals 方法生成规则信号及中间数据...")
        # 调用策略的 generate_signals 方法，传入准备好的数据和实际使用的指标配置
        data_with_all_signals = strategy.generate_signals(data=data_df, stock_code=stock_code, indicator_configs=actual_indicator_configs)
        # 检查信号生成后的数据是否为空
        if data_with_all_signals is None or data_with_all_signals.empty:
            # 记录错误并抛出运行时异常
            logger.error(f"{task_id_str} [{stock_code}]：策略 generate_signals 方法返回空或None DataFrame。")
            raise RuntimeError("策略 generate_signals 方法执行失败或未返回数据。")
        # 记录信号生成完成信息
        logger.info(f"{task_id_str} [{stock_code}]：策略 generate_signals 方法执行完毕，返回DataFrame {data_with_all_signals.shape}。")
        # 获取 Transformer 目标列名
        transformer_target_column = strategy.transformer_target_column
        # 检查返回的 DataFrame 中是否存在 Transformer 目标列
        if transformer_target_column not in data_with_all_signals.columns:
            # 记录错误并抛出运行时异常
            logger.error(f"{task_id_str} [{stock_code}]：返回的 DataFrame 中缺少 Transformer 目标列 '{transformer_target_column}'。")
            raise RuntimeError(f"策略未生成 Transformer 目标列 '{transformer_target_column}'。")
        # 记录目标列存在信息
        logger.info(f"{task_id_str} [{stock_code}]：Transformer 目标列 '{transformer_target_column}' 存在。")
    except Exception as signal_err:
        # 记录信号生成或后续检查错误信息并重新抛出异常
        logger.error(f"{task_id_str} [{stock_code}]：策略 generate_signals 或后续检查时出错: {signal_err}", exc_info=True)
        raise signal_err
    # 阶段 3: 准备 Transformer 训练数据 (特征选择、标准化、分割)
    # 使用包含所有信号的 DataFrame 作为 prepare_data_for_transformer 的输入
    data_for_prep = data_with_all_signals
    try:
        # 记录 Transformer 数据准备开始信息
        logger.info(f"{task_id_str} [{stock_code}]：开始调用 prepare_data_for_transformer...")
        # 获取 Transformer 参数和数据准备配置
        tf_params = strategy.tf_params
        data_prep_config = tf_params.get('transformer_data_prep_config', {})
        # 从 strategy.params 中获取 required_columns，如果 Transformer 参数中有特定定义，则优先使用
        # 假设 strategy.params['indicator_params']['all_feature_columns'] 包含所有可用特征名
        # 或 strategy.params['external_feature_params']['feature_columns'] 等
        # 这里需要根据您的具体实现确定 'required_columns' 的来源
        # 示例：假设所有生成信号的列（除了目标列本身）都可以作为初始特征
        all_columns = data_for_prep.columns.tolist()
        required_columns_for_transformer = [col for col in all_columns if col != transformer_target_column]
        # 如果有更精确的特征列表，应使用那个列表
        # 例如，如果 actual_indicator_configs 包含了所有生成的特征名，也可以从中提取
        # 或者从 strategy.params 中读取一个预定义的特征列表
        logger.info(f"{task_id_str} [{stock_code}]：用于 prepare_data_for_transformer 的初始特征列数: {len(required_columns_for_transformer)}")
        # --- 修改开始 ---
        # 从 data_prep_config 中提取参数，并为 prepare_data_for_transformer 提供默认值（如果config中没有）
        # 这些默认值应该与 prepare_data_for_transformer 函数定义中的默认值一致或根据需求调整
        scaler_type = data_prep_config.get('scaler_type', 'minmax')
        train_split = data_prep_config.get('train_split', 0.7)
        val_split = data_prep_config.get('val_split', 0.15)
        apply_variance_threshold = data_prep_config.get('apply_variance_threshold', False)
        variance_threshold_value = data_prep_config.get('variance_threshold_value', 0.01)
        use_pca = data_prep_config.get('use_pca', False)
        pca_n_components = data_prep_config.get('pca_n_components', 0.99)
        pca_solver = data_prep_config.get('pca_solver', 'auto')
        use_feature_selection = data_prep_config.get('use_feature_selection', True)
        feature_selector_model_type = data_prep_config.get('feature_selector_model_type', 'rf')
        fs_model_n_estimators = data_prep_config.get('fs_model_n_estimators', 100)
        fs_model_max_depth = data_prep_config.get('fs_model_max_depth', None)
        fs_max_features = data_prep_config.get('fs_max_features', 50)
        fs_selection_threshold = data_prep_config.get('fs_selection_threshold', 'median')
        target_scaler_type = data_prep_config.get('target_scaler_type', 'minmax')
        features_scaled_train, targets_scaled_train, \
        features_scaled_val, targets_scaled_val, \
        features_scaled_test, targets_scaled_test, \
        feature_scaler, target_scaler, \
        selected_feature_names = prepare_data_for_transformer(
            data=data_for_prep,
            required_columns=required_columns_for_transformer,
            target_column=transformer_target_column,
            scaler_type=scaler_type,
            train_split=train_split,
            val_split=val_split,
            apply_variance_threshold=apply_variance_threshold,
            variance_threshold_value=variance_threshold_value,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            pca_solver=pca_solver,
            use_feature_selection=use_feature_selection,
            feature_selector_model_type=feature_selector_model_type,
            fs_model_n_estimators=fs_model_n_estimators,
            fs_model_max_depth=fs_model_max_depth,
            fs_max_features=fs_max_features,
            fs_selection_threshold=fs_selection_threshold,
            target_scaler_type=target_scaler_type
        )
        # 记录 prepare_data_for_transformer 调用成功信息
        logger.info(f"{task_id_str} [{stock_code}]：prepare_data_for_transformer 调用成功。")
        # 检查数据准备结果是否有效
        if features_scaled_train is None or features_scaled_train.shape[0] == 0 or \
           targets_scaled_train is None or targets_scaled_train.shape[0] == 0 or \
           feature_scaler is None or target_scaler is None or not selected_feature_names:
             # 记录错误并抛出值错误异常
             logger.error(f"{task_id_str} [{stock_code}]：Transformer 数据准备后，训练集为空或 Scaler/特征列表未成功生成。")
             raise ValueError("Transformer 数据准备后，训练集为空或 Scaler/特征列表未成功生成。")
        # 记录数据准备完成信息
        logger.info(f"{task_id_str} [{stock_code}]：Transformer 数据准备完成。训练集 shape: {features_scaled_train.shape}, 最终特征数: {len(selected_feature_names)}")
    except Exception as data_prep_err:
        # 记录数据准备错误信息并重新抛出异常
        logger.error(f"{task_id_str} [{stock_code}]：准备 Transformer 数据时出错: {data_prep_err}", exc_info=True)
        raise data_prep_err
    # 阶段 4: 保存准备好的数据和 Scaler
    try:
        # 记录保存数据开始信息
        logger.info(f"{task_id_str} [{stock_code}]：开始保存准备好的数据和 Scaler...")
        # 调用策略的 save_prepared_data 方法保存数据和 Scaler
        strategy.save_prepared_data(
            stock_code,
            features_scaled_train, targets_scaled_train,
            features_scaled_val, targets_scaled_val,
            features_scaled_test, targets_scaled_test,
            feature_scaler, target_scaler,
            selected_feature_names
        )
        # 记录保存成功信息
        logger.info(f"{task_id_str} [{stock_code}]：准备好的 Transformer 数据和 Scaler 已成功保存。")
        # 返回成功状态和相关信息
        return {"status": "success", "stock_code": stock_code, "train_samples": features_scaled_train.shape[0], "final_features": len(selected_feature_names)}
    except Exception as save_err:
        # 记录保存错误信息并重新抛出异常
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

