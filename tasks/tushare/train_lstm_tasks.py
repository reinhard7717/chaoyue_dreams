# tasks/tushare/train_transformer_tasks.py (文件名建议修改以反映内容)
import os
import logging
# 假设 StockBasicInfoDao 存在且可用
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
import pandas as pd
import asyncio
# 假设 celery 实例存在且可用
from chaoyue_dreams.celery import app as celery_app
# 假设 IndicatorService 存在且可用
from services.indicator_services import IndicatorService
# 导入修改后的 TrendFollowingStrategy 类
from strategies.trend_following_strategy import TrendFollowingStrategy
# 导入 prepare_data_for_transformer 函数
from strategies.utils.deep_learning_utils import prepare_data_for_transformer

logger = logging.getLogger("tasks")

# 修改任务：准备 Transformer 训练数据并保存
# 任务名和函数名都反映改为 Transformer
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.batch_prepare_transformer_data')
def batch_prepare_transformer_data(self, stock_code: str, params_file: str = "strategies/indicator_parameters.json", model_dir="models", base_bars: int = 10000):
    """
    为特定股票准备 Transformer 训练数据（特征、目标、Scaler、选中特征列表）并保存到文件。
    这个任务只负责数据准备和保存，不进行模型训练。
    :param stock_code: 股票代码
    :param params_file: 策略参数文件
    :param model_dir: 模型和数据保存目录 (用于确定保存路径)
    :param base_bars: 需要的基础 K 线数据量
    """
    logger.info(f"开始执行 {stock_code} 的 Transformer 数据准备任务...")
    indicator_service = IndicatorService()
    # 实例化策略，传入 model_dir 以便策略设置正确的保存路径
    strategy = TrendFollowingStrategy(params_file, base_data_dir=model_dir)
    try:
        # 1. 获取原始数据 (调用 IndicatorService)
        logger.info(f"[{stock_code}] 获取原始数据...")
        # prepare_strategy_dataframe 应该获取并计算所有基础指标和衍生特征，这些是 prepare_data_for_transformer 的输入
        data_df, indicator_configs = asyncio.run(indicator_service.prepare_strategy_dataframe(
            stock_code=stock_code,
            params_file=params_file,
            base_needed_bars=base_bars
            # 假设 IndicatorService 内部会调用 get_required_columns 来确定需要哪些列并计算
        ))
        # TODO: 确认 IndicatorService.prepare_strategy_dataframe 返回值是否包含 indicator_configs
        # 如果不包含，需要修改 IndicatorService 或在外部重新读取参数文件来获取 indicator_configs


    except Exception as prep_err:
        logger.error(f"[{stock_code}] 调用 prepare_strategy_dataframe 获取原始数据时出错: {prep_err}", exc_info=True)
        # 任务失败，抛出异常让 Celery 标记为失败
        raise prep_err

    if data_df is None or data_df.empty:
        logger.warning(f"[{stock_code}] 未能获取原始数据 (prepare_strategy_dataframe 返回空)。")
        # 任务成功完成，但没有数据可准备
        return {"status": "warning", "message": "未能获取原始数据"}

    logger.info(f"[{stock_code}] 原始数据获取完成，共 {len(data_df)} 条，包含 {len(data_df.columns)} 列。")
    # logger.debug(f"[{stock_code}] 原始数据列名 (部分): {data_df.columns.tolist()[:20]}...") # 调试用

    # 2. 调用策略方法生成规则信号 (作为 Transformer 的训练目标)
    # generate_signals 会计算出 final_rule_signal 并添加到 data_df 中
    try:
        logger.info(f"[{stock_code}] 调用策略生成规则信号...")
        # generate_signals 方法需要 data_df, stock_code 和 indicator_configs
        # 它会返回包含规则信号 (final_rule_signal) 的 DataFrame
        # 确保 IndicatorService 返回的 data_df 包含 get_required_columns 请求的所有列
        data_with_rule_signal = strategy.generate_signals(data_df, stock_code, indicator_configs) # 传入 indicator_configs
        if data_with_rule_signal is None or data_with_rule_signal.empty or 'final_rule_signal' not in data_with_rule_signal.columns:
            logger.error(f"[{stock_code}] 策略生成规则信号失败或未生成 'final_rule_signal' 列。")
            # 任务失败
            raise RuntimeError("策略生成规则信号失败或未生成 'final_rule_signal' 列。")
        logger.info(f"[{stock_code}] 规则信号生成完成，DataFrame 现在包含 'final_rule_signal' 列。")
    except Exception as signal_err:
        logger.error(f"[{stock_code}] 调用策略生成规则信号时出错: {signal_err}", exc_info=True)
        # 任务失败
        raise signal_err
    # 使用包含规则信号的 DataFrame 作为数据准备的输入
    data_for_prep = data_with_rule_signal

    # 3. 准备 Transformer 训练数据 (调用 deep_learning_utils 中的函数)
    try:
        logger.info(f"[{stock_code}] 开始准备 Transformer 数据...")
        # 从策略参数中获取数据准备相关的配置
        tf_params = strategy.params.get('trend_following_params', {})
        data_prep_config = tf_params.get('transformer_data_prep_config', {}) # 使用正确的参数名

        # prepare_data_for_transformer 需要原始数据 (包含所有计算好的特征和目标列) 和策略定义的 required_columns 列表
        # required_columns 列表决定了初始筛选哪些列进入特征工程流程
        # prepare_data_for_transformer 内部会进行 NaN 处理、特征工程 (方差过滤, PCA, 特征选择)
        features_scaled_train, targets_scaled_train, \
        features_scaled_val, targets_scaled_val, \
        features_scaled_test, targets_scaled_test, \
        feature_scaler, target_scaler, \
        selected_feature_names = prepare_data_for_transformer( # prepare_data_for_transformer 会返回 selected_feature_names
            data=data_for_prep, # 使用包含规则信号的 DataFrame
            required_columns=strategy.get_required_columns(), # 使用策略定义的原始所需列列表
            target_column=tf_params.get('transformer_target_column', 'final_rule_signal'), # 使用 Transformer 的目标列配置
            scaler_type=data_prep_config.get('scaler_type', 'standard'),
            train_split=data_prep_config.get('train_split', 0.7),
            val_split=data_prep_config.get('val_split', 0.15),
            apply_variance_threshold=data_prep_config.get('apply_variance_threshold', False),
            variance_threshold_value=data_prep_config.get('variance_threshold_value', 0.01),
            use_pca=data_prep_config.get('use_pca', False),
            pca_n_components=data_prep_config.get('pca_n_components', 0.99),
            pca_solver=data_prep_config.get('pca_solver', 'auto'),
            use_feature_selection=data_prep_config.get('use_feature_selection', True), # 默认启用特征选择
            feature_selector_model_type=data_prep_config.get('feature_selector_model_type', 'rf'),
            fs_model_n_estimators=data_prep_config.get('fs_model_n_estimators', 100), # 特征选择模型参数
            fs_model_max_depth=data_prep_config.get('fs_model_max_depth', None),
            fs_max_features=data_prep_config.get('fs_max_features', 50), # 最大特征数参数
            fs_selection_threshold=data_prep_config.get('fs_selection_threshold', 'median'),
            target_scaler_type=data_prep_config.get('target_scaler_type', 'minmax')
        )

        # 检查数据是否有效
        if features_scaled_train.shape[0] == 0 or targets_scaled_train.shape[0] == 0 or feature_scaler is None or target_scaler is None or not selected_feature_names:
             logger.error(f"[{stock_code}] Transformer 数据准备失败，训练集为空或 Scaler/特征列表未成功。")
             # 任务失败
             raise ValueError("Transformer 数据准备失败，训练集为空或 Scaler/特征列表未成功。")

        logger.info(f"[{stock_code}] Transformer 数据准备完成。训练集 shape: {features_scaled_train.shape}, 验证集 shape: {features_scaled_val.shape}, 测试集 shape: {features_scaled_test.shape}, 最终特征数: {len(selected_feature_names)}")
    except Exception as data_prep_err:
        logger.error(f"[{stock_code}] 准备 Transformer 数据时出错: {data_prep_err}", exc_info=True)
        # 任务失败，重新抛出异常
        raise data_prep_err

    # 4. 保存准备好的数据、Scaler 和选中特征列表 (调用策略类的方法)
    try:
        # set_model_paths 在 save_prepared_data 内部调用
        strategy.save_prepared_data(
            stock_code,
            features_scaled_train, targets_scaled_train,
            features_scaled_val, targets_scaled_val,
            features_scaled_test, targets_scaled_test,
            feature_scaler, target_scaler,
            selected_feature_names # 保存选中特征列表
        )
        logger.info(f"[{stock_code}] 准备好的 Transformer 数据和 Scaler 已成功保存。")
        return {"status": "success", "stock_code": stock_code, "train_samples": features_scaled_train.shape[0], "final_features": len(selected_feature_names)}

    except Exception as save_err:
        logger.error(f"[{stock_code}] 保存准备好的 Transformer 数据或 Scaler 时出错: {save_err}", exc_info=True)
        # 任务失败，重新抛出异常
        raise save_err

# 调度器任务：仅调度数据准备任务 (修改为 Transformer 数据准备)
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_transformer_data_preparation')
def schedule_transformer_data_preparation(self, params_file: str = "strategies/indicator_parameters.json", model_dir="models", base_bars_to_request: int = 10000):
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 为每个股票创建并分派一个 Transformer 数据准备任务到指定队列。
    这个任务由 Celery Beat 调度，用于触发数据准备的多进程处理。
    :param params_file: 策略参数文件路径
    :param model_dir: 模型和数据保存目录
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: schedule_transformer_data_preparation (调度器模式) - 获取股票列表并分派数据准备任务")
    total_dispatched_tasks = 0
    try:
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())

        if not all_stocks:
            logger.warning("未获取到股票列表，跳过数据准备任务分派。")
            return {"status": "warning", "message": "未获取到股票列表", "dispatched_tasks": 0}

        for stock in all_stocks:
            stock_code = stock.stock_code
            logger.info(f"分派 {stock_code} 的 Transformer 数据准备任务到 'Train_Transformer_Prepare_Data' 队列...")

            # 定义数据准备任务签名 (调用修改后的 prepare task)
            prepare_task_signature = batch_prepare_transformer_data.s(
                stock_code=stock_code,
                params_file=params_file,
                model_dir=model_dir,
                base_bars=base_bars_to_request
            ).set(queue="Train_Transformer_Prepare_Data") # 将任务分配到新的队列名

            # 分派单个任务异步执行
            prepare_task_signature.apply_async()

            total_dispatched_tasks += 1

        logger.info(f"任务结束: schedule_transformer_data_preparation (调度器模式) - 共分派 {total_dispatched_tasks} 个数据准备任务")
        return {"status": "success", "dispatched_tasks": total_dispatched_tasks}

    except Exception as e:
        logger.error(f"执行 schedule_transformer_data_preparation (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_tasks": total_dispatched_tasks}


# 修改任务：训练 Transformer 模型 (从已准备数据加载)
# 任务名和函数名都反映改为 Transformer
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.batch_train_following_strategy_transformer')
def batch_train_following_strategy_transformer(self, stock_code: str, params_file: str = "strategies/indicator_parameters.json", model_dir="models"):
    """
    为特定股票加载已准备好的 Transformer 数据，构建并训练 Transformer 模型。
    这个任务假设数据已经通过 batch_prepare_transformer_data 准备好并保存。
    :param stock_code: 股票代码
    :param params_file: 策略参数文件
    :param model_dir: 模型和数据保存目录
    """
    logger.info(f"开始执行 {stock_code} 的 Transformer 模型训练任务 (从已准备数据加载)...")
    strategy = TrendFollowingStrategy(params_file, base_data_dir=model_dir) # 传入 model_dir

    try:
        # 调用策略类中修改后的训练方法 (已重命名并更改内部逻辑为 Transformer)
        # train_transformer_model_from_prepared_data 内部会加载数据、构建模型、训练并保存最佳权重
        strategy.train_transformer_model_from_prepared_data(stock_code)

        # 检查策略实例是否成功加载/训练了模型
        if strategy.transformer_model is not None and strategy.feature_scaler is not None and strategy.target_scaler is not None and strategy.selected_feature_names_for_transformer:
             logger.info(f"[{stock_code}] Transformer 模型训练任务完成。")
             return {"status": "success", "stock_code": stock_code}
        else:
             # 如果有任何必需组件为 None，说明训练方法内部失败
             logger.error(f"[{stock_code}] Transformer 模型训练任务失败，请检查日志获取详细信息。")
             # 任务失败，明确抛出异常
             raise RuntimeError("Transformer 模型训练任务失败，请检查日志。")


    except Exception as e:
        logger.error(f"执行股票 {stock_code} 的 Transformer 模型训练任务时发生意外错误: {e}", exc_info=True)
        # 任务失败，重新抛出异常
        raise e


# 修改调度器任务，创建任务链 (prepare -> train)
# 任务名和函数名都反映改为 Transformer
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.train_transformer_trend_following_strategy_task')
def train_transformer_trend_following_strategy_task(self, base_bars_to_request: int = 10000): # 参数名一致性
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 为每个股票创建一个任务链：先准备数据，然后训练模型。
    3. 将任务链分派到指定队列。
    这个任务由 Celery Beat 调度。
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: train_transformer_trend_following_strategy_task (调度器模式) - 获取股票列表并分派任务链")
    try:
        total_dispatched_chains = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())

        if not all_stocks:
             logger.warning("未获取到股票列表，跳过任务链分派。")
             return {"status": "warning", "message": "未获取到股票列表", "dispatched_chains": 0}

        params_file = "strategies/indicator_parameters.json" # 从参数获取或固定
        model_dir = "models" # 从参数获取或固定

        for stock in all_stocks:
            stock_code = stock.stock_code
            logger.info(f"创建 {stock_code} 的 Transformer 数据准备和模型训练任务链...")

            # 定义数据准备任务签名 (调用 batch_prepare_transformer_data)
            prepare_task_signature = batch_prepare_transformer_data.s(
                stock_code=stock_code,
                params_file=params_file,
                model_dir=model_dir,
                base_bars=base_bars_to_request
            ).set(queue="Train_Transformer_Prepare_Data") # 指定数据准备队列

            # 定义模型训练任务签名 (调用 batch_train_following_strategy_transformer)
            train_task_signature = batch_train_following_strategy_transformer.s(
                stock_code=stock_code,
                params_file=params_file,
                model_dir=model_dir
            ).set(queue="Train_Transformer_Model") # 指定模型训练队列 (新的队列名建议)

            # 创建任务链: prepare_task_signature | train_task_signature
            # Chain 的第一个任务通常不指定队列，让 Celery 自动处理。或者为 chain 指定一个默认队列。
            # 更好的做法是每个子任务指定自己的队列。
            task_chain = prepare_task_signature | train_task_signature

            # 分派任务链异步执行
            task_chain.apply_async()

            total_dispatched_chains += 1

        logger.info(f"任务结束: train_transformer_trend_following_strategy_task (调度器模式) - 共分派 {total_dispatched_chains} 个任务链")
        return {"status": "success", "dispatched_chains": total_dispatched_chains}

    except Exception as e:
        logger.error(f"执行 train_transformer_trend_following_strategy_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_chains": 0}
