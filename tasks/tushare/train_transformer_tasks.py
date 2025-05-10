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

# 任务：准备 Transformer 训练数据并保存
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.batch_prepare_transformer_data')
def batch_prepare_transformer_data(self, stock_code: str, params_file: str = "strategies/indicator_parameters.json", model_dir="models", base_bars: int = 10000):
    """
    为特定股票准备 Transformer 训练数据（特征、目标、Scaler、选中特征列表）并保存到文件。
    """
    logger.info(f"开始执行 {stock_code} 的 Transformer 数据准备任务...")
    indicator_service = IndicatorService()
    # 实例化策略，TrendFollowingStrategy 的 __init__ 会加载 params_file
    strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_dir)
    
    try:
        # 1. 获取原始数据 (包含基础指标)
        logger.info(f"[{stock_code}] 获取原始数据及基础指标...")
        data_df, indicator_configs = asyncio.run(indicator_service.prepare_strategy_dataframe(
            stock_code=stock_code,
            params_file=params_file, # IndicatorService 也可能需要参数文件来确定计算哪些指标
            base_needed_bars=base_bars
        ))
        # 确保 indicator_configs 被正确获取，因为 TrendFollowingStrategy.generate_signals 需要它
        if indicator_configs is None:
            logger.error(f"[{stock_code}] IndicatorService 未返回 indicator_configs，无法继续进行策略信号生成。")
            raise ValueError("indicator_configs 为 None，无法执行策略。")

    except Exception as prep_err:
        logger.error(f"[{stock_code}] 调用 prepare_strategy_dataframe 获取原始数据时出错: {prep_err}", exc_info=True)
        raise prep_err # 重新抛出，让 Celery 标记任务失败

    if data_df is None or data_df.empty:
        logger.warning(f"[{stock_code}] 未能获取原始数据 (prepare_strategy_dataframe 返回空)。")
        return {"status": "warning", "message": "未能获取原始数据"}

    logger.info(f"[{stock_code}] 原始数据获取完成，共 {len(data_df)} 条，包含 {len(data_df.columns)} 列。")

    # 2. 调用策略的 run 方法生成包含所有信号和中间结果的 DataFrame
    #    TrendFollowingStrategy.generate_signals (被 run 调用) 会计算规则信号并将其作为 'final_rule_signal' 列添加
    try:
        logger.info(f"[{stock_code}] 调用策略 run 方法生成信号及中间数据...")
        # 通过 strategy.run 调用，它会内部调用 strategy.generate_signals
        # 并传递 stock_code 和 indicator_configs
        # 期望返回一个包含所有计算列的 DataFrame
        data_with_all_signals = strategy.run(data=data_df, stock_code=stock_code, indicator_configs=indicator_configs)
        
        if data_with_all_signals is None or data_with_all_signals.empty:
            logger.error(f"[{stock_code}] 策略 run 方法未返回有效的 DataFrame。")
            raise RuntimeError("策略 run 方法执行失败或未返回数据。")
        
        # 检查目标列 'final_rule_signal' 是否存在，这是 Transformer 训练的关键目标
        if 'final_rule_signal' not in data_with_all_signals.columns:
            logger.error(f"[{stock_code}] 策略执行后，返回的 DataFrame 中未找到 'final_rule_signal' 列。Transformer 训练目标缺失。")
            raise RuntimeError("策略未生成 'final_rule_signal' 列。")
            
        logger.info(f"[{stock_code}] 策略信号及中间数据生成完成。DataFrame 包含 'final_rule_signal' 列。")
    
    except Exception as signal_err:
        logger.error(f"[{stock_code}] 调用策略 run 方法或后续检查时出错: {signal_err}", exc_info=True)
        raise signal_err # 重新抛出

    # 使用包含所有信号（特别是 'final_rule_signal'）的 DataFrame 进行后续数据准备
    data_for_prep = data_with_all_signals

    # 3. 准备 Transformer 训练数据 (调用 deep_learning_utils 中的函数)
    try:
        logger.info(f"[{stock_code}] 开始准备 Transformer 数据...")
        tf_params = strategy.tf_params # 从策略实例获取已加载的 tf_params
        if not tf_params:
            logger.warning(f"[{stock_code}] 策略的 'trend_following_params' (strategy.tf_params) 为空，数据准备将使用大量默认配置。")
        
        data_prep_config = tf_params.get('transformer_data_prep_config', {})

        # prepare_data_for_transformer 需要的 data 是包含所有潜在特征和目标列的 DataFrame
        # required_columns 是一个“建议”列表，prepare_data_for_transformer 内部可能会做进一步筛选
        # target_column 指定了目标变量的列名
        features_scaled_train, targets_scaled_train, \
        features_scaled_val, targets_scaled_val, \
        features_scaled_test, targets_scaled_test, \
        feature_scaler, target_scaler, \
        selected_feature_names = prepare_data_for_transformer(
            data=data_for_prep, # 传入包含所有计算列的 DataFrame
            required_columns=strategy.get_required_columns(), # 策略声明它需要哪些原始列被计算出来
            target_column=tf_params.get('transformer_target_column', 'final_rule_signal'),
            scaler_type=data_prep_config.get('scaler_type', 'standard'),
            train_split=data_prep_config.get('train_split', 0.7),
            val_split=data_prep_config.get('val_split', 0.15),
            apply_variance_threshold=data_prep_config.get('apply_variance_threshold', False),
            variance_threshold_value=data_prep_config.get('variance_threshold_value', 0.01),
            use_pca=data_prep_config.get('use_pca', False),
            pca_n_components=data_prep_config.get('pca_n_components', 0.99),
            pca_solver=data_prep_config.get('pca_solver', 'auto'),
            use_feature_selection=data_prep_config.get('use_feature_selection', True),
            feature_selector_model_type=data_prep_config.get('feature_selector_model_type', 'rf'),
            fs_model_n_estimators=data_prep_config.get('fs_model_n_estimators', 100),
            fs_model_max_depth=data_prep_config.get('fs_model_max_depth', None),
            fs_max_features=data_prep_config.get('fs_max_features', 50),
            fs_selection_threshold=data_prep_config.get('fs_selection_threshold', 'median'),
            target_scaler_type=data_prep_config.get('target_scaler_type', 'minmax')
        )

        if features_scaled_train.shape[0] == 0 or targets_scaled_train.shape[0] == 0 or \
           feature_scaler is None or target_scaler is None or not selected_feature_names:
             logger.error(f"[{stock_code}] Transformer 数据准备失败，训练集为空或 Scaler/特征列表未成功生成。")
             raise ValueError("Transformer 数据准备失败，训练集为空或 Scaler/特征列表未成功生成。")

        logger.info(f"[{stock_code}] Transformer 数据准备完成。训练集 shape: {features_scaled_train.shape}, 验证集 shape: {features_scaled_val.shape}, 测试集 shape: {features_scaled_test.shape}, 最终特征数: {len(selected_feature_names)}")
    
    except Exception as data_prep_err:
        logger.error(f"[{stock_code}] 准备 Transformer 数据时出错: {data_prep_err}", exc_info=True)
        raise data_prep_err

    # 4. 保存准备好的数据、Scaler 和选中特征列表
    try:
        strategy.save_prepared_data( # save_prepared_data 是 TrendFollowingStrategy 的方法
            stock_code,
            features_scaled_train, targets_scaled_train,
            features_scaled_val, targets_scaled_val,
            features_scaled_test, targets_scaled_test,
            feature_scaler, target_scaler,
            selected_feature_names
        )
        logger.info(f"[{stock_code}] 准备好的 Transformer 数据和 Scaler 已成功保存。")
        return {"status": "success", "stock_code": stock_code, "train_samples": features_scaled_train.shape[0], "final_features": len(selected_feature_names)}

    except Exception as save_err:
        logger.error(f"[{stock_code}] 保存准备好的 Transformer 数据或 Scaler 时出错: {save_err}", exc_info=True)
        raise save_err

# 调度器任务：仅调度数据准备任务 
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

            prepare_task_signature = batch_prepare_transformer_data.s(
                stock_code=stock_code,
                params_file=params_file,
                model_dir=model_dir,
                base_bars=base_bars_to_request
            ).set(queue="Train_Transformer_Prepare_Data")

            prepare_task_signature.apply_async()
            total_dispatched_tasks += 1

        logger.info(f"任务结束: schedule_transformer_data_preparation (调度器模式) - 共分派 {total_dispatched_tasks} 个数据准备任务")
        return {"status": "success", "dispatched_tasks": total_dispatched_tasks}

    except Exception as e:
        logger.error(f"执行 schedule_transformer_data_preparation (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_tasks": total_dispatched_tasks}


# 任务：训练 Transformer 模型 (从已准备数据加载)
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.batch_train_following_strategy_transformer')
def batch_train_following_strategy_transformer(self, stock_code: str, params_file: str = "strategies/indicator_parameters.json", model_dir="models"):
    logger.info(f"开始执行 {stock_code} 的 Transformer 模型训练任务...")
    # 实例化策略，传递参数文件和模型目录
    strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_dir)
    try:
        strategy.train_transformer_model_from_prepared_data(stock_code)
        if strategy.transformer_model and strategy.feature_scaler and \
           strategy.target_scaler and strategy.selected_feature_names_for_transformer:
            logger.info(f"[{stock_code}] Transformer 模型训练任务完成。")
            return {"status": "success", "stock_code": stock_code}
        else:
            logger.error(f"[{stock_code}] Transformer 模型训练任务失败，模型或相关组件未成功加载/训练。")
            raise RuntimeError("Transformer 模型训练任务失败，模型或相关组件未成功加载/训练。")
    except Exception as e:
        logger.error(f"执行股票 {stock_code} 的 Transformer 模型训练任务时发生意外错误: {e}", exc_info=True)
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
