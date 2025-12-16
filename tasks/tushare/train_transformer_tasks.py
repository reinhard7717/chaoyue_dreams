# tasks/tushare/train_transformer_tasks.py
import logging
import asyncio
from asgiref.sync import async_to_sync
from pathlib import Path # 导入 asyncio
from django.conf import settings
import numpy as np
# 假设 StockBasicInfoDao 存在且可用
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
import pandas as pd
# 假设 celery 实例存在且可用
from chaoyue_dreams.celery import app as celery_app
# 导入修改后的 TrendFollowStrategy 类
from strategies.trend_following_strategy import TrendFollowStrategy
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
    logger.info(f"{task_id_str} [{stock_code}]：实例化 TrendFollowStrategy...")
    # 实例化策略，传递参数文件路径和模型目录
    strategy = TrendFollowStrategy(params_file=params_file, base_data_dir=model_dir)
    # 记录策略实例化完成信息
    logger.info(f"{task_id_str} [{stock_code}]：TrendFollowStrategy 实例化完毕，策略名: '{strategy.strategy_name}'。")
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
    data_df = None # 初始化为 None
    data_for_transformer_prep = None # 初始化为 None
    # 阶段 1: 使用 IndicatorService 准备数据 (获取原始数据和指标)
    try:
        # 记录数据准备开始信息
        logger.info(f"{task_id_str} [{stock_code}]：开始使用 IndicatorService 准备数据...")
        # 调用策略的 prepare_data 方法，该方法内部调用 IndicatorService
        # prepare_data 返回包含所有 OHLCV, 指标, 外部特征, 衍生特征的 DataFrame 和实际使用的指标配置列表
        data_df, actual_indicator_configs = async_to_sync(strategy.prepare_data)(stock_code=stock_code, base_needed_count=base_bars)
        # 检查返回的数据是否为空
        if data_df is None or data_df.empty:
            # 记录警告信息
            logger.warning(f"{task_id_str} [{stock_code}]：未能获取准备好的数据 (data_df为空)。")
            return {"status": "warning", "message": "未能获取准备好的数据"}
        # 记录数据准备完成信息
        logger.info(f"{task_id_str} [{stock_code}]：准备好的数据获取完成，{len(data_df)}行，{len(data_df.columns)}列。")
        # 记录获取到的指标配置数量
        logger.info(f"{task_id_str} [{stock_code}]：获取到 {len(actual_indicator_configs)} 个实际使用的指标配置。")
        # 打印数据类型和内存使用，帮助调试
        # print(f"{task_id_str} [{stock_code}]：data_df 原始数据类型:\n{data_df.dtypes}") # 打印原始数据类型
        print(f"{task_id_str} [{stock_code}]：data_df 原始内存使用 (MB): {data_df.memory_usage(deep=True).sum() / 1024**2:.2f}") # 打印原始内存使用
        # 优化内存：尝试将所有列转换为 float32，处理潜在的 object 类型列
        # logger.info(f"{task_id_str} [{stock_code}]：尝试将所有列转换为 float32...") # 添加日志
        original_dtypes = data_df.dtypes # 保存原始数据类型用于比较
        converted_cols_count = 0 # 计数转换成功的列
        for col in data_df.columns:
            try:
                # 使用 pd.to_numeric 尝试转换，errors='coerce' 将无法转换的值变为 NaN
                # 然后转换为 float32
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce').astype(np.float32) # 转换逻辑
                if data_df[col].dtype == np.float32 and original_dtypes[col] != np.float32: # 检查是否成功转换且原类型不是 float32
                     converted_cols_count += 1 # 计数
            except Exception as e:
                logger.warning(f"{task_id_str} [{stock_code}]：转换列 '{col}' 到 float32 时出错: {e}. 列将保持原样或包含 NaN。", exc_info=True) # 记录转换错误
        # logger.info(f"{task_id_str} [{stock_code}]：尝试转换完成，成功转换 {converted_cols_count} 列到 float32。") # 添加日志
        # print(f"{task_id_str} [{stock_code}]：转换后 data_df 数据类型:\n{data_df.dtypes}") # 打印转换后数据类型
        print(f"{task_id_str} [{stock_code}]：转换后 data_df 内存使用 (MB): {data_df.memory_usage(deep=True).sum() / 1024**2:.2f}") # 打印转换后内存使用
    except Exception as prep_err:
        # 记录数据准备错误信息并重新抛出异常
        logger.error(f"{task_id_str} [{stock_code}]：使用 IndicatorService 准备数据时出错: {prep_err}", exc_info=True)
        raise prep_err
    # 阶段 2: 从准备好的数据中提取 Transformer 训练所需的特征和目标
    # 这一步取代了之前调用 generate_signals 来获取数据
    try:
        logger.info(f"{task_id_str} [{stock_code}]：开始提取 Transformer 训练所需的特征和目标...")
        # 调用策略的新方法，该方法返回一个只包含 Transformer 特征和目标列的 DataFrame
        # 这个方法内部会根据策略配置确定哪些规则信号作为特征，哪个列作为目标
        # 注意：这里传入的 data_df 已经是尝试转换为 float32 后的 DataFrame
        data_for_transformer_prep = strategy._prepare_transformer_training_data_subset(
            data=data_df, # 传入包含所有原始数据和指标的 DataFrame
            stock_code=stock_code,
            indicator_configs=actual_indicator_configs # 可能需要这些配置来确定哪些列是指标特征
        )
        # 检查返回的数据是否为空
        if data_for_transformer_prep is None or data_for_transformer_prep.empty:
            logger.error(f"{task_id_str} [{stock_code}]：策略 _prepare_transformer_training_data_subset 方法返回空或None DataFrame。")
            # 根据 _prepare_transformer_training_data_subset 的逻辑，它在失败时会返回空 DataFrame
            # 此时应该终止任务
            return {"status": "error", "message": "提取 Transformer 训练数据子集失败或返回空数据。"}
        # 记录提取完成信息
        logger.info(f"{task_id_str} [{stock_code}]：Transformer 训练数据子集提取完毕，返回DataFrame {data_for_transformer_prep.shape}。")
        # 打印数据类型和内存使用，帮助调试
        # print(f"{task_id_str} [{stock_code}]：data_for_transformer_prep 数据类型:\n{data_for_transformer_prep.dtypes}")
        print(f"{task_id_str} [{stock_code}]：data_for_transformer_prep 内存使用 (MB): {data_for_transformer_prep.memory_usage(deep=True).sum() / 1024**2:.2f}")
        # 获取 Transformer 目标列名 (从策略中获取，确保与 _prepare_transformer_training_data_subset 生成的目标列名一致)
        transformer_target_column = strategy.transformer_target_column # 假设策略中定义了目标列名
        # 检查返回的 DataFrame 中是否存在 Transformer 目标列 (在 _prepare_transformer_training_data_subset 中已检查，这里再次确认)
        if transformer_target_column not in data_for_transformer_prep.columns:
            logger.error(f"{task_id_str} [{stock_code}]：提取的 DataFrame 中缺少 Transformer 目标列 '{transformer_target_column}'。")
            # 理论上 _prepare_transformer_training_data_subset 应该处理这种情况并返回空 DataFrame
            # 但为了健壮性，这里再次检查
            return {"status": "error", "message": f"提取的 Transformer 训练数据子集缺少目标列 '{transformer_target_column}'。"}
        # 确定用于 prepare_data_for_transformer 的特征列
        # 这些列应该是 data_for_transformer_prep 中除了目标列之外的所有列
        required_columns_for_transformer = [col for col in data_for_transformer_prep.columns if col != transformer_target_column]
        if not required_columns_for_transformer:
             logger.error(f"{task_id_str} [{stock_code}]：提取的 DataFrame 中除了目标列外没有其他特征列。")
             return {"status": "error", "message": "提取的 Transformer 训练数据子集不包含特征列。"}
        logger.info(f"{task_id_str} [{stock_code}]：用于 prepare_data_for_transformer 的特征列数: {len(required_columns_for_transformer)}")
    except Exception as subset_prep_err:
        logger.error(f"{task_id_str} [{stock_code}]：提取 Transformer 训练数据子集时出错: {subset_prep_err}", exc_info=True)
        raise subset_prep_err
    finally:
        # 显式删除不再需要的 data_df，释放内存
        # data_df 在这里已经被处理并传递给下一个阶段，可以安全删除
        if data_df is not None:
            del data_df
            print(f"{task_id_str} [{stock_code}]：已删除原始 data_df。")
    # 阶段 3: 使用 prepare_data_for_transformer 准备数据 (特征选择、标准化、分割)
    # 使用精简后的 data_for_transformer_prep 作为输入
    features_scaled_train = features_scaled_val = features_scaled_test = np.array([], dtype=np.float32) # 初始化为 float32 空数组
    targets_scaled_train = targets_scaled_val = targets_scaled_test = np.array([], dtype=np.float32) # 初始化为 float32 空数组
    feature_scaler = target_scaler = pca_model = scaler_for_pca = feature_selector_model = None # 初始化为 None
    selected_feature_names = [] # 初始化为空列表
    try:
        # 记录 Transformer 数据准备开始信息
        logger.info(f"{task_id_str} [{stock_code}]：开始调用 prepare_data_for_transformer...")
        # 获取 Transformer 参数和数据准备配置
        tf_params = strategy.tf_params
        data_prep_config = tf_params.get('transformer_data_prep_config', {})
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
        random_state_seed = data_prep_config.get('random_state_seed', 42)
        features_scaled_train, targets_scaled_train, \
        features_scaled_val, targets_scaled_val, \
        features_scaled_test, targets_scaled_test, \
        feature_scaler, target_scaler, \
        selected_feature_names, \
        pca_model, scaler_for_pca, feature_selector_model = prepare_data_for_transformer(
            data=data_for_transformer_prep, # 传入已经是 float32 的 DataFrame
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
            target_scaler_type=target_scaler_type,
            random_state_seed=random_state_seed
        )
        # 记录 prepare_data_for_transformer 调用成功信息
        logger.info(f"{task_id_str} [{stock_code}]：prepare_data_for_transformer 调用成功。")
        # 检查数据准备结果是否有效
        if features_scaled_train is None or features_scaled_train.shape[0] == 0 or \
           targets_scaled_train is None or targets_scaled_train.shape[0] == 0 or \
           feature_scaler is None or target_scaler is None or not selected_feature_names:
             # 记录错误并抛出值错误异常
             logger.error(f"{task_id_str} [{stock_code}]：Transformer 数据准备后，训练集为空或 Scaler/特征列表未成功生成。")
             # prepare_data_for_transformer 在失败时会返回空数组和 None
             return {"status": "error", "message": "Transformer 数据准备后，训练集为空或 Scaler/特征列表未成功生成。任务终止。"}
        # 记录数据准备完成信息
        logger.info(f"{task_id_str} [{stock_code}]：Transformer 数据准备完成。训练集 shape: {features_scaled_train.shape}, 最终特征数: {len(selected_feature_names)}")
        # 打印最终数组形状和类型，帮助调试
        print(f"{task_id_str} [{stock_code}]：features_scaled_train shape: {features_scaled_train.shape}, dtype: {features_scaled_train.dtype}")
        print(f"{task_id_str} [{stock_code}]：targets_scaled_train shape: {targets_scaled_train.shape}, dtype: {targets_scaled_train.dtype}")
        print(f"{task_id_str} [{stock_code}]：features_scaled_val shape: {features_scaled_val.shape}, dtype: {features_scaled_val.dtype}")
        print(f"{task_id_str} [{stock_code}]：targets_scaled_val shape: {targets_scaled_val.shape}, dtype: {targets_scaled_val.dtype}")
        print(f"{task_id_str} [{stock_code}]：features_scaled_test shape: {features_scaled_test.shape}, dtype: {features_scaled_test.dtype}")
        print(f"{task_id_str} [{stock_code}]：targets_scaled_test shape: {targets_scaled_test.shape}, dtype: {targets_scaled_test.dtype}")
    except Exception as data_prep_err:
        # 记录数据准备错误信息并重新抛出异常
        logger.error(f"{task_id_str} [{stock_code}]：准备 Transformer 数据时出错: {data_prep_err}", exc_info=True)
        raise data_prep_err
    finally:
        # 显式删除不再需要的 data_for_transformer_prep，释放内存
        if data_for_transformer_prep is not None:
            del data_for_transformer_prep
            print(f"{task_id_str} [{stock_code}]：已删除 data_for_transformer_prep。")
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
            selected_feature_names,
            pca_model,
            scaler_for_pca,
            feature_selector_model
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
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_transformer_data_processing') # 修改任务名称
def schedule_transformer_data_processing(self, params_file: str = None, base_data_dir: str = None, base_bars_to_request: int = 11200):
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 检查每个股票对应的根目录 (base_data_dir / 股票代码) 是否存在。
    3. 如果根目录存在，检查下级目录 prepared_data 是否存在。
    4. 如果 prepared_data 存在，检查其中是否存在 .npz 文件。
    5. 为目录不存在、prepared_data 不存在或 prepared_data 中没有 .npz 文件的股票创建并分派一个 Transformer 数据处理任务到指定队列。
    这个任务由 Celery Beat 调度，用于触发数据处理的多进程处理。
    :param params_file: 策略参数文件路径
    :param base_data_dir: 模型和数据保存的根目录 (包含各个股票子目录)
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: schedule_transformer_data_processing (调度器模式) - 检查股票数据目录状态并分派缺失任务") # 日志信息更新
    # 优先使用传入参数，否则使用 Django settings
    if params_file is None:
        # 检查 settings 是否可用以及属性是否存在
        if not hasattr(settings, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
             logger.error("错误：指标参数文件路径未提供且 Django settings 中未配置 INDICATOR_PARAMETERS_CONFIG_PATH。")
             return {"status": "error", "message": "指标参数文件路径未配置", "dispatched_tasks": 0}
        params_file = settings.INDICATOR_PARAMETERS_CONFIG_PATH
    if base_data_dir is None:
        # 检查 settings 是否可用以及属性是否存在
        if not hasattr(settings, 'STRATEGY_DATA_DIR'):
             logger.error("错误：基础数据目录未提供且 Django settings 中未配置 STRATEGY_DATA_DIR。")
             return {"status": "error", "message": "基础数据目录未配置", "dispatched_tasks": 0}
        base_data_dir = settings.STRATEGY_DATA_DIR
    # 确保 base_data_dir 是一个 Path 对象
    base_data_path = Path(base_data_dir)
    if not base_data_path.is_dir():
         logger.error(f"错误：配置的基础数据目录 '{base_data_dir}' 不存在或不是一个目录。")
         return {"status": "error", "message": f"基础数据目录 '{base_data_dir}' 无效", "dispatched_tasks": 0}
    total_dispatched_tasks = 0
    total_skipped_tasks = 0
    total_stocks_checked = 0
    try:
        # 1. 创建 CacheManager 实例
        cache_manager = CacheManager()
        # 2. 创建 DAO 实例并注入 cache_manager
        stock_basic_dao = StockBasicInfoDao(cache_manager)
        # 使用 async_to_sync 来执行异步方法
        all_stocks = async_to_sync(stock_basic_dao.get_stock_list)() #[::-1]
        if not all_stocks:
            logger.warning("未获取到股票列表，跳过数据处理任务分派。")
            return {"status": "warning", "message": "未获取到股票列表", "dispatched_tasks": 0}
        logger.info(f"成功获取 {len(all_stocks)} 个股票代码，开始检查股票数据目录状态...") # 日志信息更新
        for stock in all_stocks:
            total_stocks_checked += 1
            stock_code = stock.stock_code
            if not stock_code.endswith('.BJ'):
                # 构建当前股票数据根目录的预期路径 (例如: /data/strategy_models/000001)
                expected_stock_data_root = base_data_path / stock_code
                # 构建 prepared_data 子目录的预期路径
                prepared_data_path = expected_stock_data_root / "prepared_data" #构建 prepared_data 目录路径
                should_schedule = False #标志是否需要调度任务
                reason = "" #记录调度或跳过的原因
                # 检查股票数据根目录是否存在
                # if not expected_stock_data_root.is_dir():
                #     should_schedule = True
                #     reason = f"股票根目录 '{expected_stock_data_root}' 不存在" #记录原因
                #     logger.info(f"股票 {stock_code}: {reason}，标记为需要调度.") #日志说明原因
                # else:
                #     # 如果根目录存在，检查 prepared_data 子目录
                #     if not prepared_data_path.is_dir(): #检查 prepared_data 目录是否存在
                #         should_schedule = True
                #         reason = f"prepared_data 子目录 '{prepared_data_path}' 不存在" #记录原因
                #         logger.info(f"股票 {stock_code}: 根目录存在，但 {reason}，标记为需要调度.") #日志说明原因
                #     else:
                #         # 如果 prepared_data 子目录存在，检查是否有 .npz 文件
                #         # 使用 glob 查找所有 .npz 文件，并用 any() 判断是否存在至少一个
                #         has_npz_files = any(prepared_data_path.glob("*.joblib")) #检查是否存在 .joblib 文件
                #         if not has_npz_files: #如果不存在 .npz 文件
                #             should_schedule = True
                #             reason = f"prepared_data 子目录 '{prepared_data_path}' 中不存在 .npz 文件" #记录原因
                #             logger.info(f"股票 {stock_code}: 根目录和 prepared_data 子目录存在，但 {reason}，标记为需要调度.") #日志说明原因
                #         else:
                #             # 根目录、prepared_data 子目录都存在，且存在 .npz 文件，则跳过
                #             should_schedule = False #明确不需要调度
                #             reason = f"股票根目录和 prepared_data 子目录存在且包含 .npz 文件" #记录原因
                #             # logger.info(f"跳过 {stock_code} 的 Transformer 数据处理任务分派 ({reason}).") # 日志信息说明跳过原因
                #             total_skipped_tasks += 1
                should_schedule = True
                # 根据 should_schedule 标志决定是否分派任务
                if should_schedule: #根据标志决定是否调度
                    logger.info(f"分派 {stock_code} 的 Transformer 数据处理任务到 'Train_Transformer_Prepare_Data' 队列 (原因: {reason}).") # 日志说明调度原因
                    # 调用任务，确保使用原始的任务函数名
                    prepare_task_signature = process_stock_data_for_transformer_training.s( # 确保使用原始的任务函数名
                        stock_code=stock_code,
                        params_file=params_file,
                        model_dir=base_data_dir, # 注意：这里传递的是基础数据目录
                        base_bars=base_bars_to_request
                    ).set(queue="Train_Transformer_Prepare_Data")
                    prepare_task_signature.apply_async()
                    total_dispatched_tasks += 1
        logger.info(f"任务结束: schedule_transformer_data_processing (调度器模式) - 共检查 {total_stocks_checked} 个股票，分派 {total_dispatched_tasks} 个任务，跳过 {total_skipped_tasks} 个任务。") # 日志信息总结
        return {
            "status": "completed",
            "message": f"共检查 {total_stocks_checked} 个股票，分派 {total_dispatched_tasks} 个任务，跳过 {total_skipped_tasks} 个任务。",
            "dispatched_tasks": total_dispatched_tasks,
            "skipped_tasks": total_skipped_tasks,
            "total_stocks_checked": total_stocks_checked
        }
    except Exception as e:
        logger.error(f"执行 schedule_transformer_data_processing (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_tasks": total_dispatched_tasks, "skipped_tasks": total_skipped_tasks, "total_stocks_checked": total_stocks_checked}

# 任务：训练 Transformer 模型 (从已准备数据加载)
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.batch_train_following_strategy_transformer', queue="Train_Transformer_Model")
def batch_train_following_strategy_transformer(self, stock_code: str, params_file: str = "", model_dir=""):
    print(f"DEBUG: !!!!! Celery task ENTRY for {stock_code} !!!!!") # 增加一个醒目的print语句
    logger.info(f"开始执行 {stock_code} 的 Transformer 模型训练任务...")
    # 实例化策略，传递参数文件和模型目录
    params_file=settings.INDICATOR_PARAMETERS_CONFIG_PATH
    model_dir = settings.STRATEGY_DATA_DIR # 使用设置的模型目录
    strategy = TrendFollowStrategy(params_file=params_file, base_data_dir=model_dir)
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

# 调度器任务: 训练 Transformer 模型
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_transformer_training_chain')
def schedule_transformer_training_chain(self): # 参数名一致性
    """
    调度器任务：训练 Transformer 模型
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: schedule_transformer_training_chain (调度器模式) - 获取股票列表并分派任务")
    try:
        total_dispatched_chains = 0
        # 1. 创建 CacheManager 实例
        cache_manager = CacheManager()
        # 2. 创建 DAO 实例并注入 cache_manager
        stock_basic_dao = StockBasicInfoDao(cache_manager)
        all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
        if not all_stocks:
             logger.warning("未获取到股票列表，跳过任务链分派。")
             return {"status": "warning", "message": "未获取到股票列表", "dispatched_chains": 0}
        for stock in all_stocks:
            stock_code = stock.stock_code
            logger.info(f"创建 {stock_code} 的 Transformer 数据处理和模型训练任务链...") # 日志信息
            # 定义模型训练任务签名 (调用 batch_train_following_strategy_transformer)
            train_task_signature = batch_train_following_strategy_transformer.s(
                stock_code=stock_code
            ).set().apply_async() # 指定模型训练队列 (新的队列名建议)
            total_dispatched_chains += 1
        logger.info(f"任务结束: schedule_transformer_training_chain (调度器模式) - 共分派 {total_dispatched_chains} 个任务") # 日志信息
        return {"status": "success", "dispatched_chains": total_dispatched_chains}
    except Exception as e:
        logger.error(f"执行 schedule_transformer_training_chain (调度器模式) 时出错: {e}", exc_info=True) # 日志信息
        return {"status": "error", "message": str(e), "dispatched_chains": 0}

