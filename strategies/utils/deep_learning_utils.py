# apps/strategies/utils/deep_learning_utils.py

# 导入必要的库
import os
import pandas as pd
import numpy as np
import tensorflow as tf
# 导入数据预处理和特征工程相关的模块
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split # 时间序列数据按时间顺序分割，无需此模块
from sklearn.feature_selection import SelectFromModel # 用于基于模型的特征选择
from sklearn.ensemble import RandomForestRegressor # 随机森林回归器，用于特征重要性评估
try:
    import xgboost as xgb # 尝试导入 xgboost
    XGB_AVAILABLE = True # 标记 XGBoost 是否可用
except ImportError:
    xgb = None # 如果未安装，则设为 None
    XGB_AVAILABLE = False
    # logger.warning("XGBoost 未安装，如果选择使用 XGBoost 进行特征选择将会失败。请运行 'pip install xgboost'") # 日志记录器可能尚未初始化，暂时注释
from sklearn.decomposition import PCA # 主成分分析
from sklearn.feature_selection import VarianceThreshold # 方差阈值特征选择
# 导入Keras模型和层
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
# 导入优化器
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
# 导入回调函数
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# 导入正则化项
from tensorflow.keras.regularizers import l2
# 导入日志、时间、绘图等辅助库
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
# 导入类型提示
from typing import Any, Tuple, List, Dict, Optional, Union, Callable
from functools import wraps
import joblib # 用于加载/保存 scaler，暂时未在代码中使用，但保留导入

# 设置日志记录器
logger = logging.getLogger("strategy_deep_learning_utils")

# 装饰器：记录执行时间
def log_execution_time(func):
    """记录函数执行时间的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # 直接使用logger打印，而不是print
        logger.info(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

# 装饰器：统一异常处理
def handle_exceptions(func):
    """处理函数异常的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行出错: {e}", exc_info=True)
            # 重新抛出异常，以便调用者可以捕获并进一步处理
            raise
    return wrapper

@log_execution_time
@handle_exceptions
def prepare_data_for_lstm(
    data: pd.DataFrame,
    required_columns: List[str], # 数据中应包含的原始特征列列表（用于初始筛选）
    target_column: str = 'final_signal',
    window_size: int = 60,
    scaler_type: str = 'minmax', # 特征缩放器类型 ('minmax' 或 'standard')
    train_split: float = 0.7,
    val_split: float = 0.15,
    apply_variance_threshold: bool = False, # 是否应用方差阈值进行特征选择 (在其他特征工程前)
    variance_threshold_value: float = 0.01, # 方差阈值
    # --- 特征工程/选择参数 ---
    use_pca: bool = False, # 是否应用PCA进行降维 (如果为 True，则忽略基于模型的特征选择)
    n_components: Union[int, float] = 0.99, # PCA保留的主成分数量或解释方差比例
    use_feature_selection: bool = True, # 是否启用基于模型的特征选择 (默认启用)
    feature_selector_model: str = 'rf', # 选择器模型: 'rf' (随机森林) 或 'xgb' (XGBoost)
    max_features_fs: Optional[int] = 50, # 选择最重要的特征数量 (例如选择前 50 个)；设为 None 则使用阈值
    feature_selection_threshold: str = 'median', # 如果 max_features_fs 为 None, 则使用此阈值 ('median', 'mean', 或浮点数)
    target_scaler_type: str = 'minmax' # 目标变量缩放器类型
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler, None], Union[MinMaxScaler, StandardScaler, None]]:
    """
    准备用于LSTM训练的时间序列数据，包括特征处理、缩放、窗口化和按时间顺序数据集分割。

    处理流程 (优化后):
    1. 检查目标列。
    2. 筛选初始特征列 (基于 required_columns 并排除目标列)。
    3. 处理 NaN 值 (前向填充后向填充，剩余 NaN 填充 0)。
    4. 应用方差阈值过滤 (可选，作为快速预过滤)。
    5. **按时间顺序将数据分割为训练集、验证集和测试集 (在模型特征选择和缩放前)。** 这是防止未来数据泄露的关键。
    6. **在训练集上拟合特征工程转换器 (PCA 或 基于模型的选择)。**
    7. **使用拟合好的转换器转换所有数据集 (训练集、验证集、测试集)。**
    8. **在处理后的训练集特征和训练集目标上拟合最终的特征缩放器和目标缩放器。**
    9. **使用拟合好的缩放器转换所有数据集 (训练集、验证集、测试集)。**
    10. 构建时间序列窗口 (在缩放后的每个数据集上单独进行)。

    将特征工程和缩放器的拟合过程严格限制在训练集上，避免未来数据泄露。

    Args:
        data (pd.DataFrame): 包含所有特征和目标列的原始DataFrame。
        required_columns (List[str]): 数据中应包含的原始特征列列表（用于初始筛选）。
        target_column (str): 目标变量列名。
        window_size (int): LSTM输入的时间步长。
        scaler_type (str): 特征缩放器类型 ('minmax' 或 'standard')。
        train_split (float): 训练集比例。
        val_split (float): 验证集比例。
        apply_variance_threshold (bool): 是否应用方差阈值进行特征选择 (在其他特征工程前)。可作为基于模型选择的预过滤。
        variance_threshold_value (float): 方差阈值。
        use_pca (bool): 是否应用PCA进行降维 (如果为 True，则忽略基于模型的特征选择)。
        n_components (Union[int, float]): PCA保留的主成分数量或解释方差比例。
        use_feature_selection (bool): 是否启用基于模型的特征选择。
        feature_selector_model (str): 特征选择模型 ('rf' (随机森林) 或 'xgb' (XGBoost))。日志显示 RF 拟合耗时较长，尝试 'xgb' 可能加速。
        max_features_fs (Optional[int]): 要选择的最大特征数量。如果为 None，则使用 threshold。
        feature_selection_threshold (str): 特征重要性阈值 (如果 max_features_fs is None)。
        target_scaler_type (str): 目标变量缩放器类型 ('minmax' 或 'standard')。

    Returns:
        Tuple containing:
        X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray): 分割、处理和窗口化后的数据集。
        feature_scaler (Union[MinMaxScaler, StandardScaler, None]): 用于最终特征的缩放器 (在训练集处理后的特征集上拟合)。如果训练集为空则为 None。
        target_scaler (Union[MinMaxScaler, StandardScaler, None]): 用于目标变量的缩放器 (在训练集原始目标集上拟合)。如果训练集为空则为 None。
    """
    logger.info(f"开始准备 LSTM 数据...")
    logger.info(f"参数: window_size={window_size}, scaler_type='{scaler_type}', train_split={train_split}, val_split={val_split}, apply_variance_threshold={apply_variance_threshold}, variance_threshold_value={variance_threshold_value}, use_pca={use_pca}, n_components={n_components}, use_feature_selection={use_feature_selection}, feature_selector_model='{feature_selector_model}', max_features_fs={max_features_fs}, feature_selection_threshold='{feature_selection_threshold}', target_scaler_type='{target_scaler_type}'")

    # --- 1. 检查目标列 ---
    if target_column not in data.columns:
        logger.error(f"目标列 '{target_column}' 不存在于输入数据中。")
        raise ValueError(f"目标列 '{target_column}' 不存在。")

    # --- 2. 初始特征列选择 (基于 required_columns 并排除目标列) ---
    # 确保只选择 data 中实际存在的列
    initial_feature_columns = [col for col in required_columns if col in data.columns and col != target_column]
    if not initial_feature_columns:
         logger.error("根据 required_columns 筛选后，没有可用的特征列。")
         raise ValueError("没有可用的特征列。")

    # --- 3. 处理 NaN 值 (在特征选择/PCA/分割之前填充整个数据集) ---
    # 使用前向填充然后后向填充，尽量保留数据
    data_filled = data.ffill().bfill()
    # 检查填充后是否仍有 NaN (如果整个列都是 NaN)
    nan_cols_after_ffill = data_filled[initial_feature_columns].isnull().sum()
    nan_cols_after_ffill = nan_cols_after_ffill[nan_cols_after_ffill > 0]
    if not nan_cols_after_ffill.empty:
        logger.warning(f"填充后以下特征列仍包含 NaN 值 (可能整列为空)，将尝试填充为 0: {nan_cols_after_ffill.index.tolist()}")
        # 对于仍然是 NaN 的列，填充为 0
        for col in nan_cols_after_ffill.index:
            data_filled[col].fillna(0, inplace=True)
            # 再次检查是否仍然有 NaN (理论上fillna(0)应该解决了，除非数据类型问题)
            if data_filled[col].isnull().any():
                 logger.warning(f"列 '{col}' 填充 0 后仍有 NaN，请检查数据类型或填充逻辑。")


    # 提取处理后的特征和目标变量 (在分割前是整个数据集)
    features_processed_flat = data_filled.loc[:, initial_feature_columns].values
    targets_original_flat = data_filled.loc[:, target_column].values # 使用填充后的目标值
    current_feature_columns = initial_feature_columns[:] # 创建副本，用于跟踪列名

    logger.info(f"初始特征维度 (处理NaN后): {features_processed_flat.shape[1]}")

    # --- 4. (可选) 方差阈值过滤 ---
    # 方差过滤可以在分割前应用于整个数据集，因为它只是基于单个特征的全局属性
    # 可以作为基于模型选择的预过滤，减少后续模型拟合的特征数量，提升速度。
    if apply_variance_threshold:
        # 检查是否有足够的样本和特征进行方差计算
        if features_processed_flat.shape[0] > 1 and features_processed_flat.shape[1] > 0:
            # 检查是否存在方差不为零的特征
            variances = np.var(features_processed_flat, axis=0)
            if np.any(variances > 1e-9): # 使用一个小阈值检查是否有非零方差
                try:
                    selector_var = VarianceThreshold(threshold=variance_threshold_value)
                    # 注意：fit_transform 会移除低方差特征
                    features_after_var = selector_var.fit_transform(features_processed_flat)
                    # 获取被保留的特征索引
                    selected_indices_var = selector_var.get_support(indices=True)

                    if len(selected_indices_var) == 0:
                         logger.error(f"方差阈值 ({variance_threshold_value}) 选择后没有剩余特征。请检查阈值或数据。")
                         # 此时不应raise error，而是警告并保留原始特征，除非是强制要求有特征
                         logger.warning("方差阈值选择后没有剩余特征，将使用之前的特征。")
                         # features_processed_flat 和 current_feature_columns 保持原样
                    else:
                        features_processed_flat = features_after_var # 更新特征矩阵
                        current_feature_columns = [current_feature_columns[i] for i in selected_indices_var]
                        logger.info(f"方差阈值 ({variance_threshold_value}) 选择后维度: {features_processed_flat.shape[1]}")

                except Exception as e:
                     logger.error(f"应用方差阈值时出错: {e}", exc_info=True)
                     logger.warning("方差阈值选择失败，将使用之前的特征。")
            else:
                 logger.warning("所有特征方差接近零，跳过方差阈值选择。")
        else:
             logger.warning("特征维度或样本不足，跳过方差阈值选择。")

    # 检查处理后的特征是否为空
    num_initial_features = features_processed_flat.shape[1]
    if num_initial_features == 0:
         logger.error("经过 NaN 处理和方差阈值过滤后，特征维度为零。无法继续。")
         # 返回空的数组和 None Scaler (因为没有数据 fit)
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None

    logger.info(f"方差过滤/初始处理后，特征维度: {num_initial_features}")


    # --- 5. 按时间顺序分割数据集 (在特征工程和缩放前) ---
    n_samples_flat = features_processed_flat.shape[0]
    if n_samples_flat == 0:
        logger.error("处理后的数据为空，无法进行分割。")
        # 返回空的数组和 None Scaler (因为没有数据 fit)
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None # 使用 None 表示 Scaler 未 fit

    if train_split + val_split > 1.0:
         logger.error("训练集和验证集比例之和必须小于等于1。")
         raise ValueError("分割比例错误。")

    # 计算分割索引
    n_train_flat = int(n_samples_flat * train_split)
    n_val_flat = int(n_samples_flat * val_split)
    # 测试集是剩余的部分
    n_test_flat = n_samples_flat - n_train_flat - n_val_flat

    # 确保训练集至少包含 window_size + 1 个样本才能进行窗口化和有效训练
    # 训练集如果小于这个数，后续窗口化会得到空集
    if n_train_flat < window_size + 1:
        logger.error(f"训练集样本数 ({n_train_flat}) 不足，无法进行窗口化或有效训练 (window_size={window_size})。至少需要 {window_size + 1} 条数据。请增加数据量或调整 train_split。")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None

    # 分割平坦的特征和目标数组
    flat_features_train = features_processed_flat[:n_train_flat]
    flat_targets_train = targets_original_flat[:n_train_flat]

    flat_features_val = features_processed_flat[n_train_flat : n_train_flat + n_val_flat]
    flat_targets_val = targets_original_flat[n_train_flat : n_train_flat + n_val_flat]

    flat_features_test = features_processed_flat[n_train_flat + n_val_flat:]
    flat_targets_test = targets_original_flat[n_train_flat + n_val_flat:]

    logger.info(f"数据分割完成 (按时间顺序)，平坦训练集: {flat_features_train.shape[0]} 条，平坦验证集: {flat_features_val.shape[0]} 条，平坦测试集: {flat_features_test.shape[0]} 条")

    # 检查训练集分割后是否为空
    if flat_features_train.shape[0] == 0 or flat_targets_train.shape[0] == 0:
         logger.error("数据分割后训练集为空。无法进行特征工程和缩放。")
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None


    # --- 6. (可选) 在训练集上拟合特征工程转换器 (PCA 或 基于模型的选择) ---
    # 然后使用拟合好的转换器转换所有数据集 (训练集、验证集、测试集)
    # 初始化转换后的特征和列名，默认使用分割后的平坦特征和列名
    features_transformed_train = flat_features_train
    features_transformed_val = flat_features_val
    features_transformed_test = flat_features_test
    transformed_feature_columns = current_feature_columns[:] # 初始为方差过滤后的列名

    # 优先使用 PCA
    if use_pca:
        # PCA 应该在训练集上拟合，并应用到所有数据集
        # 确保训练集特征数量 > 1 且样本数足够进行 PCA (样本数应大于等于特征数)
        if flat_features_train.shape[1] > 1 and flat_features_train.shape[0] >= flat_features_train.shape[1]:
            logger.info(f"启用 PCA 降维，n_components={n_components}。在训练集上拟合...")
            # PCA 前最好先标准化训练数据 (使用临时 scaler)，因为 PCA 对尺度敏感
            scaler_pca_temp = StandardScaler()
            # fit_transform 只在训练集上进行
            flat_features_train_scaled_temp = scaler_pca_temp.fit_transform(flat_features_train)

            pca = PCA(n_components=n_components)
            try:
                # 在临时缩放后的训练集上拟合 PCA
                pca.fit(flat_features_train_scaled_temp)
                num_components_pca = pca.n_components_
                logger.info(f"PCA拟合完成，保留 {num_components_pca} 个主成分，解释方差比: {sum(pca.explained_variance_ratio_):.4f}")

                if num_components_pca == 0:
                     logger.error("PCA 拟合训练集后特征维度为零。请检查 n_components 或数据。PCA 失败，将使用之前的特征。")
                     use_pca = False # 禁用 PCA 标志，不进行 PCA 转换
                else:
                    # 使用拟合好的 PCA 转换所有数据集 (需要先用临时scaler转换)
                    # 注意：这里的 transform 需要先用 fit 好的 scaler 转换原始数据
                    features_transformed_train = pca.transform(scaler_pca_temp.transform(flat_features_train))
                    # 检查验证集/测试集是否非空，再进行 transform
                    if flat_features_val.shape[0] > 0:
                        features_transformed_val = pca.transform(scaler_pca_temp.transform(flat_features_val))
                    else:
                        features_transformed_val = np.array([]) # 验证集为空则转换后也为空

                    if flat_features_test.shape[0] > 0:
                        features_transformed_test = pca.transform(scaler_pca_temp.transform(flat_features_test))
                    else:
                        features_transformed_test = np.array([]) # 测试集为空则转换后也为空


                    transformed_feature_columns = [f"pca_{i}" for i in range(num_components_pca)] # PCA后列名丢失意义
                    logger.info(f"PCA转换完成。训练集形状: {features_transformed_train.shape}, 验证集形状: {features_transformed_val.shape}, 测试集形状: {features_transformed_test.shape}")

            except Exception as e:
                 logger.error(f"应用 PCA 时出错: {e}", exc_info=True)
                 logger.warning("PCA 降维失败，将使用之前的特征进行后续处理。")
                 use_pca = False # 禁用 PCA 标志


        else:
            logger.warning(f"训练集特征维度 ({flat_features_train.shape[1]}) 或样本数 ({flat_features_train.shape[0]}) 不足，跳过PCA降维。")
            use_pca = False # 禁用 PCA 标志


    # --- 7. (可选) 基于模型的特征选择 ---
    # 仅在 PCA 未启用时执行
    if use_feature_selection and not use_pca:
        # 确保训练集有足够的特征和样本进行模型拟合
        if flat_features_train.shape[1] <= 1 or flat_features_train.shape[0] <= 1:
             logger.warning(f"训练集特征维度 ({flat_features_train.shape[1]}) 或样本数 ({flat_features_train.shape[0]}) 过低，跳过基于模型的特征选择。")
        else:
            logger.info(f"启用基于模型 '{feature_selector_model}' 的特征选择。在训练集上拟合...")
            logger.info("注意：此步骤（模型拟合）可能耗时较长，特别是特征数量多或样本量大时。尝试 'xgb' 可能加速。")

            # 特征选择模型拟合前，通常对特征进行缩放（使用临时 scaler），在训练集上拟合
            # 缩放有助于某些模型收敛，但对于基于树的模型，直接使用原始特征通常也无妨。
            # 为了通用性，这里仍使用临时缩放器对训练集进行缩放后再拟合特征重要性模型。
            # ⚠️ 注意：SelectFromModel 的 fit 方法内部会调用 estimator.fit，然后计算重要性并基于 threshold/max_features 来选择。
            # 更推荐直接在 SelectFromModel 上调用 fit，而不是先拟合 estimator 再 prefit=True。
            # SelectFromModel(estimator).fit(X_train, y_train) 是标准用法。
            # scaler_fs_temp = MinMaxScaler() # 临时 scaler，用于拟合 selector_model
            # flat_features_train_scaled_temp = scaler_fs_temp.fit_transform(flat_features_train)

            # 选择特征选择模型
            selector_model_instance = None
            model_name_lower = feature_selector_model.lower()

            if model_name_lower == 'rf':
                # RandomForestRegressor 参数可以根据需求调整，如 n_estimators, max_depth 等影响性能和选择效果
                selector_model_instance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_name_lower == 'xgb':
                if not XGB_AVAILABLE:
                    logger.error("XGBoost 未安装，无法使用 XGBoost 进行特征选择。将回退到 RandomForest。")
                    selector_model_instance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model_name_lower = 'rf' # 回退模型名称
                else:
                    # XGBoostRegressor 参数也可以调整以优化速度或效果
                    # objective='reg:squarederror' 是回归任务的默认目标
                    selector_model_instance = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
            else:
                logger.warning(f"不支持的特征选择模型: {feature_selector_model}，将使用 RandomForest。")
                selector_model_instance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model_name_lower = 'rf' # 回退模型名称


            # 使用 SelectFromModel 进行特征选择
            try:
                # 在训练集上拟合 SelectFromModel
                # 标准做法是将原始训练特征传递给 SelectFromModel 的 fit 方法
                # 它内部会处理模型的拟合和重要性计算。
                # 注意：对于基于树的模型，通常不需要对特征进行缩放来计算重要性，
                # 但为了兼容性，可以在拟合前对训练集特征进行临时缩放（如果需要）。
                # 实践中，对于RF/XGB，直接使用原始特征 fit SelectFromModel 是更常见且有效的。
                # logger.info(f"使用 {model_name_lower} 在训练集上拟合 SelectFromModel...") # 可选，更细致的日志

                # 确定选择方式 (数量或阈值)
                if max_features_fs is not None and max_features_fs > 0:
                     # 根据数量选择 (不能超过当前训练集特征数量)
                     actual_max_features = min(max_features_fs, flat_features_train.shape[1])
                     selector = SelectFromModel(selector_model_instance, max_features=actual_max_features, threshold=-np.inf) # threshold=-np.inf 确保优先按 max_features 选择
                     logger.info(f"使用 {model_name_lower} 选择最重要的 {actual_max_features} 个特征。")
                else:
                     # 根据阈值选择
                     selector = SelectFromModel(selector_model_instance, threshold=feature_selection_threshold)
                     logger.info(f"使用 {model_name_lower} 和阈值 '{feature_selection_threshold}' 选择特征。")

                # 在训练集上拟合 SelectFromModel
                # 如果模型对缩放敏感（虽然树模型不敏感，但为了通用性），可以在这里传入缩放后的训练集
                # 这里我们传递原始训练集，依靠树模型对尺度的不敏感性
                selector.fit(flat_features_train, flat_targets_train) # Fit on unscaled training features

                # 获取选中的特征索引
                selected_indices = selector.get_support(indices=True)

                if len(selected_indices) == 0:
                    logger.error("基于模型选择后没有剩余特征。请检查模型、阈值或数据。特征选择失败，将使用之前的特征。")
                    # features_transformed_train/val/test 保持为分割后、方差过滤后的结果
                    # transformed_feature_columns 保持为方差过滤后的列名
                else:
                    # 从平坦的、方差过滤后的特征矩阵中提取选中的列
                    features_transformed_train = flat_features_train[:, selected_indices]
                    # 检查验证集/测试集是否非空，再进行提取
                    if flat_features_val.shape[0] > 0:
                         features_transformed_val = flat_features_val[:, selected_indices]
                    else:
                         features_transformed_val = np.array([])

                    if flat_features_test.shape[0] > 0:
                         features_transformed_test = flat_features_test[:, selected_indices]
                    else:
                         features_transformed_test = np.array([])


                    transformed_feature_columns = [current_feature_columns[i] for i in selected_indices] # 更新选中的列名
                    num_features_selected = features_transformed_train.shape[1]
                    logger.info(f"基于模型选择转换完成。选择 {num_features_selected} 个特征。训练集形状: {features_transformed_train.shape}, 验证集形状: {features_transformed_val.shape}, 测试集形状: {features_transformed_test.shape}")
                    logger.info(f"选中的特征列名 (前10个): {transformed_feature_columns[:10]}...")


            except Exception as e:
                 logger.error(f"应用基于模型的特征选择时出错: {e}", exc_info=True)
                 logger.warning("特征选择失败，将使用分割后、方差过滤后的所有特征进行后续处理。")
                 # 如果特征选择失败，features_transformed_train/val/test 保持为分割后、方差过滤后的结果
                 # transformed_feature_columns 也保持为方差过滤后的列名


    # 确定最终处理后的特征数量 (在分割和特征工程后)
    num_final_features = features_transformed_train.shape[1]
    if num_final_features == 0:
         logger.error("经过所有预处理步骤后，训练集特征维度为零。无法继续。")
         # 返回空的数组和 Scaler (Scaler 可能已尝试创建但未 fit)
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None

    logger.info(f"最终用于缩放和窗口化的特征维度: {num_final_features}")


    # --- 8. 对处理后的特征和原始目标变量进行最终缩放 (在训练集上拟合) ---
    # 缩放器在训练集上拟合，然后转换所有数据集，这是防止数据泄露的标准做法。

    # 特征缩放 (在处理后的训练集特征上拟合和转换)
    if scaler_type.lower() == 'minmax':
        feature_scaler = MinMaxScaler()
    elif scaler_type.lower() == 'standard':
        feature_scaler = StandardScaler()
    else:
        logger.warning(f"不支持的特征缩放器类型: {scaler_type}，使用默认MinMaxScaler。")
        feature_scaler = MinMaxScaler()

    # 确保训练集特征不是空的且有维度
    if features_transformed_train.shape[0] > 0 and features_transformed_train.shape[1] > 0:
         # 在处理后的训练集特征上拟合 scaler
         feature_scaler.fit(features_transformed_train)
         # 使用拟合好的 scaler 转换所有数据集
         features_scaled_train = feature_scaler.transform(features_transformed_train)
         # 检查验证集/测试集是否非空，再进行 transform
         features_scaled_val = feature_scaler.transform(features_transformed_val) if features_transformed_val.shape[0] > 0 else np.array([])
         features_scaled_test = feature_scaler.transform(features_transformed_test) if features_transformed_test.shape[0] > 0 else np.array([])
         logger.info(f"最终特征缩放完成 (使用 {scaler_type} scaler)，在训练集上拟合，并应用到所有数据集。")
    else:
         logger.warning("处理后的训练集特征数据为空，跳过最终特征缩放。")
         # 如果训练集为空，转换后的数据也保持为空，scaler 返回 None
         features_scaled_train = features_transformed_train # 应该为空
         features_scaled_val = features_transformed_val # 应该为空
         features_scaled_test = features_transformed_test # 应该为空
         feature_scaler = None # Scaler 未 fit，返回 None


    # 目标变量缩放 (在原始训练集目标上拟合和转换)
    if target_scaler_type.lower() == 'minmax':
        target_scaler = MinMaxScaler()
    elif target_scaler_type.lower() == 'standard':
        target_scaler = StandardScaler()
    else:
        logger.warning(f"不支持的目标变量缩放器类型: {target_scaler_type}，使用默认MinMaxScaler。")
        target_scaler = MinMaxScaler()

    # 确保训练集目标不是空的
    if flat_targets_train.shape[0] > 0:
        # 训练集目标 targets_train 是一个一维数组，需要 reshape 成二维才能 fit scaler
        target_scaler.fit(flat_targets_train.reshape(-1, 1))
        # 使用拟合好的 scaler 转换所有数据集
        targets_scaled_train = target_scaler.transform(flat_targets_train.reshape(-1, 1)).flatten()
        # 检查验证集/测试集是否非空，再进行 transform
        targets_scaled_val = target_scaler.transform(flat_targets_val.reshape(-1, 1)).flatten() if flat_targets_val.shape[0] > 0 else np.array([])
        targets_scaled_test = target_scaler.transform(flat_targets_test.reshape(-1, 1)).flatten() if flat_targets_test.shape[0] > 0 else np.array([])
        logger.info(f"目标变量缩放完成 (使用 {target_scaler_type} scaler)，在训练集上拟合，并应用到所有数据集。")
    else:
        logger.warning("原始训练集目标变量数据为空，跳过目标变量缩放。")
        # 如果训练集为空，转换后的数据也保持为空，scaler 返回 None
        targets_scaled_train = flat_targets_train # 应该为空
        targets_scaled_val = flat_targets_val # 应该为空
        targets_scaled_test = flat_targets_test # 应该为空
        target_scaler = None # Scaler 未 fit，返回 None


    # --- 9. 构建时间序列窗口 (在缩放后的每个数据集上单独进行) ---

    def create_windows(features_scaled, targets_scaled, window_size):
        """辅助函数：为单个数据集构建时间序列窗口"""
        X, y = [], []
        # 确保数据长度足够构建至少一个窗口 (需要 window_size 个输入 + 1 个输出)
        if features_scaled.shape[0] < window_size + 1:
            # logger.warning(f"数据长度 {features_scaled.shape[0]} 不足以构建窗口 ({window_size})。") # 频繁调用会刷屏，只在外部检查即可
            return np.array([]), np.array([])
        for i in range(features_scaled.shape[0] - window_size):
            X.append(features_scaled[i:(i + window_size)]) # 使用缩放后的特征
            y.append(targets_scaled[i + window_size]) # 预测窗口后的一个点 (使用缩放后的目标)
        return np.array(X), np.array(y)

    logger.info(f"开始构建时间序列窗口 (window_size={window_size})...")

    X_train, y_train = create_windows(features_scaled_train, targets_scaled_train, window_size)
    X_val, y_val = create_windows(features_scaled_val, targets_scaled_val, window_size)
    X_test, y_test = create_windows(features_scaled_test, targets_scaled_test, window_size)

    # 检查训练集窗口化后是否为空
    if X_train.shape[0] == 0:
         # 这通常意味着分割后的训练集虽然非空，但长度小于 window_size + 1
         logger.error(f"窗口化训练集后数据量为零。分割后的训练集长度 {features_scaled_train.shape[0]}, 窗口大小 {window_size}。请检查数据长度和窗口大小。")
         # 返回空的数组和 Scaler (如果 fit 成功则返回，否则 None)
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), feature_scaler, target_scaler

    logger.info(f"窗口化完成。训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}, 测试集形状: {X_test.shape}")


    # 返回最终处理好的数据和在训练集上拟合的 scaler
    # Scaler 可能因为训练集为空而为 None，调用方需要检查
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler

@log_execution_time
@handle_exceptions
def train_lstm_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, model: Sequential,
    target_scaler: Union[MinMaxScaler, StandardScaler, None], # 传入目标变量缩放器 (可能为None)
    training_config: Dict[str, Any] = None, checkpoint_path: str = "models/checkpoints/best_model.keras",
    plot_training_history: bool = False # 是否绘制训练历史图
) -> Dict:
    """
    训练LSTM模型。

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray): 已准备好的数据集 (y已缩放)。
        model (Sequential): 已编译的Keras模型。
        target_scaler (Union[MinMaxScaler, StandardScaler, None]): 用于目标变量的缩放器 (在训练集上拟合)。可能为 None。用于将测试集预测结果逆缩放以估算原始范围的 MAE。
        training_config (Dict): 训练配置，如epochs, batch_size等。
        checkpoint_path (str): 模型检查点保存路径。
        plot_training_history (bool): 是否绘制训练历史图。

    Returns:
        Dict: 训练历史。
    """
    default_config = {
        'epochs': 20,
        'batch_size': 128,
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
        'reduce_lr_factor': 0.5,
        'monitor_metric': 'val_loss', # 对于回归，通常监控loss或MAE，优先监控验证集
        'verbose': 1 # 默认显示进度条，方便观察
    }
    config = default_config.copy() # 使用copy，避免修改default_config
    if training_config:
        # 深度更新字典，确保callbacks相关的参数也能被覆盖
        for key, value in training_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value


    logger.info(f"开始训练模型，训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}, 测试集样本数: {X_test.shape[0]}")
    logger.info(f"训练配置: {config}")

    # 检查训练集是否有数据
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
         logger.error("训练集数据为空，无法训练模型。")
         return {} # 返回空历史

    # 回归任务，通常不需要类别权重
    sample_weight = None

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"创建检查点目录: {checkpoint_dir}")

    # 根据验证集是否存在调整callbacks
    validation_data = (X_val, y_val) if X_val.shape[0] > 0 and y_val.shape[0] > 0 else None
    callbacks = []

    monitor_metric = config.get('monitor_metric', 'val_loss')
    # 如果没有验证集，则将监控指标切换到训练集
    if validation_data is None and monitor_metric.startswith('val_'):
        logger.warning(f"验证集为空，将监控指标 '{monitor_metric}' 切换到训练集 '{monitor_metric[4:]}'。")
        monitor_metric = monitor_metric[4:] # 例如 'val_loss' -> 'loss'
        # 如果切换后的指标仍然是 'val_something' (不可能)，或者切换后的指标不在模型metrics中，可能会有问题
        # 确保切换后的指标在 model.metrics_names 或 loss 中
        valid_monitor = False
        if monitor_metric == model.loss: valid_monitor = True
        if monitor_metric in model.metrics_names: valid_monitor = True
        if not valid_monitor:
            logger.warning(f"切换后的监控指标 '{monitor_metric}' 不在模型损失或指标列表中，EarlyStopping和ReduceLROnPlateau可能不会按预期工作。")
            # 尝试使用loss作为后备
            monitor_metric = model.loss
            logger.warning(f"后备监控指标设置为模型的损失函数: '{monitor_metric}'。")


    callbacks.append(EarlyStopping(monitor=monitor_metric, patience=config['early_stopping_patience'], restore_best_weights=True, verbose=config['verbose'], mode='min' if 'loss' in monitor_metric.lower() else 'max'))
    callbacks.append(ReduceLROnPlateau(monitor=monitor_metric, factor=config['reduce_lr_factor'], patience=config['reduce_lr_patience'], min_lr=1e-6, verbose=config['verbose'], mode='min' if 'loss' in monitor_metric.lower() else 'max'))
    # ModelCheckpoint 也可以选择监控训练集指标
    callbacks.append(ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_metric, save_best_only=True, save_weights_only=False, mode='min' if 'loss' in monitor_metric.lower() else 'max', verbose=config['verbose']))

    # 如果移除了所有依赖验证集的 callback，则打印警告 (之前的逻辑已经覆盖了大部分情况)
    if validation_data is None and any(cb.monitor.startswith('val_') for cb in callbacks):
         logger.warning("在移除依赖验证集的 callback 逻辑后，仍有 callback 监控 'val_*' 指标。请检查 monitor_metric 配置。")


    history = model.fit(
        X_train, y_train, # 使用已缩放的 y_train
        validation_data=validation_data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        sample_weight=sample_weight,  # 使用修正后的 sample_weight (None)
        callbacks=callbacks,
        verbose=config['verbose']
    )

    # 在测试集上评估 (使用缩放后的目标值)
    if X_test.shape[0] > 0 and y_test.shape[0] > 0:
        try:
            test_results = model.evaluate(X_test, y_test, verbose=0)
            # evaluate 返回的是一个列表，第一个是 loss，后面是 metrics
            test_loss = test_results[0]
            # 找到 MAE 的索引 (假设 'mae' 在 metrics 列表中)
            test_mae_scaled = np.nan # 默认 NaN
            try:
                # model.metrics_names 是一个列表，包含 loss 和 metrics 的名称
                # 例如: ['loss', 'mae']
                if 'mae' in model.metrics_names:
                    mae_index = model.metrics_names.index('mae')
                    test_mae_scaled = test_results[mae_index]
                else:
                     logger.warning("模型编译的 metrics 中没有 'mae'。")
            except Exception as e:
                 logger.warning(f"获取测试集 scaled MAE 时出错: {e}", exc_info=True)


            # 将 MAE 转换回原始范围 (仅作估算)
            mae_original_approx = np.nan # 默认 NaN
            # 只有当 target_scaler 存在且已 fit 且 scaled MAE 有效时才进行逆缩放估算
            # 检查 target_scaler 是否已 fit 的方法：查看其 n_features_in_ 属性或 feature_range 属性 (MinMaxScaler)
            is_scaler_fitted = target_scaler is not None and (
                hasattr(target_scaler, 'n_features_in_') or # StandardScaler
                (isinstance(target_scaler, MinMaxScaler) and hasattr(target_scaler, 'feature_range')) # MinMaxScaler
            )

            if is_scaler_fitted and not np.isnan(test_mae_scaled):
                 try:
                     # 估算方法：计算缩放器将 0 和 scaled_mae 转换回原始尺度的差值
                     # 创建一个包含 0 和 scaled_mae 的二维数组用于逆缩放
                     # 注意：这里假设 MAE 是非负的，且缩放器是线性的
                     dummy_values = np.array([[0.0], [test_mae_scaled]])
                     original_values = target_scaler.inverse_transform(dummy_values)
                     mae_original_approx = abs(original_values[1][0] - original_values[0][0])
                 except Exception as e:
                     logger.error(f"估算原始范围 MAE 时出错: {e}", exc_info=True)
                     mae_original_approx = np.nan # 发生错误时设置为 NaN
            else:
                 if not is_scaler_fitted:
                     logger.warning("目标变量缩放器未 fit 或为 None，无法估算原始范围 MAE。")
                 elif np.isnan(test_mae_scaled):
                     logger.warning("Scaled MAE 为 NaN，无法估算原始范围 MAE。")


            logger.info(f"LSTM模型在测试集上的损失: {test_loss:.4f}, MAE (缩放后): {test_mae_scaled:.4f}, MAE (原始范围估算): {mae_original_approx:.4f}")
        except Exception as e:
             logger.error(f"评估测试集时出错: {e}", exc_info=True)
             logger.warning("测试集评估失败。")

    else:
        logger.warning("测试集为空，无法评估LSTM模型。")

    logger.info("模型训练完成。")
    # 打印最终的训练和验证损失/指标 (从 history 中获取最后的值)
    final_loss = history.history['loss'][-1] if 'loss' in history.history else 'N/A'
    # 检查 val_loss 是否存在，只有存在时才打印
    final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 'N/A (验证集为空)' if validation_data is None else 'N/A (历史记录缺失)'
    # 检查 mae 是否存在
    final_mae = history.history['mae'][-1] if 'mae' in history.history else 'N/A (指标缺失)'
     # 检查 val_mae 是否存在
    final_val_mae = history.history['val_mae'][-1] if 'val_mae' in history.history else 'N/A (验证集为空)' if validation_data is None else 'N/A (历史记录缺失)'

    logger.info(f"训练历史: 最终损失={final_loss}, 最终验证损失={final_val_loss}, 最终MAE={final_mae}, 最终验证MAE={final_val_mae}")


    # 可选：绘制训练历史图
    if plot_training_history:
         try:
             plt.figure(figsize=(12, 6))
             plt.plot(history.history['loss'], label='训练集损失')
             # 检查是否存在验证集损失
             if 'val_loss' in history.history:
                 plt.plot(history.history['val_loss'], label='验证集损失')
             plt.title('模型损失')
             plt.xlabel('周期')
             plt.ylabel('损失')
             plt.legend()
             plt.grid(True)

             # 检查 metrics 中是否有 mae
             if 'mae' in history.history:
                plt.figure(figsize=(12, 6))
                plt.plot(history.history['mae'], label='训练集MAE')
                # 检查是否存在验证集MAE
                if 'val_mae' in history.history:
                    plt.plot(history.history['val_mae'], label='验证集MAE')
                plt.title('模型平均绝对误差 (MAE)')
                plt.xlabel('周期')
                plt.ylabel('MAE')
                plt.legend()
                plt.grid(True)
             else:
                 logger.warning("训练历史中没有MAE指标，跳过绘制MAE图。请检查model.compile的metrics参数。")

             # 为防止文件名冲突，可以考虑加上时间戳或股票代码 (如果可用)
             # 例如: plt.savefig(f'training_history_{stock_code}_{int(time.time())}.png')
             # 确保保存目录存在
             plot_dir = 'training_plots' # 可以改为配置项
             if not os.path.exists(plot_dir):
                 os.makedirs(plot_dir)
                 logger.info(f"创建绘图目录: {plot_dir}")
             plot_path = os.path.join(plot_dir, 'training_history.png') # 可以使文件名更具描述性
             plt.savefig(plot_path)
             plt.close('all') # 关闭所有图窗以释放内存
             logger.info(f"训练历史图已保存至 {plot_path}")
         except Exception as e:
             logger.error(f"绘制训练历史图出错: {e}", exc_info=True)

    return history.history

@log_execution_time
@handle_exceptions
def build_lstm_model(
    window_size: int,
    num_features: int,
    model_config: Dict[str, Any] = None,
    model_type: str = 'lstm',
    summary: bool = True,
    # 新增参数以应对目标变量类型变化 (如果需要支持分类)
    # output_units: int = 1, # 回归默认1
    # output_activation: Optional[str] = None, # 回归默认 None
    # loss: str = 'mse', # 回归默认 'mse'
    # metrics: List[str] = ['mae'] # 回归默认 ['mae']
) -> Sequential:
    """
    构建深度学习模型，支持LSTM、Bidirectional LSTM和GRU，允许自定义配置。
    默认配置为回归任务 (单输出，线性激活，MSE损失，MAE指标)。
    如需分类，需要修改这里的参数或在调用时覆盖 model_config。

    Args:
        window_size (int): 输入时间步长。
        num_features (int): 输入特征维度。
        model_config (Dict): 模型配置字典，可以覆盖默认层、优化器、损失函数等。
                             例如: {'layers': [{'units': 128, 'return_sequences': True, 'activation': 'relu'}, ...],
                                    'dense_layers': [{'units': 32, 'activation': 'relu'}],
                                    'optimizer': 'adam', 'learning_rate': 0.005,
                                    'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'output_units': 1, 'output_activation': 'sigmoid'}
        model_type (str): 模型类型 ('lstm', 'bilstm', 'gru')。
        summary (bool): 是否打印模型摘要。

    Returns:
        Sequential: 已编译的Keras模型。如果输入特征数量无效则抛出错误。
    """
    # 默认配置 (回归任务)
    default_config = {
        'layers': [
            {'units': 64, 'return_sequences': True, 'dropout': 0.3, 'l2_reg': 0.01, 'activation': 'tanh'}, # RNN层默认激活
            {'units': 32, 'return_sequences': False, 'dropout': 0.3, 'l2_reg': 0.01, 'activation': 'tanh'}
        ],
        'dense_layers': [{'units': 16, 'dropout': 0.2, 'l2_reg': 0.01, 'activation': 'relu'}], # Dense层默认激活
        'optimizer': 'adam',
        'learning_rate': 0.001, # 默认学习率
        'loss': 'mse', # 回归任务常用损失函数
        'metrics': ['mae'], # 回归任务常用指标
        'output_units': 1, # 回归任务输出单元数
        'output_activation': None # 回归任务输出层无激活函数 (线性激活)
    }
    config = default_config.copy() # 使用copy，避免修改default_config
    if model_config:
        # 深度更新字典，特别是嵌套的 layers 和 dense_layers
        for key, value in model_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                # 递归更新子字典
                config[key].update(value)
            elif isinstance(value, list) and key in config and isinstance(config[key], list):
                 # 对于列表（如层配置），直接替换整个列表
                 config[key] = value
            else:
                # 其他参数直接覆盖
                config[key] = value

    # 检查num_features是否有效
    if num_features <= 0:
        logger.error(f"特征数量无效: {num_features}。无法构建模型。")
        raise ValueError(f"无效的特征数量: {num_features}")

    # 构建模型
    model = Sequential()
    model_type_lower = model_type.lower()

    if model_type_lower not in ['lstm', 'bilstm', 'gru']:
         logger.error(f"不支持的模型类型: {model_type}")
         raise ValueError(f"不支持的模型类型: {model_type}")

    # 添加循环层 (LSTM/GRU/BiLSTM)
    if not config.get('layers'):
        logger.warning("模型配置中没有指定任何循环层 ('layers')。模型可能无法有效学习时间序列特征。")
        # 可以在这里添加一个默认的简单循环层，或者允许构建只有 Dense 层的模型
        # 为了灵活性，如果 layers 为空则跳过添加循环层
        # config['layers'] = [{'units': 32, 'return_sequences': False}] # 示例：添加一个默认层

    for i, layer_conf in enumerate(config.get('layers', [])):
        l2_reg_val = layer_conf.get('l2_reg', 0.0)
        # L2 正则化可以分别应用于核权重和循环权重
        kernel_regularizer = l2(l2_reg_val) if l2_reg_val > 0 else None
        recurrent_regularizer = l2(l2_reg_val) if l2_reg_val > 0 else None
        rnn_activation = layer_conf.get('activation', 'tanh') # RNN层默认激活函数

        layer_args = {
            'units': layer_conf.get('units', 32), # 确保units有默认值
            'return_sequences': layer_conf.get('return_sequences', False), # 默认 False 更安全
            'kernel_regularizer': kernel_regularizer,
            'recurrent_regularizer': recurrent_regularizer,
            'activation': rnn_activation
        }
        if i == 0:
            # 只有第一个循环层需要指定 input_shape
            layer_args['input_shape'] = (window_size, num_features)

        # 添加层到模型
        try:
            if model_type_lower == 'bilstm':
                 # Bidirectional Wrapper 接收 units 和 return_sequences，内部 LSTM 接收其他参数
                 model.add(Bidirectional(LSTM(units=layer_args['units'],
                                              return_sequences=layer_args['return_sequences'],
                                              kernel_regularizer=layer_args['kernel_regularizer'],
                                              recurrent_regularizer=layer_args['recurrent_regularizer'],
                                              activation=layer_args['activation']),
                                         input_shape=layer_args.get('input_shape'))) # 只有第一个 BiLSTM 需要 input_shape
            elif model_type_lower == 'lstm':
                 model.add(LSTM(**layer_args))
            elif model_type_lower == 'gru':
                 model.add(GRU(**layer_args))

            # 添加循环层后的 Dropout
            if layer_conf.get('dropout', 0.0) > 0:
                model.add(Dropout(layer_conf['dropout']))

        except Exception as e:
             logger.error(f"构建循环层 (类型: {model_type_lower}, 配置: {layer_conf}) 时出错: {e}", exc_info=True)
             raise # 抛出异常，因为层构建失败是严重问题

    # 添加全连接层 (Dense)
    if not config.get('dense_layers') and not config.get('layers'):
         logger.warning("模型配置中没有指定任何 Dense 层 ('dense_layers') 且没有循环层。模型将只有输出层。")

    for dense_conf in config.get('dense_layers', []):
        l2_reg_val = dense_conf.get('l2_reg', 0.0)
        kernel_regularizer = l2(l2_reg_val) if l2_reg_val > 0 else None
        dense_activation = dense_conf.get('activation', 'relu') # Dense层默认激活函数
        try:
            model.add(Dense(units=dense_conf.get('units', 16), # 确保units有默认值
                            kernel_regularizer=kernel_regularizer,
                            activation=dense_activation))
            # 添加全连接层后的 Dropout
            if dense_conf.get('dropout', 0.0) > 0:
                model.add(Dropout(dense_conf['dropout']))
        except Exception as e:
             logger.error(f"构建全连接层 (配置: {dense_conf}) 时出错: {e}", exc_info=True)
             raise # 抛出异常

    # 输出层
    output_units = config.get('output_units', 1)
    output_activation = config.get('output_activation', None) # 回归通常是 None
    try:
        # 最后一层通常不需要正则化和 dropout
        model.add(Dense(units=output_units, activation=output_activation)) # 默认回归输出层，无激活函数 (线性激活)
    except Exception as e:
        logger.error(f"构建输出层 (units: {output_units}, activation: {output_activation}) 时出错: {e}", exc_info=True)
        raise # 抛出异常


    # 选择优化器
    optimizer_name = config.get('optimizer', 'adam').lower() # 转换为小写以便比较
    lr = config.get('learning_rate', 0.001) # 从 config 中获取学习率
    # 优化器可以接收各种参数，这里只处理学习率和可能的 momentum (用于 SGD)
    try:
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=lr)
        elif optimizer_name == 'sgd':
            # SGD 需要 momentum 参数才能有效
            momentum_val = config.get('momentum', 0.0)
            optimizer = SGD(learning_rate=lr, momentum=momentum_val)
        else:
            logger.warning(f"不支持的优化器: {optimizer_name}，使用默认Adam。")
            optimizer = Adam(learning_rate=lr)
    except Exception as e:
        logger.error(f"构建优化器 (名称: {optimizer_name}, 学习率: {lr}) 时出错: {e}", exc_info=True)
        # 如果优化器构建失败，则尝试使用默认 Adam 并打印警告，不抛出错误
        logger.warning("优化器构建失败，尝试使用默认 Adam。")
        optimizer = Adam(learning_rate=0.001) # Hardcode default LR


    # 编译模型
    loss_func = config.get('loss', 'mse') # 损失函数
    metrics_list = config.get('metrics', ['mae']) # 评估指标列表
    # 考虑自定义损失函数和指标（如果需要）
    # if isinstance(loss_func, str):
    #     # Standard Keras loss
    #     pass
    # elif isinstance(loss_func, Callable):
    #     # Custom loss function - requires passing the function object
    #     pass
    try:
        model.compile(
            optimizer=optimizer,
            loss=loss_func,
            metrics=metrics_list
        )
    except Exception as e:
        logger.error(f"模型编译时出错 (损失函数: {loss_func}, 指标: {metrics_list}): {e}", exc_info=True)
        raise # 编译失败是严重问题，必须抛出

    if summary:
        # 使用print_fn参数，将 summary 输出到 logger
        # 防止在非终端环境中打印或与标准输出混合
        model.summary(print_fn=lambda x: logger.info(x))
    logger.info(f"模型构建完成 ({model_type.upper()})，输入形状: ({window_size}, {num_features})，输出单元: {output_units}")
    return model

# analyze_target_distribution 函数保留原样，但建议在数据缩放前调用

def analyze_target_distribution(y_train, y_val, y_test):
    """
    分析并绘制目标变量分布图。建议在数据缩放前调用此函数，以查看原始目标分布。
    """
    plt.figure(figsize=(15, 5))
    # 接受原始或缩放后的目标变量，但建议是原始的
    datasets = {'训练集': y_train, '验证集': y_val, '测试集': y_test}
    for i, (name, y) in enumerate(datasets.items()):
        plt.subplot(1, 3, i+1)
        # 检查 y 是否为空或 None
        if y is None or len(y) == 0:
            plt.title(f'{name} 目标变量分布 (数据为空)')
            # 绘制一个空的子图或者标记
            plt.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            # 打印描述性统计信息 (空数据)
            logger.info(f"{name} 目标变量描述：数据为空")
            continue

        # 根据唯一值数量调整bins，避免bins过多或过少
        # 检查数据类型，确保是数值类型
        try:
            y_numeric = pd.to_numeric(y, errors='coerce').dropna()
        except Exception as e:
             logger.error(f"转换 {name} 目标变量为数值时出错: {e}", exc_info=True)
             y_numeric = pd.Series([]) # 转换为空序列

        if len(y_numeric) == 0:
             plt.title(f'{name} 目标变量分布 (无有效数值)')
             plt.text(0.5, 0.5, 'No valid data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             logger.info(f"{name} 目标变量描述：无有效数值")
             continue

        num_unique = len(np.unique(y_numeric))
        # bins 数量不应超过数据点的数量，也不应超过唯一值的数量
        bins_to_use = min(num_unique, 50, len(y_numeric)) # 限制最大bins数量，防止数据量少时bins过多
        # 如果数据点太少，bins_to_use 可能为 0 或 1，需要处理
        if bins_to_use < 2 and len(y_numeric) > 0: # 至少有数据点且 bins < 2，设为 1
             bins_to_use = 1
        elif len(y_numeric) == 0: # 如果没有数据点
             bins_to_use = 0 # 或者可以跳过绘制，上面已经处理

        if bins_to_use > 0: # 只有在有有效 bins 时才尝试绘制
             try:
                if bins_to_use == 1: # 如果只有1个唯一值或极少数据点
                     sns.histplot(y_numeric, bins=1, kde=False, alpha=0.7) # 不绘制kde
                else:
                     sns.histplot(y_numeric, bins=bins_to_use, kde=True, alpha=0.7) # 绘制kde

                plt.title(f'{name} 目标变量分布')
                plt.xlabel('目标值')
                plt.ylabel('频次')

                # 打印描述性统计信息
                logger.info(f"{name} 目标变量描述：\n%s", pd.Series(y_numeric).describe())

             except Exception as e:
                logger.error(f"绘制 {name} 目标变量分布图出错: {e}", exc_info=True)
                plt.title(f'{name} 目标变量分布 (绘制失败)')
                plt.text(0.5, 0.5, 'Plot Error', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                logger.warning(f"无法为 {name} 绘制分布图。")
                # 尝试打印描述性统计信息，即使绘图失败
                try:
                    logger.info(f"{name} 目标变量描述 (绘图失败但尝试描述)：\n%s", pd.Series(y_numeric).describe())
                except Exception:
                     logger.warning(f"无法为 {name} 获取描述性统计信息。")

        else: # bins_to_use is 0 or 1 but len(y_numeric) is also 0
             plt.title(f'{name} 目标变量分布 (数据量不足)')
             plt.text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             logger.warning(f"{name} 目标变量数据量不足以绘制分布图。")


    plt.tight_layout()
    # 为防止文件名冲突，可以考虑加上股票代码或时间戳
    # 确保保存目录存在
    plot_dir = 'training_plots' # 可以改为配置项
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        logger.info(f"创建绘图目录: {plot_dir}")
    plot_path = os.path.join(plot_dir, 'target_distribution.png') # 可以使文件名更具描述性

    try:
        plt.savefig(plot_path)
        logger.info(f"目标变量分布图已保存至 {plot_path}")
    except Exception as e:
         logger.error(f"保存目标变量分布图出错: {e}", exc_info=True)

    plt.close('all') # 确保关闭所有图窗以释放内存


