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

# 导入 Keras Sequence
from tensorflow.keras.utils import Sequence

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

# 新增：时间序列数据生成器
class TimeSeriesSequence(Sequence):
    """
    Keras Sequence for time series data. Generates windows and targets on the fly.
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int, batch_size: int, shuffle: bool = False):
        """
        Args:
            features (np.ndarray): Scaled flat features (shape: num_samples, num_features).
            targets (np.ndarray): Scaled flat targets (shape: num_samples,).
            window_size (int): The number of time steps in each input window.
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle data indices at the end of each epoch.
                            For time series, usually False.
        """
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Features and targets must have the same number of samples.")
        if features.shape[0] < window_size + 1:
             logger.warning(f"数据长度 {features.shape[0]} 不足以构建窗口 ({window_size})。Sequence将为空。")
             self.features = np.array([])
             self.targets = np.array([])
             self.indices = np.array([])
             self.window_size = window_size
             self.batch_size = batch_size
             self.shuffle = shuffle
             self.num_samples = 0
             self.num_windows = 0
             return

        self.features = features
        self.targets = targets
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = features.shape[0]
        # 可用的窗口数量 = 总样本数 - 窗口大小
        self.num_windows = self.num_samples - self.window_size
        self.indices = np.arange(self.num_windows) # 索引对应每个窗口的起始位置

        if self.shuffle:
            self.on_epoch_end() # 初始打乱

        logger.info(f"TimeSeriesSequence 初始化完成: 样本数={self.num_samples}, 窗口大小={self.window_size}, 批次大小={self.batch_size}, 可用窗口数={self.num_windows}")


    def __len__(self):
        """Returns the number of batches per epoch."""
        if self.num_windows <= 0:
            return 0
        return int(np.ceil(self.num_windows / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # 获取当前批次的索引范围
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = []
        y_batch = []

        for i in batch_indices:
            # i 是窗口的起始索引 (在 self.indices 中，对应平坦数据中的索引)
            # 特征窗口是 [i, i + window_size - 1]
            # 目标是平坦数据中索引为 i + window_size 的点
            start_index = i
            end_index = i + self.window_size # 特征窗口的结束索引 (不包含)
            target_index = i + self.window_size # 目标的索引

            # 确保索引有效 (target_index 不能超出 self.targets 的范围)
            if target_index >= self.num_samples: # target_index 最大可以是 num_samples - 1
                 logger.warning(f"生成批次 {index} 时目标索引越界。start_index={start_index}, end_index={end_index}, target_index={target_index}, num_samples={self.num_samples}")
                 continue # 跳过此无效索引

            X_batch.append(self.features[start_index:end_index, :])
            y_batch.append(self.targets[target_index])

        # 将列表转换为 NumPy 数组
        # 检查 X_batch 是否为空，避免 np.array([]) 导致形状问题
        if not X_batch:
             # 如果 X_batch 为空，意味着这个批次没有有效数据，返回空数组
             # Keras 的 fit/evaluate 应该能处理这种情况，或者抛出更明确的错误
             # 但为了稳健，我们返回形状正确的空数组（如果可能）或就是空数组
             # LSTM 输入期望 (batch, timesteps, features)
             # LSTM 输出期望 (batch, units) 或 (batch, 1) for regression
             # 这里 num_features 可以从 self.features.shape[1] 获取，如果 self.features 非空
             num_features_dim = self.features.shape[1] if self.features.ndim == 2 and self.features.shape[1] > 0 else 0
             return np.empty((0, self.window_size, num_features_dim)), np.empty((0,))


        return np.array(X_batch), np.array(y_batch)


    def on_epoch_end(self):
        """Optional: shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
            # logger.debug("TimeSeriesSequence: Indices shuffled.") # 避免频繁日志


@log_execution_time
@handle_exceptions
def prepare_data_for_lstm(
    data: pd.DataFrame,
    required_columns: List[str], # 数据中应包含的原始特征列列表（用于初始筛选）
    target_column: str = 'final_signal',
    # window_size: int = 60, # 窗口大小不再在这里使用，而是传递给 Sequence
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
    准备用于LSTM训练的平坦、缩放后的时间序列数据，包括特征处理、缩放和按时间顺序数据集分割。
    不再进行窗口化操作，窗口化由 Keras Sequence 处理。

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
    10. **返回平坦、缩放后的数据集和 Scaler。**

    将特征工程和缩放器的拟合过程严格限制在训练集上，避免未来数据泄露。

    Args:
        data (pd.DataFrame): 包含所有特征和目标列的原始DataFrame。
        required_columns (List[str]): 数据中应包含的原始特征列列表（用于初始筛选）。
        target_column (str): 目标变量列名。
        # window_size (int): LSTM输入的时间步长。# 已移除
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
        features_scaled_train, targets_scaled_train, features_scaled_val, targets_scaled_val, features_scaled_test, targets_scaled_test (np.ndarray): 分割、处理和缩放后的平坦数据集。
        feature_scaler (Union[MinMaxScaler, StandardScaler, None]): 用于最终特征的缩放器 (在训练集处理后的特征集上拟合)。如果训练集为空则为 None。
        target_scaler (Union[MinMaxScaler, StandardScaler, None]): 用于目标变量的缩放器 (在训练集原始目标集上拟合)。如果训练集为空则为 None。
    """
    logger.info(f"开始准备 LSTM 数据 (平坦化处理)...")
    logger.info(f"参数: scaler_type='{scaler_type}', train_split={train_split}, val_split={val_split}, apply_variance_threshold={apply_variance_threshold}, variance_threshold_value={variance_threshold_value}, use_pca={use_pca}, n_components={n_components}, use_feature_selection={use_feature_selection}, feature_selector_model='{feature_selector_model}', max_features_fs={max_features_fs}, feature_selection_threshold='{feature_selection_threshold}', target_scaler_type='{target_scaler_type}'")

    # --- 1. 检查目标列 ---
    if target_column not in data.columns:
        logger.error(f"目标列 '{target_column}' 不存在于输入数据中。")
        # 返回空的数组和 None Scaler
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None

    # --- 2. 初始特征列选择 (基于 required_columns 并排除目标列) ---
    # 确保只选择 data 中实际存在的列
    initial_feature_columns = [col for col in required_columns if col in data.columns and col != target_column]
    if not initial_feature_columns:
         logger.error("根据 required_columns 筛选后，没有可用的特征列。")
         # 返回空的数组和 None Scaler
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None

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
         # 返回空的数组和 None Scaler
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None


    # 计算分割索引
    n_train_flat = int(n_samples_flat * train_split)
    n_val_flat = int(n_samples_flat * val_split)
    # 测试集是剩余的部分
    n_test_flat = n_samples_flat - n_train_flat - n_val_flat

    # 确保训练集至少包含一些样本
    if n_train_flat == 0:
        logger.error(f"训练集样本数 ({n_train_flat}) 为零。请增加数据量或调整 train_split。")
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
            # logger.info("注意：此步骤（模型拟合）可能耗时较长，特别是特征数量多或样本量大时。尝试 'xgb' 可能加速。")

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

    logger.info(f"最终用于缩放的特征维度: {num_final_features}")


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


    # --- 9. 移除窗口化步骤 ---
    # create_windows 函数不再在这里调用

    logger.info(f"数据准备完成 (平坦、缩放后)。训练集形状: {features_scaled_train.shape}, 验证集形状: {features_scaled_val.shape}, 测试集形状: {features_scaled_test.shape}")

    # 返回最终处理好的平坦数据和在训练集上拟合的 scaler
    # Scaler 可能因为训练集为空而为 None，调用方需要检查
    return features_scaled_train, targets_scaled_train, features_scaled_val, targets_scaled_val, features_scaled_test, targets_scaled_test, feature_scaler, target_scaler


@log_execution_time
@handle_exceptions
def train_lstm_model(
    # 接收 Sequence 对象而不是完整的 NumPy 数组
    train_sequence: TimeSeriesSequence,
    val_sequence: Optional[TimeSeriesSequence], # 验证集 Sequence 可能为 None
    X_test: np.ndarray, y_test: np.ndarray, # 测试集评估可以继续使用 NumPy 数组
    model: Sequential,
    target_scaler: Union[MinMaxScaler, StandardScaler, None], # 传入目标变量缩放器 (可能为None)
    training_config: Dict[str, Any] = None, checkpoint_path: str = "models/checkpoints/best_model.keras",
    plot_training_history: bool = False # 是否绘制训练历史图
) -> Dict:
    """
    训练LSTM模型，使用 Keras Sequence 进行内存高效的数据加载。
    Args:
        train_sequence (TimeSeriesSequence): 训练集数据生成器。
        val_sequence (Optional[TimeSeriesSequence]): 验证集数据生成器 (可能为 None)。
        X_test, y_test (np.ndarray): 已准备好的测试集数据 (y已缩放)。
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
        'batch_size': 128, # batch_size 现在由 Sequence 控制，但这里保留用于日志或未来可能的直接数组训练
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

    # 检查训练 Sequence 是否有效
    if len(train_sequence) == 0:
         logger.error("训练集 Sequence 为空，无法训练模型。请检查数据量和窗口大小。")
         return {} # 返回空历史

    logger.info(f"开始训练模型，训练集批次数: {len(train_sequence)}, 验证集批次数: {len(val_sequence) if val_sequence else 0}, 测试集样本数: {X_test.shape[0]}")
    logger.info(f"训练配置: {config}")

    # 回归任务，通常不需要类别权重
    sample_weight = None

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"创建检查点目录: {checkpoint_dir}")

    # 根据验证集 Sequence 是否存在调整callbacks
    validation_data = val_sequence if val_sequence and len(val_sequence) > 0 else None
    callbacks = []

    monitor_metric = config.get('monitor_metric', 'val_loss')
    # 如果没有验证集 Sequence，则将监控指标切换到训练集
    if validation_data is None and monitor_metric.startswith('val_'):
        logger.warning(f"验证集 Sequence 为空，将监控指标 '{monitor_metric}' 切换到训练集 '{monitor_metric[4:]}'。")
        monitor_metric = monitor_metric[4:] # 例如 'val_loss' -> 'loss'
        # 确保切换后的指标在 model.metrics_names 或 loss 中
        # Keras 3 中 model.loss 是一个函数或字符串，model.metrics_names 包含损失和指标的名称
        valid_monitor = False
        # 检查 monitor_metric 是否是模型已知的损失函数名称或指标名称
        # model.loss 可以是字符串 (如 'mse') 或 tf.keras.losses.Loss 实例
        # model.metrics_names 是一个列表，例如 ['loss', 'mae']
        if hasattr(model, 'loss'): # 确保 model 对象有 loss 属性
            model_loss_name = model.loss if isinstance(model.loss, str) else (model.loss.name if hasattr(model.loss, 'name') else None)
            if model_loss_name and monitor_metric == model_loss_name:
                valid_monitor = True
        if not valid_monitor and hasattr(model, 'metrics_names') and model.metrics_names is not None and monitor_metric in model.metrics_names:
            valid_monitor = True

        if not valid_monitor:
            logger.warning(f"切换后的监控指标 '{monitor_metric}' 不在模型损失或指标列表中，EarlyStopping和ReduceLROnPlateau可能不会按预期工作。")
            # 尝试使用loss作为后备 (通常 model.metrics_names[0] 是 loss)
            if hasattr(model, 'metrics_names') and model.metrics_names and len(model.metrics_names) > 0:
                monitor_metric = model.metrics_names[0] # 通常是 'loss'
                logger.warning(f"后备监控指标设置为模型的第一个度量: '{monitor_metric}'。")
            else: # 如果连 metrics_names 都没有，这是个更严重的问题
                logger.error("无法确定有效的监控指标，回调函数可能无法正常工作。")
                # 保持 monitor_metric 为切换后的值，让 Keras 处理

    callbacks.append(EarlyStopping(monitor=monitor_metric, patience=config['early_stopping_patience'], restore_best_weights=True, verbose=config['verbose'], mode='min' if 'loss' in monitor_metric.lower() or 'mse' in monitor_metric.lower() or 'mae' in monitor_metric.lower() else 'max'))
    callbacks.append(ReduceLROnPlateau(monitor=monitor_metric, factor=config['reduce_lr_factor'], patience=config['reduce_lr_patience'], min_lr=1e-6, verbose=config['verbose'], mode='min' if 'loss' in monitor_metric.lower() or 'mse' in monitor_metric.lower() or 'mae' in monitor_metric.lower() else 'max'))
    # ModelCheckpoint 也可以选择监控训练集指标
    callbacks.append(ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_metric, save_best_only=True, save_weights_only=False, mode='min' if 'loss' in monitor_metric.lower() or 'mse' in monitor_metric.lower() or 'mae' in monitor_metric.lower() else 'max', verbose=config['verbose']))

    # 如果移除了所有依赖验证集的 callback，则打印警告 (之前的逻辑已经覆盖了大部分情况)
    if validation_data is None and any(hasattr(cb, 'monitor') and cb.monitor.startswith('val_') for cb in callbacks): # 检查 cb 是否有 monitor 属性
         logger.warning("在移除依赖验证集的 callback 逻辑后，仍有 callback 监控 'val_*' 指标。请检查 monitor_metric 配置。")

    history = model.fit(
        train_sequence, # 使用训练集 Sequence
        validation_data=validation_data, # 使用验证集 Sequence 或 None
        epochs=config['epochs'],
        # batch_size 参数在 Sequence 中定义，这里不再需要
        # batch_size=config['batch_size'],
        sample_weight=sample_weight,  # 使用修正后的 sample_weight (None)
        callbacks=callbacks,
        verbose=config['verbose'],
        # workers 和 use_multiprocessing 参数可以根据需要添加，用于 Sequence 的并行加载
        # workers=os.cpu_count(), # 根据实际情况调整
        # use_multiprocessing=True # 根据实际情况调整
    )

    # 为测试集评估创建 TimeSeriesSequence 
    # 在测试集上评估 (使用缩放后的目标值)
    if X_test.shape[0] > 0 and y_test.shape[0] > 0:
        try:
            window_size_for_test = train_sequence.window_size # 从训练序列获取窗口大小
            # 测试评估的批大小可以与训练时不同，通常可以设大一些以加速，或与训练一致
            batch_size_for_test = config.get('batch_size', train_sequence.batch_size)

            # 确保测试数据足够形成至少一个窗口
            # TimeSeriesSequence 内部会检查 features.shape[0] < window_size + 1
            # 我们在这里也做一个初步判断
            if X_test.shape[0] >= window_size_for_test + 1:
                test_sequence_for_evaluation = TimeSeriesSequence(
                    features=X_test,
                    targets=y_test,
                    window_size=window_size_for_test,
                    batch_size=batch_size_for_test,
                    shuffle=False # 评估时不需要打乱
                )

                if len(test_sequence_for_evaluation) > 0:
                    logger.info(f"使用 TimeSeriesSequence 评估测试集，批次数: {len(test_sequence_for_evaluation)}。")
                    test_results = model.evaluate(test_sequence_for_evaluation, verbose=0)
                else:
                    logger.warning("创建的测试集 Sequence 为空 (len=0)，无法通过 Sequence 评估。可能是数据量不足以形成有效窗口。")
                    # 根据模型编译的度量数量，填充 NaN 结果
                    num_output_metrics = 1 # 至少有 loss
                    if hasattr(model, 'metrics_names') and model.metrics_names is not None:
                        num_output_metrics = len(model.metrics_names)
                    test_results = [np.nan] * num_output_metrics
            else:
                logger.warning(f"测试集数据量 ({X_test.shape[0]}) 不足以根据窗口大小 ({window_size_for_test}) 创建 Sequence 进行评估。")
                num_output_metrics = 1 # 至少有 loss
                if hasattr(model, 'metrics_names') and model.metrics_names is not None:
                    num_output_metrics = len(model.metrics_names)
                test_results = [np.nan] * num_output_metrics

            # evaluate 返回的是一个列表，第一个是 loss，后面是 metrics
            test_loss = test_results[0]
            # 找到 MAE 的索引 (假设 'mae' 在 metrics 列表中)
            test_mae_scaled = np.nan # 默认 NaN
            try:
                # model.metrics_names 是一个列表，包含 loss 和 metrics 的名称
                # 例如: ['loss', 'mae']
                if hasattr(model, 'metrics_names') and model.metrics_names is not None and 'mae' in model.metrics_names:
                    mae_index = model.metrics_names.index('mae') # 获取 'mae' 的实际索引
                    if len(test_results) > mae_index : # 确保 test_results 长度足够
                        test_mae_scaled = test_results[mae_index]
                    else:
                        logger.warning(f"test_results 长度 ({len(test_results)}) 不足以获取索引为 {mae_index} 的 MAE。")
                else:
                     logger.warning("模型编译的 metrics 中没有 'mae'，或者 model.metrics_names 不可用。")
            except ValueError: # 'mae' not in model.metrics_names
                 logger.warning("模型编译的 metrics 中没有 'mae' (ValueError on index)。")
            except Exception as e:
                 logger.warning(f"获取测试集 scaled MAE 时出错: {e}", exc_info=True)


            # 将 MAE 转换回原始范围 (仅作估算)
            mae_original_approx = np.nan # 默认 NaN
            # 只有当 target_scaler 存在且已 fit 且 scaled MAE 有效时才进行逆缩放估算
            # 检查 target_scaler 是否已 fit 的方法：查看其 n_features_in_ 属性或 feature_range 属性 (MinMaxScaler)
            is_scaler_fitted = target_scaler is not None and (
                (hasattr(target_scaler, 'n_features_in_') and target_scaler.n_features_in_ is not None) or # StandardScaler, MinMaxScaler (Keras 3)
                (isinstance(target_scaler, MinMaxScaler) and hasattr(target_scaler, 'data_min_') and target_scaler.data_min_ is not None) # Scikit-learn MinMaxScaler
            )

            if is_scaler_fitted and not np.isnan(test_mae_scaled):
                 try:
                     # 估算方法：计算缩放器将 0 和 scaled_mae 转换回原始尺度的差值
                     # 注意：这里假设 MAE 是非负的，且缩放器是线性的
                     # 创建一个包含 0 和 scaled_mae 的二维数组用于逆缩放
                     # 确保 dummy_values 的形状是 (n_samples, n_features=1)
                     dummy_values_for_inverse = np.array([[0.0], [test_mae_scaled]])
                     original_values = target_scaler.inverse_transform(dummy_values_for_inverse)
                     mae_original_approx = abs(original_values[1, 0] - original_values[0, 0]) # abs(val_at_mae - val_at_0)
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
    final_loss = history.history['loss'][-1] if 'loss' in history.history and history.history['loss'] else 'N/A'
    # 检查 val_loss 是否存在，只有存在时才打印
    final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history and history.history['val_loss'] else ('N/A (验证集为空或历史记录缺失)' if validation_data is None else 'N/A (历史记录缺失)')
    # 检查 mae 是否存在
    final_mae = history.history['mae'][-1] if 'mae' in history.history and history.history['mae'] else 'N/A (指标缺失)'
     # 检查 val_mae 是否存在
    final_val_mae = history.history['val_mae'][-1] if 'val_mae' in history.history and history.history['val_mae'] else ('N/A (验证集为空或历史记录缺失)' if validation_data is None else 'N/A (历史记录缺失)')

    logger.info(f"训练历史: 最终损失={final_loss}, 最终验证损失={final_val_loss}, 最终MAE={final_mae}, 最终验证MAE={final_val_mae}")

    # 可选：绘制训练历史图
    if plot_training_history:
         try:
             plt.figure(figsize=(12, 6))
             if 'loss' in history.history and history.history['loss']:
                 plt.plot(history.history['loss'], label='训练集损失')
             # 检查是否存在验证集损失
             if 'val_loss' in history.history and history.history['val_loss']:
                 plt.plot(history.history['val_loss'], label='验证集损失')
             plt.title('模型损失')
             plt.xlabel('周期')
             plt.ylabel('损失')
             plt.legend()
             plt.grid(True)
             # 可以考虑保存图像而不是显示
             # plt.savefig(os.path.join(os.path.dirname(checkpoint_path), "loss_history.png"))
             # plt.close()

             # 检查 metrics 中是否有 mae
             if 'mae' in history.history and history.history['mae']:
                plt.figure(figsize=(12, 6))
                plt.plot(history.history['mae'], label='训练集MAE')
                # 检查是否存在验证集MAE
                if 'val_mae' in history.history and history.history['val_mae']:
                    plt.plot(history.history['val_mae'], label='验证集MAE')
                plt.title('模型平均绝对误差 (MAE)')
                plt.xlabel('周期')
                plt.ylabel('MAE')
                plt.legend()
                plt.grid(True)
                # plt.savefig(os.path.join(os.path.dirname(checkpoint_path), "mae_history.png"))
                # plt.close()
             else:
                 logger.warning("训练历史中没有MAE指标，跳过绘制MAE图。请检查model.compile的metrics参数。")
             # 如果在非GUI环境，可能需要 plt.show(block=False) 或直接保存文件
             # plt.show() # 在服务器环境或无GUI环境运行时，这可能会导致问题或无效果

         except Exception as e:
             logger.error(f"绘制训练历史图时出错: {e}", exc_info=True)

    return history.history # 返回原始 history.history 字典

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
            {'units': 64, 'return_sequences': True, 'dropout': 0.2, 'l2_reg': 0.001},
            {'units': 32, 'return_sequences': False, 'dropout': 0.2, 'l2_reg': 0.001}
        ],
        'dense_layers': [{'units': 16, 'activation': 'relu', 'dropout': 0.2, 'l2_reg': 0.001}],
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'mse',
        'metrics': ['mae'],
        'output_units': 1,
        'output_activation': None # 回归任务通常是线性激活 (None)
    }
    config = default_config.copy()
    if model_config:
        # 深度更新字典，特别是 layers 和 dense_layers 列表中的字典
        for key, value in model_config.items():
            if key in ['layers', 'dense_layers'] and isinstance(value, list) and isinstance(config.get(key), list):
                # 简单列表替换，不进行深度合并，调用方需提供完整的列表
                config[key] = value
            elif isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value

    # 检查输入特征数量
    if num_features <= 0:
        logger.error(f"输入特征数量无效: {num_features}。无法构建模型。")
        raise ValueError(f"输入特征数量无效: {num_features}")

    model = Sequential()

    # 添加循环层 (LSTM, Bidirectional LSTM, 或 GRU)
    for i, layer_config in enumerate(config['layers']):
        return_sequences = layer_config.get('return_sequences', False)
        units = layer_config.get('units', 32)
        dropout = layer_config.get('dropout', 0.0)
        l2_reg = layer_config.get('l2_reg', 0.0)
        activation = layer_config.get('activation', 'tanh') # 循环层默认激活函数

        # 只有第一个循环层需要指定 batch_input_shape
        batch_input_shape_param = (None, window_size, num_features) if i == 0 else None

        # 构建层参数字典
        layer_args = {
            'units': units,
            'return_sequences': return_sequences,
            'dropout': dropout,
            'recurrent_dropout': layer_config.get('recurrent_dropout', 0.0), # 可选的循环层 dropout
            'kernel_regularizer': l2(l2_reg),
            'recurrent_regularizer': l2(layer_config.get('recurrent_l2_reg', 0.0)), # 可选的循环层 L2 正则化
            'bias_regularizer': l2(layer_config.get('bias_l2_reg', 0.0)), # 可选的偏置 L2 正则化
            'activity_regularizer': l2(layer_config.get('activity_l2_reg', 0.0)), # 可选的激活 L2 正则化
            'activation': activation,
            'recurrent_activation': layer_config.get('recurrent_activation', 'sigmoid'), # 循环激活函数
            'unroll': layer_config.get('unroll', False) # 是否展开网络
        }

        # *** 修改点：只为第一个循环层指定 batch_input_shape ***
        if i == 0:
            layer_args['batch_input_shape'] = (None, window_size, num_features)

        if model_type.lower() == 'lstm':
            model.add(LSTM(**layer_args)) # 使用字典解包传递参数
        elif model_type.lower() == 'bilstm':
             # Bidirectional 层本身也接受 batch_input_shape 参数，并传递给内部层
             bidi_layer_args = {}
             if i == 0:
                 bidi_layer_args['batch_input_shape'] = (None, window_size, num_features)
             model.add(Bidirectional(LSTM(**layer_args), **bidi_layer_args)) # 将 batch_input_shape 传递给 Bidirectional
        elif model_type.lower() == 'gru':
            model.add(GRU(**layer_args)) # 使用字典解包传递参数
        else:
            logger.warning(f"不支持的模型类型: {model_type}，使用默认 LSTM。")
            model.add(LSTM(**layer_args)) # 使用字典解包传递参数


    # 添加全连接层
    for layer_config in config['dense_layers']:
        units = layer_config.get('units', 16)
        activation = layer_config.get('activation', 'relu')
        dropout = layer_config.get('dropout', 0.0)
        l2_reg = layer_config.get('l2_reg', 0.0)
        model.add(Dense(units=units, activation=activation, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(layer_config.get('bias_l2_reg', 0.0)), activity_regularizer=l2(layer_config.get('activity_l2_reg', 0.0))))
        if dropout > 0:
            model.add(Dropout(dropout))

    # 输出层
    output_units = config.get('output_units', 1)
    output_activation = config.get('output_activation', None) # 回归通常是 None (线性)
    model.add(Dense(units=output_units, activation=output_activation))

    # 编译模型
    optimizer_type = config.get('optimizer', 'adam').lower()
    learning_rate = config.get('learning_rate', 0.001)

    if optimizer_type == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    else:
        logger.warning(f"不支持的优化器类型: {optimizer_type}，使用默认 Adam。")
        optimizer = Adam(learning_rate=learning_rate)

    loss_function = config.get('loss', 'mse')
    metrics_list = config.get('metrics', ['mae'])

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)

    if summary:
        model.summary(print_fn=logger.info) # 使用 logger 打印 summary

    logger.info(f"模型构建完成: 类型={model_type}, 输入形状=({window_size}, {num_features}), 输出形状=({output_units},)")

    return model

