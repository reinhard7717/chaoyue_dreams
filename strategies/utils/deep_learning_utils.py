# apps/strategies/utils/deep_learning_utils.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split # 不再需要，因为我们按时间顺序分割
from sklearn.feature_selection import SelectFromModel # 导入 SelectFromModel
from sklearn.ensemble import RandomForestRegressor # 导入随机森林回归器
try:
    import xgboost as xgb # 尝试导入 xgboost
except ImportError:
    xgb = None # 如果未安装，则设为 None
    # logger.warning("XGBoost 未安装，如果选择使用 XGBoost 进行特征选择将会失败。请运行 'pip install xgboost'") # 日志记录器可能尚未初始化，暂时注释
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Tuple, List, Dict, Optional, Union, Callable
from functools import wraps
import joblib # 用于加载/保存 scaler

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
            raise # 重新抛出异常以便调用者处理
    return wrapper

@log_execution_time
@handle_exceptions
def prepare_data_for_lstm(
    data: pd.DataFrame,
    required_columns: List[str], # 仍然需要知道原始可能需要的列
    target_column: str = 'final_signal',
    window_size: int = 60,
    scaler_type: str = 'minmax', # 特征缩放器类型
    train_split: float = 0.7,
    val_split: float = 0.15,
    apply_variance_threshold: bool = False, # 是否应用方差阈值进行特征选择 (在模型选择前)
    variance_threshold_value: float = 0.01,
    # --- 特征选择参数 ---
    use_feature_selection: bool = True, # 是否启用基于模型的特征选择 (默认启用)
    feature_selector_model: str = 'rf', # 选择器模型: 'rf' (随机森林) 或 'xgb' (XGBoost)
    max_features_fs: Optional[int] = 50, # 选择最重要的特征数量 (例如选择前 50 个)；设为 None 则使用阈值
    feature_selection_threshold: str = 'median', # 如果 max_features_fs 为 None, 则使用此阈值 ('median', 'mean', 或浮点数)
    # --- PCA 参数 (保留但默认关闭) ---
    use_pca: bool = False, # 默认关闭 PCA
    n_components: Union[int, float] = 0.99,
    target_scaler_type: str = 'minmax' # 目标变量缩放器类型
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler], Union[MinMaxScaler, StandardScaler]]:
    """
    准备用于LSTM训练的时间序列数据，包括特征处理(方差阈值、基于模型选择、PCA)、缩放、窗口化和数据集分割。

    处理流程:
    1. 检查目标列。
    2. 筛选初始特征列 (基于 required_columns 并排除目标列)。
    3. 处理 NaN 值 (前向填充后向填充，剩余 NaN 填充 0)。
    4. 应用方差阈值过滤 (可选)。
    5. 应用 PCA 降维 (可选，如果启用则忽略基于模型的特征选择)。
    6. 应用基于模型的特征选择 (可选，如果 PCA 未启用)。
       - 在临时缩放后的特征和原始目标上拟合选择器。
       - 从原始尺度的特征中提取选中的列。
    7. 对处理后的特征和原始目标变量进行最终缩放 (在整个数据集上拟合和转换)。
    8. 构建时间序列窗口 (在缩放后的数据上)。
    9. 按时间顺序分割窗口化后的数据集为训练集、验证集和测试集。

    Args:
        data (pd.DataFrame): 包含所有特征和目标列的原始DataFrame。
        required_columns (List[str]): 数据中应包含的原始特征列列表（用于初始筛选）。
        target_column (str): 目标变量列名。
        window_size (int): LSTM输入的时间步长。
        scaler_type (str): 特征缩放器类型 ('minmax' 或 'standard')。
        train_split (float): 训练集比例。
        val_split (float): 验证集比例。
        apply_variance_threshold (bool): 是否应用方差阈值进行特征选择 (在模型选择前)。
        variance_threshold_value (float): 方差阈值。
        use_feature_selection (bool): 是否启用基于模型的特征选择。
        feature_selector_model (str): 特征选择模型 ('rf' 或 'xgb')。
        max_features_fs (Optional[int]): 要选择的最大特征数量。如果为 None，则使用 threshold。
        feature_selection_threshold (str): 特征重要性阈值 (如果 max_features_fs is None)。
        use_pca (bool): 是否应用PCA进行降维 (如果为 True，则忽略 use_feature_selection)。
        n_components (Union[int, float]): PCA保留的主成分数量或解释方差比例。
        target_scaler_type (str): 目标变量缩放器类型 ('minmax' 或 'standard')。

    Returns:
        Tuple containing:
        X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray): 分割和处理后的数据集。
        feature_scaler (Union[MinMaxScaler, StandardScaler]): 用于最终特征的缩放器 (在整个处理后的特征集上拟合)。
        target_scaler (Union[MinMaxScaler, StandardScaler]): 用于目标变量的缩放器 (在整个原始目标集上拟合)。
    """
    logger.info(f"开始准备 LSTM 数据...")
    logger.info(f"参数: window_size={window_size}, scaler_type='{scaler_type}', train_split={train_split}, val_split={val_split}, apply_variance_threshold={apply_variance_threshold}, variance_threshold_value={variance_threshold_value}, use_feature_selection={use_feature_selection}, feature_selector_model='{feature_selector_model}', max_features_fs={max_features_fs}, feature_selection_threshold='{feature_selection_threshold}', use_pca={use_pca}, n_components={n_components}, target_scaler_type='{target_scaler_type}'")

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

    # --- 3. 处理 NaN 值 (在特征选择/PCA之前填充) ---
    # 使用前向填充然后后向填充，尽量保留数据
    data_filled = data.ffill().bfill()
    # 检查填充后是否仍有 NaN (如果整个列都是 NaN)
    nan_cols_after_ffill = data_filled[initial_feature_columns].isnull().sum()
    nan_cols_after_ffill = nan_cols_after_ffill[nan_cols_after_ffill > 0]
    if not nan_cols_after_ffill.empty:
        logger.warning(f"填充后以下特征列仍包含 NaN 值 (可能整列为空)，将尝试填充为 0: {nan_cols_after_ffill.index.tolist()}")
        # 对于仍然是 NaN 的列，填充为 0 (或者可以选择删除这些列)
        for col in nan_cols_after_ffill.index:
            data_filled[col].fillna(0, inplace=True)

    # 提取处理后的特征和目标变量
    features_processed = data_filled.loc[:, initial_feature_columns].values
    targets_original = data_filled.loc[:, target_column].values # 使用填充后的目标值
    current_feature_columns = initial_feature_columns[:] # 创建副本，用于跟踪列名

    logger.info(f"初始特征维度 (处理NaN后): {features_processed.shape[1]}")

    # --- 4. (可选) 方差阈值过滤 ---
    if apply_variance_threshold:
        if features_processed.shape[0] > 1 and features_processed.shape[1] > 0 and np.var(features_processed, axis=0).max() > 1e-9: # 检查是否有非零方差
            try:
                selector_var = VarianceThreshold(threshold=variance_threshold_value)
                features_after_var = selector_var.fit_transform(features_processed)
                selected_indices_var = selector_var.get_support(indices=True)
                current_feature_columns = [current_feature_columns[i] for i in selected_indices_var]
                features_processed = features_after_var # 更新特征矩阵
                logger.info(f"方差阈值 ({variance_threshold_value}) 选择后维度: {features_processed.shape[1]}")
                if features_processed.shape[1] == 0:
                    logger.error("方差阈值选择后没有剩余特征。请检查阈值或数据。")
                    raise ValueError("方差阈值选择后没有剩余特征。")
            except Exception as e:
                 logger.error(f"应用方差阈值时出错: {e}", exc_info=True)
                 logger.warning("方差阈值选择失败，将使用之前的特征。")
        else:
             logger.warning("特征方差过低或样本不足，跳过方差阈值选择。")

    # --- 5. (可选) PCA 降维 ---
    # 优先使用 PCA
    if use_pca:
        if features_processed.shape[1] > 1 and features_processed.shape[0] > features_processed.shape[1]:
            logger.info(f"启用 PCA 降维，n_components={n_components}")
            # PCA 前最好先标准化数据 (临时 scaler)
            scaler_pca = StandardScaler()
            features_scaled_pca = scaler_pca.fit_transform(features_processed)
            pca = PCA(n_components=n_components)
            try:
                features_processed = pca.fit_transform(features_scaled_pca) # PCA 在标准化数据上进行
                num_features = pca.n_components_
                logger.info(f"PCA降维完成，保留 {num_features} 个主成分，解释方差比: {sum(pca.explained_variance_ratio_):.4f}")
                current_feature_columns = [f"pca_{i}" for i in range(num_features)] # PCA后列名丢失意义
                if num_features == 0:
                     logger.error("PCA 降维后特征维度为零。请检查 n_components 或数据。")
                     raise ValueError("PCA 降维后特征维度为零。")
            except Exception as e:
                 logger.error(f"应用 PCA 时出错: {e}", exc_info=True)
                 logger.warning("PCA 降维失败，将使用之前的特征。")
                 use_pca = False # 禁用 PCA 标志，可能继续进行特征选择

        else:
            logger.warning(f"特征维度 ({features_processed.shape[1]}) 或样本数 ({features_processed.shape[0]}) 不足，跳过PCA降维。")
            use_pca = False # 禁用 PCA 标志

    # --- 6. (可选) 基于模型的特征选择 ---
    # 仅在 PCA 未启用时执行
    if use_feature_selection and not use_pca:
        if features_processed.shape[1] <= 1:
             logger.warning(f"特征维度 ({features_processed.shape[1]}) 过低，跳过基于模型的特征选择。")
        else:
            logger.info(f"启用基于模型 '{feature_selector_model}' 的特征选择。")
            # 特征选择前需要缩放特征 (使用临时 scaler)，在整个数据集上拟合
            scaler_fs = MinMaxScaler() # 使用 MinMax 对特征重要性模型通常更友好
            features_scaled_fs = scaler_fs.fit_transform(features_processed)

            # 选择模型
            if feature_selector_model.lower() == 'rf':
                selector_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif feature_selector_model.lower() == 'xgb':
                if xgb is None:
                    logger.error("XGBoost 未安装，无法使用 XGBoost 进行特征选择。将回退到 RandomForest。")
                    selector_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    selector_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
            else:
                logger.warning(f"不支持的特征选择模型: {feature_selector_model}，将使用 RandomForest。")
                selector_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            # 使用 SelectFromModel
            try:
                # 在缩放后的特征和原始目标上拟合选择器
                selector_model.fit(features_scaled_fs, targets_original)

                if max_features_fs is not None and max_features_fs > 0:
                     # 根据数量选择
                     # max_features 不能超过当前特征数量
                     actual_max_features = min(max_features_fs, features_processed.shape[1])
                     selector = SelectFromModel(selector_model, max_features=actual_max_features, threshold=-np.inf, prefit=True) # prefit=True 因为已经手动fit了
                     logger.info(f"使用 {feature_selector_model} 选择最重要的 {actual_max_features} 个特征。")
                else:
                     # 根据阈值选择
                     selector = SelectFromModel(selector_model, threshold=feature_selection_threshold, prefit=True) # prefit=True
                     logger.info(f"使用 {feature_selector_model} 和阈值 '{feature_selection_threshold}' 选择特征。")

                # 获取选中的特征索引
                selected_indices = selector.get_support(indices=True)

                if len(selected_indices) == 0:
                    logger.error("基于模型选择后没有剩余特征。请检查模型、阈值或数据。")
                    raise ValueError("基于模型选择后没有剩余特征。")

                # 从 *处理前* 的特征矩阵中提取选中的列 (保持原始尺度)
                # features_processed 在此步骤之前已经经过了 NaN 填充和可选的方差过滤
                features_processed = features_processed[:, selected_indices]
                current_feature_columns = [current_feature_columns[i] for i in selected_indices] # 更新选中的列名
                num_features = features_processed.shape[1]
                logger.info(f"基于模型选择后维度: {num_features}")

            except Exception as e:
                 logger.error(f"应用基于模型的特征选择时出错: {e}", exc_info=True)
                 logger.warning("特征选择失败，将使用之前的特征进行后续处理。")
                 # 如果特征选择失败，features_processed 保持为方差过滤后的结果 (或原始结果)
                 # current_feature_columns 也保持为方差过滤后的列名 (或原始列名)


    # 确定最终特征数量
    num_features = features_processed.shape[1]
    if num_features == 0:
         logger.error("经过所有预处理步骤后，特征维度为零。无法继续。")
         raise ValueError("特征维度为零。")

    logger.info(f"最终用于缩放和窗口化的特征维度: {num_features}")
    logger.info(f"最终用于缩放和窗口化的特征列名 (前10个): {current_feature_columns[:10]}...")


    # --- 7. 对处理后的特征和原始目标变量进行最终缩放 ---
    # 特征缩放 (在整个处理后的特征集上拟合和转换)
    if scaler_type.lower() == 'minmax':
        feature_scaler = MinMaxScaler()
    elif scaler_type.lower() == 'standard':
        feature_scaler = StandardScaler()
    else:
        logger.warning(f"不支持的特征缩放器类型: {scaler_type}，使用默认MinMaxScaler。")
        feature_scaler = MinMaxScaler()

    # 确保 features_processed 不是空的
    if features_processed.shape[0] > 0 and features_processed.shape[1] > 0:
         features_scaled = feature_scaler.fit_transform(features_processed)
         logger.info(f"最终特征缩放完成 (使用 {scaler_type} scaler)。")
    else:
         logger.warning("处理后的特征数据为空，跳过最终特征缩放。")
         features_scaled = features_processed # 保持原样 (空数组)


    # 目标变量缩放 (在整个原始目标集上拟合和转换)
    if target_scaler_type.lower() == 'minmax':
        target_scaler = MinMaxScaler()
    elif target_scaler_type.lower() == 'standard':
        target_scaler = StandardScaler()
    else:
        logger.warning(f"不支持的目标变量缩放器类型: {target_scaler_type}，使用默认MinMaxScaler。")
        target_scaler = MinMaxScaler()

    # 确保 targets_original 不是空的
    if targets_original.shape[0] > 0:
        # 目标变量 targets_original 是一个一维数组，需要 reshape 成二维才能 fit scaler
        targets_scaled = target_scaler.fit_transform(targets_original.reshape(-1, 1)).flatten()
        logger.info(f"目标变量缩放完成 (使用 {target_scaler_type} scaler)。")
    else:
        logger.warning("原始目标变量数据为空，跳过目标变量缩放。")
        targets_scaled = targets_original # 保持原样 (空数组)


    # --- 8. 构建时间序列窗口 (在缩放后的数据上) ---
    # 确保缩放后的数据长度足够构建至少一个窗口
    if len(features_scaled) < window_size + 1: # 需要 window_size 个输入 + 1 个输出
        logger.error(f"缩放后数据长度 {len(features_scaled)} 不足以构建窗口 (window_size={window_size})。至少需要 {window_size + 1} 条数据。")
        # 返回空的数组和 scaler (scaler 已经 fit 了，可以返回)
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), feature_scaler, target_scaler


    X_windowed_all, y_windowed_all = [], []
    # 构建所有可能的窗口
    for i in range(len(features_scaled) - window_size):
        X_windowed_all.append(features_scaled[i:(i + window_size)]) # 使用缩放后的特征
        y_windowed_all.append(targets_scaled[i + window_size]) # 预测窗口后的一个点 (使用缩放后的目标)

    X_windowed_all = np.array(X_windowed_all)
    y_windowed_all = np.array(y_windowed_all)

    if X_windowed_all.shape[0] == 0 or y_windowed_all.shape[0] == 0:
        logger.error("窗口化后数据量为零，请检查数据长度和窗口大小。")
        # 返回空的数组和 scaler
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), feature_scaler, target_scaler

    logger.info(f"窗口化完成，X_windowed_all 形状: {X_windowed_all.shape}, y_windowed_all 形状: {y_windowed_all.shape}")

    # --- 9. 分割数据集 (按时间顺序) ---
    n_samples_windowed = X_windowed_all.shape[0]
    if train_split + val_split > 1.0:
         logger.error("训练集和验证集比例之和必须小于等于1。")
         raise ValueError("分割比例错误。")

    n_train = int(n_samples_windowed * train_split)
    n_val = int(n_samples_windowed * val_split)
    n_test = n_samples_windowed - n_train - n_val

    # 确保各数据集至少有一个样本 (如果比例允许)
    if n_train == 0:
        logger.error("训练集样本数为零，请检查 train_split 和窗口化后的数据长度。")
        # 返回空的数组和 scaler
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), feature_scaler, target_scaler

    # 验证集和测试集可以为空，但给出警告
    if n_val == 0 and val_split > 1e-9: logger.warning("验证集样本数为零。")
    if n_test == 0 and (1.0 - train_split - val_split) > 1e-9 : logger.warning("测试集样本数为零。")


    X_train, y_train = X_windowed_all[:n_train], y_windowed_all[:n_train]
    X_val, y_val = X_windowed_all[n_train : n_train + n_val], y_windowed_all[n_train : n_train + n_val]
    X_test, y_test = X_windowed_all[n_train + n_val:], y_windowed_all[n_train + n_val:]

    logger.info(f"数据分割完成 (按时间顺序)，训练集: {X_train.shape[0]} 条，验证集: {X_val.shape[0]} 条，测试集: {X_test.shape[0]} 条")

    # 返回最终处理好的数据和在整个数据集上拟合的 scaler
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler

@log_execution_time
@handle_exceptions
def train_lstm_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, model: Sequential,
    target_scaler: Union[MinMaxScaler, StandardScaler], # 传入目标变量缩放器
    training_config: Dict[str, Any] = None, checkpoint_path: str = "models/checkpoints/best_model.keras",
    plot_training_history: bool = False # 是否绘制训练历史图
) -> Dict:
    """
    训练LSTM模型。

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray): 已准备好的数据集 (y已缩放)。
        model (Sequential): 已编译的Keras模型。
        target_scaler (Union[MinMaxScaler, StandardScaler]): 用于目标变量的缩放器。
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
        'monitor_metric': 'val_loss', # 对于回归，通常监控loss或MAE
        'verbose': 1 # 默认显示进度条，方便观察
    }
    config = default_config.copy() # 使用copy，避免修改default_config
    if training_config:
        config.update(training_config)

    logger.info(f"开始训练模型，训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}, 测试集样本数: {X_test.shape[0]}")
    logger.info(f"训练配置: {config}")


    # 检查目标变量缩放后的范围 (仅作提醒，实际缩放已在prepare_data_for_lstm完成)
    if isinstance(target_scaler, MinMaxScaler):
        logger.info(f"目标变量 MinMaxScaler range: {target_scaler.feature_range}")
    # else: StandardScaler 的范围没有特定保证

    # 回归任务，通常不需要类别权重
    sample_weight = None

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"创建检查点目录: {checkpoint_dir}")

    callbacks = [
        # monitor Metric应与model.compile中的metrics或loss一致
        # mode 应根据 monitor_metric 是 loss (min) 还是 metric (max) 来设置
        EarlyStopping(monitor=config['monitor_metric'], patience=config['early_stopping_patience'], restore_best_weights=True, verbose=config['verbose'], mode='min' if 'loss' in config['monitor_metric'].lower() else 'max'),
        ReduceLROnPlateau(monitor=config['monitor_metric'], factor=config['reduce_lr_factor'], patience=config['reduce_lr_patience'], min_lr=1e-6, verbose=config['verbose'], mode='min' if 'loss' in config['monitor_metric'].lower() else 'max'),
        ModelCheckpoint(filepath=checkpoint_path, monitor=config['monitor_metric'], save_best_only=True, save_weights_only=False, mode='min' if 'loss' in config['monitor_metric'].lower() else 'max', verbose=config['verbose'])
    ]

    # 检查验证集是否存在，如果不存在则移除依赖验证集的 callback
    validation_data = (X_val, y_val) if X_val.shape[0] > 0 else None
    if validation_data is None:
        logger.warning("验证集为空，将不使用依赖验证集的 EarlyStopping 和 ReduceLROnPlateau。")
        # 过滤掉 monitor 以 'val_' 开头的 callback
        callbacks = [cb for cb in callbacks if not cb.monitor.startswith('val_')]
        # 如果 ModelCheckpoint 监控的是 val_loss/val_metric，也需要调整
        if callbacks and isinstance(callbacks[-1], ModelCheckpoint) and callbacks[-1].monitor.startswith('val_'):
             logger.warning("ModelCheckpoint 监控指标为 val_*，但验证集为空，将移除 ModelCheckpoint。")
             callbacks = callbacks[:-1] # 移除最后一个 ModelCheckpoint

        # 如果移除了所有 callback，则打印警告
        if not callbacks:
             logger.warning("没有可用的训练回调函数。")


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
    if X_test.shape[0] > 0:
        try:
            test_results = model.evaluate(X_test, y_test, verbose=0)
            # evaluate 返回的是一个列表，第一个是 loss，后面是 metrics
            test_loss = test_results[0]
            # 找到 MAE 的索引 (假设 'mae' 在 metrics 列表中)
            mae_index = model.metrics_names.index('mae') if 'mae' in model.metrics_names else None
            test_mae_scaled = test_results[mae_index] if mae_index is not None else np.nan

            # 将 MAE 转换回原始范围
            mae_original_approx = np.nan # 默认 NaN
            try:
                # 估算方法：计算缩放器将 0 和 scaled_mae 转换回原始尺度的差值
                # 需要确保 target_scaler 已经被 fit 并且 test_mae_scaled 不是 NaN
                if test_mae_scaled is not np.nan and hasattr(target_scaler, 'inverse_transform'):
                     # 创建一个包含 0 和 scaled_mae 的二维数组用于逆缩放
                     # 注意：这里假设 MAE 是非负的，且缩放器是线性的
                     dummy_values = np.array([[0.0], [test_mae_scaled]])
                     original_values = target_scaler.inverse_transform(dummy_values)
                     mae_original_approx = abs(original_values[1][0] - original_values[0][0])
                else:
                     logger.warning("目标变量缩放器未 fit 或类型不支持，或 scaled MAE 为 NaN，无法估算原始范围 MAE。")

            except Exception as e:
                logger.error(f"估算原始范围 MAE 时出错: {e}", exc_info=True)
                mae_original_approx = np.nan # 发生错误时设置为 NaN

            logger.info(f"LSTM模型在测试集上的损失: {test_loss:.4f}, MAE (缩放后): {test_mae_scaled:.4f}, MAE (原始范围估算): {mae_original_approx:.4f}")
        except Exception as e:
             logger.error(f"评估测试集时出错: {e}", exc_info=True)
             logger.warning("测试集评估失败。")

    else:
        logger.warning("测试集为空，无法评估LSTM模型。")

    logger.info("模型训练完成。")
    # 打印最终的训练和验证损失/指标
    final_loss = history.history['loss'][-1] if 'loss' in history.history else 'N/A'
    final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 'N/A'
    final_mae = history.history['mae'][-1] if 'mae' in history.history else 'N/A'
    final_val_mae = history.history['val_mae'][-1] if 'val_mae' in history.history else 'N/A'
    logger.info(f"训练历史: 最终损失={final_loss}, 最终验证损失={final_val_loss}, 最终MAE={final_mae}, 最终验证MAE={final_val_mae}")


    # 可选：绘制训练历史图
    if plot_training_history:
         try:
             plt.figure(figsize=(12, 6))
             plt.plot(history.history['loss'], label='训练集损失')
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
             plt.savefig('training_history.png')
             plt.close('all') # 关闭所有图，避免内存泄露
             logger.info("训练历史图已保存至 training_history.png")
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
    默认配置为回归任务。如需分类，需要修改这里的参数或在调用时覆盖。

    Args:
        window_size (int): 输入时间步长。
        num_features (int): 输入特征维度。
        model_config (Dict): 模型配置字典，可以覆盖默认层、优化器、损失函数等。
        model_type (str): 模型类型 ('lstm', 'bilstm', 'gru')。
        summary (bool): 是否打印模型摘要。

    Returns:
        Sequential: 已编译的Keras模型。
    """
    # 默认配置 (回归任务)
    default_config = {
        'layers': [
            {'units': 64, 'return_sequences': True, 'dropout': 0.3, 'l2_reg': 0.01},
            {'units': 32, 'return_sequences': False, 'dropout': 0.3, 'l2_reg': 0.01}
        ],
        'dense_layers': [{'units': 16, 'dropout': 0.2, 'l2_reg': 0.01}],
        'optimizer': 'adam',
        'learning_rate': 0.001, # 默认学习率
        'loss': 'mse',
        'metrics': ['mae'],
        'output_units': 1, # 回归任务输出单元数
        'output_activation': None # 回归任务输出层无激活函数
    }
    config = default_config.copy() # 使用copy，避免修改default_config
    if model_config:
        # 深度更新字典，特别是嵌套的 layers 和 dense_layers
        for key, value in model_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            elif isinstance(value, list) and key in config and isinstance(config[key], list):
                 # 对于列表，直接替换
                 config[key] = value
            else:
                config[key] = value

    # 构建模型
    model = Sequential()
    model_type_lower = model_type.lower()

    if model_type_lower == 'lstm':
        layer_class = LSTM
    elif model_type_lower == 'bilstm':
        # Bidirectional Wrapper handles input_shape in the first layer
        # 注意：Bidirectional Wrapper 需要一个 RNN 层作为参数
        def bilstm_layer(units, return_sequences=False, **kwargs):
             # 从 kwargs 中移除 activation, kernel_regularizer, recurrent_regularizer 等，这些应该传给内部的 LSTM
             lstm_kwargs = {k: v for k, v in kwargs.items() if k not in ['activation']} # activation 应该在 LSTM 层内部
             return Bidirectional(LSTM(units, return_sequences=return_sequences, **lstm_kwargs))
        layer_class = bilstm_layer
    elif model_type_lower == 'gru':
        layer_class = GRU
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 添加循环层
    for i, layer_conf in enumerate(config['layers']):
        l2_reg_val = layer_conf.get('l2_reg', 0.0)
        regularizer = l2(l2_reg_val) if l2_reg_val > 0 else None
        layer_args = {
            'units': layer_conf['units'],
            'return_sequences': layer_conf.get('return_sequences', False), # 默认 False 更安全
            'kernel_regularizer': regularizer,
            'recurrent_regularizer': regularizer,
            'activation': layer_conf.get('activation', 'tanh') # 显式指定RNN激活函数，默认tanh
        }
        if i == 0:
            # 只有第一个循环层需要指定 input_shape
            if model_type_lower == 'bilstm':
                 # 对于 Bidirectional，input_shape 传递给其内部的 LSTM 层
                 # layer_class (bilstm_layer) 会处理这个参数
                 layer_args['input_shape'] = (window_size, num_features)
                 model.add(layer_class(**layer_args))
            else:
                 layer_args['input_shape'] = (window_size, num_features)
                 model.add(layer_class(**layer_args))
        else:
             # 非第一个循环层不需要 input_shape 参数
             # 对于 Bidirectional，units 和 return_sequences 传递给 wrapper，其他参数传递给内部 LSTM
             if model_type_lower == 'bilstm':
                  model.add(layer_class(units=layer_conf['units'],
                                        return_sequences=layer_conf.get('return_sequences', False),
                                        kernel_regularizer=regularizer, # 这些传给内部 LSTM
                                        recurrent_regularizer=regularizer,
                                        activation=layer_conf.get('activation', 'tanh')))
             else:
                  model.add(layer_class(units=layer_conf['units'],
                                        return_sequences=layer_conf.get('return_sequences', False),
                                        kernel_regularizer=regularizer,
                                        recurrent_regularizer=regularizer,
                                        activation=layer_conf.get('activation', 'tanh')))


        if layer_conf.get('dropout', 0.0) > 0:
            # 循环层后的 Dropout
            model.add(Dropout(layer_conf['dropout']))

    # 添加全连接层
    for dense_conf in config.get('dense_layers', []):
        l2_reg_val = dense_conf.get('l2_reg', 0.0)
        regularizer = l2(l2_reg_val) if l2_reg_val > 0 else None
        model.add(Dense(units=dense_conf['units'], kernel_regularizer=regularizer, activation=dense_conf.get('activation', 'relu'))) # 显式指定Dense激活函数，默认relu
        if dense_conf.get('dropout', 0.0) > 0:
            # 全连接层后的 Dropout
            model.add(Dropout(dense_conf['dropout']))

    # 输出层
    output_units = config.get('output_units', 1)
    output_activation = config.get('output_activation', None)
    model.add(Dense(units=output_units, activation=output_activation)) # 默认回归输出层，无激活函数

    # 选择优化器
    optimizer_name = config.get('optimizer', 'adam').lower() # 转换为小写以便比较
    lr = config.get('learning_rate', 0.001) # 从 config 中获取学习率
    # 优化器可以接收各种参数，这里只处理学习率和可能的 momentum
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=config.get('momentum', 0.0)) # 新增momentum参数
    else:
        logger.warning(f"不支持的优化器: {optimizer_name}，使用默认Adam。")
        optimizer = Adam(learning_rate=lr)

    # 编译模型
    loss_func = config.get('loss', 'mse')
    metrics_list = config.get('metrics', ['mae'])
    # 考虑自定义损失函数和指标（如果需要）
    # if isinstance(loss_func, str):
    #     # Standard Keras loss
    #     pass
    # elif isinstance(loss_func, Callable):
    #     # Custom loss function
    #     pass

    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=metrics_list
    )

    if summary:
        model.summary(print_fn=lambda x: logger.info(x))
    logger.info(f"模型构建完成 ({model_type.upper()})，输入形状: ({window_size}, {num_features})") # 修正日志形状显示
    return model

# analyze_target_distribution 函数保留原样，但在调用时应在数据缩放前调用

# 保留原有的 analyze_target_distribution 函数
def analyze_target_distribution(y_train, y_val, y_test):
    """
    分析并绘制目标变量分布图。建议在数据缩放前调用此函数。
    """
    plt.figure(figsize=(15, 5))
    datasets = {'训练集': y_train, '验证集': y_val, '测试集': y_test}
    for i, (name, y) in enumerate(datasets.items()):
        plt.subplot(1, 3, i+1)
        # 检查 y 是否为空，为空则跳过绘制和分析
        if len(y) == 0:
            plt.title(f'{name} 目标变量分布 (数据为空)')
            # 绘制一个空的子图或者标记
            plt.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            continue

        # 根据唯一值数量调整bins，避免bins过多或过少
        # 检查数据类型，确保是数值类型
        y_numeric = pd.to_numeric(y, errors='coerce').dropna()
        if len(y_numeric) == 0:
             plt.title(f'{name} 目标变量分布 (无有效数值)')
             plt.text(0.5, 0.5, 'No valid data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             continue

        num_unique = len(np.unique(y_numeric))
        bins_to_use = min(num_unique, 50) if num_unique > 1 else 1 # 如果只有一个值，bins设为1
        if bins_to_use <= 1: # 如果数据只有1个或0个唯一值
             sns.histplot(y_numeric, bins=1, kde=False, alpha=0.7) # 不绘制kde，bins设为1
        else:
             sns.histplot(y_numeric, bins=bins_to_use, kde=True, alpha=0.7) # 绘制kde

        plt.title(f'{name} 目标变量分布')
        plt.xlabel('目标值')
        plt.ylabel('频次')

        # 打印描述性统计信息
        logger.info(f"{name} 目标变量描述：\n%s", pd.Series(y_numeric).describe())


    plt.tight_layout()
    # 为防止文件名冲突，可以考虑加上股票代码或时间戳
    plt.savefig('target_distribution.png')
    plt.close('all') # 确保关闭所有图窗以释放内存
    logger.info("目标变量分布图已保存至 target_distribution.png")

