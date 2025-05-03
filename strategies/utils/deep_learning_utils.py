# apps/strategies/utils/deep_learning_utils.py
import os

from sklearn.ensemble import RandomForestRegressor
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns  # 添加seaborn导入
from typing import Any, Tuple, List, Dict, Optional, Union, Callable
from functools import wraps

logger = logging.getLogger("strategy_deep_learning_utils")

# 装饰器：记录执行时间
def log_execution_time(func):
    """记录函数执行时间的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.2f} 秒")
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
            raise
    return wrapper

@log_execution_time
@handle_exceptions
def prepare_data_for_lstm(
    data: pd.DataFrame, required_columns: List[str], target_column: str = 'final_signal', window_size: int = 60,
    feature_scaler_type: str = 'minmax', train_split: float = 0.7, val_split: float = 0.15,
    # 调整特征选择/降维参数
    apply_variance_threshold: bool = False, variance_threshold_value: float = 0.01,
    use_pca: bool = False, # 默认关闭PCA，除非明确需要
    n_components: Union[int, float] = 0.99, # 如果使用PCA，保留更多方差
    target_scaler_type: str = 'minmax' # 新增目标变量缩放器类型参数
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler], Union[MinMaxScaler, StandardScaler]]:
    """
    准备用于LSTM训练的时间序列数据，包括特征选择、缩放、窗口化和数据集分割。

    Args:
        data (pd.DataFrame): 包含所有特征和目标列的原始DataFrame。
        required_columns (List[str]): 数据中应包含的最小特征列列表。
        target_column (str): 目标变量列名。
        window_size (int): LSTM输入的时间步长。
        feature_scaler_type (str): 特征缩放器类型 ('minmax' 或 'standard').
        train_split (float): 训练集比例。
        val_split (float): 验证集比例。
        apply_variance_threshold (bool): 是否应用方差阈值进行特征选择。
        variance_threshold_value (float): 方差阈值。
        use_pca (bool): 是否应用PCA进行降维。
        n_components (Union[int, float]): PCA保留的主成分数量或解释方差比例。
        target_scaler_type (str): 目标变量缩放器类型 ('minmax' 或 'standard').

    Returns:
        Tuple containing:
        X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray): 分割和处理后的数据集。
        feature_scaler (Union[MinMaxScaler, StandardScaler]): 用于特征的缩放器。
        target_scaler (Union[MinMaxScaler, StandardScaler]): 用于目标变量的缩放器。
    """
    # 确保目标列存在
    if target_column not in data.columns:
        logger.error(f"目标列 '{target_column}' 不存在于输入数据中。")
        raise ValueError(f"目标列 '{target_column}' 不存在。")

    # 选择原始特征列（排除目标列）
    feature_columns = [col for col in data.columns if col != target_column]
    if not feature_columns:
         logger.error("输入数据中没有可用的特征列 (除了目标列)。")
         raise ValueError("没有可用的特征列。")

    features = data.loc[:, feature_columns].values
    targets = data.loc[:, target_column].values

    logger.info(f"初始特征维度: {features.shape[1]}")

    # 特征选择：移除低方差特征 (可选)
    if apply_variance_threshold:
        selector = VarianceThreshold(threshold=variance_threshold_value)
        features = selector.fit_transform(features)
        # 更新 feature_columns 列表以便后续PCA或日志记录
        selected_indices = selector.get_support(indices=True)
        feature_columns = [feature_columns[i] for i in selected_indices]
        logger.info(f"方差阈值 {variance_threshold_value} 选择后维度: {features.shape[1]}")
        if features.shape[1] == 0:
             logger.error("方差阈值选择后没有剩余特征。请检查阈值或数据。")
             raise ValueError("没有剩余特征。")

    # 使用PCA降维（可选，在方差选择后进行）
    if use_pca and features.shape[1] > 1: # PCA至少需要2个特征
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(features)
        logger.info(f"PCA降维完成，保留 {pca.n_components_} 个主成分，解释方差比: {sum(pca.explained_variance_ratio_):.4f}")
        # 注意：PCA后的特征不再有原始列名，维度即 n_components_
    elif use_pca and features.shape[1] <= 1:
        logger.warning(f"特征维度为 {features.shape[1]} <= 1，跳过PCA降维。")

    # 构建时间序列窗口
    X, y = [], []
    # 调整循环范围，确保 y[i + window_size] 不越界
    for i in range(len(features) - window_size):
        X.append(features[i:(i + window_size)])
        y.append(targets[i + window_size]) # 预测窗口后的一个点
    
    X = np.array(X)
    y = np.array(y)

    if X.shape[0] == 0 or y.shape[0] == 0:
        logger.error("窗口化后数据量为零，请检查数据长度和窗口大小。")
        raise ValueError("窗口化后数据量为零。")

    logger.info(f"窗口化完成，X 形状: {X.shape}, y 形状: {y.shape}")

    # 分割数据集 (按时间顺序)
    # 检查分割比例和总样本数
    if train_split + val_split >= 1.0:
        logger.error("训练集和验证集比例之和必须小于1。")
        raise ValueError("分割比例错误。")
    if n_samples < window_size + 1:
         logger.error(f"数据长度 {len(data)} 不足以构建窗口 (window_size={window_size})。")
         raise ValueError("数据长度不足。")

    n_samples_windowed = X.shape[0]
    n_train = int(n_samples_windowed * train_split)
    n_val = int(n_samples_windowed * val_split)
    n_test = n_samples_windowed - n_train - n_val

    # 确保每个集合至少有一个样本
    if n_train == 0:
        logger.error("训练集样本数为零，请检查 train_split 和数据长度。")
        raise ValueError("训练集样本数为零。")
    # 验证集和测试集可以为零，如果 val_split 或 train_split + val_split 接近 1.0
    # if n_val == 0 and val_split > 0:
    #      logger.warning("验证集样本数为零。")
    # if n_test == 0 and test_split_ratio > 0:
    #      logger.warning("测试集样本数为零。")


    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    logger.info(f"数据分割完成 (按时间顺序)，训练集: {X_train.shape[0]} 条，验证集: {X_val.shape[0]} 条，测试集: {X_test.shape[0]} 条 (计算值: n_test={n_test})")

    # 缩放特征数据
    if feature_scaler_type == 'minmax':
        feature_scaler = MinMaxScaler()
    else: # default to StandardScaler
        feature_scaler = StandardScaler()
    
    num_features = X_train.shape[2]
    # 为了fit scaler，需要将所有时间步的特征展平
    X_train_reshaped_for_scaler = X_train.reshape(-1, num_features)
    feature_scaler.fit(X_train_reshaped_for_scaler) # 只在训练集上fit

    # 对所有数据集应用缩放，并恢复到原始形状
    X_train_scaled = feature_scaler.transform(X_train_reshaped_for_scaler).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape) if X_val.shape[0] > 0 else X_val
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape) if X_test.shape[0] > 0 else X_test

    logger.info(f"特征缩放完成。")

    # 缩放目标数据
    # 目标变量也需要缩放，特别是对于回归任务
    if target_scaler_type == 'minmax':
        target_scaler = MinMaxScaler()
    else: # default to StandardScaler
        target_scaler = StandardScaler()

    # 目标变量 y_train 是一个一维数组，需要reshape成二维才能fit scaler
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten() # 只在训练集目标变量上fit
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten() if y_val.shape[0] > 0 else y_val
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten() if y_test.shape[0] > 0 else y_test

    logger.info(f"目标变量缩放完成。")
    logger.info(f"数据准备完成 (已缩放)，训练集: {X_train_scaled.shape[0]} 条，验证集: {X_val_scaled.shape[0]} 条，测试集: {X_test_scaled.shape[0]} 条")
    logger.info(f"最终输入模型特征维度: {num_features}") # PCA后 num_features 就是主成分数量
    
    # 目标变量分布分析应该在缩放*之前*进行，以查看原始分布
    # analyze_target_distribution(y_train, y_val, y_test) # 移动到 prepare_data_for_lstm 调用前
    
    # 返回缩放后的数据和缩放器
    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler

@log_execution_time
@handle_exceptions
def train_lstm_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, model: Sequential,
    target_scaler: Union[MinMaxScaler, StandardScaler], # 传入目标变量缩放器
    training_config: Dict[str, Any] = None, checkpoint_path: str = "models/checkpoints/best_model.keras",
    plot_training_history: bool = False, # 保留绘图选项
    # 移除 sample_weight 参数，除非确认目标变量是分类并使用相应的模型/loss
    # use_sample_weight: bool = False # 考虑是否需要根据目标类型调整
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
        'verbose': 0 # 默认静默，可通过config覆盖
    }
    config = default_config.copy() # 使用copy，避免修改default_config
    if training_config:
        config.update(training_config)

    logger.info(f"开始训练模型，X_train: {X_train.shape}, y_train: {y_train.shape}")

    # 检查目标变量缩放后的范围 (仅作提醒，实际缩放已在prepare_data_for_lstm完成)
    # 如果 target_scaler 是 MinMaxScaler 并且其 feature_range 是 (0, 1)，则 y_scaled 应该在 [0, 1]
    # 对于 StandardScaler，它将数据转换为均值为 0，标准差为 1，范围不固定
    if isinstance(target_scaler, MinMaxScaler):
        logger.info(f"目标变量 MinMaxScaler range: {target_scaler.feature_range}")
    # else: StandardScaler 的范围没有特定保证

    # --- 移除不恰当的分类样本权重计算 ---
    # from sklearn.utils.class_weight import compute_sample_weight
    # sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_scaled) # y_train_scaled 现在是连续值
    # logger.info("已计算样本权重以平衡目标变量分布。") # 此日志也不再相关

    # 根据模型目标类型 (回归 vs 分类) 和目标变量的实际意义来决定是否使用样本权重。
    # 当前模型是回归，目标是连续值，通常不使用 compute_sample_weight('balanced')。
    # 如果目标实际上是离散信号(-1, 0, 1)，则需要修改模型、loss、metric 并可能使用样本权重。
    sample_weight = None # 默认不使用样本权重

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"创建检查点目录: {checkpoint_dir}")

    callbacks = [
        # monitor Metric应与model.compile中的metrics或loss一致
        EarlyStopping(monitor=config['monitor_metric'], patience=config['early_stopping_patience'], restore_best_weights=True, verbose=config['verbose']),
        ReduceLROnPlateau(monitor=config['monitor_metric'], factor=config['reduce_lr_factor'], patience=config['reduce_lr_patience'], min_lr=1e-6, verbose=config['verbose']),
        # mode 应根据 monitor_metric 是 loss (min) 还是 metric (max) 来设置
        ModelCheckpoint(filepath=checkpoint_path, monitor=config['monitor_metric'], save_best_only=True, save_weights_only=False, mode='min' if 'loss' in config['monitor_metric'].lower() else 'max', verbose=config['verbose'])
    ]

    # 检查验证集是否存在
    validation_data = (X_val, y_val) if X_val.shape[0] > 0 else None
    if validation_data is None:
        logger.warning("验证集为空，将不使用 EarlyStopping 和 ReduceLROnPlateau 的 val_* 监控。")
        # 如果没有验证集，需要修改回调函数的 monitor 参数
        for cb in callbacks:
            if isinstance(cb, (EarlyStopping, ReduceLROnPlateau)):
                cb.monitor = cb.monitor.replace('val_', '') # 例如 'val_loss' -> 'loss'
        # ModelCheckpoint 可能也需要调整 mode 和 monitor

    history = model.fit(
        X_train, y_train, # 使用已缩放的 y_train
        validation_data=validation_data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        sample_weight=sample_weight,  # 使用修正后的 sample_weight (None)
        callbacks=callbacks,
        verbose=config['verbose']
    )

    if X_test.shape[0] > 0:
        # 在测试集上评估 (使用缩放后的目标值)
        test_loss, test_mae_scaled = model.evaluate(X_test, y_test, verbose=0)

        # 将 MAE 转换回原始范围
        # scaler.inverse_transform 需要二维输入
        # 创建一个包含 scaled MAE 的数组，其形状与 y_test 相同
        # 然后用 target_scaler 对一个全零数组进行逆缩放，得到原始范围的0点对应值
        # 再对一个全Scaled MAE值的数组进行逆缩放，得到原始范围的Scaled MAE点对应值
        # 两者的差的绝对值就是原始范围的 MAE
        # 这是一个估算，更严谨是预测后逐点逆缩放计算MAE，但 evaluate 返回的是平均MAE
        # 这里用一个简化的估算方法，只适用于线性缩放器且目标范围是连续的
        mae_original_approx = abs(target_scaler.inverse_transform([[test_mae_scaled]])[0][0] - target_scaler.inverse_transform([[0]])[0][0])
        # 注意：对于 MinMaxScaler(0,1)，这将是 test_mae_scaled * (original_max - original_min)

        logger.info(f"LSTM模型在测试集上的损失: {test_loss:.4f}, MAE (缩放后): {test_mae_scaled:.4f}, MAE (原始范围估算): {mae_original_approx:.4f}")
    else:
        logger.warning("测试集为空，无法评估LSTM模型。")

    logger.info("模型训练完成。")

    # 可选：绘制训练历史图
    if plot_training_history:
         try:
             plt.figure(figsize=(12, 6))
             plt.plot(history.history['loss'], label='Train Loss')
             if 'val_loss' in history.history:
                 plt.plot(history.history['val_loss'], label='Val Loss')
             plt.title('Model Loss')
             plt.xlabel('Epoch')
             plt.ylabel('Loss')
             plt.legend()
             plt.grid(True)

             plt.figure(figsize=(12, 6))
             plt.plot(history.history['mae'], label='Train MAE')
             if 'val_mae' in history.history:
                 plt.plot(history.history['val_mae'], label='Val MAE')
             plt.title('Model MAE')
             plt.xlabel('Epoch')
             plt.ylabel('MAE')
             plt.legend()
             plt.grid(True)

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
    # output_units: int = 1,
    # output_activation: Optional[str] = None, # None for regression, 'sigmoid' or 'softmax' for classification
    # loss: str = 'mse', # 'mse' for regression, 'binary_crossentropy' or 'categorical_crossentropy' for classification
    # metrics: List[str] = ['mae'] # 'mae' for regression, ['accuracy', 'precision', 'recall', 'f1'] for classification
) -> Sequential:
    """
    构建深度学习模型，支持LSTM、Bidirectional LSTM和GRU，允许自定义配置。
    默认配置为回归任务。如需分类，需要修改这里的参数或在调用时覆盖。
    """
    # 默认配置 (回归任务)
    default_config = {
        'layers': [
            {'units': 64, 'return_sequences': True, 'dropout': 0.3, 'l2_reg': 0.01},
            {'units': 32, 'return_sequences': False, 'dropout': 0.3, 'l2_reg': 0.01}
        ],
        'dense_layers': [{'units': 16, 'dropout': 0.2, 'l2_reg': 0.01}],
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'mse',
        'metrics': ['mae']
    }
    config = default_config.copy() # 使用copy，避免修改default_config
    if model_config:
        config.update(model_config)

    # 构建模型
    model = Sequential()
    if model_type.lower() == 'lstm':
        layer_class = LSTM
    elif model_type.lower() == 'bilstm':
        layer_class = lambda units, return_sequences=False, **kwargs: Bidirectional(LSTM(units, return_sequences=return_sequences, **kwargs))
    elif model_type.lower() == 'gru':
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
            layer_args['input_shape'] = (window_size, num_features)  # 确保输入形状正确

        # 特别处理 Bidirectional 的输入
        if model_type.lower() == 'bilstm':
             model.add(layer_class(**layer_args)) # Bidirectional Wrapper handles input_shape
        else:
             model.add(layer_class(**layer_args))


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

    # 输出层 (默认回归任务)
    # output_units = config.get('output_units', 1) # 可通过config控制输出单元数
    # output_activation = config.get('output_activation', None) # 可通过config控制输出激活函数
    model.add(Dense(units=1, activation=None)) # 默认回归输出层

    # 选择优化器
    optimizer_name = config.get('optimizer', 'adam').lower() # 转换为小写以便比较
    lr = config.get('learning_rate', 0.001)
    # 优化器可以接收各种参数，这里只处理学习率
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
    logger.info(f"模型构建完成 ({model_type.upper()})，输入形状: {window_size} x {num_features}")
    return model

# analyze_target_distribution 函数保留原样，但在调用时应在 prepare_data_for_lstm *之前* 调用

# 保留原有的 analyze_target_distribution 函数
def analyze_target_distribution(y_train, y_val, y_test):
    plt.figure(figsize=(15, 5))
    for i, (y, name) in enumerate(zip([y_train, y_val, y_test], ['训练集', '验证集', '测试集'])):
        plt.subplot(1, 3, i+1)
        # 检查 y 是否为空，为空则跳过绘制和分析
        if len(y) == 0:
            plt.title(f'{name} 目标变量分布 (数据为空)')
            continue
        
        sns.histplot(y, bins=min(len(np.unique(y)), 30), kde=True, alpha=0.7) # 根据唯一值数量调整bins
        plt.title(f'{name} 目标变量分布')
        plt.xlabel('目标值')
        plt.ylabel('频次')

        # 检查值集中情况 (在原始值上检查更有意义，如果 analyze_target_distribution 在缩放后调用，则需要重新考虑此逻辑)
        # value_counts = pd.Series(y).value_counts(normalize=True)
        # if value_counts.max() > 0.8:
        #     dominant_value = value_counts.idxmax()
        #     logger.warning(f"{name} 目标变量值集中，值 {dominant_value} 占比 {value_counts.max():.2%}")

    plt.tight_layout()
    # 为防止文件名冲突，可以考虑加上股票代码或时间戳
    plt.savefig('target_distribution.png')
    plt.close('all') # 确保关闭图窗
    logger.info("目标变量分布图已保存至 target_distribution.png")

    if len(y_train) > 0: logger.info("训练集目标变量描述：\n%s", pd.Series(y_train).describe())
    if len(y_val) > 0: logger.info("验证集目标变量描述：\n%s", pd.Series(y_val).describe())
    if len(y_test) > 0: logger.info("测试集目标变量描述：\n%s", pd.Series(y_test).describe())












