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
    scaler_type: str = 'minmax', train_split: float = 0.7, val_split: float = 0.15,
    feature_selection: Optional[List[str]] = None, augment_data: bool = False,
    use_pca: bool = True, n_components: Union[int, float] = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler]]:
    """
    准备LSTM模型的训练数据，修正了Scaler的使用以防止数据泄露。
    Args:
        data (pd.DataFrame): 包含特征和目标变量的DataFrame。
        required_columns (List[str]): 模型输入特征列名列表。
        target_column (str): 目标变量列名，默认为'final_signal'。
        window_size (int): 时间序列窗口大小，默认为60。
        scaler_type (str): 归一化方法，'minmax' 或 'standard'，默认为'minmax'。
        train_split (float): 训练集比例，默认为0.7。
        val_split (float): 验证集比例，默认为0.15（剩余为测试集）。
        feature_selection (Optional[List[str]]): 可选的特征子集列名列表，若为None则使用required_columns。
        augment_data (bool): 是否进行数据增强（如添加噪声），默认为False。
        use_pca (bool): 是否使用PCA降维，默认为True。
        n_components (Union[int, float]): PCA保留的主成分数量或方差比例，默认为0.95。
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler]]:
            X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    if data.empty:
        raise ValueError("输入数据为空，无法准备训练数据。")
    
    # 选择特征列
    selected_columns = feature_selection if feature_selection is not None else required_columns
    
    # 特征选择：移除低方差特征
    features = data.loc[:, selected_columns].values
    if feature_selection is None:
        selector = VarianceThreshold(threshold=0.01)
        features_reduced = selector.fit_transform(features)
        selected_columns = [selected_columns[i] for i in selector.get_support(indices=True)]
        features = features_reduced
        logger.info(f"特征选择完成，保留 {len(selected_columns)} 个特征。")

    # 使用PCA降维（可选）
    if use_pca:
        pca = PCA(n_components=n_components if isinstance(n_components, int) else n_components)
        features_pca = pca.fit_transform(features)
        logger.info(f"PCA降维完成，保留 {pca.n_components_} 个主成分，解释方差比: {sum(pca.explained_variance_ratio_):.4f}")
        features = features_pca
    
    # 提取目标变量
    targets = data.loc[:, target_column].values

    # 构建时间序列窗口
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:(i + window_size)])
        y.append(targets[i + window_size])
    
    if not X:
        raise ValueError(f"数据长度 ({len(features)}) 不足以构建窗口大小为 {window_size} 的时间序列。至少需要 {window_size + 1} 条数据。")

    X = np.array(X)
    y = np.array(y)
    
    # 分割数据集
    test_split_ratio = 1.0 - train_split - val_split
    if test_split_ratio < 0 or test_split_ratio > 1:
        raise ValueError(f"训练集({train_split})、验证集({val_split})和测试集({test_split_ratio})的比例设置无效。")

    n_samples = X.shape[0]
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    n_test = n_samples - n_train - n_val  # 计算测试集样本数量

    if n_train == 0 or n_val == 0 or n_test == 0:
        logger.warning(f"数据集过小或分割比例不当，导致某个子集为空。样本总数: {n_samples}, 训练集: {n_train}, 验证集: {n_val}, 测试集: {n_test}")
        if n_train == 0:
            raise ValueError("训练集样本数为0，无法继续。")

    # 按时间顺序分割
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    logger.info(f"数据分割完成 (按时间顺序)，训练集: {X_train.shape[0]} 条，验证集: {X_val.shape[0]} 条，测试集: {X_test.shape[0]} 条 (计算值: n_test={n_test})")

    # 初始化并拟合Scaler (仅在训练集上)
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"不支持的归一化方法: {scaler_type}")

    num_features = X_train.shape[2]
    X_train_reshaped = X_train.reshape(-1, num_features)
    logger.info(f"拟合Scaler: 使用 {X_train_reshaped.shape[0]} 个时间点的数据 (来自训练集)。")
    scaler.fit(X_train_reshaped)
    logger.info(f"Scaler拟合完成。Scaler对象: {scaler}")

    # 使用拟合好的Scaler转换所有数据集
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape) if X_val.shape[0] > 0 else X_val
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape) if X_test.shape[0] > 0 else X_test

    # 数据增强（可选）
    if augment_data:
        noise = np.random.normal(0, 0.01, X_train_scaled.shape)
        X_train_scaled = X_train_scaled + noise
        logger.info("已对训练集应用数据增强（添加高斯噪声）。")

    logger.info(f"数据准备完成 (已缩放)，训练集: {X_train_scaled.shape[0]} 条，验证集: {X_val_scaled.shape[0]} 条，测试集: {X_test_scaled.shape[0]} 条")
    logger.info(f"X_train_scaled样本 (前1个窗口): {X_train_scaled[:1]}, y_train样本 (前5个): {y_train[:5]}")
    
    analyze_target_distribution(y_train, y_val, y_test)
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler

@log_execution_time
@handle_exceptions
def train_lstm_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray, model: Sequential,
    training_config: Dict[str, Any] = None, checkpoint_path: str = "models/checkpoints/best_model.keras", 
    plot_training_history: bool = False
) -> Dict:
    default_config = {
        'epochs': 20,
        'batch_size': 128,
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
        'reduce_lr_factor': 0.5,
        'monitor_metric': 'val_loss',
        'verbose': 0
    }
    config = training_config if training_config is not None else default_config
    logger.info(f"开始训练模型，X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    y_train_scaled = y_train / 100.0
    y_val_scaled = y_val / 100.0
    y_test_scaled = y_test / 100.0
    
    if np.any(y_train_scaled < 0) or np.any(y_train_scaled > 1):
        logger.warning(f"训练集目标缩放后超出 [0, 1] 范围。最小值: {np.min(y_train_scaled)}, 最大值: {np.max(y_train_scaled)}")
    if np.any(y_val_scaled < 0) or np.any(y_val_scaled > 1):
        logger.warning(f"验证集目标缩放后超出 [0, 1] 范围。最小值: {np.min(y_val_scaled)}, 最大值: {np.max(y_val_scaled)}")
    if np.any(y_test_scaled < 0) or np.any(y_test_scaled > 1):
        logger.warning(f"测试集目标缩放后超出 [0, 1] 范围。最小值: {np.min(y_test_scaled)}, 最大值: {np.max(y_test_scaled)}")
    
    # 计算样本权重以平衡目标变量分布
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_scaled)
    logger.info("已计算样本权重以平衡目标变量分布。")
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"创建检查点目录: {checkpoint_dir}")
    
    callbacks = [
        EarlyStopping(monitor=config['monitor_metric'], patience=config['early_stopping_patience'], restore_best_weights=True, verbose=config['verbose']),
        ReduceLROnPlateau(monitor=config['monitor_metric'], factor=config['reduce_lr_factor'], patience=config['reduce_lr_patience'], min_lr=1e-6, verbose=config['verbose']),
        ModelCheckpoint(filepath=checkpoint_path, monitor=config['monitor_metric'], save_best_only=True, save_weights_only=False, mode='min' if 'loss' in config['monitor_metric'].lower() else 'max', verbose=config['verbose'])
    ]
    
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        sample_weight=sample_weights,  # 添加样本权重
        callbacks=callbacks,
        verbose=config['verbose']
    )
    
    if X_test.shape[0] > 0:
        test_loss, test_mae = model.evaluate(X_test, y_test_scaled, verbose=0)
        test_mae_original = test_mae * 100.0
        logger.info(f"LSTM模型在测试集上的损失: {test_loss:.4f}, MAE: {test_mae_original:.4f} (原始范围)")
    else:
        logger.warning("测试集为空，无法评估LSTM模型。")
    
    logger.info("模型训练完成。")
    return history.history

@log_execution_time
@handle_exceptions
def build_lstm_model(
    window_size: int,
    num_features: int,  # 确保传入正确的特征维度
    model_config: Dict[str, Any] = None,
    model_type: str = 'lstm',
    summary: bool = True
) -> Sequential:
    """
    构建深度学习模型，支持LSTM、Bidirectional LSTM和GRU，允许自定义配置。
    """
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
    config = model_config if model_config is not None else default_config
    
    model = Sequential()
    if model_type == 'lstm': 
        layer_class = LSTM
    elif model_type == 'bilstm': 
        layer_class = lambda *args, **kwargs: Bidirectional(LSTM(*args, **kwargs))
    elif model_type == 'gru': 
        layer_class = GRU
    else: 
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 添加循环层
    for i, layer_conf in enumerate(config['layers']):
        l2_reg_val = layer_conf.get('l2_reg', 0.0)
        regularizer = l2(l2_reg_val) if l2_reg_val > 0 else None
        layer_args = {
            'units': layer_conf['units'],
            'return_sequences': layer_conf['return_sequences'],
            'kernel_regularizer': regularizer,
            'recurrent_regularizer': regularizer
        }
        if i == 0:
            layer_args['input_shape'] = (window_size, num_features)  # 确保输入形状正确
        model.add(layer_class(**layer_args))
        if layer_conf.get('dropout', 0.0) > 0:
            model.add(Dropout(layer_conf['dropout']))

    # 添加全连接层
    for dense_conf in config.get('dense_layers', []):
        l2_reg_val = dense_conf.get('l2_reg', 0.0)
        regularizer = l2(l2_reg_val) if l2_reg_val > 0 else None
        model.add(Dense(units=dense_conf['units'], kernel_regularizer=regularizer, activation='relu'))
        if dense_conf.get('dropout', 0.0) > 0:
            model.add(Dropout(dense_conf['dropout']))

    # 输出层
    model.add(Dense(units=1, activation=None))
    
    # 选择优化器
    optimizer_name = config.get('optimizer', 'adam')
    lr = config.get('learning_rate', 0.001)
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=config.get('loss', 'mse'),
        metrics=config.get('metrics', ['mae'])
    )
    
    if summary:
        model.summary(print_fn=lambda x: logger.info(x))
    logger.info(f"模型构建完成 ({model_type.upper()})，输入形状: {window_size} x {num_features}")
    return model

def analyze_target_distribution(y_train, y_val, y_test):
    plt.figure(figsize=(15, 5))
    for i, (y, name) in enumerate(zip([y_train, y_val, y_test], ['训练集', '验证集', '测试集'])):
        plt.subplot(1, 3, i+1)
        sns.histplot(y, bins=30, kde=True, alpha=0.7)
        plt.title(f'{name} 目标变量分布')
        plt.xlabel('目标值')
        plt.ylabel('频次')
        
        # 检查值集中情况
        value_counts = pd.Series(y).value_counts(normalize=True)
        if value_counts.max() > 0.8:  # 如果某一值占比超过80%
            dominant_value = value_counts.idxmax()
            logger.warning(f"{name} 目标变量值集中，值 {dominant_value} 占比 {value_counts.max():.2%}")
    
    plt.tight_layout()
    plt.savefig('target_distribution.png')
    plt.close()
    logger.info("目标变量分布图已保存至 target_distribution.png")
    
    logger.info("训练集目标变量描述：\n%s", pd.Series(y_train).describe())
    logger.info("验证集目标变量描述：\n%s", pd.Series(y_val).describe())
    logger.info("测试集目标变量描述：\n%s", pd.Series(y_test).describe())












