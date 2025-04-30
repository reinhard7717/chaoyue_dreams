# apps/strategies/utils/deep_learning_utils.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from typing import Any, Tuple, List, Dict, Optional, Union, Callable
import logging
import time
import os
from functools import wraps
import matplotlib.pyplot as plt

logger = logging.getLogger("strategy_deep_learning_utils")

# 装饰器：记录执行时间
def log_execution_time(func):
    """记录函数执行时间的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
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
            raise
    return wrapper

@log_execution_time
@handle_exceptions
def prepare_data_for_lstm(
    data: pd.DataFrame,
    required_columns: List[str],
    target_column: str = 'final_signal',
    window_size: int = 60,
    scaler_type: str = 'minmax',
    train_split: float = 0.7,
    val_split: float = 0.15,
    fill_na_method: str = 'ffill',
    feature_selection: Optional[List[str]] = None,
    augment_data: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler]]:
    """
    准备LSTM模型的训练数据，支持多种预处理选项和数据集分割。
    
    参数:
        data: 包含特征和目标变量的DataFrame。
        required_columns: 模型输入特征列名列表。
        target_column: 目标变量列名，默认为'final_signal'。
        window_size: 时间序列窗口大小，默认为60。
        scaler_type: 归一化方法，'minmax' 或 'standard'，默认为'minmax'。
        train_split: 训练集比例，默认为0.7。
        val_split: 验证集比例，默认为0.15（剩余为测试集）。
        fill_na_method: 缺失值填充方法，'ffill'（向前填充）、'bfill'（向后填充）或'mean'（均值填充），默认为'ffill'。
        feature_selection: 可选的特征子集列名列表，若为None则使用required_columns。
        augment_data: 是否进行数据增强（如添加噪声），默认为False。
    
    返回:
        X_train, y_train: 训练集特征和目标。
        X_val, y_val: 验证集特征和目标。
        X_test, y_test: 测试集特征和目标。
        scaler: 用于归一化的Scaler对象。
    """
    if data.empty:
        raise ValueError("输入数据为空，无法准备训练数据。")
    
    # 选择特征列
    selected_columns = feature_selection if feature_selection is not None else required_columns
    missing_cols = [col for col in selected_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必需特征列: {missing_cols}")
    
    # 缺失值处理
    data_copy = data.copy()
    if fill_na_method == 'ffill':
        data_copy[selected_columns] = data_copy[selected_columns].ffill()
    elif fill_na_method == 'bfill':
        data_copy[selected_columns] = data_copy[selected_columns].bfill()
    elif fill_na_method == 'mean':
        data_copy[selected_columns] = data_copy[selected_columns].fillna(data_copy[selected_columns].mean())
    else:
        raise ValueError(f"不支持的缺失值填充方法: {fill_na_method}")
    
    # 再次检查缺失值
    data_copy = data_copy.dropna(subset=selected_columns + [target_column])
    if data_copy.empty:
        raise ValueError("处理缺失值后数据为空，无法准备训练数据。")
    
    # 提取特征和目标
    features = data_copy[selected_columns].values
    targets = data_copy[target_column].values
    
    # 归一化特征
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"不支持的归一化方法: {scaler_type}")
    features_scaled = scaler.fit_transform(features)
    
    # 数据增强（可选）
    if augment_data:
        noise = np.random.normal(0, 0.01, features_scaled.shape)
        features_scaled = features_scaled + noise
        logger.info("已应用数据增强（添加高斯噪声）。")
    
    # 构建时间序列窗口
    X, y = [], []
    for i in range(len(features_scaled) - window_size):
        X.append(features_scaled[i:i + window_size])
        y.append(targets[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    if X.shape[0] == 0:
        raise ValueError(f"数据不足以构建窗口大小为 {window_size} 的时间序列。")
    
    # 分割数据集
    test_split = 1.0 - train_split - val_split
    if test_split < 0:
        raise ValueError("训练集和验证集比例之和不能大于1.0。")
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_split, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_split/(train_split + val_split), shuffle=False)
    
    logger.info(f"数据准备完成，训练集: {X_train.shape[0]} 条，验证集: {X_val.shape[0]} 条，测试集: {X_test.shape[0]} 条")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

@log_execution_time
@handle_exceptions
def build_lstm_model(
    window_size: int,
    num_features: int,
    model_config: Dict[str, Any] = None,
    model_type: str = 'lstm',
    summary: bool = True
) -> Sequential:
    """
    构建深度学习模型，支持LSTM、Bidirectional LSTM和GRU，允许自定义配置。
    
    参数:
        window_size: 时间序列窗口大小。
        num_features: 输入特征数量。
        model_config: 模型配置字典，包含层数、单元数、dropout率、正则化等参数。
        model_type: 模型类型，'lstm'、'bilstm'（双向LSTM）或'gru'，默认为'lstm'。
        summary: 是否打印模型摘要，默认为True。
    
    返回:
        model: 编译好的Keras模型。
    """
    # 默认配置
    default_config = {
        'layers': [
            {'units': 50, 'return_sequences': True, 'dropout': 0.2, 'l2_reg': 0.01},
            {'units': 50, 'return_sequences': False, 'dropout': 0.2, 'l2_reg': 0.01}
        ],
        'dense_layers': [{'units': 25, 'dropout': 0.1, 'l2_reg': 0.01}],
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'mse',
        'metrics': ['mae']
    }
    
    config = model_config if model_config is not None else default_config
    
    # 构建模型
    model = Sequential()
    
    # 选择模型类型
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
        l2_reg = l2(layer_conf.get('l2_reg', 0.01))
        if i == 0:
            # 第一层需要指定input_shape
            model.add(layer_class(
                units=layer_conf['units'],
                return_sequences=layer_conf['return_sequences'],
                input_shape=(window_size, num_features),
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg
            ))
        else:
            model.add(layer_class(
                units=layer_conf['units'],
                return_sequences=layer_conf['return_sequences'],
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg
            ))
        if layer_conf.get('dropout', 0.0) > 0:
            model.add(Dropout(layer_conf['dropout']))
    
    # 添加全连接层
    for dense_conf in config.get('dense_layers', []):
        l2_reg = l2(dense_conf.get('l2_reg', 0.01))
        model.add(Dense(units=dense_conf['units'], kernel_regularizer=l2_reg))
        if dense_conf.get('dropout', 0.0) > 0:
            model.add(Dropout(dense_conf['dropout']))
    
    # 输出层
    model.add(Dense(units=1, activation='sigmoid'))  # 输出0-1范围，稍后缩放到0-100
    
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
    
    return model

@log_execution_time
@handle_exceptions
def train_lstm_model( X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, model: Sequential,
    training_config: Dict[str, Any] = None, checkpoint_path: str = "models/checkpoints/best_model.h5", plot_training_history: bool = False
) -> Dict:
    """
    训练深度学习模型，支持早停、学习率调度和模型检查点保存。
    参数:
        X_train, y_train: 训练集特征和目标。
        X_val, y_val: 验证集特征和目标。
        model: 未训练的Keras模型。
        training_config: 训练配置字典，包含epochs、batch_size、callbacks等参数。
        checkpoint_path: 模型检查点保存路径。
        plot_training_history: 是否绘制训练历史曲线，默认为False。
    返回:
        history: 训练历史记录字典。
    """
    # 默认训练配置
    default_config = {
        'epochs': 50,
        'batch_size': 32,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'reduce_lr_factor': 0.5,
        'monitor_metric': 'val_loss',
        'verbose': 1
    }
    
    config = training_config if training_config is not None else default_config
    
    # 将目标变量缩放到0-1范围
    y_train_scaled = y_train / 100.0
    y_val_scaled = y_val / 100.0
    
    # 确保检查点目录存在
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"创建检查点目录: {checkpoint_dir}")
    
    # 设置回调函数
    callbacks = []
    if config.get('early_stopping_patience', 0) > 0:
        callbacks.append(EarlyStopping(
            monitor=config['monitor_metric'],
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=config['verbose']
        ))
    if config.get('reduce_lr_patience', 0) > 0:
        callbacks.append(ReduceLROnPlateau(
            monitor=config['monitor_metric'],
            factor=config['reduce_lr_factor'],
            patience=config['reduce_lr_patience'],
            min_lr=1e-6,
            verbose=config['verbose']
        ))
    callbacks.append(ModelCheckpoint(
        checkpoint_path,
        monitor=config['monitor_metric'],
        save_best_only=True,
        mode='min' if 'loss' in config['monitor_metric'] else 'max',
        verbose=config['verbose']
    ))
    
    # 训练模型
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=config['verbose']
    )
    
    # 绘制训练历史曲线（可选）
    if plot_training_history:
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 绘制指标曲线（假设有mae）
        if 'mae' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='训练MAE')
            plt.plot(history.history['val_mae'], label='验证MAE')
            plt.title('模型MAE')
            plt.xlabel('轮数')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        logger.info("训练历史曲线已保存至 training_history.png")
    
    logger.info("模型训练完成。")
    return history.history
