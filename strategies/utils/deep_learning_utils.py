# apps/strategies/utils/deep_learning_utils.py
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU,放置在代码的顶部（在导入TensorFlow之前）
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from typing import Any, Tuple, List, Dict, Optional, Union, Callable
import logging
import time

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
    use_pca: bool = True, n_components: Union[int, float] = 0.95,
    use_feature_importance: bool = False, importance_threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler]]:
    selected_columns = feature_selection if feature_selection is not None else required_columns
    features = data.loc[:, selected_columns].values
    targets = data.loc[:, target_column].values

    # 基于特征重要性筛选
    if use_feature_importance:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(features, targets)
        importances = model.feature_importances_
        important_features = [selected_columns[i] for i in range(len(selected_columns)) if importances[i] >= importance_threshold]
        if len(important_features) == 0:
            logger.warning("特征重要性筛选后无特征保留，使用原始特征。")
        else:
            selected_columns = important_features
            features = data.loc[:, selected_columns].values
            logger.info(f"特征重要性筛选完成，保留 {len(selected_columns)} 个特征。")

    # PCA降维
    if use_pca:
        pca = PCA(n_components=n_components if isinstance(n_components, int) else n_components)
        features_pca = pca.fit_transform(features)
        logger.info(f"PCA降维完成，保留 {pca.n_components_} 个主成分，解释方差比: {sum(pca.explained_variance_ratio_):.4f}")
        features = features_pca

    # --- 2. 检查并处理剩余的NaN (防御性编程) ---
    # 假设上游已填充，但仍检查一下。如果还有NaN，说明上游填充不完全。
    # 之前的dropna()可能过于激进，这里改为打印警告并继续（或根据需要决定是否填充/删除）
    nan_check = data[selected_columns + [target_column]].isna().sum()
    nan_cols = nan_check[nan_check > 0]
    if not nan_cols.empty:
        logger.warning(f"输入数据在进行窗口化前仍存在NaN值: {nan_cols.to_dict()}。这可能影响模型训练。考虑检查上游填充逻辑。")
        # 根据策略决定如何处理，例如再次填充或删除包含NaN的行
        # data = data.dropna(subset=selected_columns + [target_column]) # 如果选择删除
        # 或者再次尝试填充
        data = data.copy() # 避免SettingWithCopyWarning
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        # 再次检查
        nan_check_after = data[selected_columns + [target_column]].isna().sum()
        if nan_check_after.any():
             logger.error(f"填充后仍有NaN，无法继续: {nan_check_after[nan_check_after > 0].to_dict()}")
             raise ValueError("数据填充后仍存在NaN，无法准备训练数据。")

    # --- 3. 提取特征和目标 ---
    # 使用 .loc 避免潜在的链式索引警告
    features = data.loc[:, selected_columns].values
    targets = data.loc[:, target_column].values

    # --- 4. 构建时间序列窗口 (X是3D, y是1D) ---
    X, y = [], []
    # 循环范围修正：应该是 len(features) - window_size
    # 因为 X[i] 包含 features[i] 到 features[i+window_size-1]
    # 对应的 y[i] 应该是 features[i+window_size] 的目标值
    for i in range(len(features) - window_size):
        X.append(features[i:(i + window_size)])
        y.append(targets[i + window_size]) # y[i] 对应 X[i] 窗口之后的值

    if not X: # 检查列表是否为空
        raise ValueError(f"数据长度 ({len(features)}) 不足以构建窗口大小为 {window_size} 的时间序列。至少需要 {window_size + 1} 条数据。")

    X = np.array(X)
    y = np.array(y)

    # --- 5. 分割数据集 (在缩放之前！) ---
    test_split_ratio = 1.0 - train_split - val_split
    if test_split_ratio < 0 or test_split_ratio > 1: # 增加检查
        raise ValueError(f"训练集({train_split})、验证集({val_split})和测试集({test_split_ratio})的比例设置无效。")

    # 计算分割点索引 (整数)
    n_samples = X.shape[0]
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    n_test = n_samples - n_train - n_val # 确保加起来是总数

    if n_train == 0 or n_val == 0 or n_test == 0:
        logger.warning(f"数据集过小或分割比例不当，导致某个子集为空。样本总数: {n_samples}, 训练集: {n_train}, 验证集: {n_val}, 测试集: {n_test}")
        # 可以选择抛出错误或返回空值，这里先继续，但后续步骤可能失败
        if n_train == 0: raise ValueError("训练集样本数为0，无法继续。")

    # 按时间顺序分割
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"数据分割完成 (按时间顺序)，训练集: {X_train.shape[0]} 条，验证集: {X_val.shape[0]} 条，测试集: {X_test.shape[0]} 条")

    # --- 6. 初始化并拟合Scaler (仅在训练集上！) ---
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"不支持的归一化方法: {scaler_type}")

    # Scaler 需要 2D 输入: (样本数 * 时间步, 特征数)
    # Reshape X_train for fitting the scaler
    # X_train 的形状是 (n_train_samples, window_size, num_features)
    num_features = X_train.shape[2]
    X_train_reshaped = X_train.reshape(-1, num_features) # 变成 (n_train_samples * window_size, num_features)

    print(f"拟合Scaler: 使用 {X_train_reshaped.shape[0]} 个时间点的数据 (来自训练集)。")
    # --- 关键：只在训练数据上 fit ---
    scaler.fit(X_train_reshaped)
    print(f"Scaler拟合完成。Scaler对象: {scaler}")

    # --- 7. 使用拟合好的Scaler转换所有数据集 ---
    # Transform training data
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape) # Reshape back to 3D

    # Transform validation data
    if X_val.shape[0] > 0: # 确保验证集非空
        X_val_reshaped = X_val.reshape(-1, num_features)
        X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape)
    else:
        X_val_scaled = X_val # 如果为空，保持为空

    # Transform test data
    if X_test.shape[0] > 0: # 确保测试集非空
        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
    else:
        X_test_scaled = X_test # 如果为空，保持为空

    # --- 8. 数据增强（可选，在缩放后应用） ---
    if augment_data:
        # 仅对训练集应用增强
        noise = np.random.normal(0, 0.01, X_train_scaled.shape)
        X_train_scaled = X_train_scaled + noise
        print("已对训练集应用数据增强（添加高斯噪声）。")

    print(f"数据准备完成 (已缩放)，训练集: {X_train_scaled.shape[0]} 条，验证集: {X_val_scaled.shape[0]} 条，测试集: {X_test_scaled.shape[0]} 条")
    print(f"X_train_scaled样本 (前1个窗口): {X_train_scaled[:1]}, y_train样本 (前5个): {y_train[:5]}") # y 保持原始值，在训练函数中缩放

    analyze_target_distribution(y_train, y_val, y_test)

    # 返回缩放后的特征数据和原始的目标数据，以及拟合好的 scaler
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler

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
    """
    # 默认配置，优化为更轻量级模型以提高CPU训练效率
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
            layer_args['input_shape'] = (window_size, num_features)
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

@log_execution_time
@handle_exceptions
def train_lstm_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray, model: Sequential,
    training_config: Dict[str, Any] = None, checkpoint_path: str = "models/checkpoints/best_model.keras", 
    plot_training_history: bool = False
) -> Dict:
    """
    训练深度学习模型，支持早停、学习率调度和模型检查点保存。
    目标变量 y 会在此函数内部缩放到 0-1 范围以匹配模型的 sigmoid 输出。
    Args:
        X_train (np.ndarray): 训练集特征 (已缩放)。
        y_train (np.ndarray): 训练集目标 (原始值, e.g., 0-100)。
        X_val (np.ndarray): 验证集特征 (已缩放)。
        y_val (np.ndarray): 验证集目标 (原始值, e.g., 0-100)。
        X_test (np.ndarray): 测试集特征 (已缩放)。
        y_test (np.ndarray): 测试集目标 (原始值, e.g., 0-100)。
        model (Sequential): 未训练或已编译的Keras模型。
        training_config (Dict[str, Any]): 训练配置字典。
        checkpoint_path (str): 模型检查点保存路径。
        plot_training_history (bool): 是否绘制训练历史曲线。
    Returns:
        Dict: 训练历史记录字典。
    """
    # 默认训练配置
    default_config = {
        'epochs': 30,
        'batch_size': 128,  # 增加批次大小
        'early_stopping_patience': 8,
        'reduce_lr_patience': 3,
        'reduce_lr_factor': 0.5,
        'monitor_metric': 'val_loss',
        'verbose': 0  # 减少日志输出
    }
    config = training_config if training_config is not None else default_config
    logger.info(f"开始训练模型，X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # --- 将目标变量 y 缩放到 0-1 范围 ---
    y_train_scaled = y_train / 100.0
    y_val_scaled = y_val / 100.0
    y_test_scaled = y_test / 100.0  # 缩放测试集目标变量
    
    # 检查缩放后的值是否在合理范围 (可选)
    if np.any(y_train_scaled < 0) or np.any(y_train_scaled > 1):
        logger.warning(f"训练集目标缩放后存在超出 [0, 1] 范围的值。最小值: {np.min(y_train_scaled)}, 最大值: {np.max(y_train_scaled)}")
    if np.any(y_val_scaled < 0) or np.any(y_val_scaled > 1):
        logger.warning(f"验证集目标缩放后存在超出 [0, 1] 范围的值。最小值: {np.min(y_val_scaled)}, 最大值: {np.max(y_val_scaled)}")
    if np.any(y_test_scaled < 0) or np.any(y_test_scaled > 1):
        logger.warning(f"测试集目标缩放后存在超出 [0, 1] 范围的值。最小值: {np.min(y_test_scaled)}, 最大值: {np.max(y_test_scaled)}")
    
    # 确保检查点目录存在
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"创建检查点目录: {checkpoint_dir}")
    
    # --- 设置回调函数 ---
    callbacks = []
    # 早停：如果在 patience 个轮次内，监控指标没有改善，则停止训练
    if config.get('early_stopping_patience', 0) > 0:
        callbacks.append(EarlyStopping(
            monitor=config['monitor_metric'],
            patience=config['early_stopping_patience'],
            restore_best_weights=True,  # 关键：训练结束后，模型权重会恢复到最佳轮次的状态
            verbose=config['verbose']
        ))
    # 学习率衰减：当监控指标停止改善时，降低学习率
    if config.get('reduce_lr_patience', 0) > 0:
        callbacks.append(ReduceLROnPlateau(
            monitor=config['monitor_metric'],
            factor=config['reduce_lr_factor'],  # 学习率乘以的因子
            patience=config['reduce_lr_patience'],
            min_lr=1e-6,  # 学习率下限
            verbose=config['verbose']
        ))
    # 模型检查点：只保存在验证集上性能最好的模型
    callbacks.append(ModelCheckpoint(
        filepath=checkpoint_path,  # 使用 filepath 参数
        monitor=config['monitor_metric'],
        save_best_only=True,
        save_weights_only=False,  # 可以选择只保存权重或整个模型，False表示保存整个模型
        mode='min' if 'loss' in config['monitor_metric'].lower() else 'max',  # 根据监控指标判断模式
        verbose=config['verbose']
    ))
    
    # --- 训练模型 ---
    # 使用缩放后的 y 值进行训练
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=config['verbose']
    )
    logger.info(f"训练历史 (部分): { {k: v[:3] for k, v in history.history.items()} }...")  # 打印部分历史记录

    # --- 绘制训练历史曲线 ---
    if plot_training_history:
        try:
            plt.figure(figsize=(12, 5))  # 调整图形大小

            # 绘制损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='训练损失 (loss)')
            plt.plot(history.history['val_loss'], label='验证损失 (val_loss)')
            plt.title('模型损失')
            plt.xlabel('轮数 (Epoch)')
            plt.ylabel('损失值')
            plt.legend()
            plt.grid(True)

            # 绘制第一个指标曲线 (通常是 mae)
            metric_key = config.get('metrics', ['mae'])[0]  # 获取第一个指标的名称
            val_metric_key = f"val_{metric_key}"
            if metric_key in history.history and val_metric_key in history.history:
                plt.subplot(1, 2, 2)
                plt.plot(history.history[metric_key], label=f'训练 {metric_key.upper()}')
                plt.plot(history.history[val_metric_key], label=f'验证 {metric_key.upper()}')
                plt.title(f'模型 {metric_key.upper()}')
                plt.xlabel('轮数 (Epoch)')
                plt.ylabel(f'{metric_key.upper()} 值')
                plt.legend()
                plt.grid(True)

            plt.tight_layout()  # 调整子图布局
            plot_filename = 'training_history.png'
            plt.savefig(plot_filename)  # 直接保存在当前目录
            plt.close()  # 关闭图形，释放内存
            logger.info(f"训练历史曲线已保存至 {plot_filename}")
        except Exception as plot_err:
            logger.error(f"绘制训练历史图时出错: {plot_err}", exc_info=True)

    # --- 模型训练完成后的评估 ---
    if X_test.shape[0] > 0:
        test_loss, test_mae = model.evaluate(X_test, y_test_scaled, verbose=0)
        # 反缩放MAE值以反映原始范围
        test_mae_original = test_mae * 100.0
        logger.info(f"LSTM模型在测试集上的损失: {test_loss:.4f}, MAE: {test_mae_original:.4f} (原始范围)")
    else:
        logger.warning("测试集为空，无法评估LSTM模型。")
    
    logger.info("模型训练完成。")
    # 因为使用了 restore_best_weights=True，model 对象现在持有最佳权重
    # history.history 包含了所有轮次的记录
    return history.history

def analyze_target_distribution(y_train, y_val, y_test):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(15, 5))
    for i, (y, name) in enumerate(zip([y_train, y_val, y_test], ['训练集', '验证集', '测试集'])):
        plt.subplot(1, 3, i+1)
        sns.histplot(y, bins=30, kde=True, alpha=0.7)
        plt.title(f'{name} 目标变量分布')
        plt.xlabel('目标值')
        plt.ylabel('频次')
    plt.tight_layout()
    plt.savefig('target_distribution.png')
    plt.close()
    logger.info("目标变量分布图已保存至 target_distribution.png")
    
    # 输出描述性统计信息
    logger.info("训练集目标变量描述：\n%s", pd.Series(y_train).describe())
    logger.info("验证集目标变量描述：\n%s", pd.Series(y_val).describe())
    logger.info("测试集目标变量描述：\n%s", pd.Series(y_test).describe())














