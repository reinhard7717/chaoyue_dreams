# apps/strategies/utils/deep_learning_utils.py

# 导入必要的库 (PyTorch 相关)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter # 用于 TensorBoard 日志

# 导入必要的库 (数据处理和特征工程)
import os
import pandas as pd
import numpy as np
# 导入数据预处理和特征工程相关的模块 (Scikit-learn 部分保持不变，因为它处理 NumPy 数组)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False
    # logger.warning("XGBoost 未安装，如果选择使用 XGBoost 进行特征选择将会失败。请运行 'pip install xgboost'")
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
# 导入 PyTorch 回调函数 (手动实现或使用库)
from torch.optim.lr_scheduler import ReduceLROnPlateau # PyTorch 的学习率调度器

# 导入日志、时间、绘图等辅助库
import logging
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# 导入类型提示
from typing import Any, Tuple, List, Dict, Optional, Union, Callable
from functools import wraps
import joblib # 用于加载/保存 scaler

# 设置日志记录器
logger = logging.getLogger("strategy_deep_learning_utils")

# 装饰器：记录执行时间
def log_execution_time(func: Callable) -> Callable:
    """
    记录函数执行时间的装饰器。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

# 装饰器：统一异常处理
def handle_exceptions(func: Callable) -> Callable:
    """
    处理函数执行过程中产生的异常的装饰器。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行出错: {e}", exc_info=True)
            raise # 重新抛出，让调用者处理
    return wrapper

# --- PyTorch 相关组件 ---

class TimeSeriesDataset(Dataset):
    """
    用于时间序列数据的 PyTorch Dataset。
    用于加载经过预处理和缩放的 NumPy 数组，并在 __getitem__ 中进行窗口化并转换为 PyTorch Tensor。
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int):
        """
        初始化时间序列数据集。

        Args:
            features (np.ndarray): 经过缩放的平坦特征数据 (形状: num_samples, num_features)。
            targets (np.ndarray): 经过缩放的平坦目标数据 (形状: num_samples,).
                                 这里的 target[i] 对应 features[i] 时刻的 final_signal。
            window_size (int): 每个输入窗口的时间步长。
        """
        if features.shape[0] != targets.shape[0]:
            raise ValueError("特征和目标的样本数量必须相同。")

        # 窗口需要 window_size 个点。预测目标是窗口最后一个点 (t+window_size-1) 的信号。
        # 需要 features[t : t + window_size] 来预测 targets[t + window_size - 1].
        # 最后一个可用的窗口起始索引是 num_samples - window_size.
        self.num_samples = features.shape[0]
        if self.num_samples < window_size: # 至少需要 window_size 条数据才能构成第一个窗口的输入和对应目标
            logger.warning(
                f"提供的特征数据长度 ({self.num_samples}) 不足以根据窗口大小 ({window_size}) 构建任何样本。 "
                f"至少需要 {window_size} 条数据。TimeSeriesDataset 将为空。"
            )
            self.features = np.array([])
            self.targets = np.array([])
            self.num_windows = 0
        else:
            self.features = features
            self.targets = targets
            # 可生成的窗口数量 = 总样本数 - 窗口大小 + 1
            self.num_windows = self.num_samples - window_size + 1 # 例如：数据 [0,1,2,3,4], window_size=3. 窗口起始索引 0, 1, 2. Num windows = 5 - 3 + 1 = 3.
        self.window_size = window_size

        logger.info(
            f"TimeSeriesDataset 初始化: "
            f"原始样本数={self.num_samples}, 窗口大小={self.window_size}, "
            f"可生成窗口数={self.num_windows}"
        )


    def __len__(self) -> int:
        """返回数据集中可生成的窗口数量（即总样本数）。"""
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的窗口数据和目标值。
        目标是：使用时间步 [idx, idx+1, ..., idx+window_size-1] 的特征，
                 来预测/拟合时间步 idx+window_size-1 的 target。

        Args:
            idx (int): 窗口的起始索引（在可生成窗口列表中的索引，不是原始数据索引）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 一个包含输入特征窗口 (X) 和目标值 (y) 的元组。
                                               X 形状: (window_size, num_features)
                                               y 形状: (1,)
        """
        if self.num_windows == 0:
            # 返回空的 tensor，保持形状一致性，但 batch size 为 0
            num_features_dim = self.features.shape[1] if self.features.ndim == 2 and self.features.shape[0] > 0 else 0
            return torch.empty((self.window_size, num_features_dim), dtype=torch.float32), torch.empty((1,), dtype=torch.float32)

        # 窗口的起始索引在原始平坦数据中是 idx
        window_start_flat_idx = idx
        # 窗口的结束索引（不包含）在原始平坦数据中是 idx + window_size
        window_end_flat_idx = idx + self.window_size

        # 目标 (target) 对应于窗口中最后一个时间步的信号。
        # 窗口中最后一个时间步在原始平坦数据中的索引是: window_end_flat_idx - 1
        target_flat_idx = window_end_flat_idx - 1

        # 从平坦特征中提取窗口
        # X_window 包含从 window_start_flat_idx 到 window_end_flat_idx-1 的特征
        X_window_np = self.features[window_start_flat_idx:window_end_flat_idx, :]

        # 从平坦目标中提取对应于窗口最后一个时间步的目标值
        y_target_np = self.targets[target_flat_idx]

        # 转换为 PyTorch Tensor
        X_tensor = torch.tensor(X_window_np, dtype=torch.float32)
        y_tensor = torch.tensor([y_target_np], dtype=torch.float32) # 回归目标通常是单值

        return X_tensor, y_tensor

class PositionalEncoding(nn.Module):
    """
    标准正弦位置编码层。
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 创建一个足够大的PE矩阵，max_len是序列最大长度
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # 形状变为 (max_len, 1, d_model)
        self.register_buffer('pe', pe) # 将PE矩阵注册为buffer，不会被视为模型参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入序列，形状 (seq_len, batch_size, d_model)
        Returns:
            torch.Tensor: 加上位置编码后的输出，形状 (seq_len, batch_size, d_model)
        """
        # 将输入形状从 (batch_size, seq_len, num_features) 转换为 (seq_len, batch_size, d_model)
        # 假设输入 x 已经通过线性层调整到 d_model 维度
        x = x.transpose(0, 1) # (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :] # 只取与当前序列长度相同的PE部分
        return x.transpose(0, 1) # 变回 (batch_size, seq_len, d_model)

class TransformerModel(nn.Module):
    """
    TransformerEncoder 用于时间序列回归预测的模型。
    """
    def __init__(self, num_features: int, d_model: int, nhead: int, dim_feedforward: int, nlayers: int,
                 dropout: float = 0.5, activation: str = 'relu', window_size: int = 60):
        """
        Args:
            num_features (int): 输入特征的维度。
            d_model (int): Transformer模型的特征维度 (必须是 nhead 的整数倍)。
            nhead (int): 多头注意力机制的头数。
            dim_feedforward (int): Transformer内部前馈网络的维度。
            nlayers (int): TransformerEncoder层的数量。
            dropout (float): Dropout率。
            activation (str): Transformer内部前馈网络的激活函数 ('relu' 或 'gelu')。
            window_size (int): 输入序列长度（窗口大小），用于位置编码。
        """
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.window_size = window_size

        # 输入层：将 num_features 映射到 d_model 维度
        self.embedding = nn.Linear(num_features, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size) # 使用 window_size 作为 max_len

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        # 输出层：TransformerEncoder 输出形状是 (batch_size, seq_len, d_model)
        # 我们需要一个固定大小的输出，可以通过池化或只取最后一个时间步的输出来实现
        # 这里使用平均池化，也可以尝试其他池化方式
        self.pooling = nn.AdaptiveAvgPool1d(1) # 池化到长度 1

        # 回归头：一个或多个全连接层
        # 池化后的形状是 (batch_size, d_model, 1)，squeeze后是 (batch_size, d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(), # 或其他激活函数
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # 输出一个标量预测值 (例如信号得分)
        )

        self.init_weights() # 初始化权重

    def init_weights(self):
        """
        初始化模型权重。
        """
        # 对线性层使用 Xavier 初始化
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.regressor[-1].bias.data.zero_()
        self.regressor[-1].weight.data.uniform_(-initrange, initrange)
        # Transformer Encoder 内部权重由其模块自身初始化

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            src (torch.Tensor): 输入序列，形状 (batch_size, window_size, num_features)。

        Returns:
            torch.Tensor: 输出预测值，形状 (batch_size, 1)。
        """
        # 输入形状: (batch_size, window_size, num_features)
        # 经过 embedding: (batch_size, window_size, d_model)
        src = self.embedding(src)
        # 加上位置编码: (batch_size, window_size, d_model)
        src = self.pos_encoder(src) # 位置编码已经处理了输入形状转换

        # 经过 Transformer Encoder: (batch_size, window_size, d_model)
        output = self.transformer_encoder(src)

        # 池化: (batch_size, d_model, window_size) -> (batch_size, d_model, 1)
        # TransformerEncoder 输出是 (batch_size, seq_len, d_model)
        # PyTorch AdaptiveAvgPool1d 期望输入 (N, C, L)，所以需要转置
        pooled_output = self.pooling(output.transpose(1, 2)).squeeze(2) # shape (batch_size, d_model)

        # 经过回归头: (batch_size, 1)
        prediction = self.regressor(pooled_output)

        return prediction

@log_execution_time
@handle_exceptions
def prepare_data_for_transformer(
    data: pd.DataFrame,
    required_columns: List[str],
    target_column: str = 'final_signal',
    scaler_type: str = 'minmax',
    train_split: float = 0.7,
    val_split: float = 0.15,
    apply_variance_threshold: bool = False,
    variance_threshold_value: float = 0.01,
    use_pca: bool = False,
    pca_n_components: Union[int, float] = 0.99,
    pca_solver: str = 'auto',
    use_feature_selection: bool = True,
    feature_selector_model_type: str = 'rf',
    fs_model_n_estimators: int = 100,
    fs_model_max_depth: Optional[int] = None,
    fs_max_features: Optional[int] = 50,
    fs_selection_threshold: Union[str, float] = 'median',
    target_scaler_type: str = 'minmax'
) -> Tuple[
    np.ndarray, np.ndarray, # train_features, train_targets
    np.ndarray, np.ndarray, # val_features, val_targets
    np.ndarray, np.ndarray, # test_features, test_targets
    Union[MinMaxScaler, StandardScaler, None], # feature_scaler
    Union[MinMaxScaler, StandardScaler, None], # target_scaler
    List[str] # selected_feature_names
]:
    """
    为 Transformer 模型准备平坦化、经过缩放的时间序列数据 (NumPy 数组)。
    此函数负责数据清洗、特征工程（方差过滤、PCA、基于模型的选择）、数据分割和缩放。
    数据分割严格按时间顺序，确保无未来数据泄露。
    窗口化操作由后续的 TimeSeriesDataset 处理。

    Args:
        data (pd.DataFrame): 包含所有特征和目标列的原始DataFrame。
        required_columns (List[str]): 用于初始筛选的原始特征列名列表。
        target_column (str): 目标变量的列名。
        scaler_type (str): 特征缩放器类型 ('minmax' 或 'standard')。
        train_split (float): 训练集在总数据中的比例。
        val_split (float): 验证集在总数据中的比例。测试集为剩余部分。
        apply_variance_threshold (bool): 是否应用方差阈值进行特征选择。
        variance_threshold_value (float): 方差阈值的具体值。
        use_pca (bool): 是否应用PCA进行降维。如果为 True，则忽略基于模型的特征选择。
        pca_n_components (Union[int, float]): PCA保留的主成分数量或解释方差的比例。
        pca_solver (str): PCA求解器类型，如 'auto', 'full', 'arpack', 'randomized'。
        use_feature_selection (bool): 是否启用基于模型的特征选择 (在PCA之后，如果PCA未启用)。
        feature_selector_model_type (str): 特征选择模型类型 ('rf' 代表 RandomForest, 'xgb' 代表 XGBoost)。
        fs_model_n_estimators (int): 特征选择模型 (如RandomForest) 的 `n_estimators` 参数。
        fs_model_max_depth (Optional[int]): 特征选择模型 (如RandomForest) 的 `max_depth` 参数。
        fs_max_features (Optional[int]): 基于模型选择时，要保留的最大特征数量。若为 None，则使用阈值选择。
        fs_selection_threshold (Union[str, float]): 特征重要性阈值 (如 'median', 'mean', 或一个浮点数).
                                                    仅在 `fs_max_features` 为 None 时生效。
        target_scaler_type (str): 目标变量缩放器类型 ('minmax' 或 'standard')。

    Returns:
        Tuple:
            - features_scaled_train (np.ndarray): 缩放后的训练集特征 (NumPy)。
            - targets_scaled_train (np.ndarray): 缩放后的训练集目标 (NumPy)。
            - features_scaled_val (np.ndarray): 缩放后的验证集特征 (NumPy)。
            - targets_scaled_val (np.ndarray): 缩放后的验证集目标 (NumPy)。
            - features_scaled_test (np.ndarray): 缩放后的测试集特征 (NumPy)。
            - targets_scaled_test (np.ndarray): 缩放后的测试集目标 (NumPy)。
            - feature_scaler (Union[MinMaxScaler, StandardScaler, None]): 拟合好的特征缩放器。
            - target_scaler (Union[MinMaxScaler, StandardScaler, None]): 拟合好的目标缩放器。
            - final_selected_feature_names (List[str]): 最终被选入模型的特征名列表。
    """
    logger.info("开始为 Transformer 模型准备平坦化输入数据...")
    log_params = {k: v for k, v in locals().items() if k not in ['data', 'required_columns']}
    logger.info(f"数据准备参数: {log_params}")

    # --- 0. 复制数据 ---
    data_processed = data.copy()

    # --- 1. 检查目标列 ---
    if target_column not in data_processed.columns:
        logger.error(f"目标列 '{target_column}' 不存在于输入数据中。可用列: {data_processed.columns.tolist()}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None, []

    # --- 2. 初始特征列选择 ---
    initial_feature_names = [col for col in required_columns if col in data_processed.columns and col != target_column]
    if not initial_feature_names:
         logger.error("根据 required_columns 筛选后，没有可用的特征列。")
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None, []
    logger.info(f"初始筛选后，特征列数量: {len(initial_feature_names)}。部分列名: {initial_feature_names[:10]}...")

    # --- 3. 处理 NaN 值 ---
    features_df = data_processed[initial_feature_names]
    targets_series = data_processed[target_column]

    features_filled_df = features_df.ffill().bfill()
    nan_cols_features = features_filled_df.isnull().sum()
    nan_cols_features = nan_cols_features[nan_cols_features > 0]
    if not nan_cols_features.empty:
        logger.warning(f"特征数据在ffill().bfill()后，以下列仍包含NaN (将被填充为0): {nan_cols_features.index.tolist()}")
        features_filled_df.fillna(0, inplace=True)

    targets_filled_series = targets_series.ffill().bfill()
    if targets_filled_series.isnull().any():
        logger.warning(f"目标列 '{target_column}' 在ffill().bfill()后仍包含NaN (将被填充为0)。请检查目标数据质量。")
        targets_filled_series.fillna(0, inplace=True)

    current_features_np = features_filled_df.values
    current_feature_names = initial_feature_names[:]
    targets_np = targets_filled_series.values

    logger.info(f"NaN值处理完成。特征矩阵形状: {current_features_np.shape}, 目标数组形状: {targets_np.shape}")

    # --- 4. (可选) 方差阈值过滤 ---
    if apply_variance_threshold:
        if current_features_np.shape[0] > 1 and current_features_np.shape[1] > 0:
            variances = np.var(current_features_np, axis=0)
            if np.any(variances > 1e-9):
                try:
                    selector_var = VarianceThreshold(threshold=variance_threshold_value)
                    features_after_var = selector_var.fit_transform(current_features_np)
                    selected_indices_var = selector_var.get_support(indices=True)

                    if features_after_var.shape[1] == 0:
                        logger.warning(f"方差阈值 ({variance_threshold_value}) 过高，所有特征均被移除。将跳过方差过滤。")
                    elif features_after_var.shape[1] < current_features_np.shape[1]:
                        num_removed = current_features_np.shape[1] - features_after_var.shape[1]
                        current_features_np = features_after_var
                        current_feature_names = [current_feature_names[i] for i in selected_indices_var]
                        logger.info(f"方差阈值过滤完成，移除了 {num_removed} 个低方差特征。剩余特征数: {current_features_np.shape[1]}")
                        logger.debug(f"方差过滤后特征名 (部分): {current_feature_names[:10]}...")
                    else:
                        logger.info("方差阈值过滤未移除任何特征。")
                except Exception as e_var:
                     logger.error(f"应用方差阈值时出错: {e_var}", exc_info=True)
                     logger.warning("方差阈值选择失败，将使用之前的特征。")
            else:
                 logger.warning("所有特征方差接近零，跳过方差阈值选择。")
        else:
             logger.warning("特征数据不足 (样本或特征数为0/1)，跳过方差阈值选择。")

    if current_features_np.shape[1] == 0:
         logger.error("经过初始处理和方差过滤后，特征数量为零。无法继续。")
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None, []

    # --- 5. 按时间顺序分割数据集 ---
    n_samples_total = current_features_np.shape[0]
    if n_samples_total == 0:
        logger.error("数据为空，无法进行分割。")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None, []

    if not (0 < train_split < 1 and 0 <= val_split < 1 and train_split + val_split <= 1):
        logger.error(f"无效的数据集分割比例: train_split={train_split}, val_split={val_split}。比例应在(0,1)或[0,1)范围内，且总和<=1。")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None, []

    n_train = int(n_samples_total * train_split)
    n_val = int(n_samples_total * val_split)
    # n_test = n_samples_total - n_train - n_val # 测试集是剩余部分

    if n_train == 0:
        logger.error(f"计算得到的训练集样本数为零 (总样本数: {n_samples_total}, 训练比例: {train_split})。请增加数据量或调整比例。")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None, []

    features_train_raw = current_features_np[:n_train]
    features_val_raw = current_features_np[n_train : n_train + n_val]
    features_test_raw = current_features_np[n_train + n_val:]
    targets_train_raw = targets_np[:n_train]
    targets_val_raw = targets_np[n_train : n_train + n_val]
    targets_test_raw = targets_np[n_train + n_val:]

    logger.info(f"数据按时间顺序分割完成: "
                f"训练集 {features_train_raw.shape[0]}条, "
                f"验证集 {features_val_raw.shape[0]}条, "
                f"测试集 {features_test_raw.shape[0]}条。")

    features_eng_train = features_train_raw
    features_eng_val = features_val_raw
    features_eng_test = features_test_raw
    feature_names_after_eng = current_feature_names[:]

    # --- 6. (可选) PCA 降维 ---
    pca_applied = False
    if use_pca:
        if features_eng_train.shape[1] <= 1:
            logger.warning(f"训练集特征维度 ({features_eng_train.shape[1]}) 过低，跳过PCA。")
        elif features_eng_train.shape[0] < features_eng_train.shape[1] and isinstance(pca_n_components, float):
             logger.warning(f"训练集样本数 ({features_eng_train.shape[0]}) 小于特征数 ({features_eng_train.shape[1]})，PCA解释方差比例可能不可靠。建议使用固定数量的主成分或增加样本。")
        else:
            logger.info(f"启用 PCA 降维 (n_components={pca_n_components}, solver='{pca_solver}')。在训练集上拟合...")
            scaler_for_pca = StandardScaler()
            try:
                features_train_scaled_for_pca = scaler_for_pca.fit_transform(features_eng_train)

                pca = PCA(n_components=pca_n_components, svd_solver=pca_solver, random_state=42)
                pca.fit(features_train_scaled_for_pca)
                num_components_retained = pca.n_components_
                explained_variance_ratio = sum(pca.explained_variance_ratio_)
                logger.info(f"PCA 拟合完成。保留主成分数: {num_components_retained}, 累计解释方差比: {explained_variance_ratio:.4f}")

                if num_components_retained == 0:
                    logger.error("PCA 降维后特征维度为零。PCA失败，将使用原始特征。")
                else:
                    features_eng_train = pca.transform(features_train_scaled_for_pca)
                    if features_eng_val.shape[0] > 0:
                        features_eng_val = pca.transform(scaler_for_pca.transform(features_eng_val))
                    if features_eng_test.shape[0] > 0:
                        features_eng_test = pca.transform(scaler_for_pca.transform(features_eng_test))

                    feature_names_after_eng = [f"pca_comp_{i}" for i in range(num_components_retained)]
                    logger.info(f"PCA 转换完成。处理后特征维度: {num_components_retained}")
                    logger.debug(f"PCA转换后特征名 (部分): {feature_names_after_eng[:10]}...")
                    pca_applied = True
            except Exception as e_pca:
                logger.error(f"应用 PCA 时出错: {e_pca}", exc_info=True)
                logger.warning("PCA 降维失败，将使用之前的特征。")

    # --- 7. (可选) 基于模型的特征选择 (仅当PCA未应用时) ---
    if use_feature_selection and not pca_applied:
        if features_eng_train.shape[1] <= 1:
            logger.warning(f"训练集特征维度 ({features_eng_train.shape[1]}) 过低，跳过基于模型的特征选择。")
        elif features_eng_train.shape[0] <= 1:
             logger.warning(f"训练集样本数 ({features_eng_train.shape[0]}) 过低，无法进行基于模型的特征选择。")
        else:
            logger.info(f"启用基于模型 '{feature_selector_model_type}' 的特征选择。在训练集上拟合...")
            logger.info(f"特征选择模型参数: n_estimators={fs_model_n_estimators}, max_depth={fs_model_max_depth}, "
                        f"选择方式: max_features={fs_max_features} 或 threshold='{fs_selection_threshold}'")

            selector_model_instance = None
            model_type_lower = feature_selector_model_type.lower()

            if model_type_lower == 'rf':
                selector_model_instance = RandomForestRegressor(
                    n_estimators=fs_model_n_estimators,
                    max_depth=fs_model_max_depth,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type_lower == 'xgb':
                if not XGB_AVAILABLE:
                    logger.warning("XGBoost 未安装，无法使用 XGBoost 进行特征选择。将回退到 RandomForest。")
                    selector_model_instance = RandomForestRegressor(
                        n_estimators=fs_model_n_estimators,
                        max_depth=fs_model_max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    selector_model_instance = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        n_estimators=fs_model_n_estimators,
                        max_depth=fs_model_max_depth,
                        random_state=42,
                        n_jobs=-1,
                    )
            else:
                logger.warning(f"不支持的特征选择模型: {feature_selector_model_type}。将使用 RandomForest。")
                selector_model_instance = RandomForestRegressor(
                    n_estimators=fs_model_n_estimators,
                    max_depth=fs_model_max_depth,
                    random_state=42,
                    n_jobs=-1
                )

            try:
                if fs_max_features is not None and fs_max_features > 0:
                    actual_max_fs = min(fs_max_features, features_eng_train.shape[1])
                    selector_fs = SelectFromModel(
                        selector_model_instance,
                        max_features=actual_max_fs,
                        threshold=-np.inf
                    )
                    logger.info(f"特征选择方式: 最多选择 {actual_max_fs} 个特征。")
                else:
                    selector_fs = SelectFromModel(
                        selector_model_instance,
                        threshold=fs_selection_threshold
                    )
                    logger.info(f"特征选择方式: 使用阈值 '{fs_selection_threshold}'。")

                selector_fs.fit(features_eng_train, targets_train_raw)

                selected_indices_fs = selector_fs.get_support(indices=True)
                num_selected_fs = len(selected_indices_fs)

                if num_selected_fs == 0:
                    logger.error("基于模型选择后没有剩余特征。特征选择失败，将使用之前的特征。")
                elif num_selected_fs < features_eng_train.shape[1]:
                    num_removed_fs = features_eng_train.shape[1] - num_selected_fs
                    features_eng_train = selector_fs.transform(features_eng_train)
                    if features_eng_val.shape[0] > 0:
                        features_eng_val = selector_fs.transform(features_eng_val)
                    if features_eng_test.shape[0] > 0:
                        features_eng_test = selector_fs.transform(features_eng_test)

                    feature_names_after_eng = [feature_names_after_eng[i] for i in selected_indices_fs]
                    logger.info(f"基于模型选择完成，移除了 {num_removed_fs} 个特征。剩余特征数: {num_selected_fs}")
                    logger.debug(f"模型选择后特征名 (部分): {feature_names_after_eng[:10]}...")
                else:
                    logger.info("基于模型选择未移除任何特征 (可能所有特征都很重要或已达到最大选择数)。")

            except Exception as e_fs:
                logger.error(f"应用基于模型的特征选择时出错: {e_fs}", exc_info=True)
                logger.warning("特征选择失败，将使用之前的特征。")
    elif pca_applied:
        logger.info("PCA已应用，跳过基于模型的特征选择。")
    else:
        logger.info("未启用基于模型的特征选择。")

    final_features_train = features_eng_train
    final_features_val = features_eng_val
    final_features_test = features_eng_test
    final_selected_feature_names = feature_names_after_eng[:]

    if final_features_train.shape[1] == 0:
         logger.error("经过所有特征工程步骤后，训练集特征维度为零。无法继续。")
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None, []
    logger.info(f"所有特征工程处理完成。最终用于缩放的特征维度: {final_features_train.shape[1]}")

    # --- 8. 特征缩放 (在处理后的训练集特征上拟合) ---
    feature_scaler: Union[MinMaxScaler, StandardScaler, None] = None
    if scaler_type.lower() == 'minmax':
        feature_scaler = MinMaxScaler()
    elif scaler_type.lower() == 'standard':
        feature_scaler = StandardScaler()
    else:
        logger.warning(f"不支持的特征缩放器类型: {scaler_type}。将使用 MinMaxScaler。")
        feature_scaler = MinMaxScaler()

    features_scaled_train = np.array([])
    features_scaled_val = np.array([])
    features_scaled_test = np.array([])

    if final_features_train.shape[0] > 0 and final_features_train.shape[1] > 0:
        feature_scaler.fit(final_features_train)
        features_scaled_train = feature_scaler.transform(final_features_train)
        if final_features_val.shape[0] > 0:
            features_scaled_val = feature_scaler.transform(final_features_val)
        if final_features_test.shape[0] > 0:
            features_scaled_test = feature_scaler.transform(final_features_test)
        logger.info(f"特征缩放完成 (使用 {scaler_type} scaler)。")
    else:
        logger.warning("处理后的训练集特征为空或无特征维度，跳过特征缩放。特征缩放器未拟合。")
        feature_scaler = None
        features_scaled_train = final_features_train
        features_scaled_val = final_features_val
        features_scaled_test = final_features_test

    # --- 9. 目标变量缩放 (在原始训练集目标上拟合) ---
    target_scaler: Union[MinMaxScaler, StandardScaler, None] = None
    if target_scaler_type.lower() == 'minmax':
        target_scaler = MinMaxScaler()
    elif target_scaler_type.lower() == 'standard':
        target_scaler = StandardScaler()
    else:
        logger.warning(f"不支持的目标变量缩放器类型: {target_scaler_type}。将使用 MinMaxScaler。")
        target_scaler = MinMaxScaler()

    targets_scaled_train = np.array([])
    targets_scaled_val = np.array([])
    targets_scaled_test = np.array([])

    if targets_train_raw.shape[0] > 0:
        target_scaler.fit(targets_train_raw.reshape(-1, 1))
        targets_scaled_train = target_scaler.transform(targets_train_raw.reshape(-1, 1)).flatten()
        if targets_val_raw.shape[0] > 0:
            targets_scaled_val = target_scaler.transform(targets_val_raw.reshape(-1, 1)).flatten()
        if targets_test_raw.shape[0] > 0:
            targets_scaled_test = target_scaler.transform(targets_test_raw.reshape(-1, 1)).flatten()
        logger.info(f"目标变量缩放完成 (使用 {target_scaler_type} scaler)。")
    else:
        logger.warning("训练集目标数据为空，跳过目标变量缩放。目标缩放器未拟合。")
        target_scaler = None
        targets_scaled_train = targets_train_raw

    logger.info("Transformer 数据准备流程结束。")
    # 返回 NumPy 数组和 scaler 对象
    return (
        features_scaled_train.astype(np.float32), targets_scaled_train.astype(np.float32),
        features_scaled_val.astype(np.float32), targets_scaled_val.astype(np.float32),
        features_scaled_test.astype(np.float32), targets_scaled_test.astype(np.float32),
        feature_scaler, target_scaler,
        final_selected_feature_names
    )


@log_execution_time
@handle_exceptions
def build_transformer_model(
    num_features: int,
    model_config: Dict[str, Any],
    summary: bool = True,
    window_size: int = 60 # 需要窗口大小来初始化位置编码
) -> TransformerModel:
    """
    构建 TransformerEncoder 模型。

    Args:
        num_features (int): 输入特征的维度。
        model_config (Dict[str, Any]): 包含模型配置参数的字典。
            - 'd_model': Transformer模型的特征维度。
            - 'nhead': 多头注意力机制的头数。
            - 'dim_feedforward': 前馈网络的维度。
            - 'nlayers': TransformerEncoder层的数量。
            - 'dropout': Dropout率。
            - 'activation': 激活函数 ('relu' 或 'gelu')。
            - 'learning_rate': 优化器学习率 (这里不用于构建模型，但可供参考)。
            - 'weight_decay': 优化器权重衰减 (L2 正则化)。
        summary (bool): 是否打印模型结构 (简略)。
        window_size (int): 输入序列长度（窗口大小），用于位置编码。

    Returns:
        TransformerModel: 构建好的 PyTorch Transformer 模型。
    """
    logger.info("开始构建 Transformer 模型...")
    logger.info(f"模型配置: num_features={num_features}, window_size={window_size}, config={model_config}")

    if num_features <= 0:
        raise ValueError(f"特征数量必须大于0，当前为: {num_features}")
    if window_size <= 0:
        raise ValueError(f"窗口大小必须大于0，当前为: {window_size}")

    d_model = model_config.get('d_model', 128)
    nhead = model_config.get('nhead', 8)
    dim_feedforward = model_config.get('dim_feedforward', 512)
    nlayers = model_config.get('nlayers', 4)
    dropout = model_config.get('dropout', 0.2)
    activation = model_config.get('activation', 'relu')

    # 检查 d_model 必须是 nhead 的整数倍
    if d_model % nhead != 0:
        raise ValueError(f"模型的特征维度 d_model ({d_model}) 必须是注意力头数 nhead ({nhead}) 的整数倍。")


    model = TransformerModel(
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        nlayers=nlayers,
        dropout=dropout,
        activation=activation,
        window_size=window_size # 传递窗口大小
    )

    # 可以选择打印模型结构，但对于复杂的 PyTorch 模型，summary() 方法不像 Keras 那么标准化和详细
    # 可以手动打印层信息
    if summary:
        logger.info("Transformer 模型结构:")
        logger.info(model) # 直接打印模型对象会显示模块结构
        # logger.info(f"总参数数量: {sum(p.numel() for p in model.parameters())}") # 统计参数数量

    logger.info("Transformer 模型构建完成。")
    return model


@log_execution_time
@handle_exceptions
def train_transformer_model(
    model: TransformerModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    target_scaler: Union[MinMaxScaler, StandardScaler], # 用于反标准化以评估真实MAE
    training_config: Dict[str, Any],
    checkpoint_dir: str,
    stock_code: Optional[str] = "STOCK",
    plot_training_history: bool = False
) -> Tuple[TransformerModel, pd.DataFrame]:
    """
    训练 Transformer 模型。

    Args:
        model (TransformerModel): 已构建好的 PyTorch Transformer 模型。
        train_loader (DataLoader): 训练数据加载器。
        val_loader (Optional[DataLoader]): 验证数据加载器 (可选)。
        target_scaler (Union[MinMaxScaler, StandardScaler]): 用于目标变量的缩放器，以便在评估时计算反标准化的MAE。
        training_config (Dict[str, Any]): 包含训练参数的字典。
            - 'epochs': 训练轮数。
            - 'learning_rate': 学习率。
            - 'weight_decay': 权重衰减 (L2 正则化)。
            - 'optimizer': 优化器名称 (如 'adam', 'adamw', 'rmsprop', 'sgd')。
            - 'loss': 损失函数名称 (如 'mse', 'mae', 'huber')。
            - 'early_stopping_patience': 早停的耐心轮数。
            - 'reduce_lr_patience': 降低学习率的耐心轮数。
            - 'reduce_lr_factor': 学习率降低因子。
            - 'monitor_metric': 早停和降低学习率监控的指标 (如 'val_loss', 'val_mae')。
            - 'tensorboard_log_dir': TensorBoard 日志的基础目录 (可选)。
            - 'verbose': 训练过程的打印详细程度 (0: 无输出, 1: 每轮输出, 2: 每批输出)。
            - 'clip_grad_norm': 梯度裁剪的范数最大值 (可选)。
        checkpoint_dir (str): 用于保存最佳模型检查点和TensorBoard日志的目录。
        stock_code (Optional[str]): 股票代码，用于 TensorBoard 日志命名和模型文件名。
        plot_training_history (bool): 是否绘制训练历史曲线。

    Returns:
        Tuple[TransformerModel, pd.DataFrame]:
            - model (TransformerModel): 训练完成（或加载的最佳）模型。
            - history_df (pd.DataFrame): 包含训练历史的DataFrame。
    """
    logger.info("开始训练 Transformer 模型...")
    logger.info(f"训练配置: {training_config}")

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model.to(device)

    # --- 优化器设置 ---
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0) # PyTorch AdamW 内置 weight_decay
    momentum = training_config.get('momentum', 0.0) # 用于 SGD, RMSprop

    optimizer: optim.Optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # weight_decay is L2
    elif optimizer_name == 'adamw':
         # AdamW 有一个不同的 weight decay 实现
         # 如果 model_config 也定义了 weight_decay，优先使用 training_config 的
         effective_weight_decay = training_config.get('weight_decay', model.regressor[0].weight.decay) # Example: Check regressor layer for decay
         optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=effective_weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        logger.warning(f"未知的优化器名称: {optimizer_name}。将使用 Adam (lr={learning_rate}, weight_decay={weight_decay})。")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- 损失函数设置 ---
    loss_name = training_config.get('loss', 'mse').lower()
    criterion: nn.Module # PyTorch 损失函数是 nn.Module
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'mae':
        criterion = nn.L1Loss() # L1Loss 就是 MAE
    elif loss_name == 'huber':
         # HuberLoss 结合 MSE 和 MAE，对异常值不敏感
         # delta 参数需要根据目标值范围调整，例如，如果目标范围 0-100，delta 可以设为 10
         huber_delta = training_config.get('huber_delta', 1.0) # Default delta is 1.0
         # 如果目标变量范围差异大，可能需要根据 target_scaler 的信息调整 delta
         # 或者在训练前对目标进行标准化，再使用一个固定的 delta
         # 考虑到我们已经缩放了目标变量到 0-1 或其他范围，delta=1.0 通常是合理的起点
         criterion = nn.HuberLoss(delta=huber_delta)
         logger.info(f"使用 HuberLoss (delta={huber_delta:.2f})")
    else:
        logger.warning(f"未知的损失函数名称: {loss_name}。将使用 MSELoss。")
        criterion = nn.MSELoss()

    # --- 评估指标设置 ---
    # 除了 loss，我们通常还需要 MAE 作为评估指标
    # PyTorch 没有内置的 metric 列表，需要手动计算
    # 这里我们只关注 MAE
    mae_metric = nn.L1Loss() # 用于计算反标准化前的 MAE
    true_mae_metric = nn.L1Loss() # 用于计算反标准化后的真实 MAE

    # --- 回调函数/训练过程控制 ---
    epochs = training_config.get('epochs', 100)
    early_stopping_patience = training_config.get('early_stopping_patience', 30)
    reduce_lr_patience = training_config.get('reduce_lr_patience', 10)
    reduce_lr_factor = training_config.get('reduce_lr_factor', 0.5)
    # monitor_metric: 'val_loss' 或 'val_mae'
    monitor_metric = training_config.get('monitor_metric', 'val_loss')
    verbose_level = training_config.get('verbose', 1)
    clip_grad_norm_value = training_config.get('clip_grad_norm', None) # 梯度裁剪

    # 学习率调度器 (监控验证集指标)
    scheduler = None
    if val_loader is not None and len(val_loader) > 0:
         # PyTorch ReduceLROnPlateau 根据监控指标调整学习率
         # mode='min' for loss/mae, 'max' for accuracy/R2
         scheduler_mode = 'min' if 'loss' in monitor_metric.lower() or 'mae' in monitor_metric.lower() else 'max'
         scheduler = ReduceLROnPlateau(
             optimizer,
             mode=scheduler_mode,
             factor=reduce_lr_factor,
             patience=reduce_lr_patience,
             verbose=True, # PyTorch scheduler has its own verbose
             min_lr=1e-7
         )
         logger.info(f"启用 ReduceLROnPlateau: monitor='{monitor_metric}', mode='{scheduler_mode}', factor={reduce_lr_factor}, patience={reduce_lr_patience}")
    elif reduce_lr_patience > 0:
         logger.warning("验证集为空或未提供，ReduceLROnPlateau 将被禁用。")
         scheduler = None

    # 早停逻辑
    best_val_metric = float('inf') if 'loss' in monitor_metric.lower() or 'mae' in monitor_metric.lower() else float('-inf')
    epochs_no_improve = 0
    early_stop = False

    # 模型检查点路径
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_filepath = os.path.join(checkpoint_dir, f"best_model_{stock_code}.pth") # PyTorch typically saves .pth

    # TensorBoard 设置
    writer = None
    tensorboard_log_dir_base = training_config.get('tensorboard_log_dir', None)
    if tensorboard_log_dir_base:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir = os.path.join(tensorboard_log_dir_base, f"{stock_code}_{current_time}_Transformer") # 区分Transformer
        os.makedirs(run_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_log_dir)
        logger.info(f"启用 TensorBoard: 日志保存到 '{run_log_dir}'")


    # 训练历史记录
    history = {
        'loss': [], 'mae': [],
        'val_loss': [], 'val_mae': [],
        'lr': []
    }


    # --- 训练循环 ---
    logger.info(f"使用 {len(train_loader)} 个训练批次和 {len(val_loader) if val_loader else 0} 个验证批次进行训练。")

    if len(train_loader) == 0:
        logger.error("训练 DataLoader 为空，无法进行模型训练。")
        if writer: writer.close()
        return model, pd.DataFrame(history)


    for epoch in range(epochs):
        start_time = time.time()
        model.train() # 设置模型为训练模式
        total_loss = 0
        total_mae = 0
        num_batches = 0

        # 训练阶段
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device) # 移动数据到设备

            optimizer.zero_grad() # 梯度清零
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, targets) # 计算损失

            loss.backward() # 反向传播

            # 梯度裁剪 (可选)
            if clip_grad_norm_value is not None:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)

            optimizer.step() # 更新权重

            total_loss += loss.item()
            total_mae += mae_metric(outputs, targets).item() # 计算 MAE
            num_batches += 1

            if verbose_level == 2:
                 logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss={loss.item():.4f}, MAE={mae_metric(outputs, targets).item():.4f}")


        avg_train_loss = total_loss / num_batches
        avg_train_mae = total_mae / num_batches
        current_lr = optimizer.param_groups[0]['lr']

        history['loss'].append(avg_train_loss)
        history['mae'].append(avg_train_mae)
        history['lr'].append(current_lr)


        # 验证阶段 (如果提供了验证集)
        avg_val_loss = float('inf') # 默认设置为无穷大
        avg_val_mae = float('inf')
        avg_val_true_mae = float('inf') # 反标准化后的 MAE

        if val_loader is not None and len(val_loader) > 0:
            model.eval() # 设置模型为评估模式
            total_val_loss = 0
            total_val_mae = 0
            total_val_true_mae = 0
            num_val_batches = 0

            with torch.no_grad(): # 在验证阶段不计算梯度
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_targets)
                    val_mae = mae_metric(val_outputs, val_targets)

                    total_val_loss += val_loss.item()
                    total_val_mae += val_mae.item()

                    # 计算反标准化后的真实 MAE
                    # 将 tensor 转移回 CPU 并转换为 numpy
                    val_outputs_np = val_outputs.cpu().numpy()
                    val_targets_np = val_targets.cpu().numpy()
                    # 反标准化
                    val_outputs_original = target_scaler.inverse_transform(val_outputs_np)
                    val_targets_original = target_scaler.inverse_transform(val_targets_np)
                    # 计算真实 MAE
                    true_mae_batch = np.mean(np.abs(val_outputs_original - val_targets_original))
                    total_val_true_mae += true_mae_batch

                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches
            avg_val_mae = total_val_mae / num_val_batches
            avg_val_true_mae = total_val_true_mae / num_val_batches

            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(avg_val_mae)

            # 学习率调度器 step (基于验证集指标)
            # 确定监控的验证集指标的值
            monitored_val_value = avg_val_loss if monitor_metric.lower() == 'val_loss' else avg_val_mae # Assuming 'val_mae' is the other option
            scheduler.step(monitored_val_value)

            # 早停判断
            if (scheduler_mode == 'min' and monitored_val_value < best_val_metric) or \
               (scheduler_mode == 'max' and monitored_val_value > best_val_metric):
                best_val_metric = monitored_val_value
                epochs_no_improve = 0
                # 保存最佳模型权重
                torch.save(model.state_dict(), best_model_filepath) # 只保存模型状态字典
                logger.info(f"Epoch {epoch+1}: 验证集 {monitor_metric} 提升 ({best_val_metric:.4f})。保存最佳模型权重。")
            else:
                epochs_no_improve += 1
                logger.info(f"Epoch {epoch+1}: 验证集 {monitor_metric} 未提升。未提升轮数: {epochs_no_improve}")
                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"验证集 {monitor_metric} 连续 {early_stopping_patience} 轮未提升，触发早停。")
                    early_stop = True

        end_time = time.time()
        epoch_duration = end_time - start_time

        if verbose_level >= 1:
             log_message = f"Epoch {epoch+1}/{epochs} - {epoch_duration:.2f}s - loss: {avg_train_loss:.4f} - mae: {avg_train_mae:.4f}"
             if val_loader is not None and len(val_loader) > 0:
                 log_message += f" - val_loss: {avg_val_loss:.4f} - val_mae: {avg_val_mae:.4f} - val_true_mae: {avg_val_true_mae:.2f}"
             log_message += f" - lr: {current_lr:.6f}"
             logger.info(log_message)


        # TensorBoard 记录
        if writer:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('MAE/train', avg_train_mae, epoch)
            if val_loader is not None and len(val_loader) > 0:
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('MAE/val', avg_val_mae, epoch)
                writer.add_scalar('True MAE/val', avg_val_true_mae, epoch)
            writer.add_scalar('Learning Rate', current_lr, epoch)

        if early_stop:
            break

    # --- 训练结束 ---
    logger.info("模型训练完成。")

    # 加载在训练过程中保存的最佳模型权重
    if os.path.exists(best_model_filepath):
        try:
            logger.info(f"从 '{best_model_filepath}' 加载训练过程中的最佳模型权重...")
            model.load_state_dict(torch.load(best_model_filepath))
            logger.info("最佳模型权重加载成功。")
        except Exception as e_load:
            logger.error(f"加载最佳模型权重 '{best_model_filepath}' 失败: {e_load}。将使用训练结束时的模型权重。", exc_info=True)
    else:
        logger.warning(f"未找到已保存的最佳模型权重文件: '{best_model_filepath}'。将使用训练结束时的模型权重。")

    # 将训练历史转换为 DataFrame
    history_df = pd.DataFrame(history)
    logger.debug(f"训练历史记录:\n{history_df.tail()}")

    # 可选：在测试集上进行最终评估 (这里只计算，不影响早停和模型保存)
    # test_metrics = evaluate_transformer_model(model, test_loader, criterion, mae_metric, target_scaler, device)
    # logger.info(f"测试集评估: Loss={test_metrics['loss']:.4f}, MAE={test_metrics['mae']:.4f}, True MAE={test_metrics['true_mae']:.2f}")

    # --- (可选) 绘制训练历史 ---
    if plot_training_history and history_df is not None and not history_df.empty:
        try:
            plt.figure(figsize=(12, 8))

            # 绘制损失曲线
            plt.subplot(2, 1, 1)
            if 'loss' in history_df.columns:
                plt.plot(history_df.index, history_df['loss'], label='训练损失 (Loss)')
            if 'val_loss' in history_df.columns and not history_df['val_loss'].isnull().all():
                plt.plot(history_df.index, history_df['val_loss'], label='验证损失 (Validation Loss)')
            plt.title(f'{stock_code} 模型训练损失')
            plt.xlabel('Epoch')
            plt.ylabel('损失 (Loss)')
            plt.legend()
            plt.grid(True)

            # 绘制评估指标曲线 (例如 MAE)
            plt.subplot(2, 1, 2)
            if 'mae' in history_df.columns:
                plt.plot(history_df.index, history_df['mae'], label='训练 MAE')
            if 'val_mae' in history_df.columns and not history_df['val_mae'].isnull().all():
                plt.plot(history_df.index, history_df['val_mae'], label='验证 MAE')
            # 可以选择绘制反标准化后的 MAE
            # if 'val_true_mae' in history_df.columns and not history_df['val_true_mae'].isnull().all():
            #     plt.plot(history_df.index, history_df['val_true_mae'], label='验证 True MAE (Original Scale)', linestyle='--')
            plt.title(f'{stock_code} 模型训练评估指标 (MAE)')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plot_save_path = os.path.join(checkpoint_dir, f"training_history_{stock_code}.png")
            plt.savefig(plot_save_path)
            logger.info(f"训练历史曲线图已保存到: {plot_save_path}")
            plt.close()
        except Exception as e_plot:
            logger.error(f"绘制训练历史图表时出错: {e_plot}", exc_info=True)

    if writer: writer.close() # 关闭 TensorBoard writer
    return model, history_df

# 新增函数：在 Transformer 模型上进行预测
@log_execution_time
@handle_exceptions
def predict_with_transformer_model(
    model: TransformerModel,
    data: pd.DataFrame,
    feature_scaler: Union[MinMaxScaler, StandardScaler],
    target_scaler: Union[MinMaxScaler, StandardScaler],
    selected_feature_names: List[str],
    window_size: int,
    device: torch.device
) -> float:
    """
    使用训练好的 Transformer 模型对最新数据进行预测。

    Args:
        model (TransformerModel): 已加载权重的 PyTorch Transformer 模型。
        data (pd.DataFrame): 包含所有原始特征列的最新数据 DataFrame。
        feature_scaler (Union[MinMaxScaler, StandardScaler]): 拟合好的特征缩放器。
        target_scaler (Union[MinMaxScaler, StandardScaler]): 拟合好的目标缩放器。
        selected_feature_names (List[str]): 最终用于模型训练的特征名列表。
        window_size (int): 模型期望的输入窗口大小。
        device (torch.device): 预测设备 (CPU 或 GPU)。

    Returns:
        float: 预测的信号值 (原始尺度)。如果无法预测，返回 50.0。
    """
    logger.info("开始使用 Transformer 模型进行预测...")

    if data is None or data.empty or len(data) < window_size:
        logger.warning(f"输入数据为空或长度 ({len(data)}) 小于窗口大小 ({window_size})，无法进行预测。")
        return 50.0 # 返回中性预测

    # --- 准备预测数据 ---
    # 1. 选择用于预测的特征列 (必须与训练时使用的特征一致)
    features_for_prediction_df = data.loc[:, [col for col in selected_feature_names if col in data.columns]].copy()

    # 检查选中的特征列是否都存在于当前数据中
    missing_selected_features = [col for col in selected_feature_names if col not in features_for_prediction_df.columns]
    if missing_selected_features:
         logger.error(f"用于 Transformer 预测的选中特征列在当前数据中缺失: {missing_selected_features}。无法进行预测。")
         return 50.0

    # 2. 处理 NaN 值 (与数据准备时一致)
    features_for_prediction_df = features_for_prediction_df.ffill().bfill()
    if features_for_prediction_df.isnull().any().any():
        logger.warning("用于 Transformer 预测的数据中仍有 NaN 值，将尝试填充为0。")
        features_for_prediction_df.fillna(0, inplace=True)

    # 3. 应用特征缩放 (使用训练时拟合好的 scaler)
    # 确保 feature_scaler 已加载
    if feature_scaler is None:
        logger.error("特征 Scaler 未加载，无法进行预测。")
        return 50.0

    try:
        features_scaled_np = feature_scaler.transform(features_for_prediction_df.values)
    except Exception as e:
        logger.error(f"应用特征 Scaler 时出错: {e}", exc_info=True)
        return 50.0


    # 4. 构建预测所需的窗口数据 (只取最后一个窗口进行最新预测)
    # PyTorch Transformer 期望输入形状 (batch_size, sequence_length, features_dim)
    # 我们需要最后一个窗口的数据，形状 (window_size, num_features)
    # 然后为了喂给模型，需要加一个 batch dimension，形状 (1, window_size, num_features)
    if features_scaled_np.shape[0] < window_size:
        logger.warning(f"缩放后的数据长度 ({features_scaled_np.shape[0]}) 小于窗口大小 ({window_size})，无法构建预测窗口。")
        return 50.0

    # 取最后 window_size 个时间步的数据
    X_predict_np = features_scaled_np[-window_size:, :] # Shape: (window_size, num_features)

    # 转换为 PyTorch Tensor 并添加 batch dimension
    X_predict_tensor = torch.tensor(X_predict_np, dtype=torch.float32).unsqueeze(0) # Shape: (1, window_size, num_features)

    # 移动到设备
    X_predict_tensor = X_predict_tensor.to(device)
    model.to(device) # 确保模型也在同一设备

    # --- 进行预测 ---
    model.eval() # 设置模型为评估模式 (禁用 dropout 等)
    with torch.no_grad(): # 在预测阶段不计算梯度
        try:
            lstm_pred_scaled_tensor = model(X_predict_tensor) # Shape: (1, 1)
        except Exception as e:
            logger.error(f"Transformer 模型前向传播出错: {e}", exc_info=True)
            return 50.0
    # --- 逆缩放预测结果 ---
    # 将预测结果 tensor 转移回 CPU 并转换为 numpy
    lstm_pred_scaled_np = lstm_pred_scaled_tensor.cpu().numpy() # Shape: (1, 1)

    # 确保 target_scaler 已加载
    if target_scaler is None:
        logger.error("目标 Scaler 未加载，无法逆缩放预测结果。")
        # 可以选择返回缩放后的预测值，或者一个默认值
        return 50.0

    try:
        # target_scaler.inverse_transform 期望二维数组
        lstm_pred_original_scale_np = target_scaler.inverse_transform(lstm_pred_scaled_np) # Shape: (1, 1)
        lstm_signal_score = lstm_pred_original_scale_np[0][0] # 提取最终预测值
    except Exception as e:
        logger.error(f"应用目标 Scaler 逆缩放时出错: {e}", exc_info=True)
        return 50.0


    # 将预测分数限制在 0-100 范围内并四舍五入
    final_predicted_signal = np.clip(lstm_signal_score, 0, 100).round(2)

    logger.debug(f"Transformer 模型预测完成，预测信号 (原始尺度): {final_predicted_signal:.2f}")
    return float(final_predicted_signal) # 返回 float 类型

# 新增函数：在测试集上评估模型
@log_execution_time
def evaluate_transformer_model(
    model: TransformerModel,
    test_loader: DataLoader,
    criterion: nn.Module,
    mae_metric: nn.Module,
    target_scaler: Union[MinMaxScaler, StandardScaler],
    device: torch.device
) -> Dict[str, float]:
    """
    在测试集上评估 Transformer 模型性能。

    Args:
        model (TransformerModel): 已加载权重的 PyTorch Transformer 模型。
        test_loader (DataLoader): 测试数据加载器。
        criterion (nn.Module): 损失函数实例 (例如 MSELoss)。
        mae_metric (nn.Module): MAE 度量实例 (例如 L1Loss)。
        target_scaler (Union[MinMaxScaler, StandardScaler]): 目标变量缩放器。
        device (torch.device): 评估设备。

    Returns:
        Dict[str, float]: 包含 Loss, MAE, True MAE (反标准化后) 的字典。
    """
    logger.info("开始在测试集上评估 Transformer 模型...")

    if len(test_loader) == 0:
        logger.warning("测试 DataLoader 为空，无法进行评估。")
        return {'loss': float('nan'), 'mae': float('nan'), 'true_mae': float('nan')}

    model.eval() # 设置模型为评估模式
    total_test_loss = 0
    total_test_mae = 0
    total_test_true_mae = 0
    num_test_batches = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae = mae_metric(outputs, targets)

            total_test_loss += loss.item()
            total_test_mae += mae.item()

            # 计算反标准化后的真实 MAE
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_original = target_scaler.inverse_transform(outputs_np)
            targets_original = target_scaler.inverse_transform(targets_np)
            true_mae_batch = np.mean(np.abs(outputs_original - targets_original))
            total_test_true_mae += true_mae_batch

            num_test_batches += 1

    avg_test_loss = total_test_loss / num_test_batches
    avg_test_mae = total_test_mae / num_test_batches
    avg_test_true_mae = total_test_true_mae / num_test_batches

    logger.info(f"测试集评估完成: Loss={avg_test_loss:.4f}, MAE={avg_test_mae:.4f}, True MAE={avg_test_true_mae:.2f}")

    return {'loss': avg_test_loss, 'mae': avg_test_mae, 'true_mae': avg_test_true_mae}

