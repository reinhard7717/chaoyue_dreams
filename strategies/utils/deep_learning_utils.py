# apps/strategies/utils/deep_learning_utils.py

# 导入必要的库 (PyTorch 相关)
import optuna
import torch
torch.backends.cudnn.benchmark = True # 在输入尺寸不变时加速卷积操作。
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter # 用于 TensorBoard 日志
# 导入 PyTorch AMP (自动混合精度) 相关模块
from torch.amp import autocast, GradScaler #: 导入 AMP 模块

# 导入必要的库 (数据处理和特征工程)
import os
import pandas as pd
import numpy as np
# 导入数据预处理和特征工程相关的模块 (Scikit-learn 部分保持不变，因为它处理 NumPy 数组)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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
import torch.optim as optim
# 导入 PyTorch 回调函数 (手动实现或使用库)
from torch.optim.lr_scheduler import ReduceLROnPlateau # PyTorch 的学习率调度器


# 导入日志、时间、绘图等辅助库
import logging
import psutil # 导入 psutil 库用于获取系统信息
import time   # 导入 time 库用于实现等待
from pathlib import Path
import datetime
import math
import matplotlib
# 使用 'Agg' 后端，这是一个非交互式的后端，不依赖于任何 GUI 库
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
import seaborn as sns
# 导入类型提示
from typing import Any, Tuple, List, Dict, Optional, Union, Callable
from functools import wraps
import joblib # 用于加载/保存 scaler

# 导入 tqdm 库用于显示进度条
from tqdm.auto import tqdm

# 设置日志记录器
logger = logging.getLogger("strategy_deep_learning_utils")

# 定义CPU检查阈值和等待间隔
CPU_THRESHOLD = 75.0 # CPU使用率阈值 (百分比)
CPU_CHECK_INTERVAL = 5 # 检查间隔和等待时间 (秒)

# 装饰器：记录执行时间
def log_execution_time(func: Callable) -> Callable:
    """
    记录函数执行时间的装饰器。
    Args:
        func (Callable): 被装饰的函数。
    Returns:
        Callable: 装饰后的函数。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # logger.info(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

# 装饰器：统一异常处理
def handle_exceptions(func: Callable) -> Callable:
    """
    处理函数执行过程中产生的异常的装饰器。
    Args:
        func (Callable): 被装饰的函数。
    Returns:
        Callable: 装饰后的函数。
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
    此类用于加载经过预处理和缩放的 NumPy 数组，并在 `__getitem__` 方法中
    根据窗口大小（`window_size`）动态构建输入序列 (X) 和对应的目标值 (y)，
    并将它们转换为 PyTorch Tensor。
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int):
        """
        初始化时间序列数据集。
        Args:
            features (np.ndarray): 经过缩放的平坦特征数据 (形状: num_samples, num_features)。
                                   这些是模型训练的输入特征。
            targets (np.ndarray): 经过缩放的平坦目标数据 (形状: num_samples,)。
                                  `targets[i]` 对应 `features[i]` 时刻的目标值。
            window_size (int): 每个输入窗口的时间步长。模型将使用 `window_size` 长度的
                               特征序列来预测该序列最后一个时间步的目标值。
        """
        if features.shape[0] != targets.shape[0]:
            raise ValueError("特征 (features) 和目标 (targets) 的样本数量必须相同。")
        self.num_samples = features.shape[0]
        self.window_size = window_size
        self.num_original_features = features.shape[1] if features.ndim == 2 and features.shape[0] > 0 else 0
        # 预测目标是窗口最后一个点 (t+window_size-1) 的信号。
        # 需要 features[t : t + window_size] 来预测 targets[t + window_size - 1].
        # 至少需要 window_size 条数据才能构成第一个窗口的输入和对应目标。
        if self.num_samples < window_size:
            logger.warning(
                f"提供的特征数据长度 ({self.num_samples}) 不足以根据窗口大小 ({self.window_size}) 构建任何样本。 "
                f"至少需要 {self.window_size} 条数据。TimeSeriesDataset 将为空。"
            )
            # 保持 features 为二维数组，即使是空的
            self.features = np.empty((0, self.num_original_features), dtype=features.dtype)
            self.targets = np.array([], dtype=targets.dtype)
            self.num_windows = 0
        else:
            self.features = features
            self.targets = targets
            # 可生成的窗口数量 = 总样本数 - 窗口大小 + 1
            # 例如：数据 [0,1,2,3,4], window_size=3. 窗口起始索引 0, 1, 2. Num windows = 5 - 3 + 1 = 3.
            self.num_windows = self.num_samples - window_size + 1
        # logger.info(f"TimeSeriesDataset 初始化: 原始样本数={self.num_samples}, 窗口大小={self.window_size}, "
        #             f"特征维度={self.num_original_features}, 可生成窗口数={self.num_windows}")
    def __len__(self) -> int:
        """
        返回数据集中可生成的窗口数量。
        这是 DataLoader 用来确定数据集大小的。
        Returns:
            int: 可生成的窗口数量。
        """
        return self.num_windows
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的窗口数据和目标值。
        目标是：使用时间步 `[idx, idx+1, ..., idx+window_size-1]` 的特征，
                来预测/拟合时间步 `idx+window_size-1` 的 `target`。
        Args:
            idx (int): 窗口的起始索引（在可生成窗口列表中的索引，不是原始数据索引）。
                    范围是 `0` 到 `self.num_windows - 1`。
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - X_tensor (torch.Tensor): 输入特征窗口，形状 `(window_size, num_features)`。
                - y_tensor (torch.Tensor): 对应的目标值，形状 `(1,)`。
        """
        if not (0 <= idx < self.num_windows):
            raise IndexError(f"索引 {idx} 超出范围 (0 至 {self.num_windows - 1})。")
        window_start_flat_idx = idx
        window_end_flat_idx = idx + self.window_size
        target_flat_idx = window_end_flat_idx - 1
        X_window_np = self.features[window_start_flat_idx:window_end_flat_idx, :]
        y_target_np = self.targets[target_flat_idx]
        # 保证 X_tensor 是 float32
        if isinstance(X_window_np, torch.Tensor):
            X_tensor = X_window_np.clone().detach().float()
        else:
            X_tensor = torch.tensor(X_window_np, dtype=torch.float32)
        # 保证 y_tensor 形状为 (1,)
        y_tensor = torch.as_tensor(y_target_np, dtype=torch.float32).view(1)  # <--- 只用 view(1)
        return X_tensor, y_tensor
    
class PositionalEncoding(nn.Module):
    """
    标准正弦位置编码层。
    为序列中的每个位置添加一个独特的编码，以帮助模型理解元素的顺序。
    Transformer 模型本身不包含任何关于序列顺序的信息，位置编码弥补了这一点。
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码层。
        Args:
            d_model (int): 模型的特征维度（嵌入维度）。
            max_len (int): 预先计算编码的最大序列长度。
                           应大于或等于模型可能遇到的任何序列的长度。
        """
        super().__init__()
        # 创建一个足够大的PE矩阵，max_len是序列最大长度
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        # div_term 计算频率项，用于正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数索引应用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数索引应用 cos (如果d_model是奇数，最后一个会被截断)
        if d_model % 2 != 0: # 处理 d_model 为奇数的情况，确保 pe[:, d_model-1] 被填充
            # 当d_model为奇数时，pe[:, 1::2] 的最后一列可能没有对应的 div_term
            # 应该使用与倒数第二个偶数项相同的 div_term
            pe[:, d_model-1] = torch.cos(position * div_term[d_model//2 -1]) # 确保奇数维度的cos项使用正确的频率
        # pe 形状 (max_len, d_model)
        # Transformer通常期望 (seq_len, batch_size, d_model) 或 (batch_size, seq_len, d_model)
        # 这里注册的 pe 是 (max_len, d_model)，在使用时会根据输入调整
        # self.register_buffer 将 pe 注册为模型的持久状态，但不是模型参数（即不会被优化器更新）
        self.register_buffer('pe', pe) # pe 形状 (max_len, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码添加到输入序列中。
        Args:
            x (torch.Tensor): 输入序列。
                              期望形状 `(batch_size, seq_len, d_model)`。
        Returns:
            torch.Tensor: 加上位置编码后的输出，形状与输入 `x` 相同。
        """
        # x 形状: (batch_size, seq_len, d_model)
        # self.pe 形状: (max_len, d_model)
        # 我们需要 self.pe 的一部分，形状为 (seq_len, d_model)，然后扩展到 (1, seq_len, d_model) 以便广播
        # 或者直接取 self.pe[:x.size(1), :] 并加到 x 上 (PyTorch支持广播)
        # x.size(1) 是序列长度 (seq_len)
        # self.pe[:x.size(1), :] 的形状是 (seq_len, d_model)
        # x + self.pe[:x.size(1), :].unsqueeze(0) -> if pe is (1, seq_len, d_model)
        # x + self.pe[:x.size(1), :] -> if pe is (seq_len, d_model) and broadcasting works
        # PyTorch 的广播机制允许 (B, S, D) + (S, D) -> (B, S, D)
        # 使用切片操作获取与当前序列长度匹配的位置编码部分
        pe_slice = self.pe[:x.size(1), :] # 形状 (seq_len, d_model)
        # 将位置编码添加到输入张量中
        x = x + pe_slice # PyTorch 广播机制会自动处理 batch_size 维度
        return x

class TransformerModel(nn.Module):
    """
    基于 Transformer Encoder 的时间序列回归预测模型。
    该模型使用 Transformer Encoder 来捕捉时间序列数据中的依赖关系，
    并通过一个回归头输出最终的预测值。
    """
    def __init__(self, num_features: int, d_model: int, nhead: int, dim_feedforward: int, nlayers: int,
                 dropout: float = 0.5, activation: str = 'relu', window_size: int = 60):
        """
        初始化 Transformer 模型。
        Args:
            num_features (int): 输入特征的原始维度。
            d_model (int): Transformer 模型的内部特征维度 (也称为嵌入维度)。
                           此值必须是 `nhead` 的整数倍。
            nhead (int): 多头注意力机制中的头数。
            dim_feedforward (int): Transformer Encoder 层内部前馈网络 (FFN) 的隐藏层维度。
            nlayers (int): Transformer Encoder 中 Encoder 层的数量。
            dropout (float): 应用于模型中多个位置的 Dropout 比率。
            activation (str): Transformer Encoder 层内部 FFN 使用的激活函数 ('relu' 或 'gelu')。
            window_size (int): 输入序列的长度（窗口大小），主要用于位置编码的最大长度。
        """
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.window_size = window_size # 主要给 PositionalEncoding 的 max_len 参考
        # 1. 输入嵌入层 (Input Embedding)
        # 将原始的 num_features 维输入映射到 d_model 维，这是 Transformer 的工作维度。
        self.embedding = nn.Linear(num_features, d_model)
        # 2. 位置编码 (Positional Encoding)
        # 为嵌入后的序列添加位置信息。
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size) # max_len 可以设为预期的最大窗口大小
        # 3. Transformer Encoder 层
        # batch_first=True 表示输入和输出的张量形状为 (batch_size, seq_len, feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # 重要：设置批次维度在前
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        # 4. 输出处理与回归头 (Output Processing & Regression Head)
        # TransformerEncoder 的输出形状是 (batch_size, seq_len, d_model)。
        # 对于序列到单值的回归任务，我们需要将序列信息汇总。
        # 常用方法：
        #   a) 取序列最后一个时间步的输出: output[:, -1, :]
        #   b) 对序列所有时间步的输出进行平均池化或最大池化。
        # 这里使用平均池化，作用于序列长度维度。
        # AdaptiveAvgPool1d(1) 会将 (N, C, L_in) -> (N, C, 1)，然后 squeeze 掉最后一维。
        # C 在这里是 d_model, L_in 是 window_size。
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # 回归头：一个或多个全连接层，将池化后的 d_model 维特征映射到单个预测值。
        # 池化后形状 (batch_size, d_model, 1)，squeeze(2) 后是 (batch_size, d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(), # 或其他激活函数如 GELU
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # 输出一个标量预测值
        )
        self.init_weights() # 初始化模型权重
    def init_weights(self):
        """
        初始化模型权重。
        一个好的权重初始化策略有助于模型训练。
        """
        initrange = 0.1 # 初始化范围
        # 对嵌入层和回归头的线性层权重进行均匀分布初始化
        self.embedding.weight.data.uniform_(-initrange, initrange)
        if self.embedding.bias is not None:
            self.embedding.bias.data.zero_()
        # 初始化回归器中的线性层
        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        # TransformerEncoder 内部的权重由其模块自身进行默认初始化，通常是合理的。
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播过程。
        Args:
            src (torch.Tensor): 输入序列，形状 `(batch_size, window_size, num_features)`。
        Returns:
            torch.Tensor: 模型的输出预测值，形状 `(batch_size, 1)`。
        """
        # 1. 输入嵌入: (batch_size, window_size, num_features) -> (batch_size, window_size, d_model)
        # 在 AMP 模式下，嵌入层通常在 autocast 区域内执行，以利用 float16 计算
        # 乘以 sqrt(d_model) 是一种常见的缩放技巧，有助于位置编码和后续层
        src_embedded = self.embedding(src) * math.sqrt(self.d_model) # 使用 math.sqrt
        # 2. 添加位置编码: (batch_size, window_size, d_model)
        src_pos_encoded = self.pos_encoder(src_embedded)
        # 3. 通过 Transformer Encoder:
        # 由于 TransformerEncoderLayer 初始化时 batch_first=True,
        # 输入形状应为 (batch_size, window_size, d_model)
        # 输出形状同样为 (batch_size, window_size, d_model)
        transformer_output = self.transformer_encoder(src_pos_encoded)
        # 4. 池化操作:
        # AdaptiveAvgPool1d 期望输入 (N, C, L_in), 即 (batch_size, d_model, window_size)
        # 当前 transformer_output 是 (batch_size, window_size, d_model)
        # 需要转置: .transpose(1, 2)
        # (batch_size, window_size, d_model) -> (batch_size, d_model, window_size)
        pooled_output = self.pooling(transformer_output.transpose(1, 2))
        # (batch_size, d_model, 1) -> (batch_size, d_model)
        pooled_output = pooled_output.squeeze(2)
        # 5. 通过回归头: (batch_size, d_model) -> (batch_size, 1)
        prediction = self.regressor(pooled_output)
        return prediction

@log_execution_time
@handle_exceptions
def prepare_data_for_transformer(
    data: pd.DataFrame,
    required_columns: List[str],
    target_column: str = 'transformer_train_target', # 修改默认目标列名以匹配新逻辑
    scaler_type: str = 'minmax', # 可选: 'minmax', 'standard', 'robust'
    train_split: float = 0.7,
    val_split: float = 0.15,
    apply_variance_threshold: bool = False,
    variance_threshold_value: float = 0.01,
    use_pca: bool = False,
    pca_n_components: Union[int, float] = 0.99,
    pca_solver: str = 'auto',
    use_feature_selection: bool = True,
    feature_selector_model_type: str = 'rf', # 修改默认值为 'rf'
    fs_model_n_estimators: int = 100,
    fs_model_max_depth: Optional[int] = None,
    fs_max_features: Optional[int] = None,
    fs_selection_threshold: Union[str, float] = 'median',
    target_scaler_type: str = 'minmax', # 可选: 'minmax', 'standard', 'robust'
    random_state_seed: Optional[int] = 42 #: 随机种子参数
) -> Tuple[
    np.ndarray, np.ndarray, # train_features, train_targets
    np.ndarray, np.ndarray, # val_features, val_targets
    np.ndarray, np.ndarray, # test_features, test_targets
    Union[MinMaxScaler, StandardScaler, RobustScaler, None], # feature_scaler
    Union[MinMaxScaler, StandardScaler, RobustScaler, None], # target_scaler
    List[str], # selected_feature_names
    Optional[PCA], # pca_model
    Optional[StandardScaler], # scaler_for_pca
    Optional[SelectFromModel] # feature_selector_model
]:
    """
    为 Transformer 模型准备平坦化、经过缩放的时间序列数据 (NumPy 数组)。
    此函数负责数据清洗、特征工程（方差过滤、PCA、基于模型的选择）、数据分割和缩放。
    数据分割严格按时间顺序，确保无未来数据泄露。
    窗口化操作由后续的 TimeSeriesDataset 类处理。
    Args:
        data (pd.DataFrame): 包含所有特征和目标列的原始DataFrame。
        required_columns (List[str]): 用于初始筛选的原始特征列名列表。
        target_column (str): 目标变量的列名。
        scaler_type (str): 特征缩放器类型 ('minmax', 'standard', 'robust')。
        train_split (float): 训练集在总数据中的比例。
        val_split (float): 验证集在总数据中的比例。测试集为剩余部分。
        apply_variance_threshold (bool): 是否应用方差阈值进行特征选择。
        variance_threshold_value (float): 方差阈值的具体值。
        use_pca (bool): 是否应用PCA进行降维。如果为 True，则通常会跳过后续基于模型的特征选择。
        pca_n_components (Union[int, float]): PCA保留的主成分数量或解释方差的比例。
        pca_solver (str): PCA求解器类型，如 'auto', 'full', 'arpack', 'randomized'。
        use_feature_selection (bool): 是否启用基于模型的特征选择 (在PCA之后，如果PCA未启用)。
        feature_selector_model_type (str): 特征选择模型类型 ('rf' 代表 RandomForest, 'xgb' 代表 XGBoost)。
        fs_model_n_estimators (int): 特征选择模型 (如RandomForest) 的 `n_estimators` 参数。
        fs_model_max_depth (Optional[int]): 特征选择模型 (如RandomForest) 的 `max_depth` 参数。
        fs_max_features (Optional[int]): 基于模型选择时，要保留的最大特征数量。若为 None，则使用阈值选择。
        fs_selection_threshold (Union[str, float]): 特征重要性阈值 (如 'median', 'mean', 或一个浮点数)。
                                                    仅在 `fs_max_features` 为 None 时生效。
        target_scaler_type (str): 目标变量缩放器类型 ('minmax', 'standard', 'robust')。
        random_state_seed (Optional[int]): 用于需要随机性的步骤 (如 PCA, RandomForest, XGBoost)。
    Returns:
        Tuple:
            - features_scaled_train (np.ndarray): 缩放后的训练集特征 (NumPy 数组)。
            - targets_scaled_train (np.ndarray): 缩放后的训练集目标 (NumPy 数组)。
            - features_scaled_val (np.ndarray): 缩放后的验证集特征 (NumPy 数组)。
            - targets_scaled_val (np.ndarray): 缩放后的验证集目标 (NumPy 数组)。
            - features_scaled_test (np.ndarray): 缩放后的测试集特征 (NumPy 数组)。
            - targets_scaled_test (np.ndarray): 缩放后的测试集目标 (NumPy 数组)。
            - feature_scaler (Union[MinMaxScaler, StandardScaler, RobustScaler, None]): 拟合好的特征缩放器。
            - target_scaler (Union[MinMaxScaler, StandardScaler, RobustScaler, None]): 拟合好的目标缩放器。
            - final_selected_feature_names (List[str]): 最终被选入模型的特征名列表。
                                                        如果应用了PCA，这些名称将是 'pca_comp_0', 'pca_comp_1', ...
            - pca_model (Optional[PCA]): 拟合好的 PCA 模型实例 (如果使用了 PCA)。
            - scaler_for_pca (Optional[StandardScaler]): PCA 前使用的 StandardScaler 实例 (如果使用了 PCA)。
            - feature_selector_model (Optional[SelectFromModel]): 拟合好的特征选择器模型实例 (如果使用了基于模型的选择)。
    """
    logger.info("开始为 Transformer 模型准备平坦化输入数据...")
    # 记录除大型数据对象外的所有参数
    log_params = {k: v for k, v in locals().items() if k not in ['data', 'required_columns']}
    logger.info(f"数据准备参数: {log_params}")
    # --- 0. 复制数据，避免修改原始 DataFrame ---
    # 复制是必要的，因为后续的 NaN 处理会修改 DataFrame
    data_processed = data.copy()
    print(f"prepare_data_for_transformer: 复制输入数据完成。形状: {data_processed.shape}, 内存使用 (MB): {data_processed.memory_usage(deep=True).sum() / 1024**2:.2f}")
    # --- 1. 检查目标列是否存在 ---
    if target_column not in data_processed.columns:
        logger.error(f"目标列 '{target_column}' 不存在于输入数据中。可用列: {data_processed.columns.tolist()}")
        # 返回空数组和 None 对象，保持返回类型一致性
        empty_np_array = np.array([], dtype=np.float32)
        return empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, None, None, [], None, None, None
    # --- 2. 初始特征列选择 ---
    # 从 required_columns 中筛选出实际存在于数据中且不是目标列的列名
    initial_feature_names = [col for col in required_columns if col in data_processed.columns and col != target_column]
    if not initial_feature_names:
        logger.error("根据 `required_columns` 筛选后，没有可用的特征列。请检查 `required_columns` 的内容和数据列名。")
        empty_np_array = np.array([], dtype=np.float32)
        return empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, None, None, [], None, None, None
    logger.info(f"初始筛选后，特征列数量: {len(initial_feature_names)}。部分列名 (最多10个): {initial_feature_names[:10]}...")
    # --- 3. 处理 NaN 值 ---
    # 分离特征和目标
    features_df = data_processed[initial_feature_names]
    targets_series = data_processed[target_column]
    # 使用前向填充 (ffill) 和后向填充 (bfill) 处理特征中的 NaN
    features_filled_df = features_df.ffill().bfill()
    # 检查是否仍有 NaN (例如，如果整列都是 NaN)
    nan_cols_features = features_filled_df.isnull().sum()
    nan_cols_features = nan_cols_features[nan_cols_features > 0]
    if not nan_cols_features.empty:
        # logger.warning(f"特征数据在 ffill().bfill() 后，以下列仍包含 NaN (将被填充为0): {nan_cols_features.index.tolist()}")
        features_filled_df.fillna(0, inplace=True) # 对剩余 NaN 用 0 填充
    # 同样处理目标列中的 NaN
    targets_filled_series = targets_series.ffill().bfill()
    if targets_filled_series.isnull().any():
        logger.warning(f"目标列 '{target_column}' 在 ffill().bfill() 后仍包含 NaN (将被填充为0)。请检查目标数据质量。")
        targets_filled_series.fillna(0, inplace=True)
    # 转换为 NumPy 数组进行后续处理
    # 使用 astype(np.float32) 确保数据类型一致并节省内存
    current_features_np = features_filled_df.values.astype(np.float32)
    current_feature_names = initial_feature_names[:] # 创建副本
    targets_np = targets_filled_series.values.astype(np.float32)
    logger.info(f"NaN 值处理完成。特征矩阵形状: {current_features_np.shape}, 目标数组形状: {targets_np.shape}")
    print(f"prepare_data_for_transformer: NaN处理后，特征数组形状: {current_features_np.shape}, dtype: {current_features_np.dtype}")
    print(f"prepare_data_for_transformer: NaN处理后，目标数组形状: {targets_np.shape}, dtype: {targets_np.dtype}")
    # --- 3.5. 终极 NaN/Inf 处理 (确保所有数值都有限且在 float32 范围内) ---
    # 尽管前面进行了 ffill/bfill/fillna(0)，但可能存在原始 Inf 值，或者极少数情况下 NaN 未被完全清除。
    # np.nan_to_num 是一个非常鲁棒的函数，可以处理 NaN, Inf, -Inf
    # 记录处理前的 Inf/NaN 情况
    if np.any(np.isinf(current_features_np)) or np.any(np.isnan(current_features_np)):
        logger.warning(f"特征数据在转换为 NumPy 数组后发现 Inf 或 NaN 值。将使用 np.nan_to_num 进行最终清理。")
        # 统计 Inf 和 NaN 数量
        num_inf = np.sum(np.isinf(current_features_np))
        num_nan = np.sum(np.isnan(current_features_np))
        logger.warning(f"处理前特征中 Inf 数量: {num_inf}, NaN 数量: {num_nan}")
    current_features_np = np.nan_to_num(
        current_features_np,
        nan=0.0, # 将 NaN 替换为 0
        posinf=np.finfo(np.float32).max, # 将正无穷替换为 float32 的最大值
        neginf=np.finfo(np.float32).min # 将负无穷替换为 float32 的最小值
    ).astype(np.float32) # 再次确保 dtype 为 float32，虽然 nan_to_num 会尝试保持原始dtype，但显式转换更安全
    if np.any(np.isinf(targets_np)) or np.any(np.isnan(targets_np)):
        logger.warning(f"目标数据在转换为 NumPy 数组后发现 Inf 或 NaN 值。将使用 np.nan_to_num 进行最终清理。")
        num_inf_target = np.sum(np.isinf(targets_np))
        num_nan_target = np.sum(np.isnan(targets_np))
        logger.warning(f"处理前目标中 Inf 数量: {num_inf_target}, NaN 数量: {num_nan_target}")
    targets_np = np.nan_to_num(
        targets_np,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min
    ).astype(np.float32)
    logger.info(f"最终 NaN/Inf 处理完成。特征矩阵形状: {current_features_np.shape}, 目标数组形状: {targets_np.shape}")
    print(f"prepare_data_for_transformer: 最终NaN/Inf处理后，特征数组形状: {current_features_np.shape}, dtype: {current_features_np.dtype}")
    print(f"prepare_data_for_transformer: 最终NaN/Inf处理后，目标数组形状: {targets_np.shape}, dtype: {targets_np.dtype}")
    # 显式删除不再需要的 DataFrame，释放内存
    del features_df, targets_series, features_filled_df, targets_filled_series, data_processed
    print("prepare_data_for_transformer: 已删除中间 DataFrame。")
    # --- 4. (可选) 方差阈值过滤 ---
    # 移除方差过低的特征，这些特征可能对模型贡献不大
    if apply_variance_threshold:
        if current_features_np.shape[0] > 1 and current_features_np.shape[1] > 0: # 至少需要2个样本才能计算方差
            variances = np.var(current_features_np, axis=0)
            # 仅在存在显著方差时应用，避免对全零或常数特征应用时出错
            if np.any(variances > 1e-9): # 1e-9 是一个小的阈值，避免浮点数精度问题
                try:
                    selector_var = VarianceThreshold(threshold=variance_threshold_value)
                    # VarianceThreshold 直接在 NumPy 数组上工作
                    features_after_var = selector_var.fit_transform(current_features_np)
                    selected_indices_var = selector_var.get_support(indices=True)
                    if features_after_var.shape[1] == 0:
                        logger.warning(f"方差阈值 ({variance_threshold_value}) 过高，所有特征均被移除。将跳过方差过滤。")
                    elif features_after_var.shape[1] < current_features_np.shape[1]:
                        num_removed = current_features_np.shape[1] - features_after_var.shape[1]
                        current_features_np = features_after_var # 更新特征数组
                        current_feature_names = [current_feature_names[i] for i in selected_indices_var] # 更新特征名
                        logger.info(f"方差阈值过滤完成，移除了 {num_removed} 个低方差特征。剩余特征数: {current_features_np.shape[1]}")
                        logger.debug(f"方差过滤后特征名 (部分): {current_feature_names[:10]}...")
                    else:
                        logger.info("方差阈值过滤未移除任何特征。")
                except Exception as e_var:
                     logger.error(f"应用方差阈值时出错: {e_var}", exc_info=True)
                     logger.warning("方差阈值选择失败，将使用之前的特征集。")
            else:
                 logger.warning("所有特征方差接近零或为零，跳过方差阈值选择。")
        else:
             logger.warning("特征数据不足 (样本数 <= 1 或特征数为 0)，跳过方差阈值选择。")
    # 如果经过处理后没有特征了，则无法继续
    if current_features_np.shape[1] == 0:
        logger.error("经过初始处理和方差过滤后，特征数量为零。无法继续进行数据准备。")
        empty_np_array = np.array([], dtype=np.float32)
        return empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, None, None, [], None, None, None
    print(f"prepare_data_for_transformer: 方差过滤后，特征数组形状: {current_features_np.shape}, dtype: {current_features_np.dtype}")
    # --- 5. 按时间顺序分割数据集 ---
    # 确保数据分割的正确性对于时间序列至关重要，以防数据泄露
    n_samples_total = current_features_np.shape[0]
    if n_samples_total == 0:
        logger.error("数据为空，无法进行分割。")
        empty_np_array = np.array([], dtype=np.float32)
        return empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, None, None, [], None, None, None
    # 校验分割比例
    if not (0 < train_split < 1 and 0 <= val_split < 1 and train_split + val_split <= 1.00001): # 增加浮点容差
        logger.error(f"无效的数据集分割比例: train_split={train_split}, val_split={val_split}。"
                     f"比例应在 (0,1) (训练集) 或 [0,1) (验证集) 范围内，且总和 <= 1。")
        empty_np_array = np.array([], dtype=np.float32)
        return empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, None, None, [], None, None, None
    n_train = int(n_samples_total * train_split)
    n_val = int(n_samples_total * val_split)
    # n_test = n_samples_total - n_train - n_val # 测试集是剩余部分
    if n_train == 0:
        logger.error(f"计算得到的训练集样本数为零 (总样本数: {n_samples_total}, 训练比例: {train_split})。"
                     f"请增加数据量或调整训练集比例。")
        empty_np_array = np.array([], dtype=np.float32)
        return empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, None, None, [], None, None, None
    # 分割特征
    features_train_raw = current_features_np[:n_train]
    features_val_raw = current_features_np[n_train : n_train + n_val]
    features_test_raw = current_features_np[n_train + n_val:]
    # 分割目标
    targets_train_raw = targets_np[:n_train]
    targets_val_raw = targets_np[n_train : n_train + n_val]
    targets_test_raw = targets_np[n_train + n_val:]
    logger.info(f"数据按时间顺序分割完成: "
                f"训练集 {features_train_raw.shape[0]}条 (特征形状: {features_train_raw.shape}), "
                f"验证集 {features_val_raw.shape[0]}条 (特征形状: {features_val_raw.shape}), "
                f"测试集 {features_test_raw.shape[0]}条 (特征形状: {features_test_raw.shape})。")
    print(f"prepare_data_for_transformer: 分割后，训练集特征形状: {features_train_raw.shape}, 目标形状: {targets_train_raw.shape}")
    print(f"prepare_data_for_transformer: 分割后，验证集特征形状: {features_val_raw.shape}, 目标形状: {targets_val_raw.shape}")
    print(f"prepare_data_for_transformer: 分割后，测试集特征形状: {features_test_raw.shape}, 目标形状: {targets_test_raw.shape}")
    # 初始化用于后续特征工程的变量
    features_eng_train = features_train_raw
    features_eng_val = features_val_raw
    features_eng_test = features_test_raw
    feature_names_after_eng = current_feature_names[:] # 创建副本
    # --- 6. (可选) PCA 降维 ---
    # PCA 应在数据分割后，且仅在训练集上拟合，然后应用到验证集和测试集
    pca_applied = False
    pca_model: Optional[PCA] = None
    scaler_for_pca: Optional[StandardScaler] = None
    if use_pca:
        if features_eng_train.shape[1] <= 1:
            logger.warning(f"训练集特征维度 ({features_eng_train.shape[1]}) 过低 (<=1)，跳过PCA。")
        # 稍微调整PCA n_components的检查逻辑和日志
        elif isinstance(pca_n_components, float) and not (0 < pca_n_components < 1.0) and pca_n_components != 1.0:
             logger.warning(f"PCA n_components 作为浮点数 ({pca_n_components}) 通常应在 (0,1) 表示解释方差比例。当前值可能导致意外行为或错误。")
        elif isinstance(pca_n_components, int) and pca_n_components <= 0:
             logger.warning(f"PCA n_components 作为整数 ({pca_n_components}) 应为正数。")
        elif features_eng_train.shape[0] < features_eng_train.shape[1] and isinstance(pca_n_components, float) and pca_n_components >= 1.0 : # 原有检查
             logger.warning(f"训练集样本数 ({features_eng_train.shape[0]}) 小于特征数 ({features_eng_train.shape[1]})，"
                            f"PCA解释方差比例 ({pca_n_components}) 可能不可靠。")
        else:
            logger.info(f"启用 PCA 降维 (n_components={pca_n_components}, solver='{pca_solver}')。在训练集上拟合...")
            # PCA 对数据尺度敏感，通常在 PCA 前进行标准化
            scaler_for_pca = StandardScaler()
            try:
                # StandardScaler 在 NumPy 数组上工作
                features_train_scaled_for_pca = scaler_for_pca.fit_transform(features_eng_train)
                pca_model = PCA(n_components=pca_n_components, svd_solver=pca_solver, random_state=random_state_seed)
                pca_model.fit(features_train_scaled_for_pca)
                num_components_retained = pca_model.n_components_
                explained_variance_ratio = sum(pca_model.explained_variance_ratio_)
                logger.info(f"PCA 拟合完成。保留主成分数: {num_components_retained}, "
                            f"累计解释方差比: {explained_variance_ratio:.4f}")
                if num_components_retained == 0:
                    logger.error("PCA 降维后特征维度为零。PCA失败，将使用原始特征。")
                    pca_model = None # 重置
                    scaler_for_pca = None # 重置
                else:
                    # 应用 PCA 转换
                    features_eng_train = pca_model.transform(features_train_scaled_for_pca)
                    if features_eng_val.shape[0] > 0: # 确保验证集非空
                        features_eng_val = pca_model.transform(scaler_for_pca.transform(features_eng_val))
                    if features_eng_test.shape[0] > 0: # 确保测试集非空
                        features_eng_test = pca_model.transform(scaler_for_pca.transform(features_eng_test))
                    # 更新特征名 (PCA后的特征是原始特征的线性组合，不再具有原名)
                    feature_names_after_eng = [f"pca_comp_{i}" for i in range(num_components_retained)]
                    logger.info(f"PCA 转换完成。处理后特征维度: {num_components_retained}")
                    logger.debug(f"PCA转换后特征名 (部分): {feature_names_after_eng[:10]}...")
                    pca_applied = True
            except Exception as e_pca:
                logger.error(f"应用 PCA 时出错: {e_pca}", exc_info=True)
                logger.warning("PCA 降维失败，将使用之前的特征集。")
                pca_model = None # 重置
                scaler_for_pca = None # 重置
    print(f"prepare_data_for_transformer: PCA处理后，训练集特征形状: {features_eng_train.shape}, dtype: {features_eng_train.dtype}")
    # --- CPU 占用度检查 ---
    # 在进行基于模型的特征选择（可能使用多核）之前检查CPU占用
    print(f"prepare_data_for_transformer: 准备进行基于模型的特征选择，检查CPU占用...") # 使用 print 输出调试信息
    while True: # 循环检查直到CPU占用低于阈值
        try:
            # 获取系统整体CPU使用率，interval=1 表示在1秒内采样
            cpu_usage = psutil.cpu_percent(interval=1) # 使用 psutil 获取 CPU 使用率
            print(f"prepare_data_for_transformer: 当前系统CPU占用率: {cpu_usage:.2f}%") # 使用 print 输出调试信息
            if cpu_usage > CPU_THRESHOLD: # 判断是否超过阈值
                print(f"prepare_data_for_transformer: 警告 - CPU占用过高 ({cpu_usage:.2f}%)，等待 {CPU_CHECK_INTERVAL} 秒...") # 使用 print 输出调试信息
                time.sleep(CPU_CHECK_INTERVAL) # 暂停执行
            else:
                print(f"prepare_data_for_transformer: CPU占用正常 ({cpu_usage:.2f}%)，继续。") # 使用 print 输出调试信息
                break # CPU占用正常，退出循环
        except ImportError: # 处理 psutil 未安装的情况
            print("prepare_data_for_transformer: 警告 - 未安装 psutil，跳过CPU检查。") # 使用 print 输出调试信息
            break # psutil 未安装，退出循环
        except Exception as e_cpu: # 捕获其他可能的异常
            print(f"prepare_data_for_transformer: 错误 - 检查CPU占用失败: {e_cpu}") # 使用 print 输出调试信息
            break # 发生错误，退出循环
    # --- 7. (可选) 基于模型的特征选择 (仅当PCA未应用或PCA后仍希望进一步选择时) ---
    # 通常 PCA 和基于模型的特征选择是互斥的，或按特定顺序进行。这里假设 PCA 优先。
    feature_selector_model: Optional[SelectFromModel] = None # 用于可能保存或后续使用的选择器模型
    if use_feature_selection and not pca_applied: # 如果PCA已应用，通常跳过此步骤
        if features_eng_train.shape[1] <= 1: # 对单特征或无特征无意义
            logger.warning(f"训练集特征维度 ({features_eng_train.shape[1]}) 过低 (<=1)，跳过基于模型的特征选择。")
        elif features_eng_train.shape[0] <= 1: # 样本过少无法训练选择器
             logger.warning(f"训练集样本数 ({features_eng_train.shape[0]}) 过低 (<=1)，无法进行基于模型的特征选择。")
        else:
            logger.info(f"启用基于模型 '{feature_selector_model_type}' 的特征选择。在训练集上拟合...")
            logger.info(f"特征选择模型参数: n_estimators={fs_model_n_estimators}, max_depth={fs_model_max_depth}, "
                        f"选择方式: max_features={fs_max_features if fs_max_features is not None else 'N/A'} 或 threshold='{fs_selection_threshold if fs_max_features is None else 'N/A'}'")
            selector_model_instance = None # 实际用于 SelectFromModel 的模型实例
            model_type_lower = feature_selector_model_type.lower()
            if model_type_lower == 'rf':
                selector_model_instance = RandomForestRegressor(
                    n_estimators=fs_model_n_estimators,
                    max_depth=fs_model_max_depth,
                    random_state=random_state_seed,
                    n_jobs=-1 # 使用所有可用核心
                )
            elif model_type_lower == 'xgb':
                if not XGB_AVAILABLE:
                    logger.warning("XGBoost 未安装，无法使用 XGBoost 进行特征选择。将回退到 RandomForest。")
                    selector_model_instance = RandomForestRegressor(
                        n_estimators=fs_model_n_estimators, max_depth=fs_model_max_depth,
                        random_state=random_state_seed, n_jobs=-1
                    )
                else: # pragma: no cover (依赖外部XGBoost)
                    selector_model_instance = xgb.XGBRegressor(
                        objective='reg:squarederror', n_estimators=fs_model_n_estimators,
                        max_depth=fs_model_max_depth, random_state=random_state_seed,
                        n_jobs=-1, verbosity=0 # XGBoost 1.6+ use verbosity
                    )
            else:
                logger.warning(f"不支持的特征选择模型类型: {feature_selector_model_type}。将使用 RandomForest。")
                selector_model_instance = RandomForestRegressor(
                    n_estimators=fs_model_n_estimators, max_depth=fs_model_max_depth,
                    random_state=random_state_seed, n_jobs=-1
                )
            try:
                # SelectFromModel 用于根据特征重要性选择特征
                if fs_max_features is not None and fs_max_features > 0:
                    # 确保选择的特征数不超过可用特征数
                    actual_max_fs = min(fs_max_features, features_eng_train.shape[1])
                    feature_selector_model = SelectFromModel(
                        selector_model_instance,
                        max_features=actual_max_fs,
                        threshold=-np.inf # 当 max_features 生效时，threshold 应设为-inf
                    )
                    logger.info(f"特征选择方式: 最多选择 {actual_max_fs} 个特征。")
                else:
                    feature_selector_model = SelectFromModel(
                        selector_model_instance,
                        threshold=fs_selection_threshold # 例如 'median', '0.1*mean', 或一个浮点数
                    )
                    logger.info(f"特征选择方式: 使用阈值 '{fs_selection_threshold}'。")
                # 在训练数据上拟合选择器模型
                # SelectFromModel 在 NumPy 数组上工作
                feature_selector_model.fit(features_eng_train, targets_train_raw)
                selected_indices_fs = feature_selector_model.get_support(indices=True)
                num_selected_fs = len(selected_indices_fs)
                if num_selected_fs == 0:
                    logger.error("基于模型选择后没有剩余特征。特征选择失败，将使用之前的特征集。")
                    feature_selector_model = None # 重置
                elif num_selected_fs < features_eng_train.shape[1]:
                    num_removed_fs = features_eng_train.shape[1] - num_selected_fs
                    # 应用转换
                    features_eng_train = feature_selector_model.transform(features_eng_train)
                    if features_eng_val.shape[0] > 0:
                        features_eng_val = feature_selector_model.transform(features_eng_val)
                    if features_eng_test.shape[0] > 0:
                        features_eng_test = feature_selector_model.transform(features_eng_test)
                    # 更新特征名列表
                    feature_names_after_eng = [feature_names_after_eng[i] for i in selected_indices_fs]
                    logger.info(f"基于模型选择完成，移除了 {num_removed_fs} 个特征。剩余特征数: {num_selected_fs}")
                    logger.debug(f"模型选择后特征名 (部分): {feature_names_after_eng[:10]}...")
                else:
                    logger.info("基于模型选择未移除任何特征 (可能所有特征都很重要或已达到最大选择数)。")
            except Exception as e_fs:
                logger.error(f"应用基于模型的特征选择时出错: {e_fs}", exc_info=True)
                logger.warning("特征选择失败，将使用之前的特征集。")
                feature_selector_model = None # 重置
    elif pca_applied:
        logger.info("PCA已应用，跳过基于模型的特征选择。")
    else: # neither use_feature_selection nor pca_applied
        logger.info("未启用基于模型的特征选择。")
    # 最终用于缩放的特征集
    final_features_train = features_eng_train
    final_features_val = features_eng_val
    final_features_test = features_eng_test
    final_selected_feature_names = feature_names_after_eng[:] # 创建副本
    if final_features_train.shape[1] == 0:
         logger.error("经过所有特征工程步骤后，训练集特征维度为零。无法继续。")
         empty_np_array = np.array([], dtype=np.float32)
         return empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, empty_np_array, None, None, [], None, None, None
    logger.info(f"所有特征工程处理完成。最终用于缩放的特征维度: {final_features_train.shape[1]}")
    logger.debug(f"最终选定特征名 (部分): {final_selected_feature_names[:10]}...")
    print(f"prepare_data_for_transformer: 特征工程后，训练集特征形状: {final_features_train.shape}, dtype: {final_features_train.dtype}")
    # --- 8. 特征缩放 (Feature Scaling) ---
    # 缩放器应在处理后的训练集特征上拟合，然后应用到所有数据集
    feature_scaler: Union[MinMaxScaler, StandardScaler, RobustScaler, None] = None
    if scaler_type.lower() == 'minmax':
        feature_scaler = MinMaxScaler()
    elif scaler_type.lower() == 'standard':
        feature_scaler = StandardScaler()
    elif scaler_type.lower() == 'robust':
        feature_scaler = RobustScaler()
    else:
        logger.warning(f"不支持的特征缩放器类型: '{scaler_type}'。将默认使用 MinMaxScaler。")
        feature_scaler = MinMaxScaler()
    features_scaled_train = np.array([], dtype=np.float32)
    features_scaled_val = np.array([], dtype=np.float32)
    features_scaled_test = np.array([], dtype=np.float32)
    # 仅当训练集有数据且有特征时才进行拟合和转换
    if final_features_train.shape[0] > 0 and final_features_train.shape[1] > 0:
        feature_scaler.fit(final_features_train)
        features_scaled_train = feature_scaler.transform(final_features_train).astype(np.float32)
        if final_features_val.shape[0] > 0: # 确保验证集非空
            features_scaled_val = feature_scaler.transform(final_features_val).astype(np.float32)
        if final_features_test.shape[0] > 0: # 确保测试集非空
            features_scaled_test = feature_scaler.transform(final_features_test).astype(np.float32)
        logger.info(f"特征缩放完成 (使用 {scaler_type} scaler)。")
        print(f"prepare_data_for_transformer: 特征缩放后，训练集形状: {features_scaled_train.shape}, dtype: {features_scaled_train.dtype}")
    else:
        logger.warning("处理后的训练集特征为空或无特征维度，跳过特征缩放。特征缩放器未拟合。")
        feature_scaler = None # 明确设为 None
        # 保持数组为空或原始状态（如果它们已经是空的）
        # 确保即使跳过缩放，返回的也是 float32 数组
        features_scaled_train = final_features_train.astype(np.float32) if final_features_train.size > 0 else np.array([], dtype=np.float32)
        features_scaled_val = final_features_val.astype(np.float32) if final_features_val.size > 0 else np.array([], dtype=np.float32)
        features_scaled_test = final_features_test.astype(np.float32) if final_features_test.size > 0 else np.array([], dtype=np.float32)
    # --- 9. 目标变量缩放 (Target Scaling) ---
    # 目标变量缩放器在原始训练集目标上拟合
    target_scaler: Union[MinMaxScaler, StandardScaler, RobustScaler, None] = None
    if target_scaler_type.lower() == 'minmax':
        target_scaler = MinMaxScaler()
    elif target_scaler_type.lower() == 'standard':
        target_scaler = StandardScaler()
    elif target_scaler_type.lower() == 'robust':
        target_scaler = RobustScaler()
    else:
        logger.warning(f"不支持的目标变量缩放器类型: '{target_scaler_type}'。将默认使用 MinMaxScaler。")
        target_scaler = MinMaxScaler()
    targets_scaled_train = np.array([], dtype=np.float32)
    targets_scaled_val = np.array([], dtype=np.float32)
    targets_scaled_test = np.array([], dtype=np.float32)
    if targets_train_raw.shape[0] > 0:
        # reshape(-1, 1) 是因为 scaler 期望二维输入
        target_scaler.fit(targets_train_raw.reshape(-1, 1))
        targets_scaled_train = target_scaler.transform(targets_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
        if targets_val_raw.shape[0] > 0:
            targets_scaled_val = target_scaler.transform(targets_val_raw.reshape(-1, 1)).flatten().astype(np.float32)
        if targets_test_raw.shape[0] > 0:
            targets_scaled_test = target_scaler.transform(targets_test_raw.reshape(-1, 1)).flatten().astype(np.float32)
        logger.info(f"目标变量缩放完成 (使用 {target_scaler_type} scaler)。")
        print(f"prepare_data_for_transformer: 目标缩放后，训练集形状: {targets_scaled_train.shape}, dtype: {targets_scaled_train.dtype}")
    else:
        logger.warning("训练集目标数据为空，跳过目标变量缩放。目标缩放器未拟合。")
        target_scaler = None # 明确设为 None
        # 确保即使跳过缩放，返回的也是 float32 数组
        targets_scaled_train = targets_train_raw.astype(np.float32) if targets_train_raw.size > 0 else np.array([], dtype=np.float32)
        targets_scaled_val = targets_val_raw.astype(np.float32) if targets_val_raw.size > 0 else np.array([], dtype=np.float32)
        targets_scaled_test = targets_test_raw.astype(np.float32) if targets_test_raw.size > 0 else np.array([], dtype=np.float32)
    logger.info("Transformer 数据准备流程结束。")
    # 返回 NumPy 数组 (确保 float32 类型以匹配 PyTorch 默认) 和 scaler 对象
    # 注意：如果 PCA 或特征选择器被使用，它们也应该被返回，以便在预测新数据时能够复现相同的转换。
    # 当前函数签名已包含这些返回。
    return (
        features_scaled_train, targets_scaled_train,
        features_scaled_val, targets_scaled_val,
        features_scaled_test, targets_scaled_test,
        feature_scaler, target_scaler,
        final_selected_feature_names,
        pca_model,
        scaler_for_pca,
        feature_selector_model
    )

@log_execution_time
@handle_exceptions
def build_transformer_model( num_features: int, model_config: Dict[str, Any], summary: bool = True, window_size: int = 60 ) -> TransformerModel:
    """
    构建并初始化一个 TransformerEncoder 模型。
    Args:
        num_features (int): 输入特征的维度 (经过特征工程后，送入模型的特征数量)。
        model_config (Dict[str, Any]): 包含模型超参数的字典。
            - 'd_model' (int): Transformer模型的内部特征维度。
            - 'nhead' (int): 多头注意力机制的头数。
            - 'dim_feedforward' (int): 前馈网络的隐藏层维度。
            - 'nlayers' (int): TransformerEncoder层的数量。
            - 'dropout' (float): Dropout比率。
            - 'activation' (str): 激活函数 ('relu' 或 'gelu')。
            - (其他参数如 'learning_rate', 'weight_decay' 主要用于训练配置，此处不直接使用)
        summary (bool): 是否打印模型结构摘要。
        window_size (int): 输入序列的长度（窗口大小），用于位置编码的 `max_len`。
    Returns:
        TransformerModel: 构建好的 PyTorch Transformer 模型实例。
    Raises:
        ValueError: 如果 `num_features` 或 `window_size` 小于等于0，
                    或者 `d_model` 不是 `nhead` 的整数倍。
    """
    # logger.info("开始构建 Transformer 模型...")
    # logger.info(f"模型配置: num_features={num_features}, window_size={window_size}, config={model_config}")
    if num_features <= 0:
        raise ValueError(f"输入特征数量 (num_features) 必须大于0，当前为: {num_features}")
    if window_size <= 0:
        raise ValueError(f"窗口大小 (window_size) 必须大于0，当前为: {window_size}")
    # 从配置字典中获取模型参数，提供默认值
    d_model = model_config.get('d_model', 128)
    nhead = model_config.get('nhead', 8)
    dim_feedforward = model_config.get('dim_feedforward', 512)
    nlayers = model_config.get('nlayers', 4)
    dropout = model_config.get('dropout', 0.2)
    activation = model_config.get('activation', 'relu')
    # Transformer 的一个重要约束：d_model 必须是 nhead 的整数倍
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
        window_size=window_size # 传递窗口大小给模型，主要用于 PositionalEncoding
    )
    # if summary:
    #     logger.info("Transformer 模型结构:")
    #     logger.info(str(model)) # 打印模型对象会显示其模块结构
    #     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     logger.info(f"模型可训练参数总量: {total_params:,}")
    logger.info("Transformer 模型构建完成。")
    return model

@log_execution_time
@handle_exceptions
def train_transformer_model(
    model: TransformerModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    target_scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
    training_config: Dict[str, Any],
    checkpoint_dir: str, # 这是股票专属的基础检查点目录，如 /root/chaoyue_dreams/models/000089.SZ/trained_model
    stock_code: Optional[str] = "STOCK",
    plot_training_history: bool = False,
    enable_anomaly_detection: bool = False,
    trial=None  # Optuna.Trial 对象，或 None
) -> Tuple[TransformerModel, pd.DataFrame]:
    """
    训练 Transformer 模型。
    Args:
        model (TransformerModel): 要训练的 PyTorch Transformer 模型实例。
        train_loader (DataLoader): 训练数据加载器。建议设置 num_workers > 0 和 pin_memory=True
                                   以提高数据加载效率和 GPU 利用率。
        val_loader (Optional[DataLoader]): 验证数据加载器。如果为 None，则不进行验证。
                                           建议设置 num_workers > 0 和 pin_memory=True。
        target_scaler (Union[MinMaxScaler, StandardScaler, RobustScaler]):
            用于目标变量反向缩放以计算真实 MAE 的缩放器。
        training_config (Dict[str, Any]): 包含训练超参数的字典，例如：
            - 'optimizer' (str): 优化器名称 ('adam', 'adamw', 'rmsprop', 'sgd')。
            - 'learning_rate' (float): 学习率 (Warmup 后的目标学习率)。
            - 'warmup_epochs' (int): Warmup 阶段的轮数。若 <= 0 则不启用 Warmup。
            - 'warmup_start_lr' (float): Warmup 开始时的学习率。
            - 'weight_decay' (float): 权重衰减。
            - 'momentum' (float): SGD 或 RMSprop 的动量。
            - 'loss' (str): 损失函数名称 ('mse', 'mae', 'huber')。
            - 'huber_delta' (float): HuberLoss 的 delta 参数 (如果使用)。
            - 'epochs' (int): 训练轮数。
            - 'early_stopping_patience' (int): 早停的耐心轮数 (基于监控指标)。
            - 'nan_metric_patience' (int): 监控指标连续为 NaN 时触发早停的耐心轮数。
            - 'reduce_lr_patience' (int): ReduceLROnPlateau 调度器的耐心轮数。
            - 'reduce_lr_factor' (float): ReduceLROnPlateau 调度器的学习率衰减因子。
            - 'min_lr' (float): 学习率的下限，用于 ReduceLROnPlateau 和 NaN/Inf 重试降学习率。
            - 'monitor_metric' (str): 用于早停和学习率调度的监控指标
                                      ('val_loss', 'val_mae', 'val_true_mae')。
            - 'clip_grad_norm' (Optional[float]): 梯度裁剪范数值。若为 None 或 <=0，则不裁剪。
            - 'use_amp' (bool): 是否使用自动混合精度 (AMP) 训练 (仅限 CUDA)。
            - 'tensorboard_log_dir' (Optional[str]): TensorBoard 日志的根目录。
                                                     若为 None，则不使用 TensorBoard。
        checkpoint_dir (str): 保存最佳模型检查点的目录。
        stock_code (Optional[str]): 用于日志记录和文件命名的股票代码或标识符。
        plot_training_history (bool): 是否在训练结束后绘制训练历史图表。
        enable_anomaly_detection (bool): 是否启用 PyTorch 的异常检测 (torch.autograd.set_detect_detect_anomaly(True))。
                                         用于调试 NaN/Inf 问题，但会显著降低训练速度。默认为 False。
        trial: (Optuna.Trial, optional): Optuna Trial 对象，如果当前在 Optuna 优化中。默认为 None。
    Returns:
        Tuple[TransformerModel, pd.DataFrame]:
            - model (TransformerModel): 训练完成或从最佳检查点加载的模型。
            - history_df (pd.DataFrame): 包含训练历史（损失、MAE、学习率等）的 DataFrame。
    """
    start_time = time.time()
    logger.info(f"开始训练 Transformer 模型 (股票/标识: {stock_code})...")
    logger.info(f"训练配置: {training_config}")
    # 根据参数启用 PyTorch 异常检测
    if enable_anomaly_detection:
        torch.autograd.set_detect_anomaly(True) # pylint: disable=not-callable
        logger.warning("PyTorch 异常检测已启用 (torch.autograd.set_detect_anomaly(True))。这将显著降低训练速度，请仅用于调试目的。")
    else:
        # 如果异常检测被禁用，则确保其设置为 False（避免残留状态）
        # 兼容旧版本PyTorch没有 set_detect_anomaly 的情况，增加 hasattr 检查
        if hasattr(torch.autograd, 'set_detect_anomaly'):
            torch.autograd.set_detect_anomaly(False) # pylint: disable=not-callable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model.to(device) # 将模型移动到指定设备
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    # 获取 Warmup 后的目标学习率
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0)
    momentum = training_config.get('momentum', 0.0)
    # 存储 Warmup 后的目标学习率
    initial_optimizer_lr = learning_rate
    optimizer: optim.Optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=initial_optimizer_lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=initial_optimizer_lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=initial_optimizer_lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=initial_optimizer_lr, weight_decay=weight_decay, momentum=momentum)
    else:
        logger.warning(f"未知的优化器名称: '{optimizer_name}'。将默认使用 Adam.")
        optimizer = optim.Adam(model.parameters(), lr=initial_optimizer_lr, weight_decay=weight_decay)
    logger.info(f"优化器: {optimizer_name}, 初始学习率 (Warmup后目标): {initial_optimizer_lr:.2e}, 权重衰减: {weight_decay}")
    loss_name = training_config.get('loss', 'mse').lower()
    criterion: nn.Module
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'mae':
        criterion = nn.L1Loss()
    elif loss_name == 'huber':
        huber_delta = training_config.get('huber_delta', 1.0)
        criterion = nn.HuberLoss(delta=huber_delta)
        logger.info(f"使用 HuberLoss (delta={huber_delta:.2f})")
    else:
        logger.warning(f"未知的损失函数名称: '{loss_name}'。将默认使用 MSELoss.")
        criterion = nn.MSELoss()
    logger.info(f"损失函数: {loss_name}")
    mae_eval_metric = nn.L1Loss() # 用于评估 MAE (scaled)
    epochs = training_config.get('epochs', 100)
    early_stopping_patience = training_config.get('early_stopping_patience', 30)
    nan_metric_patience = training_config.get('nan_metric_patience', 5) # NaN 指标的耐心
    reduce_lr_patience = training_config.get('reduce_lr_patience', 10)
    reduce_lr_factor = training_config.get('reduce_lr_factor', 0.5)
    min_lr = training_config.get('min_lr', 1e-7) # 获取最小学习率配置
    monitor_metric = training_config.get('monitor_metric', 'avg_val_mae').lower()
    clip_grad_norm_value = training_config.get('clip_grad_norm', None) # 梯度裁剪值
    use_amp_config = training_config.get('use_amp', False) # 是否使用AMP
    # 初始化梯度缩放器 (仅在 CUDA 且启用 AMP 时创建)
    grad_scaler = None
    if use_amp_config and device.type == 'cuda':
        grad_scaler = GradScaler()
        logger.info("自动混合精度训练 (AMP) 已启用 (CUDA)。")
    elif use_amp_config: # device.type != 'cuda'
        logger.warning(f"AMP 配置为启用但设备为 '{device.type}'。AMP 将被禁用。")
    else: # use_amp_config is False
        logger.info("自动混合精度训练 (AMP) 未启用。")
    # --- Warmup Configuration ---
    warmup_epochs = training_config.get('warmup_epochs', 0)
    # 如果未提供 warmup_start_lr，则默认为 Warmup 后的目标学习率 (即不进行 Warmup)
    warmup_start_lr = training_config.get('warmup_start_lr', initial_optimizer_lr)
    # 初始化全局步数计数器 (用于 Warmup)
    global_step = 0
    total_train_batches = len(train_loader)
    total_warmup_steps = warmup_epochs * total_train_batches
    # 检查 Warmup 配置是否有效
    if warmup_epochs > 0 and warmup_start_lr < initial_optimizer_lr:
        logger.info(f"启用学习率 Warmup: 从 {warmup_start_lr:.2e} 线性增加到 {initial_optimizer_lr:.2e}，持续 {warmup_epochs} 轮 ({total_warmup_steps} 步).")
        # 在第一个训练步之前，将优化器的学习率设置为 Warmup 开始的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_start_lr
    elif warmup_epochs > 0 and warmup_start_lr >= initial_optimizer_lr:
         logger.warning(f"Warmup 开始学习率 ({warmup_start_lr:.2e}) >= 目标学习率 ({initial_optimizer_lr:.2e})。Warmup 将被跳过。")
         warmup_epochs = 0 # 禁用 Warmup
         total_warmup_steps = 0
    else:
        logger.info("学习率 Warmup 未启用。")
    # 初始化学习率调度器
    lr_scheduler_type = training_config.get('lr_scheduler', 'ReduceLROnPlateau').lower()
    scheduler = None
    scheduler_mode = 'min' if 'loss' in monitor_metric or 'mae' in monitor_metric else 'max'
    if val_loader is not None and len(val_loader) > 0 and reduce_lr_patience > 0:
         scheduler = ReduceLROnPlateau(
             optimizer, mode=scheduler_mode, factor=reduce_lr_factor,
             patience=reduce_lr_patience, min_lr=min_lr
         )
         logger.info(f"启用 ReduceLROnPlateau: monitor='{monitor_metric}', mode='{scheduler_mode}', factor={reduce_lr_factor}, patience={reduce_lr_patience}, min_lr={min_lr:.2e}. 调度器将在 Warmup ({warmup_epochs} 轮) 后生效。")
    elif reduce_lr_patience > 0:
         logger.warning("验证集 (val_loader) 为空或未提供，ReduceLROnPlateau 将被禁用。")
    # --- 模型文件路径和 TensorBoard 日志目录的动态生成 ---
    # checkpoint_dir 是外部传入的股票专属目录 (e.g., /.../000089.SZ/trained_model)
    base_checkpoint_path = Path(checkpoint_dir)
    os.makedirs(base_checkpoint_path, exist_ok=True) # 确保目录存在
    best_model_filename = "" # 实际用于保存的检查点文件名
    if trial is not None:
        # 如果是 Optuna Trial，使用 Trial ID 作为后缀，避免并行冲突
        trial_id = trial.number
        best_model_filename = f"best_transformer_model_{stock_code}_trial_{trial_id}.pth"
        logger.info(f"Optuna Trial {trial_id}: 最佳模型将保存到 Trial-specific 文件: {best_model_filename}")
    else:
        # 对于最终训练或独立运行，使用标准文件名
        best_model_filename = f"best_transformer_model_{stock_code}.pth"
        logger.info(f"非 Optuna Trial: 最佳模型将保存到标准文件: {best_model_filename}")
    # 这是在训练过程中实际保存和加载的路径
    current_best_model_filepath = base_checkpoint_path / best_model_filename
    # TensorBoard writer 的路径也需要区分 Trial
    writer = None
    tensorboard_log_dir_base = training_config.get('tensorboard_log_dir', None)
    if tensorboard_log_dir_base:
        current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_run_id = f"{stock_code}_{current_time_str}"
        if trial is not None:
            log_run_id = f"{stock_code}_trial_{trial.number}_{current_time_str}"
        run_log_dir = os.path.join(tensorboard_log_dir_base, f"transformer_{log_run_id}")
        os.makedirs(run_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_log_dir)
        logger.info(f"启用 TensorBoard: 日志将保存到 '{run_log_dir}'")
    # --- 动态路径生成结束 ---
    # 早停和最佳模型保存相关变量
    best_monitored_value = float('inf') if scheduler_mode == 'min' else float('-inf')
    epochs_no_improve = 0
    consecutive_nan_metric_epochs = 0 # 连续出现 NaN 监控指标的轮数
    early_stop_triggered = False
    training_halted_due_to_retries = False # 标记因 NaN/Inf 重试次数用尽导致训练停止
    best_model_saved = False # 标记是否成功保存过最佳模型
    # 存储训练历史
    history = {'epoch': [], 'loss': [], 'mae': [], 'lr': [], 'val_loss': [], 'val_mae': [], 'val_true_mae': []}
    logger.info(f"开始训练，共 {epochs} 轮。每轮训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader) if val_loader else 0}.")
    if len(train_loader) == 0:
        logger.error("训练 DataLoader 为空，无法进行模型训练。")
        if writer: writer.close()
        if enable_anomaly_detection:
            if hasattr(torch.autograd, 'set_detect_anomaly'):
                torch.autograd.set_detect_anomaly(False) # pylint: disable=not-callable
            logger.info("PyTorch 异常检测已禁用。")
        return model, pd.DataFrame(history)
    current_epoch = 0 # 当前训练的 Epoch 编号 (从 0 开始)
    max_epoch_retries = 10 # 每个 Epoch 的最大重试次数
    try:
        while current_epoch < epochs and not early_stop_triggered and not training_halted_due_to_retries:
            epoch_start_time = time.time()
            epoch_retries = 0 # 当前 Epoch 的重试计数器
            epoch_completed_successfully = False # 标记当前 Epoch 是否成功完成 (无 NaN/Inf)
            while epoch_retries <= max_epoch_retries:
                nan_inf_in_this_attempt = False # 标记当前重试尝试中是否出现 NaN/Inf
                # --- 训练阶段 (当前尝试) ---
                model.train() # 设置为训练模式
                epoch_train_loss_sum = 0.0
                epoch_train_mae_sum = 0.0 # 存储缩放后的 MAE
                num_train_samples = 0
                train_loop = tqdm(train_loader, leave=False, desc=f"Epoch {current_epoch+1}/{epochs} [Train, Retry {epoch_retries+1}]")
                for batch_idx, (inputs, targets) in enumerate(train_loop):
                    inputs, targets = inputs.to(device), targets.to(device)
                    current_batch_size = inputs.size(0)
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss_train = torch.tensor(np.nan, device=device)
                    batch_mae_train = torch.tensor(np.nan, device=device)
                    outputs = None
                    performed_optimizer_step = False
                    try:
                        # --- 学习率 Warmup 逻辑 (按步调整) ---
                        current_lr_for_step = initial_optimizer_lr
                        if warmup_epochs > 0 and global_step < total_warmup_steps:
                            current_lr_for_step = warmup_start_lr + (initial_optimizer_lr - warmup_start_lr) * (global_step / total_warmup_steps)
                            current_lr_for_step = min(current_lr_for_step, initial_optimizer_lr)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = current_lr_for_step
                        # --- Warmup 逻辑结束 ---
                        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                            logger.error(f"检测到 NaN/Inf 在输入数据中！Epoch {current_epoch+1}, Batch {batch_idx+1}.")
                            nan_inf_in_this_attempt = True
                            break
                        if torch.isnan(targets).any() or torch.isinf(targets).any():
                            logger.error(f"检测到 NaN/Inf 在目标数据中！Epoch {current_epoch+1}, Batch {batch_idx+1}.")
                            nan_inf_in_this_attempt = True
                            break
                        with autocast(device_type=device.type, enabled=grad_scaler is not None):
                            outputs = model(inputs)
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                logger.error(f"检测到 NaN/Inf 在模型输出中！Epoch {current_epoch+1}, Batch {batch_idx+1}.")
                                nan_inf_in_this_attempt = True
                                break
                            batch_loss_train = criterion(outputs, targets)
                            if torch.isnan(batch_loss_train) or torch.isinf(batch_loss_train):
                                 logger.error(f"检测到 NaN/Inf 在损失值中！Epoch {current_epoch+1}, Batch {batch_idx+1}.")
                                 nan_inf_in_this_attempt = True
                                 break
                        if grad_scaler:
                            grad_scaler.scale(batch_loss_train).backward()
                            if clip_grad_norm_value is not None and clip_grad_norm_value > 0:
                                grad_scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                            performed_optimizer_step = True
                        else:
                            batch_loss_train.backward()
                            found_nan_grad = False
                            for name, param in model.named_parameters():
                                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                    logger.error(f"检测到 NaN/Inf 在参数 '{name}' 的梯度中！Epoch {current_epoch+1}, Batch {batch_idx+1}.")
                                    found_nan_grad = True
                                    break
                            if found_nan_grad:
                                logger.error(f"由于梯度中存在NaN/Inf，已跳过 optimizer.step()。Epoch {current_epoch+1}, Batch {batch_idx+1}")
                                nan_inf_in_this_attempt = True
                                break
                            if clip_grad_norm_value is not None and clip_grad_norm_value > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)
                            optimizer.step()
                            performed_optimizer_step = True
                        if performed_optimizer_step:
                            for name, param in model.named_parameters():
                                if param.data is not None and (torch.isnan(param.data).any() or torch.isinf(param.data).any()):
                                    logger.error(f"严重错误: 模型参数 '{name}' 在 optimizer.step() 后包含 NaN/Inf！Epoch {current_epoch+1}, Batch {batch_idx+1}.")
                                    nan_inf_in_this_attempt = True
                                    break
                            if nan_inf_in_this_attempt:
                                break
                        if performed_optimizer_step:
                            global_step += 1
                        with torch.no_grad():
                            batch_mae_train = mae_eval_metric(outputs, targets)
                            if torch.isnan(batch_mae_train) or torch.isinf(batch_mae_train):
                                 logger.warning(f"Epoch {current_epoch+1}, Batch {batch_idx+1}: 训练批次MAE(scaled)为 NaN/Inf。")
                    except Exception as e_batch:
                        logger.error(f"训练批次 Epoch {current_epoch+1}, Batch {batch_idx+1} 发生未知错误: {e_batch}", exc_info=True)
                        nan_inf_in_this_attempt = True
                        if enable_anomaly_detection and isinstance(e_batch, RuntimeError) and "Traceback of forward call that caused the error" in str(e_batch):
                            logger.error("PyTorch anomaly detection pinpointed an error in the forward pass. Check logs above for details.")
                        break
                    if not nan_inf_in_this_attempt:
                        if not (torch.isnan(batch_loss_train) or torch.isinf(batch_loss_train)):
                            epoch_train_loss_sum += batch_loss_train.item() * current_batch_size
                        if outputs is not None and not (torch.isnan(outputs).any() or torch.isinf(outputs).any()):
                            if not (torch.isnan(batch_mae_train) or torch.isinf(batch_mae_train)):
                                 epoch_train_mae_sum += batch_mae_train.item() * current_batch_size
                        num_train_samples += current_batch_size
                        if not (torch.isnan(batch_loss_train) or torch.isinf(batch_loss_train)) and \
                           not (torch.isnan(batch_mae_train) or torch.isinf(batch_mae_train)):
                            train_loop.set_postfix(loss=batch_loss_train.item(), mae=batch_mae_train.item())
                        else:
                            train_loop.set_postfix(loss="NaN/Inf", mae="NaN/Inf")
                if nan_inf_in_this_attempt:
                    epoch_retries += 1
                    if epoch_retries <= max_epoch_retries:
                        current_lr_before_retry_adjust = optimizer.param_groups[0]['lr']
                        new_lr = max(current_lr_before_retry_adjust * reduce_lr_factor, min_lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.warning(f"Epoch {current_epoch+1}: 检测到 NaN/Inf，降低学习率至 {new_lr:.2e} 并重试本轮 (尝试 {epoch_retries+1}/{max_epoch_retries+1})。")
                        epoch_train_loss_sum = 0.0
                        epoch_train_mae_sum = 0.0
                        num_train_samples = 0
                    else:
                        logger.error(f"Epoch {current_epoch+1}: 重试次数 ({max_epoch_retries}) 已用尽，训练停止。")
                        training_halted_due_to_retries = True
                        break
                else:
                    epoch_completed_successfully = True
                    break
            if training_halted_due_to_retries:
                logger.error(f"Epoch {current_epoch+1} 因 NaN/Inf 重试次数用尽而提前终止整个训练。")
                history['epoch'].append(current_epoch + 1)
                history['loss'].append(np.nan)
                history['mae'].append(np.nan)
                history['lr'].append(optimizer.param_groups[0]['lr'])
                history['val_loss'].append(np.nan)
                history['val_mae'].append(np.nan)
                history['val_true_mae'].append(np.nan)
                break
            if epoch_completed_successfully:
                avg_train_loss = epoch_train_loss_sum / num_train_samples if num_train_samples > 0 else np.nan
                avg_train_mae = epoch_train_mae_sum / num_train_samples if num_train_samples > 0 else np.nan
                avg_val_loss, avg_val_mae, avg_val_true_mae = np.nan, np.nan, np.nan
                # --- 验证阶段 ---
                if val_loader is not None and len(val_loader) > 0:
                    model.eval()
                    epoch_loss_sum_val, epoch_mae_sum_val, epoch_true_mae_sum_val = 0.0, 0.0, 0.0
                    num_valid_samples_for_loss, num_valid_samples_for_mae, num_valid_samples_for_true_mae = 0, 0, 0
                    nan_batches_in_val_epoch = 0
                    val_loop = tqdm(val_loader, leave=False, desc=f"Epoch {current_epoch+1}/{epochs} [Validate]")
                    with torch.no_grad():
                        with autocast(device_type=device.type, enabled=grad_scaler is not None):
                            for val_batch_idx, (val_inputs, val_targets) in enumerate(val_loop):
                                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                                current_batch_size_val = val_inputs.size(0)
                                val_outputs, val_loss_batch, val_mae_batch = None, torch.tensor(np.nan), torch.tensor(np.nan)
                                try:
                                    if torch.isnan(val_inputs).any() or torch.isinf(val_inputs).any():
                                        logger.error(f"检测到 NaN/Inf 在验证输入数据中！Epoch {current_epoch+1}, Val_Batch {val_batch_idx+1}.")
                                        raise ValueError("Validation input is NaN/Inf")
                                    if torch.isnan(val_targets).any() or torch.isinf(val_targets).any():
                                        logger.error(f"检测到 NaN/Inf 在验证目标数据中！Epoch {current_epoch+1}, Val_Batch {val_batch_idx+1}.")
                                        raise ValueError("Validation target is NaN/Inf")
                                    val_outputs = model(val_inputs)
                                    if torch.isnan(val_outputs).any() or torch.isinf(val_outputs).any():
                                        logger.error(f"检测到 NaN/Inf 在验证模型输出中！Epoch {current_epoch+1}, Val_Batch {val_batch_idx+1}.")
                                        raise ValueError("Validation model output is NaN/Inf")
                                    val_loss_batch = criterion(val_outputs, val_targets)
                                    if torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch):
                                        logger.warning(f"Epoch {current_epoch+1}, Val_Batch {val_batch_idx+1}: 验证批次损失为 NaN/Inf。")
                                        raise ValueError("Validation loss is NaN/Inf")
                                    val_mae_batch = mae_eval_metric(val_outputs, val_targets)
                                    if torch.isnan(val_mae_batch) or torch.isinf(val_mae_batch):
                                        logger.warning(f"Epoch {current_epoch+1}, Val_Batch {val_batch_idx+1}: 验证批次MAE(scaled)为 NaN/Inf。")
                                except ValueError as e_val_nan_inf:
                                    logger.warning(f"Epoch {current_epoch+1}, Val_Batch {val_batch_idx+1}: 验证中因 '{e_val_nan_inf}' 中止此批次指标计算。")
                                    nan_batches_in_val_epoch += 1
                                except Exception as e_val_batch:
                                    logger.error(f"验证批次 Epoch {current_epoch+1}, Val_Batch {val_batch_idx+1} 发生未知错误: {e_val_batch}", exc_info=True)
                                    nan_batches_in_val_epoch += 1
                                if not (torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch)):
                                    epoch_loss_sum_val += val_loss_batch.item() * current_batch_size_val
                                    num_valid_samples_for_loss += current_batch_size_val
                                if val_outputs is not None and not (torch.isnan(val_outputs).any() or torch.isinf(val_outputs).any()):
                                    if not (torch.isnan(val_mae_batch) or torch.isinf(val_mae_batch)):
                                         epoch_mae_sum_val += val_mae_batch.item() * current_batch_size_val
                                         num_valid_samples_for_mae += current_batch_size_val
                                    if target_scaler:
                                        val_outputs_np = val_outputs.cpu().numpy()
                                        val_targets_np = val_targets.cpu().numpy()
                                        true_mae_batch_val = np.nan
                                        can_compute_true_mae = not (np.isnan(val_outputs_np).any() or np.isinf(val_outputs_np).any() or \
                                                                    np.isnan(val_targets_np).any() or np.isinf(val_targets_np).any())
                                        if can_compute_true_mae:
                                            try:
                                                if val_outputs_np.ndim == 1:
                                                     val_outputs_np = val_outputs_np.reshape(-1, 1)
                                                if val_targets_np.ndim == 1:
                                                     val_targets_np = val_targets_np.reshape(-1, 1)
                                                val_outputs_original = target_scaler.inverse_transform(val_outputs_np)
                                                val_targets_original = target_scaler.inverse_transform(val_targets_np)
                                                if not (np.isnan(val_outputs_original).any() or np.isinf(val_outputs_original).any() or \
                                                        np.isnan(val_targets_original).any() or np.isinf(val_targets_original).any()):
                                                    if val_outputs_original.shape == val_targets_original.shape:
                                                        true_mae_batch_val = np.mean(np.abs(val_outputs_original - val_targets_original))
                                                    else:
                                                        logger.warning(f"Epoch {current_epoch+1}, Val_Batch {val_batch_idx+1}: 反向缩放后的形状不匹配 ({val_outputs_original.shape} vs {val_targets_original.shape})，无法计算真实 MAE。")
                                                        true_mae_batch_val = np.nan
                                            except Exception as e_inverse:
                                                pass
                                        if not np.isnan(true_mae_batch_val):
                                            epoch_true_mae_sum_val += true_mae_batch_val * current_batch_size_val
                                            num_valid_samples_for_true_mae += current_batch_size_val
                                if not (torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch)) and \
                                   not (torch.isnan(val_mae_batch) or torch.isinf(val_mae_batch)):
                                    val_loop.set_postfix(val_loss=val_loss_batch.item(), val_mae=val_mae_batch.item())
                                else:
                                    val_loop.set_postfix(val_loss="NaN/Inf", val_mae="NaN/Inf")
                    if nan_batches_in_val_epoch > 0:
                        logger.warning(f"Epoch {current_epoch+1}: 验证中有 {nan_batches_in_val_epoch}/{len(val_loader)} 个批次出现NaN/Inf问题。")
                    avg_val_loss = epoch_loss_sum_val / num_valid_samples_for_loss if num_valid_samples_for_loss > 0 else np.nan
                    avg_val_mae = epoch_mae_sum_val / num_valid_samples_for_mae if num_valid_samples_for_mae > 0 else np.nan
                    avg_val_true_mae = epoch_true_mae_sum_val / num_valid_samples_for_true_mae if num_valid_samples_for_true_mae > 0 else np.nan
                    # --- 根据 val_mae 调整早停耐心 ---
                    if not np.isnan(avg_val_mae) and avg_val_mae < 0.01:
                        if early_stopping_patience > 4:
                            early_stopping_patience = 4
                            logger.info(f"Epoch {current_epoch+1}: 验证MAE(缩放) {avg_val_mae:.4f} 小于 0.01，早停耐心已设置为 4。")
                    elif not np.isnan(avg_val_mae) and avg_val_mae < 0.015:
                        if early_stopping_patience > 8:
                            early_stopping_patience = 8
                            logger.info(f"Epoch {current_epoch+1}: 验证MAE(缩放) {avg_val_mae:.4f} 小于 0.02，早停耐心已设置为 8。")
                    # --- Optuna 早停机制 ---
                    if trial is not None:
                        prune_metric = avg_val_mae if not np.isnan(avg_val_mae) else avg_val_loss
                        # 确保 Optuna 报告的是一个非 NaN 的值，否则它可能会中断
                        if not np.isnan(prune_metric):
                            print(f"[Epoch {current_epoch+1}] 向 Optuna 报告指标: {prune_metric:.6f}")
                            trial.report(prune_metric, current_epoch)
                        else:
                            print(f"[Epoch {current_epoch+1}] 监控指标为 NaN，不向 Optuna 报告。")
                        if trial.should_prune():
                            print(f"[Epoch {current_epoch+1}] Optuna 触发早停，Trial 被中断。")
                            raise optuna.exceptions.TrialPruned()
                    # --- 学习率调度与早停逻辑 ---
                    monitored_value_for_scheduler = np.nan
                    if monitor_metric == 'val_loss': monitored_value_for_scheduler = avg_val_loss
                    elif monitor_metric == 'val_mae': monitored_value_for_scheduler = avg_val_mae
                    elif monitor_metric == 'val_true_mae': monitored_value_for_scheduler = avg_val_true_mae
                    monitored_value_for_scheduler_for_lr_es = np.nan
                    if np.isnan(monitored_value_for_scheduler):
                        logger.warning(f"监控指标 '{monitor_metric}' 在 Epoch {current_epoch+1} 为 NaN。")
                        consecutive_nan_metric_epochs += 1
                        if monitor_metric != 'val_loss' and not np.isnan(avg_val_loss):
                             logger.info(f"回退到使用 'val_loss' ({avg_val_loss:.4f}) 进行学习率调度/早停决策。")
                             monitored_value_for_scheduler_for_lr_es = avg_val_loss
                    else:
                        consecutive_nan_metric_epochs = 0
                        monitored_value_for_scheduler_for_lr_es = monitored_value_for_scheduler
                    lr_before_scheduler_step = optimizer.param_groups[0]['lr']
                    if current_epoch >= warmup_epochs and lr_scheduler_type == 'reducelronplateau' and scheduler and not np.isnan(monitored_value_for_scheduler_for_lr_es):
                        scheduler.step(monitored_value_for_scheduler_for_lr_es)
                        new_lr_after_scheduler = optimizer.param_groups[0]['lr']
                        if new_lr_after_scheduler < lr_before_scheduler_step:
                             logger.info(f"Epoch {current_epoch+1}: ReduceLROnPlateau 学习率从 {lr_before_scheduler_step:.2e} 降低到 {new_lr_after_scheduler:.2e} (基于 {monitor_metric}={monitored_value_for_scheduler_for_lr_es:.4f})")
                    elif current_epoch >= warmup_epochs and lr_scheduler_type in ['cosineannealinglr'] and scheduler :
                         scheduler.step()
                         new_lr_after_scheduler = optimizer.param_groups[0]['lr']
                         if new_lr_after_scheduler != lr_before_scheduler_step:
                              logger.info(f"Epoch {current_epoch+1}: 学习率调度器调整学习率至 {new_lr_after_scheduler:.2e}.")
                    # 早停逻辑 (使用 current_best_model_filepath)
                    if early_stopping_patience > 0:
                        if consecutive_nan_metric_epochs >= nan_metric_patience:
                            logger.info(f"监控指标 '{monitor_metric}' 连续 {consecutive_nan_metric_epochs} 轮为 NaN，触发早停。")
                            early_stop_triggered = True
                        elif not np.isnan(monitored_value_for_scheduler_for_lr_es):
                            improved = False
                            if scheduler_mode == 'min' and monitored_value_for_scheduler_for_lr_es < best_monitored_value:
                                improved = True
                            elif scheduler_mode == 'max' and monitored_value_for_scheduler_for_lr_es > best_monitored_value:
                                improved = True
                            if improved:
                                best_monitored_value = monitored_value_for_scheduler_for_lr_es
                                epochs_no_improve = 0
                                # 保存最佳模型到 Trial 专属或标准路径
                                try:
                                    torch.save(model.state_dict(), current_best_model_filepath) # <--- 关键修改点
                                    best_model_saved = True # 标记最佳模型已保存
                                    logger.info(f"Epoch {current_epoch+1}: {monitor_metric} 提升 ({best_monitored_value:.4f})。保存最佳模型到 '{current_best_model_filepath.name}' ({stock_code})")
                                except Exception as e:
                                    logger.error(f"保存最佳模型到 '{current_best_model_filepath}' 失败: {e}", exc_info=True)
                                    best_model_saved = False # 如果保存失败，确保此标记为 False
                            else:
                                epochs_no_improve += 1
                                logger.info(f"Epoch {current_epoch+1}: {monitor_metric} ({monitored_value_for_scheduler_for_lr_es:.4f}) 未能超越最佳 ({best_monitored_value:.4f})。连续未提升: {epochs_no_improve}/{early_stopping_patience}")
                                if epochs_no_improve >= early_stopping_patience:
                                    logger.info(f"{monitor_metric} 连续 {early_stopping_patience} 轮未提升，触发早停。")
                                    early_stop_triggered = True
                        else:
                            epochs_no_improve +=1
                            logger.warning(f"Epoch {current_epoch+1}: 监控指标 '{monitor_metric}' 为 NaN，视为未提升。连续未提升轮次: {epochs_no_improve}/{early_stopping_patience}")
                            if epochs_no_improve >= early_stopping_patience:
                                logger.info(f"监控指标连续 {early_stopping_patience} 轮为 NaN 或未提升，触发早停。")
                                early_stop_triggered = True
                else:
                    if np.isnan(avg_train_loss) or np.isinf(avg_train_loss) :
                        consecutive_nan_metric_epochs +=1
                        if consecutive_nan_metric_epochs >= nan_metric_patience:
                            logger.error(f"训练损失连续 {consecutive_nan_metric_epochs} 轮为NaN，且无验证集，触发早停。")
                            early_stop_triggered = True
                    else:
                        consecutive_nan_metric_epochs = 0
                # --- Epoch 结束日志与 TensorBoard ---
                epoch_duration = time.time() - epoch_start_time
                train_loss_str = f"{avg_train_loss:.4f}" if not np.isnan(avg_train_loss) else "N/A"
                train_mae_str = f"{avg_train_mae:.4f}" if not np.isnan(avg_train_mae) else "N/A"
                val_loss_str = f"{avg_val_loss:.4f}" if not np.isnan(avg_val_loss) else "N/A"
                val_mae_str = f"{avg_val_mae:.4f}" if not np.isnan(avg_val_mae) else "N/A"
                val_true_mae_str = f"{avg_val_true_mae:.4f}" if not np.isnan(avg_val_true_mae) else "N/A"
                lr_for_logging = optimizer.param_groups[0]['lr']
                log_msg = (
                    f"轮次 {current_epoch+1}/{epochs} [{epoch_duration:.2f}秒] - "
                    f"学习率: {lr_for_logging:.2e} - "
                    f"训练损失: {train_loss_str}, 训练MAE(缩放): {train_mae_str}"
                )
                if val_loader is not None and len(val_loader) > 0:
                    log_msg += f" - 验证损失: {val_loss_str}, 验证MAE(缩放): {val_mae_str}, 验证MAE(真实): {val_true_mae_str}"
                logger.info(log_msg)
                if writer:
                    if not np.isnan(avg_train_loss): writer.add_scalar('Loss/train', avg_train_loss, current_epoch + 1)
                    if not np.isnan(avg_train_mae): writer.add_scalar('MAE_scaled/train', avg_train_mae, current_epoch + 1)
                    writer.add_scalar('LearningRate', lr_for_logging, current_epoch + 1)
                    if val_loader is not None and len(val_loader) > 0:
                        if not np.isnan(avg_val_loss): writer.add_scalar('Loss/validation', avg_val_loss, current_epoch + 1)
                        if not np.isnan(avg_val_mae): writer.add_scalar('MAE_scaled/validation', avg_val_mae, current_epoch + 1)
                        if not np.isnan(avg_val_true_mae): writer.add_scalar('MAE_true/validation', avg_val_true_mae, current_epoch + 1)
                history['epoch'].append(current_epoch + 1)
                history['loss'].append(avg_train_loss)
                history['mae'].append(avg_train_mae)
                history['lr'].append(lr_for_logging)
                history['val_loss'].append(avg_val_loss)
                history['val_mae'].append(avg_val_mae)
                history['val_true_mae'].append(avg_val_true_mae)
                current_epoch += 1
                if early_stop_triggered:
                    print(f"[Epoch {current_epoch}] 训练完成，早停触发。") # 注意：current_epoch 此时已加 1
                    break
    finally: # 确保在函数退出前（无论正常或异常）关闭异常检测和 TensorBoard writer
        if enable_anomaly_detection:
            if hasattr(torch.autograd, 'set_detect_anomaly'):
                torch.autograd.set_detect_anomaly(False) # pylint: disable=not-callable
            logger.info("PyTorch 异常检测已禁用。")
        if writer:
            writer.close()
    # --- 训练循环结束后的最终模型处理逻辑 ---
    # 如果训练因 NaN/Inf 重试次数用尽而停止，且之前没有保存过最佳模型，则返回当前模型状态。
    # 否则，会尝试加载保存的最佳模型。
    if training_halted_due_to_retries:
        if not best_model_saved:
            logger.warning("训练因 NaN/Inf 重试次数用尽而停止，且没有保存最佳模型。将返回当前模型状态，不加载或保存模型文件。")
        else:
            logger.warning(f"训练因 NaN/Inf 重试次数用尽而停止，但之前已保存最佳模型到 '{current_best_model_filepath}'。将尝试加载最佳模型。")
            try:
                model.load_state_dict(torch.load(current_best_model_filepath, map_location=device)) # <--- 关键修改点
                logger.info(f"已从 '{current_best_model_filepath}' 加载最佳模型。")
            except Exception as e_load:
                logger.error(f"加载最佳模型 '{current_best_model_filepath}' 失败: {e_load}。将返回停止时的模型状态。", exc_info=True)
                # 再次强调：如果加载失败，应该抛出异常，让 Optuna 知道这个 Trial 失败
                raise RuntimeError(f"Error(s) in loading state_dict for TransformerModel (on retry/early stop load): {e_load}")
    elif early_stop_triggered:
        logger.info(f"早停在 Epoch {current_epoch} 被触发。尝试加载最佳模型。")
        if best_model_saved and current_best_model_filepath.exists():
            try:
                model.load_state_dict(torch.load(current_best_model_filepath, map_location=device)) # <--- 关键修改点
                logger.info(f"已从 '{current_best_model_filepath}' 加载最佳模型 (基于 {monitor_metric}={best_monitored_value:.4f})。")
            except Exception as e_load:
                logger.error(f"加载最佳模型 '{current_best_model_filepath}' 失败: {e_load}。将返回当前模型状态。", exc_info=True)
                # 再次强调：如果加载失败，应该抛出异常
                raise RuntimeError(f"Error(s) in loading state_dict for TransformerModel (on early stop load): {e_load}")
        else:
            logger.warning(f"早停已触发，但未找到最佳模型文件 '{current_best_model_filepath}' 或之前保存失败。将返回当前模型状态。")
    else: # 自然结束训练 (跑满了所有 Epoch)
        logger.info(f"训练完成 {epochs} 轮。")
        # 训练自然完成时，将当前模型状态保存为“最终模型”。
        # 如果前面已经有最佳模型被保存 (best_model_saved)，那么 current_best_model_filepath 已经包含它。
        # 如果没有（例如，没有验证集导致没有“最佳”，或者耐心设为0），则保存当前训练结束时的模型。
        # 确保使用相同的路径，这样最终的最好模型会被保留。
        try:
             torch.save(model.state_dict(), current_best_model_filepath) # <--- 关键修改点
             logger.info(f"训练完成，已保存最终模型到 '{current_best_model_filepath}'。")
        except Exception as e_save:
             logger.error(f"保存最终模型 '{current_best_model_filepath}' 失败: {e_save}.", exc_info=True)
    # 将 history 字典转换为 DataFrame
    history_df = pd.DataFrame(history)
    # --- Optuna Trial 文件清理 (仅当在 Optuna Trial 模式下) ---
    # 这一步旨在删除 Optuna Trial 过程中产生的临时模型文件，
    # 避免磁盘被大量临时文件占满，同时保留最终的 `best_transformer_model_{stock_code}.pth` 文件。
    if trial is not None:
        # 如果当前文件路径包含 Trial ID (即是临时文件)
        if "_trial_" in current_best_model_filepath.name:
            if current_best_model_filepath.exists():
                try:
                    os.remove(current_best_model_filepath)
                    logger.info(f"Optuna Trial {trial.number}: 已删除临时模型文件 '{current_best_model_filepath.name}'。")
                except Exception as e_del:
                    logger.warning(f"Optuna Trial {trial.number}: 删除临时模型文件 '{current_best_model_filepath.name}' 失败: {e_del}")
    # --- 清理结束 ---
    # 可选：绘制训练历史
    # if plot_training_history:
    #     try:
    #         import matplotlib.pyplot as plt # 确保导入 matplotlib
    #         fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    #         if 'loss' in history_df.columns:
    #             axes[0].plot(history_df['epoch'], history_df['loss'], label='Train Loss')
    #         if 'val_loss' in history_df.columns and not history_df['val_loss'].isnull().all():
    #             axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    #         axes[0].set_ylabel('Loss')
    #         axes[0].legend()
    #         axes[0].set_title(f'Training and Validation Loss ({stock_code})')
    #         if 'mae' in history_df.columns:
    #             axes[1].plot(history_df['epoch'], history_df['mae'], label='Train MAE (Scaled)')
    #         if 'val_mae' in history_df.columns and not history_df['val_mae'].isnull().all():
    #             axes[1].plot(history_df['epoch'], history_df['val_mae'], label='Validation MAE (Scaled)')
    #         if 'val_true_mae' in history_df.columns and not history_df['val_true_mae'].isnull().all():
    #             axes[1].plot(history_df['epoch'], history_df['val_true_mae'], label='Validation MAE (True)', linestyle='--')
    #         axes[1].set_xlabel('Epoch')
    #         axes[1].set_ylabel('Mean Absolute Error (MAE)')
    #         axes[1].legend()
    #         axes[1].set_title(f'Training and Validation MAE ({stock_code})')
    #         plt.tight_layout()
    #         # 这里的 plot_filename 也应该考虑 Optuna Trial ID，如果需要在每个 trial 结束时保存图表
    #         plot_filename_suffix = ""
    #         if trial is not None:
    #             plot_filename_suffix = f"_trial_{trial.number}"
    #         plot_filename = base_checkpoint_path / f"training_history_{stock_code}{plot_filename_suffix}.png"
    #         plt.savefig(plot_filename)
    #         logger.info(f"训练历史图表已保存到: {plot_filename}")
    #         plt.close(fig)
    #     except ImportError:
    #         logger.warning("Matplotlib 未安装，无法绘制训练历史图表。")
    #     except Exception as e_plot:
    #         logger.error(f"绘制训练历史图表时出错: {e_plot}", exc_info=True)
    logger.info(f"Transformer 模型训练流程结束 (股票/标识: {stock_code})。")
    return model, history_df

@log_execution_time
@handle_exceptions
def predict_with_transformer_model(
    model: TransformerModel,
    data: pd.DataFrame,
    feature_scaler: Union[MinMaxScaler, StandardScaler, RobustScaler], # 修改类型提示
    target_scaler: Union[MinMaxScaler, StandardScaler, RobustScaler], # 修改类型提示
    selected_feature_names: List[str],
    window_size: int,
    device: Optional[torch.device] = None
) -> float:
    """
    使用训练好的 Transformer 模型对最新数据进行单步预测。
    重要提示:
    传入的 `data` DataFrame 应该包含模型训练时最终使用的特征列 (由 `selected_feature_names` 指定)。
    如果训练时应用了 PCA 或复杂的特征选择，`data` 应该已经经过了这些转换，
    `selected_feature_names` 应该是这些转换后特征的名称 (例如 'pca_comp_0', ...)。
    此函数仅负责对这些选定特征进行缩放、窗口化和模型预测。
    Args:
        model (TransformerModel): 已加载权重的 PyTorch Transformer 模型。
        data (pd.DataFrame): 包含模型所需特征列的最新数据 DataFrame。
                             应至少包含 `window_size` 行数据。
        feature_scaler (Union[MinMaxScaler, StandardScaler, RobustScaler]): 用于特征的已拟合缩放器。
        target_scaler (Union[MinMaxScaler, StandardScaler, RobustScaler]): 用于目标的已拟合缩放器，用于反转预测结果。
        selected_feature_names (List[str]): 模型训练时最终使用的特征名列表。
        window_size (int): 模型期望的输入窗口大小。
        device (Optional[torch.device]): 预测设备 (CPU 或 GPU)。如果为 None，则自动检测。
    Returns:
        float: 预测的信号值 (原始尺度)。如果无法预测，则返回一个中性值 (例如 50.0)。
    """
    logger.info("开始使用 Transformer 模型进行预测...")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"自动选择预测设备: {device}")
    # --- 0. 输入数据校验 ---
    if data is None or data.empty:
        logger.warning("输入数据 (data) 为空，无法进行预测。返回默认值 50.0。")
        return 50.0
    if len(data) < window_size:
        logger.warning(f"输入数据长度 ({len(data)}) 小于窗口大小 ({window_size})，无法构建预测窗口。返回默认值 50.0。")
        return 50.0
    if feature_scaler is None:
        logger.error("特征缩放器 (feature_scaler) 未提供，无法进行预测。返回默认值 50.0。")
        return 50.0
    if target_scaler is None:
        logger.error("目标缩放器 (target_scaler) 未提供，无法反转预测结果。返回默认值 50.0。")
        return 50.0
    if not selected_feature_names:
        logger.error("选定特征列表 (selected_feature_names) 为空。无法进行预测。返回默认值 50.0。")
        return 50.0
    # --- 1. 准备预测数据 ---
    # 1.1. 选择用于预测的特征列 (必须与训练时使用的特征一致)
    try:
        # 确保只选择存在的列，并按 selected_feature_names 的顺序排列
        features_for_prediction_df = data[selected_feature_names].copy()
    except KeyError as e:
        missing_cols = set(selected_feature_names) - set(data.columns)
        logger.error(f"部分选定特征列在输入数据中缺失: {missing_cols}。原始错误: {e}。无法进行预测。返回默认值 50.0。")
        return 50.0
    # 1.2. 处理 NaN 值 (与数据准备时一致，例如 ffill().bfill().fillna(0) )
    # 假设预测时拿到的数据已经是相对干净的，或者上游已处理。这里做一个基本保障。
    if features_for_prediction_df.isnull().any().any():
        logger.warning("用于 Transformer 预测的数据中检测到 NaN 值，将尝试使用 ffill().bfill().fillna(0) 进行填充。")
        features_for_prediction_df = features_for_prediction_df.ffill().bfill().fillna(0)
        if features_for_prediction_df.isnull().any().any(): # 再次检查
             logger.error("填充后，预测数据中仍存在无法处理的 NaN 值。无法进行预测。返回默认值 50.0。")
             return 50.0
    # 1.3. 应用特征缩放 (使用训练时拟合好的 scaler)
    try:
        features_scaled_np = feature_scaler.transform(features_for_prediction_df.values)
    except Exception as e_scale:
        logger.error(f"应用特征缩放器 (feature_scaler) 时出错: {e_scale}。无法进行预测。返回默认值 50.0。", exc_info=True)
        return 50.0
    # 1.4. 构建预测所需的窗口数据 (只取最后一个窗口进行最新预测)
    # PyTorch Transformer 期望输入形状 (batch_size, sequence_length, num_features)
    # 我们需要最后 window_size 行数据构成一个序列。
    if features_scaled_np.shape[0] < window_size: # 再次确认，虽然前面已检查原始数据长度
        logger.warning(f"缩放后的数据长度 ({features_scaled_np.shape[0]}) 小于窗口大小 ({window_size})，无法构建预测窗口。返回默认值 50.0。")
        return 50.0
    # 取最后 window_size 个时间步的数据，形状 (window_size, num_features)
    X_predict_window_np = features_scaled_np[-window_size:, :]
    # 转换为 PyTorch Tensor 并添加 batch dimension (1, window_size, num_features)
    X_predict_tensor = torch.tensor(X_predict_window_np, dtype=torch.float32).unsqueeze(0)
    # --- 2. 进行预测 ---
    model.to(device) # 确保模型在指定设备
    X_predict_tensor = X_predict_tensor.to(device) # 数据也移至相同设备
    model.eval() # 设置模型为评估模式 (禁用 dropout, batchnorm更新 等)
    with torch.no_grad(): # 在预测阶段不计算梯度，节省内存和计算
        # 预测阶段也可以使用 autocast 来加速，特别是如果模型在训练时使用了 AMP
        with autocast(device_type=device.type, enabled=device.type == 'cuda'): # 预测阶段也使用 autocast (仅限 CUDA)
            try:
                # 模型输出形状通常是 (batch_size, 1)，这里 batch_size 是 1
                predicted_scaled_tensor = model(X_predict_tensor)
            except Exception as e_predict:
                logger.error(f"Transformer 模型前向传播 (预测) 出错: {e_predict}。返回默认值 50.0。", exc_info=True)
                return 50.0
    # --- 3. 逆缩放预测结果 ---
    # 将预测结果 tensor 转移回 CPU 并转换为 numpy
    predicted_scaled_np = predicted_scaled_tensor.cpu().numpy() # Shape: (1, 1)
    try:
        # target_scaler.inverse_transform 期望二维数组
        predicted_original_scale_np = target_scaler.inverse_transform(predicted_scaled_np) # Shape: (1, 1)
        predicted_signal_score = predicted_original_scale_np[0, 0] # 提取单个预测值
    except Exception as e_inv_scale:
        logger.error(f"应用目标缩放器 (target_scaler) 逆缩放时出错: {e_inv_scale}。返回默认值 50.0。", exc_info=True)
        return 50.0
    # --- 4. 后处理预测值 ---
    # 将预测分数限制在合理范围内 (例如 0-100) 并四舍五入
    # 这个范围取决于具体业务场景中 final_signal 的定义
    final_predicted_signal = np.clip(predicted_signal_score, 0, 100)
    final_predicted_signal = round(float(final_predicted_signal), 2) # 四舍五入到两位小数
    # logger.info(f"Transformer 模型预测完成。预测信号 (原始尺度, 0-100范围, 保留2位小数): {final_predicted_signal:.2f}")
    return final_predicted_signal

@log_execution_time
@handle_exceptions # 确保 evaluate 函数也有异常处理
def evaluate_transformer_model(
    model: TransformerModel,
    test_loader: DataLoader,
    criterion: nn.Module,
    target_scaler: Union[MinMaxScaler, StandardScaler, RobustScaler], # 修改类型提示
    mae_metric: nn.Module, # mae_metric 参数，用于计算 MAE
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    在测试集上评估 Transformer 模型性能。
    Args:
        model (TransformerModel): 已加载权重的 PyTorch Transformer 模型。
        test_loader (DataLoader): 测试数据加载器。建议设置 num_workers > 0 和 pin_memory=True
                                  以提高数据加载效率和 GPU 利用率。
        criterion (nn.Module): 损失函数实例 (例如 MSELoss)。
        target_scaler (Union[MinMaxScaler, StandardScaler, RobustScaler]): 用于目标变量的已拟合缩放器，
                                                             用于计算反标准化后的真实 MAE。
        mae_metric (nn.Module): 用于计算 MAE 的度量实例 (例如 nn.L1Loss())。
        device (Optional[torch.device]): 评估设备 (CPU 或 GPU)。如果为 None，则自动检测。
    Returns:
        Dict[str, float]: 包含评估指标的字典:
            - 'loss': 测试集上的平均损失值 (例如 MSE)。
            - 'mae_scaled': 测试集上缩放后数据的平均绝对误差 (MAE)。
            - 'mae_true': 测试集上反标准化 (真实尺度) 数据的平均绝对误差 (MAE)。
    """
    # logger.info("开始在测试集上评估 Transformer 模型...")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"自动选择评估设备: {device}")
    if len(test_loader) == 0:
        logger.warning("测试 DataLoader 为空，无法进行评估。返回 NaN 指标。")
        return {'loss': np.nan, 'mae_scaled': np.nan, 'mae_true': np.nan}
    if target_scaler is None:
        logger.warning("目标缩放器 (target_scaler) 未提供，无法计算真实 MAE。'mae_true' 将为 NaN。")
    model.to(device) # 确保模型在设备上
    model.eval()     # 设置模型为评估模式
    total_test_loss = 0.0
    total_test_mae_scaled = 0.0
    total_test_mae_true = 0.0
    total_samples = 0
    # MAE 评估指标
    # mae_eval_metric = nn.L1Loss(reduction='sum') # 移除内部定义，使用传入的 mae_metric
    with torch.no_grad(): # 在评估阶段不计算梯度
        # 评估阶段也可以使用 autocast 来加速，特别是如果模型在训练时使用了 AMP
        with autocast(device_type=device.type, enabled=device.type == 'cuda'): # 评估阶段也使用 autocast (仅限 CUDA)
            # 使用 tqdm 包装 test_loader，显示测试进度条
            test_loop = tqdm(test_loader, leave=False, desc="Evaluating Test Set")
            for inputs, targets_scaled in test_loop:
                inputs, targets_scaled = inputs.to(device), targets_scaled.to(device)
                batch_size = inputs.size(0)
                outputs_scaled = model(inputs) # 模型输出的是缩放后的值
                # 1. 计算损失 (基于缩放后的值)
                loss_batch = criterion(outputs_scaled, targets_scaled)
                total_test_loss += loss_batch.item() * batch_size # 累加批次总损失
                # 2. 计算缩放后的 MAE
                # 使用传入的 mae_metric。
                # 假设 mae_metric 可能返回批次的平均MAE (如 nn.L1Loss() 默认)，
                # 乘以 batch_size 来得到批次的总MAE，以与原始累加逻辑保持一致。
                mae_scaled_batch_val = mae_metric(outputs_scaled, targets_scaled)
                total_test_mae_scaled += mae_scaled_batch_val.item() * batch_size
                # 3. 计算反标准化后的真实 MAE
                if target_scaler:
                    try:
                        # 在进行 numpy 转换和反向缩放之前，确保 tensor 在 CPU 上
                        outputs_np_scaled = outputs_scaled.cpu().numpy() # 确保在 CPU 上
                        targets_np_scaled = targets_scaled.cpu().numpy() # 确保在 CPU 上
                        outputs_original = target_scaler.inverse_transform(outputs_np_scaled)
                        targets_original = target_scaler.inverse_transform(targets_np_scaled)
                        # 计算这个批次的真实 MAE (逐元素绝对差后求和)
                        mae_true_batch = np.sum(np.abs(outputs_original - targets_original))
                        total_test_mae_true += mae_true_batch
                    except Exception as e_inv_transform_eval:
                        # 将警告级别降低，避免频繁打印
                        # logger.warning(f"评估中反标准化预测/目标时出错: {e_inv_transform_eval}。部分真实MAE可能不准确。")
                        pass # 忽略单个批次的转换警告
                total_samples += batch_size
                # 更新 tqdm 进度条的 postfix 信息，显示当前批次的损失和 MAE
                test_loop.set_postfix(loss=loss_batch.item(), mae_scaled=mae_scaled_batch_val.item())
    if total_samples == 0: # 避免除以零
        logger.warning("测试集中总样本数为零，无法计算平均指标。返回 NaN。")
        return {'loss': np.nan, 'mae_scaled': np.nan, 'mae_true': np.nan}
    avg_test_loss = total_test_loss / total_samples
    avg_test_mae_scaled = total_test_mae_scaled / total_samples
    avg_test_mae_true = total_test_mae_true / total_samples if target_scaler and total_samples > 0 else np.nan # 确保 total_samples > 0
    mae_true_str = f"{avg_test_mae_true:.4f}" if not np.isnan(avg_test_mae_true) else "N/A"
    logger.info(f"测试集评估完成: "
                f"平均损失 (Loss) = {avg_test_loss:.4f}, "
                f"平均MAE (缩放后, Scaled MAE) = {avg_test_mae_scaled:.4f}, "
                f"平均MAE (真实值, True MAE) = {mae_true_str}") # 使用预先格式化好的字符串
    return {
        'loss': avg_test_loss,
        'mae_scaled': avg_test_mae_scaled,
        'mae_true': avg_test_mae_true
    }

