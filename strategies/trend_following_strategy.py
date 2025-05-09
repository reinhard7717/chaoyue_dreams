# 此策略侧重于识别和跟随趋势，主要使用 EMA 排列、DMI、SAR 等指标，并以 30 分钟级别为主要权重。
# trend_following_strategy.py
import pandas as pd
import numpy as np
import json
import os
import logging
from django.conf import settings
import joblib # 用于加载/保存 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 替换 tensorflow 导入为 pytorch 导入
import torch
import torch.nn as nn
# 导入新的 PyTorch 深度学习工具函数和类
from .utils.deep_learning_utils import (
    build_transformer_model,
    evaluate_transformer_model,
    train_transformer_model,
    predict_with_transformer_model,
    TimeSeriesDataset, # 使用新的 Dataset 类
    prepare_data_for_transformer # 使用新的数据准备函数
)
from torch.utils.data import DataLoader # 导入 DataLoader
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas_ta as ta
from .base import BaseStrategy
from .utils import strategy_utils # 导入公共工具
# deep_learning_utils 已经通过 from .utils.deep_learning_utils import ... 导入了所需函数

logger = logging.getLogger("strategy_trend_following")

class TrendFollowingStrategy(BaseStrategy):
    """
    趋势跟踪策略：
    - 基于多时间框架指标评分，并根据参数侧重特定时间框架 (`focus_timeframe`)。
    - 主要关注趋势指标 (DMI, SAR, MACD, EMA排列, OBV趋势) 和趋势强度 (ADX)。
    - 结合量能确认、波动率、STOCH、VWAP、BOLL等辅助判断趋势的持续性、增强或衰竭。
    - 增加量价背离检测作为潜在反转的警示信号。
    - 适应A股 T+1 交易制度，增强假信号过滤，动态调整参数。
    - 集成 Transformer 模型进行信号预测增强。
    """
    strategy_name = "TrendFollowingStrategy" # 默认策略名称
    default_focus_timeframe = '30' # 默认主要关注的时间框架

    def __init__(self, params_file: str = "strategies/indicator_parameters.json", base_data_dir: str = settings.STRATEGY_DATA_DIR):
        """
        初始化趋势跟踪策略。

        Args:
            params_file (str): 策略参数JSON文件的路径。
            base_data_dir (str): 存储策略相关数据（如模型、scalers）的基础目录。
                                 默认为 Django settings中的 STRATEGY_DATA_DIR。
        """
        super().__init__(params_file) # 调用父类的初始化，会加载参数到 self.params

        self.strategy_name = self.params.get('trend_following_strategy_name', self.strategy_name)
        self.base_data_dir = base_data_dir # 用于存储模型、scaler等

        # --- 加载趋势跟踪特定参数 (trend_following_params) ---
        self.tf_params: Dict[str, Any] = self.params.get('trend_following_params', {}) # 获取趋势跟踪参数块，若不存在则为空字典

        # 策略主要关注的时间框架
        self.focus_timeframe: str = str(self.tf_params.get('focus_timeframe', self.default_focus_timeframe))

        # 时间框架权重 (用于多时间框架评分)
        self.timeframe_weights: Optional[Dict[str, float]] = self.tf_params.get('timeframe_weights', None)

        # 策略关注的核心趋势指标列表 (用于分析和信号组合)
        self.trend_indicators: List[str] = self.tf_params.get('trend_indicators', ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi'])

        # 不同信号成分的权重 (用于最终规则信号组合)
        self.rule_signal_weights: Dict[str, float] = self.tf_params.get('rule_signal_weights', {
            'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15,
            'ema_cross': 0.15, 'boll_breakout': 0.1, 'adx_strength': 0.1, 'vwap_deviation': 0.05,
            'volume_spike': 0.05
        })

        # 量能确认相关参数
        vc_global_params = self.params.get('volume_confirmation', {})
        self.volume_boost_factor: float = self.tf_params.get('volume_boost_factor', vc_global_params.get('boost_factor', 1.2))
        self.volume_penalty_factor: float = self.tf_params.get('volume_penalty_factor', vc_global_params.get('penalty_factor', 0.8))
        self.volume_spike_threshold: float = self.tf_params.get('volume_spike_threshold', 2.0)

        # 波动率阈值
        self.volatility_threshold_high: float = self.tf_params.get('volatility_threshold_high', 10.0)
        self.volatility_threshold_low: float = self.tf_params.get('volatility_threshold_low', 5.0)
        self.volatility_adjust_factor: float = 1.0 # 用于动态调整的因子

        # 其他趋势判断辅助参数
        self.adx_strong_threshold: int = self.tf_params.get('adx_strong_threshold', 30)
        self.adx_moderate_threshold: int = self.tf_params.get('adx_moderate_threshold', 20)
        self.trend_duration_threshold_strong: int = self.tf_params.get('trend_duration_threshold_strong', 3)
        self.trend_duration_threshold_moderate: int = self.tf_params.get('trend_duration_threshold_moderate', 5)
        self.stoch_oversold_threshold: int = self.tf_params.get('stoch_oversold_threshold', 20)
        self.stoch_overbought_threshold: int = self.tf_params.get('stoch_overbought_threshold', 80)
        self.vwap_deviation_threshold: float = self.tf_params.get('vwap_deviation_threshold', 0.01)
        self.trend_confirmation_periods: int = self.tf_params.get('trend_confirmation_periods', 3)

        # --- Transformer 模型相关配置 (PyTorch) ---
        # Transformer 输入窗口大小
        self.transformer_window_size: int = self.tf_params.get('transformer_window_size', 60) # 参数名改为 transformer_window_size
        # Transformer 训练批大小
        self.transformer_batch_size: int = self.tf_params.get('transformer_batch_size', 128) # 参数名改为 transformer_batch_size
        # Transformer 目标列名
        self.transformer_target_column: str = self.tf_params.get('transformer_target_column', 'final_rule_signal') # 目标是规则信号

        # Transformer 模型结构配置 (适应 build_transformer_model 函数参数)
        self.transformer_model_config: Dict[str, Any] = self.tf_params.get('transformer_model_config', {
            'd_model': 128, # 特征维度
            'nhead': 8, # 注意力头数
            'dim_feedforward': 512, # 前馈网络维度
            'nlayers': 4, # Encoder层数
            'dropout': 0.2,
            'activation': 'relu',
            # learning_rate, weight_decay 等通常在 training_config 中设置
        })
        # Transformer 训练过程配置 (适应 train_transformer_model 函数参数)
        self.transformer_training_config: Dict[str, Any] = self.tf_params.get('transformer_training_config', {
            'epochs': 100,
            'batch_size': 128, # 训练时实际使用的批大小 (覆盖 self.transformer_batch_size)
            'learning_rate': 0.0001,
            'weight_decay': 0.004, # L2 正则化/权重衰减
            'optimizer': 'adamw', # 推荐 AdamW
            'loss': 'mse', # 回归任务常用 MSE, MAE, Huber
            'early_stopping_patience': 30,
            'reduce_lr_patience': 10,
            'reduce_lr_factor': 0.5,
            'monitor_metric': 'val_loss', # 监控指标
            'verbose': 1,
            'tensorboard_log_dir': None,
            'clip_grad_norm': 1.0 # 梯度裁剪，可选
        })

        # 确保训练配置中的学习率优先于模型配置中的学习率 (如果模型配置也定义了)
        if 'learning_rate' in self.transformer_training_config:
            self.transformer_model_config['learning_rate'] = self.transformer_training_config['learning_rate'] # 虽然 build_model 不直接用，但可以记录
        if 'weight_decay' in self.transformer_training_config:
            self.transformer_model_config['weight_decay'] = self.transformer_training_config['weight_decay'] # 同上
        # 确保训练配置中的 batch_size 覆盖实例属性的 transformer_batch_size
        if 'batch_size' in self.transformer_training_config:
             self.transformer_batch_size = self.transformer_training_config['batch_size']

        # Transformer 数据准备参数 (传递给 prepare_data_for_transformer)
        self.transformer_data_prep_config: Dict[str, Any] = self.tf_params.get('transformer_data_prep_config', {
            'scaler_type': 'standard', # Transformer 通常对 Standard scaling 更友好
            'train_split': 0.7,
            'val_split': 0.15,
            'apply_variance_threshold': False,
            'variance_threshold_value': 0.01,
            'use_pca': False,
            'pca_n_components': 0.99,
            'pca_solver': 'auto',
            'use_feature_selection': True,
            'feature_selector_model_type': 'rf', # 'rf' or 'xgb'
            'fs_model_n_estimators': 100,
            'fs_model_max_depth': None,
            'fs_max_features': 50, # 增加最大特征数，Transformer 可以处理更多特征
            'fs_selection_threshold': 'median',
            'target_scaler_type': 'minmax' # 目标缩放器可以保持 minmax
        })

        # --- 初始化模型和scaler相关属性 ---
        self.transformer_model: Optional[nn.Module] = None # 存储加载或训练好的PyTorch模型
        self.feature_scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None # 特征缩放器
        self.target_scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None # 目标缩放器
        self.selected_feature_names_for_transformer: List[str] = [] # 存储最终用于Transformer的特征名

        # PyTorch 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch 使用设备: {self.device}")


        # 股票特定的模型和scaler路径将在 set_model_paths 中设置
        self.model_path: Optional[str] = None # 模型权重文件路径 (.pth)
        self.feature_scaler_path: Optional[str] = None
        self.target_scaler_path: Optional[str] = None
        self.selected_features_path: Optional[str] = None

        # 存储中间数据和分析结果的DataFrame (可选)
        self.intermediate_data: Optional[pd.DataFrame] = None

        if ta is None:
             logger.error(f"[{self.strategy_name}] pandas_ta 未成功加载，策略部分功能可能不可用。")

        self._validate_params() # 执行参数验证
        logger.info(f"策略 '{self.strategy_name}' 初始化完成。主要关注时间框架: {self.focus_timeframe}")

    def _normalize_weights(self, weights: Dict[str, float]):
        """归一化权重字典，使其总和为1。"""
        total_weight = sum(weights.values())
        if total_weight > 0 and not np.isclose(total_weight, 1.0):
            # logger.debug(f"归一化权重 (总和: {total_weight:.4f})") # 太频繁，改为 debug
            for key in weights:
                weights[key] /= total_weight
        elif total_weight == 0:
             logger.warning("权重总和为零，无法归一化。")

    def _validate_params(self):
        """
        验证策略特定参数的有效性。
        """
        super()._validate_params() # 调用父类的验证

        if 'trend_following_params' not in self.params:
            logger.warning("参数中缺少 'trend_following_params' 部分，将使用大量默认值。")

        bs_params = self.params.get('base_scoring', {})
        if not bs_params.get('timeframes'):
            raise ValueError("参数 'base_scoring.timeframes' 未定义或为空，无法确定操作时间级别。")
        if self.focus_timeframe not in bs_params['timeframes']:
            logger.warning(f"主要关注时间框架 '{self.focus_timeframe}' 不在 'base_scoring.timeframes' ({bs_params['timeframes']}) 中。可能导致错误。")

        if self.timeframe_weights:
            if not isinstance(self.timeframe_weights, dict):
                raise ValueError("'trend_following_params.timeframe_weights' 必须是一个字典。")

        if not self.trend_indicators:
            logger.warning("'trend_following_params.trend_indicators' 为空，策略可能无法有效识别趋势。")

        # 验证 Transformer 相关配置 (简单检查是否存在关键项)
        # 检查 Transformer 模型结构配置中的关键参数
        model_conf = self.transformer_model_config
        required_model_keys = ['d_model', 'nhead', 'dim_feedforward', 'nlayers']
        if not all(key in model_conf for key in required_model_keys):
             logger.warning(f"Transformer模型结构配置 'transformer_model_config' 缺少关键参数: {required_model_keys}。将使用默认值。")
             # 可以考虑在这里补充默认值，或者依赖 build_transformer_model 中的默认值

        # 检查 Transformer 训练配置中的关键参数
        train_conf = self.transformer_training_config
        required_train_keys = ['epochs', 'batch_size', 'learning_rate', 'loss']
        if not all(key in train_conf for key in required_train_keys):
             logger.warning(f"Transformer训练配置 'transformer_training_config' 缺少关键参数: {required_train_keys}。将使用默认值。")
             # 同样，可以依赖 train_transformer_model 中的默认值

        # 验证信号组合权重
        if not isinstance(self.rule_signal_weights, dict) or not self.rule_signal_weights:
             logger.warning("'rule_signal_weights' 参数无效或为空，将使用默认权重。")
             self.rule_signal_weights = {
                'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15,
                'ema_cross': 0.15, 'boll_breakout': 0.1, 'adx_strength': 0.1, 'vwap_deviation': 0.05,
                'volume_spike': 0.05
            }
        self._normalize_weights(self.rule_signal_weights) # 归一化规则信号权重

        # 验证 Transformer 信号组合权重
        lstm_combination_weights = self.tf_params.get('signal_combination_weights', {}) # 参数名保持 lstm_... 兼容旧配置
        if not isinstance(lstm_combination_weights, dict) or not lstm_combination_weights:
             logger.warning("'signal_combination_weights' 参数无效或为空，将使用默认权重 0.7/0.3。")
             self.tf_params['signal_combination_weights'] = {'rule_weight': 0.7, 'lstm_weight': 0.3} # 兼容旧的 key names
        self._normalize_weights(self.tf_params['signal_combination_weights']) # 归一化组合信号权重

        logger.debug(f"[{self.strategy_name}] 特定参数验证完成。")

    def set_model_paths(self, stock_code: str):
        """
        为特定股票设置模型、scaler 和准备好的数据的保存/加载路径，增加子目录区分。
        结构为: base_data_dir/stock_code/prepared_data/... 和 base_data_dir/stock_code/trained_model/...
        模型文件 확장名改为 .pth 适应 PyTorch 状态字典保存。
        """
        # 股票特定的根目录
        stock_root_dir = os.path.join(self.base_data_dir, stock_code)
        if not os.path.exists(stock_root_dir):
            os.makedirs(stock_root_dir)
            logger.info(f"创建股票根目录: {stock_root_dir}")

        # 数据准备阶段的子目录
        prepared_data_dir = os.path.join(stock_root_dir, "prepared_data")
        os.makedirs(prepared_data_dir, exist_ok=True)
        logger.debug(f"确保股票数据准备目录存在: {prepared_data_dir}")

        # 模型训练阶段的子目录
        trained_model_dir = os.path.join(stock_root_dir, "trained_model")
        os.makedirs(trained_model_dir, exist_ok=True)
        logger.debug(f"确保股票模型训练目录存在: {trained_model_dir}")


        # 模型路径 (存放在 trained_model 目录下, 使用 .pth 扩展名)
        self.model_path = os.path.join(trained_model_dir, "trend_following_transformer_weights.pth") # 明确是权重文件

        # Scaler 和数据路径 (存放在 prepared_data 目录下)
        self.feature_scaler_path = os.path.join(prepared_data_dir, "trend_following_transformer_feature_scaler.save")
        self.target_scaler_path = os.path.join(prepared_data_dir, "trend_following_transformer_target_scaler.save")
        self.selected_features_path = os.path.join(prepared_data_dir, "trend_following_transformer_selected_features.json")

        # 准备好的数据的路径 (使用 .npz 格式保存 NumPy 数组)
        # 这个文件包含了训练、验证、测试集的 NumPy 数组
        self.all_prepared_data_npz_path = os.path.join(prepared_data_dir, "all_prepared_data_transformer.npz") # 区分文件

        logger.debug(f"设置股票 {stock_code} 的文件路径:")
        logger.debug(f"  模型权重: {self.model_path}")
        logger.debug(f"  特征Scaler: {self.feature_scaler_path}")
        logger.debug(f"  目标Scaler: {self.target_scaler_path}")
        logger.debug(f"  准备数据NPZ: {self.all_prepared_data_npz_path}")
        logger.debug(f"  选中特征: {self.selected_features_path}")


    # 重命名并修改为训练 Transformer 模型
    def train_transformer_model_from_prepared_data(self, stock_code: str):
        """
        为特定股票加载已准备好的数据，构建并训练 Transformer 模型，然后保存模型权重。
        这个方法假设数据已经通过 save_prepared_data 任务准备好并保存。
        """
        self.set_model_paths(stock_code) # 设置股票特定的路径

        logger.info(f"开始为股票 {stock_code} 训练 Transformer 模型 (从已准备数据加载)...")

        # --- 尝试加载已准备好的数据和 Scaler ---
        # load_prepared_data 现在返回 NumPy 数组和 scaler 对象
        features_scaled_train_np, targets_scaled_train_np, \
        features_scaled_val_np, targets_scaled_val_np, \
        features_scaled_test_np, targets_scaled_test_np, \
        feature_scaler, target_scaler = self.load_prepared_data(stock_code) # 此方法内部会加载 selected_feature_names_for_transformer

        # 检查加载的数据是否有效
        if features_scaled_train_np.shape[0] == 0 or targets_scaled_train_np.shape[0] == 0 or feature_scaler is None or target_scaler is None or not self.selected_feature_names_for_transformer:
             logger.error(f"股票 {stock_code} 加载已准备好的数据或 Scaler 或选中特征列表失败或数据无效，无法继续训练。请先运行数据准备任务。")
             self.transformer_model = None
             self.feature_scaler = None
             self.target_scaler = None
             self.selected_feature_names_for_transformer = []
             return # 停止训练

        # 将加载的 scaler 赋值给实例属性 (以便后续预测使用)
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        # 动态获取实际使用的特征数量 (在缩放后)
        num_features = features_scaled_train_np.shape[1]
        logger.info(f"[{stock_code}] 最终用于训练的平坦数据集 shape: train_features={features_scaled_train_np.shape}, train_targets={targets_scaled_train_np.shape}, "
                    f"val_features={features_scaled_val_np.shape}, val_targets={targets_scaled_val_np.shape}, "
                    f"test_features={features_scaled_test_np.shape}, test_targets={targets_scaled_test_np.shape}")
        logger.info(f"[{stock_code}] 实际用于训练的特征维度: {num_features}")


        # --- 创建 PyTorch Dataset 和 DataLoader ---
        try:
            # 训练集 Dataset 和 DataLoader
            train_dataset = TimeSeriesDataset(
                features=features_scaled_train_np,
                targets=targets_scaled_train_np,
                window_size=self.transformer_window_size # 使用正确的窗口大小参数名
            )
            # 验证集 Dataset 和 DataLoader (如果存在)
            val_dataset = None
            val_loader = None
            if features_scaled_val_np.shape[0] > 0 and targets_scaled_val_np.shape[0] > 0:
                 val_dataset = TimeSeriesDataset(
                     features=features_scaled_val_np,
                     targets=targets_scaled_val_np,
                     window_size=self.transformer_window_size # 使用正确的窗口大小参数名
                 )
                 # 注意：验证集也需要足够长度才能构建窗口
                 if len(val_dataset) > 0:
                      val_loader = DataLoader(val_dataset, batch_size=self.transformer_batch_size, shuffle=False) # 验证集不打乱
                 else:
                      logger.warning(f"股票 {stock_code} 验证集 Dataset 为空。请检查数据量和窗口大小 ({self.transformer_window_size})。验证阶段将跳过。")

            # 测试集 Dataset (评估可选)
            test_dataset = None
            test_loader = None
            if features_scaled_test_np.shape[0] > 0 and targets_scaled_test_np.shape[0] > 0:
                 test_dataset = TimeSeriesDataset(
                     features=features_scaled_test_np,
                     targets=targets_scaled_test_np,
                     window_size=self.transformer_window_size # 使用正确的窗口大小参数名
                 )
                 if len(test_dataset) > 0:
                     test_loader = DataLoader(test_dataset, batch_size=self.transformer_batch_size, shuffle=False) # 测试集不打乱
                 else:
                      logger.warning(f"股票 {stock_code} 测试集 Dataset 为空。请检查数据量和窗口大小 ({self.transformer_window_size})。测试评估将跳过。")


            # 检查训练集 Dataset 是否有效
            if len(train_dataset) == 0:
                 logger.error(f"股票 {stock_code} 训练集 Dataset 为空。请检查数据量和窗口大小 ({self.transformer_window_size})。")
                 return # 停止训练

            # 创建 DataLoader
            train_loader = DataLoader(train_dataset, batch_size=self.transformer_batch_size, shuffle=True) # 训练集打乱


        except Exception as e:
            logger.error(f"股票 {stock_code} 创建 PyTorch Dataset/DataLoader 出错: {e}", exc_info=True)
            return # 停止训练


        # --- 2. 构建模型 ---
        try:
            # 使用实际的特征数量构建模型
            model = build_transformer_model(
                num_features=num_features, # <--- 使用实际的特征数量
                model_config=self.transformer_model_config,
                summary=True,
                window_size=self.transformer_window_size # 传递窗口大小用于位置编码
            )
            self.transformer_model = model # 暂存模型对象
        except Exception as e:
            logger.error(f"股票 {stock_code} 构建 Transformer 模型出错: {e}", exc_info=True)
            self.transformer_model = None
            return # 停止训练


        # --- 3. 训练模型 (使用 DataLoader) ---
        try:
            # train_transformer_model 函数会处理设备移动和模型权重保存
            self.transformer_model, history_df = train_transformer_model(
                model=self.transformer_model,
                train_loader=train_loader, # 传递训练集 DataLoader
                val_loader=val_loader,     # 传递验证集 DataLoader (可能为 None)
                target_scaler=self.target_scaler, # 传入目标变量 scaler
                training_config=self.transformer_training_config, # 传递训练配置
                checkpoint_dir=os.path.dirname(self.model_path), # 保存到股票特定的模型目录
                stock_code=stock_code, # 传递股票代码用于文件名和 TensorBoard
                plot_training_history=self.tf_params.get('transformer_plot_history', False), # 参数名改为 transformer_plot_history
            )
            logger.info(f"股票 {stock_code} Transformer 模型训练完成，最佳模型权重已保存到 {self.model_path}")

            # 可选：在测试集上评估
            if test_loader is not None and len(test_loader) > 0 and self.transformer_model is not None:
                 logger.info(f"开始在测试集上评估股票 {stock_code} 的 Transformer 模型...")
                 test_metrics = evaluate_transformer_model(
                      model=self.transformer_model,
                      test_loader=test_loader,
                      criterion=nn.MSELoss(), # 使用 MSE 或训练时的损失函数
                      mae_metric=nn.L1Loss(), # 使用 MAE
                      target_scaler=self.target_scaler,
                      device=self.device
                 )
                 logger.info(f"股票 {stock_code} 测试集评估结果: {test_metrics}")


        except Exception as e:
            logger.error(f"股票 {stock_code} 训练 Transformer 模型出错: {e}", exc_info=True)
            self.transformer_model = None # 训练失败，模型对象置空

    def get_required_columns(self) -> List[str]:
        """
        根据策略参数，动态生成并返回 IndicatorService 需要准备的所有数据列名。
        这些列名将作为 `prepare_data_for_transformer` 函数的 `required_columns` 参数。
        此方法仅依赖于 self.params 中的配置信息来推断所需的列名。
        假设 IndicatorService 根据参数和预设规则生成列名。
        """
        required = set() # 使用集合避免重复
        bs_params = self.params.get('base_scoring', {})
        timeframes = bs_params.get('timeframes', [])
        if not timeframes:
            logger.error("无法获取所需列，因为 'base_scoring.timeframes' 未定义。")
            return []

        # 1. 基础OHLCV数据 (所有时间级别都需要)
        for tf_str in timeframes:
            for col_base in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                required.add(f"{col_base}_{tf_str}")

        # 2. 基础评分指标 (base_scoring.score_indicators)
        score_indicators_config = bs_params.get('score_indicators', [])
        # 缓存参数，避免重复查找
        macd_fast = bs_params.get('macd_fast', 12)
        macd_slow = bs_params.get('macd_slow', 26)
        macd_sig = bs_params.get('macd_signal', 9)
        rsi_period = bs_params.get('rsi_period', 14)
        kdj_k = bs_params.get('kdj_period_k', 9)
        kdj_d = bs_params.get('kdj_period_d', 3)
        kdj_j = bs_params.get('kdj_period_j', 3)
        boll_period = bs_params.get('boll_period', 20)
        boll_std_dev = bs_params.get('boll_std_dev', 2.0)
        boll_std_str = f"{boll_std_dev:.1f}" # 格式化浮点数参数
        cci_period = bs_params.get('cci_period', 14)
        mfi_period = bs_params.get('mfi_period', 14)
        roc_period = bs_params.get('roc_period', 12)
        dmi_period = bs_params.get('dmi_period', 14)
        sar_step = bs_params.get('sar_step', 0.02)
        sar_max_af = bs_params.get('sar_max', 0.2)
        ema_p = bs_params.get('ema_params',{}).get('period', 20)
        sma_p = bs_params.get('sma_params',{}).get('period', 20)


        for indi_key in score_indicators_config:
            for tf_str in timeframes:
                if indi_key == 'macd':
                    # 修正 MACD 系列的列名，确保使用正确的参数和分隔符
                    required.add(f'MACD_{macd_fast}_{macd_slow}_{macd_sig}_{tf_str}')
                    required.add(f'MACDh_{macd_fast}_{macd_slow}_{macd_sig}_{tf_str}') # 修正了 tf_sig 的笔误
                    required.add(f'MACDs_{macd_fast}_{macd_slow}_{macd_sig}_{tf_str}')
                elif indi_key == 'rsi':
                    required.add(f'RSI_{rsi_period}_{tf_str}')
                elif indi_key == 'kdj':
                    required.add(f'K_{kdj_k}_{kdj_d}_{kdj_j}_{tf_str}')
                    required.add(f'D_{kdj_k}_{kdj_d}_{kdj_j}_{tf_str}')
                    required.add(f'J_{kdj_k}_{kdj_d}_{kdj_j}_{tf_str}')
                elif indi_key == 'boll':
                    required.add(f'BBL_{boll_period}_{boll_std_str}_{tf_str}')
                    required.add(f'BBM_{boll_period}_{boll_std_str}_{tf_str}')
                    required.add(f'BBU_{boll_period}_{boll_std_str}_{tf_str}')
                    required.add(f'BBW_{boll_period}_{boll_std_str}_{tf_str}')
                    required.add(f'BBP_{boll_period}_{boll_std_str}_{tf_str}')
                elif indi_key == 'cci':
                    required.add(f'CCI_{cci_period}_{tf_str}')
                elif indi_key == 'mfi':
                    required.add(f'MFI_{mfi_period}_{tf_str}')
                elif indi_key == 'roc':
                    required.add(f'ROC_{roc_period}_{tf_str}')
                elif indi_key == 'dmi':
                    # 修正 DMI 列名以匹配 pandas_ta 默认输出
                    required.add(f'PLUS_DI_{dmi_period}_{tf_str}')
                    required.add(f'MINUS_DI_{dmi_period}_{tf_str}')
                    required.add(f'ADX_{dmi_period}_{tf_str}')
                elif indi_key == 'sar':
                    # 格式化 SAR 参数以匹配列名，通常是 step_max_af
                    sar_step_str = f"{sar_step:.2f}".rstrip('0').rstrip('.') if isinstance(sar_step, float) else str(sar_step)
                    sar_max_af_str = f"{sar_max_af:.2f}".rstrip('0').rstrip('.') if isinstance(sar_max_af, float) else str(sar_max_af)
                    required.add(f'SAR_{sar_step_str}_{sar_max_af_str}_{tf_str}')
                elif indi_key == 'ema':
                    required.add(f'EMA_{ema_p}_{tf_str}')
                elif indi_key == 'sma':
                    required.add(f'SMA_{sma_p}_{tf_str}')

        # 3. 量能确认指标 (volume_confirmation)
        vc_params = self.params.get('volume_confirmation', {})
        if vc_params.get('enabled', False) or vc_params.get('volume_analysis_enabled', False):
            vc_tf_list = vc_params.get('tf', [self.focus_timeframe]) # 确保是列表
            if not isinstance(vc_tf_list, list): vc_tf_list = [vc_tf_list] # 再次检查
            amount_ma_period = vc_params.get('amount_ma_period',10)
            cmf_period = vc_params.get('cmf_period',20)

            for vc_tf_str in vc_tf_list:
                if vc_tf_str not in timeframes:
                    logger.warning(f"量能确认时间框架 '{vc_tf_str}' 未在 'base_scoring.timeframes' 中定义，可能无法获取其数据。")
                    continue
                required.add(f"AMT_MA_{amount_ma_period}_{vc_tf_str}")
                required.add(f"CMF_{cmf_period}_{vc_tf_str}")
                # OBV (无参数) 会在后面统一添加
        # 4. 其他分析指标 (indicator_analysis_params)
        ia_params = self.params.get('indicator_analysis_params', {})
        ia_timeframes = bs_params.get('timeframes', [])

        # 缓存 IA 参数
        stoch_k = ia_params.get('stoch_k', 9)
        stoch_d = ia_params.get('stoch_d', 3)
        stoch_smooth_k = ia_params.get('stoch_smooth_k', 3)
        vol_ma_period = ia_params.get('volume_ma_period', 20)
        vwap_anchor = ia_params.get('vwap_anchor', None)
        calculate_adl = ia_params.get('calculate_adl', True)
        ichimoku_enabled = ia_params.get('calculate_ichimoku', True)
        ichimoku_tenkan = ia_params.get('ichimoku_tenkan', 9)
        ichimoku_kijun = ia_params.get('ichimoku_kijun', 26)
        ichimoku_senkou = ia_params.get('ichimoku_senkou', 52)
        calculate_pivot_points = ia_params.get('calculate_pivot_points', True)

        for tf_str_ia in ia_timeframes:
            # STOCH
            required.add(f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}_{tf_str_ia}")
            required.add(f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}_{tf_str_ia}")
            # VOL_MA
            required.add(f"VOL_MA_{vol_ma_period}_{tf_str_ia}")
            # VWAP
            vwap_col_name = 'VWAP' if vwap_anchor is None else f'VWAP_{vwap_anchor}'
            required.add(f"{vwap_col_name}_{tf_str_ia}")
            # ADL
            if calculate_adl:
                required.add(f"ADL_{tf_str_ia}")
            # Ichimoku
            if ichimoku_enabled:
                required.add(f"TENKAN_{ichimoku_tenkan}_{tf_str_ia}")
                required.add(f"KIJUN_{ichimoku_kijun}_{tf_str_ia}")
                required.add(f"CHIKOU_{ichimoku_kijun}_{tf_str_ia}")
                required.add(f"SENKOU_A_{ichimoku_tenkan}_{ichimoku_kijun}_{tf_str_ia}")
                required.add(f"SENKOU_B_{ichimoku_senkou}_{tf_str_ia}")
            # Pivot Points (仅日线)
            if calculate_pivot_points and tf_str_ia == 'D':
                required.add("PP_D")
                for i in range(1, 5): required.add(f"S{i}_D"); required.add(f"R{i}_D")
                for i in range(1, 4): required.add(f"F_S{i}_D"); required.add(f"F_R{i}_D") # Floor pivots

        # 5. 特征工程产生的指标 (feature_engineering_params)
        fe_params = self.params.get('feature_engineering_params', {})
        fe_tf_list = fe_params.get('apply_on_timeframes', timeframes)
        # 缓存 FE 参数
        atr_period = fe_params.get('atr_params',{}).get('period',14)
        hv_period = fe_params.get('hv_params',{}).get('period',20)
        kc_ema_period = fe_params.get('kc_params',{}).get('ema_period',20)
        kc_atr_period = fe_params.get('kc_params',{}).get('atr_period',10)
        mom_period = fe_params.get('mom_params',{}).get('period',10)
        willr_period = fe_params.get('willr_params',{}).get('period',14)
        vroc_period = fe_params.get('vroc_params',{}).get('period',10)
        aroc_period = fe_params.get('aroc_params',{}).get('period',10)
        ema_periods_fe = fe_params.get('ema_periods', [])
        sma_periods_fe = fe_params.get('sma_periods', [])
        ema_periods_rel = fe_params.get('ema_periods_for_relation', [])
        sma_periods_rel = fe_params.get('sma_periods_for_relation', [])
        indicators_to_diff_fe = fe_params.get('indicators_for_difference', [])
        diff_periods_fe = fe_params.get('difference_periods', [1])


        for tf_str_fe in fe_tf_list:
            if tf_str_fe not in timeframes: continue
            if fe_params.get('calculate_atr', True):
                required.add(f"ATR_{atr_period}_{tf_str_fe}")
            if fe_params.get('calculate_hv', True):
                required.add(f"HV_{hv_period}_{tf_str_fe}")
            if fe_params.get('calculate_kc', True):
                required.add(f"KCL_{kc_ema_period}_{kc_atr_period}_{tf_str_fe}")
                required.add(f"KCM_{kc_ema_period}_{kc_atr_period}_{tf_str_fe}")
                required.add(f"KCU_{kc_ema_period}_{kc_atr_period}_{tf_str_fe}")
            if fe_params.get('calculate_mom', True):
                required.add(f"MOM_{mom_period}_{tf_str_fe}")
            if fe_params.get('calculate_willr', True):
                required.add(f"WILLR_{willr_period}_{tf_str_fe}")
            if fe_params.get('calculate_vroc', True):
                required.add(f"VROC_{vroc_period}_{tf_str_fe}")
            if fe_params.get('calculate_aroc', True):
                required.add(f"AROC_{aroc_period}_{tf_str_fe}")

            # EMA 和 SMA 特征 (EMA/SMA 本身可能也是特征)
            for p_fe in ema_periods_fe:
                required.add(f"EMA_{p_fe}_{tf_str_fe}")
            for p_fe in sma_periods_fe:
                 required.add(f"SMA_{p_fe}_{tf_str_fe}")

            # 衍生特征 (价格与均线关系)
            for ma_type_deriv_fe, periods_list in [('EMA', ema_periods_rel), ('SMA', sma_periods_rel)]:
                 for p_deriv_fe in periods_list:
                     required.add(f'CLOSE_{ma_type_deriv_fe}_RATIO_{p_deriv_fe}_{tf_str_fe}')
                     required.add(f'CLOSE_{ma_type_deriv_fe}_NDIFF_{p_deriv_fe}_{tf_str_fe}')

            # 指标差分
            # 这里需要更通用地构建原始指标的列名，然后添加差分后缀
            # 遍历 feature_engineering.indicators_for_difference 配置
            for indi_diff_conf_fe in indicators_to_diff_fe:
                base_name_fe = indi_diff_conf_fe['base_name'].upper() # 基础指标名称（大写）
                param_keys_fe = indi_diff_conf_fe.get('params_key', []) # 参数键列表
                default_p_values_fe = indi_diff_conf_fe.get('default_period', []) # 默认参数值列表

                param_values_for_col_fe = []
                # 尝试从策略参数中查找每个参数键对应的值
                param_sources = {**bs_params, **ia_params, **fe_params} # 合并所有可能的参数来源
                try:
                    if isinstance(param_keys_fe, list):
                        for i_fe, pk_fe in enumerate(param_keys_fe):
                            # 查找参数值，如果找不到则使用默认值列表中的对应值
                            val_fe = param_sources.get(pk_fe, default_p_values_fe[i_fe] if i_fe < len(default_p_values_fe) else None)
                            if val_fe is not None:
                                # 格式化参数值
                                if isinstance(val_fe, float):
                                     param_values_for_col_fe.append(f"{val_fe:.2f}".rstrip('0').rstrip('.'))
                                else:
                                     param_values_for_col_fe.append(str(val_fe))
                            else:
                                 logger.warning(f"指标差分 '{base_name_fe}' 参数 '{pk_fe}' 未在策略参数中找到对应值，也无默认值。跳过此指标的差分列。")
                                 param_values_for_col_fe = [] # 清空，表示无法构建正确的列名
                                 break # 跳出参数查找循环
                    else: # 如果 param_keys_fe 不是列表 (可能只有一个参数键)
                         pk_fe = param_keys_fe
                         val_fe = param_sources.get(pk_fe, default_p_values_fe[0] if isinstance(default_p_values_fe, list) and default_p_values_fe else (default_p_values_fe if not isinstance(default_p_values_fe, list) else None) )
                         if val_fe is not None:
                              if isinstance(val_fe, float):
                                   param_values_for_col_fe.append(f"{val_fe:.2f}".rstrip('0').rstrip('.'))
                              else:
                                   param_values_for_col_fe.append(str(val_fe))
                         else:
                             logger.warning(f"指标差分 '{base_name_fe}' 参数 '{pk_fe}' 未在策略参数中找到对应值，也无默认值。跳过此指标的差分列。")
                             param_values_for_col_fe = []


                except Exception as e_param_lookup:
                    logger.warning(f"构建指标差分 '{base_name_fe}' 的参数时出错: {e_param_lookup}. 跳过此指标的差分列。", exc_info=True)
                    param_values_for_col_fe = []


                if param_values_for_col_fe: # 如果成功构建了参数部分
                    param_str_for_col_fe = "_".join(param_values_for_col_fe)
                    # 构建基础列名（不含时间框架和差分后缀）
                    # 注意：有些指标的列名可能与 base_name 不同 (如 KDJ -> K, D, J)。
                    # 这里假设 base_name_fe 是指标名称，IndicatorService 会根据参数生成完整的原始列名
                    # 例如 RSI_14, MACD_12_26_9 等
                    # 最安全的方式是 IndicatorService 提供一个方法，根据指标 key 和 params 返回其所有生成的列名
                    # 但在不知道 IndicatorService 具体实现的情况下，这里只能假设一个命名规则。
                    # 假设规则是: BASE_NAME_PARAM1_PARAM2...
                    base_indi_col_prefix = base_name_fe # 例如 'RSI', 'MACD', 'ADX'
                    # 找到所有以 base_indi_col_prefix_PARAM1_PARAM2..._tf_str_fe 开头的列名，并为其添加差分后缀
                    # 这种方式太不可靠，因为无法确定原始列名是什么。
                    # 比如 KDJ 会生成 K, D, J 三列，参数是共享的。
                    # MACD 会生成 MACD, MACDh, MACDs 三列。
                    # 正确的做法是 strategy_utils.py 的 calculate_all_indicator_scores 方法返回的 DataFrame
                    # 包含了所有计算好的原始指标和衍生指标列。这个 DataFrame 的列名是 IndicatorService 实际生成的。
                    # prepare_data_for_transformer 函数接收这个 DataFrame 并从中选择列。
                    # get_required_columns 的作用只是告诉 IndicatorService 需要哪些“基础”计算。
                    # 那么，get_required_columns 是否需要列出所有可能的衍生特征列名？
                    # 如果 IndicatorService 足够智能，它计算了基础指标后，会根据 strategy_params 中 feature_engineering_params
                    # 的配置自动计算并添加衍生特征列。get_required_columns 只需请求基础指标。
                    # 如果 IndicatorService 不智能，get_required_columns 就需要列出所有需要的列。
                    # 鉴于 prepare_data_for_transformer 函数接收的是完整的 DataFrame (由 IndicatorService 提供)
                    # 并在内部进行特征选择，那么 get_required_columns 只需确保 IndicatorService
                    # 计算了所有可能的、策略可能用到的基础指标和衍生特征。
                    # 目前的 strategy_utils.py -> calculate_all_indicator_scores 确实只返回了 SCORE 列。
                    # 而 IndicatorService 的 calculate_indicators 方法返回的是包含所有计算列的 DataFrame。
                    # prepare_data_for_transformer 接收的是 IndicatorService 的完整输出。
                    # 因此，get_required_columns 应该列出 prepare_data_for_transformer 可能需要的 *所有* 原始和衍生特征列。
                    # 这样 IndicatorService 才知道需要计算哪些列。
                    # 之前的代码逻辑是试图在这里构建所有可能的列名。这仍然是正确的方向，但需要精确匹配 IndicatorService 的命名。
                    # 重新审视这段代码：它在尝试构建原始指标的列名，然后添加差分后缀。这看起来是对的。
                    # 问题在于如何知道原始指标列名？例如，对于 KDJ，它可能生成 K, D, J 列。如果我们想对 K 线差分，
                    # 我们需要请求 'K_9_3_3_DIFF1_30'.
                    # 让我们继续按照这个思路修正列名构建。需要针对不同的 base_name_fe 硬编码或查找其可能的原始列名。
                    # 这段逻辑确实复杂，而且耦合了 IndicatorService 的内部命名。
                    # simplest approach: just list the base indicators and hope IndicatorService does the rest, or
                    # assume a fixed naming convention and list everything based on params.
                    # Let's assume a fixed naming convention based on the parameters used for the base indicator.

                    base_indi_col_name_parts = [base_name_fe] + param_values_for_col_fe
                    base_indi_col_prefix_for_diff = "_".join(base_indi_col_name_parts)

                    # For indicators that produce multiple columns (like KDJ, MACD, BOLL), we need to list diffs for each relevant output column.
                    # This requires knowing which columns a base indicator produces. This information is implicitly in IndicatorService and indicator_parameters.json config.
                    # Let's simplify for now and assume the diff is applied to the primary column, or we need to iterate through possible output columns if known.
                    # Example: For MACD diff, apply to MACD, MACDh, MACDs? For KDJ, apply to K, D, J?
                    # The `indicators_for_difference` config only lists `base_name`. It should perhaps list `output_columns` too.
                    # Assuming for simplicity it applies to the main output, or all outputs if not specified.
                    # Let's assume it applies to the main output identified by `base_name_fe`.

                    # Let's refine based on common outputs:
                    output_cols_to_diff = []
                    if base_name_fe == 'KDJ': output_cols_to_diff = ['K', 'D', 'J']
                    elif base_name_fe == 'MACD': output_cols_to_diff = ['MACD', 'MACDh', 'MACDs'] # Note: MACD naming already includes params, so prefix is enough
                    elif base_name_fe == 'BOLL': output_cols_to_diff = ['BBP', 'BBW'] # %B and Bandwidth are common to diff
                    elif base_name_fe == 'RSI': output_cols_to_diff = ['RSI']
                    # Add other specific cases if needed.
                    # If not in the specific list, assume the base_name itself is the column prefix.
                    if not output_cols_to_diff:
                         output_cols_to_diff = [base_name_fe] # Default to the base name

                    # Now construct diff column names for each output column
                    for output_col_prefix in output_cols_to_diff:
                         # Need to rebuild the full original column name structure for the base indicator first
                         original_base_col_name_parts = [output_col_prefix] + param_values_for_col_fe + [tf_str_fe]
                         original_base_col_name = "_".join(original_base_col_name_parts)

                         # Check if this constructed base column name is plausible (optional but good practice)
                         # We don't have the full list of plausible names here easily.
                         # Let's just assume the convention is strict.

                         for diff_p_fe in diff_periods_fe:
                             # Final diff column name format: ORIGINAL_BASE_COL_NAME_DIFF_PERIOD
                             # Example: K_9_3_3_30_DIFF1
                             # No, the diff period usually comes before the timeframe.
                             # Let's assume the format is: ORIGINAL_BASE_NAME_PARAMS_DIFF_PERIOD_TF
                             # Example: K_9_3_3_DIFF1_30
                             # This matches the previous code's intent, but the base_indi_col_name_parts logic needs refinement.

                             # Let's try again: Original base column name parts are like [ 'K', '9', '3', '3' ] for KDJ
                             # Or ['MACD', '12', '26', '9'] for MACD
                             # The previous logic built base_indi_col_prefix_for_diff = "KDJ_9_3_3". This seems wrong.
                             # It should build the *prefix* that matches the *start* of the original column names.
                             # For KDJ(9,3,3), original columns are K_9_3_3, D_9_3_3, J_9_3_3.
                             # For MACD(12,26,9), original columns are MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9.
                             # The parameter string part (e.g., '9_3_3' or '12_26_9') is consistent for all outputs of one indicator calculation.

                             param_suffix_fe = "_".join(param_values_for_col_fe) if param_values_for_col_fe else ""
                             if param_suffix_fe:
                                 param_suffix_fe = "_" + param_suffix_fe

                             # Now, for each potential output column prefix (K, D, J, MACD, MACDh, MACDs, RSI, etc.)
                             # Construct the expected original column name prefix + params + DIFF + period
                             for output_col_prefix in output_cols_to_diff:
                                 # Format: OUTPUT_PREFIX_PARAMS_DIFF_PERIOD_TF
                                 diff_col_name = f'{output_col_prefix}{param_suffix_fe}_DIFF{diff_p_fe}_{tf_str_fe}'
                                 required.add(diff_col_name)

                    # If the base_name_fe doesn't have explicit output_cols_to_diff defined above,
                    # assume the base_name_fe itself is the column prefix.
                    if not output_cols_to_diff and base_name_fe not in ['KDJ', 'MACD', 'BOLL']: # Add other multi-output indicators here
                         param_suffix_fe = "_".join(param_values_for_col_fe) if param_values_for_col_fe else ""
                         if param_suffix_fe:
                             param_suffix_fe = "_" + param_suffix_fe
                         for diff_p_fe in diff_periods_fe:
                             diff_col_name = f'{base_name_fe}{param_suffix_fe}_DIFF{diff_p_fe}_{tf_str_fe}'
                             required.add(diff_col_name)


            # 价格在通道中位置
            # 列名格式: CLOSE_CHANNEL_POS_PARAM1_PARAM2..._TF
            required.add(f'CLOSE_BB_POS_{boll_period}_{boll_std_str}_{tf_str_fe}')
            required.add(f'CLOSE_KC_POS_{kc_ema_period}_{kc_atr_period}_{tf_str_fe}')

        # 6. K线形态检测 (如果启用)
        kp_params = self.params.get('kline_pattern_detection', {})
        if kp_params.get('enabled', False):
             kp_tf_list = kp_params.get('timeframes', timeframes)
             # K线形态的输出列名由 pandas_ta 的 cdl_xxxx 函数生成，格式为 CDL_PATTERNNAME_TF
             # 策略只需要知道它可能需要这些列作为特征输入。
             # 这里无法枚举所有形态，所以无法精确列出。
             # 一个务实的方法是，如果启用了形态检测，就假设所有可能的形态列都需要。
             # 或者，如果参数文件中有配置需要特定形态，就列出。
             # 假设 IndicatorService 会生成所有形态，这里不列出具体的形态列名，依赖 IndicatorService 的完整输出
             pass # 依赖后续特征工程从 IndicatorService 的完整输出中选择

        # 确保 OBV (无参数) 和 ADL (无参数) 被包含 (通常在所有时间级别计算)
        for tf_str_basic_vol in timeframes:
            required.add(f"OBV_{tf_str_basic_vol}")
            # ADL 已在 IA 部分处理过，如果计算 ADL 启用，则会添加 ADL_TF 列

        final_columns = sorted(list(required))
        logger.info(f"策略 '{self.strategy_name}' 共需要 {len(final_columns)} 个数据列。")
        logger.debug(f"所需列名 (部分): {final_columns[:20]}...")
        return final_columns

    def _calculate_rule_based_signal(self, data: pd.DataFrame, stock_code: str, indicator_configs: List[Dict]) -> Tuple[pd.Series, Dict]:
        """
        计算基于规则的信号，并返回中间结果。
        Args:
            data (pd.DataFrame): 输入数据 (包含 IndicatorService 计算的所有列)。
            stock_code (str): 股票代码。
            indicator_configs (List[Dict]): IndicatorService 生成的指标配置列表。
        Returns:
            Tuple[pd.Series, Dict]: 最终规则信号 Series 和中间结果字典。
        """
        # 这个方法的逻辑基本不变，它计算的是基于规则的信号，作为 Transformer 的训练目标
        # 只需要确保引用的列名与 IndicatorService 实际生成的列名一致 (例如 DMI 的列名修正)
        if data is None or data.empty:
            logger.warning("输入数据为空，无法生成规则信号。")
            return pd.Series(dtype=float), {}

        # --- 检查必需列 ---
        bs_params = self.params.get('base_scoring', {})
        vc_params = self.params.get('volume_confirmation', {})
        ia_params = self.params.get('indicator_analysis_params', {})
        dd_params = self.params.get('divergence_detection', {})

        focus_tf = self.focus_timeframe
        # 检查焦点时间框架的关键列
        critical_cols = [f'close_{focus_tf}', f'volume_{focus_tf}']
        if bs_params.get('score_indicators'):
             first_score_indi_key = bs_params['score_indicators'][0]
             # 需要更精确地查找该指标在焦点时间框架下的列名，基于 indicator_configs
             potential_score_col_prefix = first_score_indi_key.upper()
             potential_score_col = next((col for col in data.columns if col.startswith(potential_score_col_prefix) and col.endswith(f'_{focus_tf}')), None)
             if potential_score_col:
                 critical_cols.append(potential_score_col)
             else:
                 logger.warning(f"无法找到评分指标 '{first_score_indi_key}' 在焦点时间框架 '{focus_tf}' 的列。")

        missing_critical_cols = [col for col in critical_cols if col not in data.columns or data[col].isnull().all()]
        if missing_critical_cols:
            logger.error(f"[{stock_code}] 规则信号计算缺少关键输入列或数据全为 NaN: {missing_critical_cols}。")
            return pd.Series(50.0, index=data.index), {}

        # --- 动态调整参数 ---
        self._adjust_volatility_parameters(data)

        # --- 计算所有配置的指标评分 ---
        indicator_scores_df = strategy_utils.calculate_all_indicator_scores(data, bs_params, indicator_configs)

        # --- 计算多时间框架加权的基础评分 (base_score_raw) ---
        current_weights: Dict[str, float]
        timeframes_from_config = bs_params.get('timeframes', [])
        if self.timeframe_weights:
            current_weights = self.timeframe_weights.copy()
            defined_tfs_set = set(timeframes_from_config)
            for tf_w in list(current_weights.keys()):
                if tf_w not in defined_tfs_set:
                    del current_weights[tf_w]
            for tf_d in defined_tfs_set:
                if tf_d not in current_weights:
                    current_weights[tf_d] = 0.0
        else:
            focus_weight_val = self.tf_params.get('focus_weight', 0.45)
            num_other_tfs = len(timeframes_from_config) - 1
            if num_other_tfs > 0:
                base_weight_val = (1.0 - focus_weight_val) / num_other_tfs
            elif len(timeframes_from_config) == 1:
                base_weight_val = 0.0
                focus_weight_val = 1.0
            else:
                base_weight_val = 0.0
                focus_weight_val = 0.0
            current_weights = {tf: base_weight_val for tf in timeframes_from_config if tf != self.focus_timeframe}
            current_weights[self.focus_timeframe] = focus_weight_val

        self._normalize_weights(current_weights)

        base_score_raw = pd.Series(0.0, index=data.index)
        total_effective_weight = 0.0

        for tf_s in timeframes_from_config:
            tf_weight = current_weights.get(tf_s, 0)
            if tf_weight == 0:
                continue

            tf_score_cols = [col for col in indicator_scores_df.columns if col.endswith(f'_{tf_s}') and col.startswith('SCORE_')]
            if tf_score_cols:
                tf_average_score = indicator_scores_df[tf_score_cols].mean(axis=1).fillna(50.0)
                base_score_raw = base_score_raw.add(tf_average_score * tf_weight, fill_value=0.0)
                total_effective_weight += tf_weight
            else:
                 logger.debug(f"时间框架 '{tf_s}' 没有找到任何指标评分列。")
                 if tf_weight > 0:
                      base_score_raw = base_score_raw.add(pd.Series(50.0, index=data.index) * tf_weight, fill_value=0.0)
                      total_effective_weight += tf_weight


        if total_effective_weight > 0 and not np.isclose(total_effective_weight, 1.0):
             base_score_raw = base_score_raw / total_effective_weight * sum(current_weights.values())

        base_score_raw = base_score_raw.clip(0, 100).fillna(50.0)

        # --- 应用量能调整 ---
        vc_params_adjusted = vc_params.copy()
        vc_params_adjusted['boost_factor'] = self.volume_boost_factor
        vc_params_adjusted['penalty_factor'] = self.volume_penalty_factor
        vc_params_adjusted['volume_spike_threshold'] = self.volume_spike_threshold
        volume_adjusted_results_df = strategy_utils.adjust_score_with_volume(
            base_score_raw, data, vc_params_adjusted
        )
        base_score_volume_adjusted = volume_adjusted_results_df['ADJUSTED_SCORE']

        # --- 执行趋势分析 (基于量能调整后的分数) ---
        trend_analysis_df = self._perform_trend_analysis(data, base_score_volume_adjusted)

        # --- 检测背离信号 ---
        divergence_signals_df = pd.DataFrame(index=data.index)
        if dd_params.get('enabled', True):
            try:
                divergence_signals_df = strategy_utils.detect_divergence(data, dd_params, indicator_configs)
                logger.debug(f"[{stock_code}] 背离检测完成，发现信号: {divergence_signals_df.iloc[-1].to_dict() if not divergence_signals_df.empty else '无'}")
            except Exception as e:
                logger.error(f"[{stock_code}] 执行背离检测时出错: {e}", exc_info=True)

        # --- 组合最终规则信号 ---
        final_rule_signal = pd.Series(50.0, index=data.index)

        # 归一化各个贡献项到 -1 到 1 范围
        base_score_norm = (base_score_volume_adjusted.fillna(50.0) - 50) / 50
        alignment_norm = trend_analysis_df.get('alignment_signal', pd.Series(0, index=data.index)).fillna(0) / 3
        long_context_norm = trend_analysis_df.get('long_term_context', pd.Series(0, index=data.index)).fillna(0)
        score_momentum_series = trend_analysis_df.get('score_momentum', pd.Series(0, index=data.index)).fillna(0)
        momentum_norm = np.sign(score_momentum_series)
        ema_cross_norm = trend_analysis_df.get('ema_cross_signal', pd.Series(0, index=data.index)).fillna(0)
        boll_breakout_norm = trend_analysis_df.get('boll_breakout_signal', pd.Series(0, index=data.index)).fillna(0)
        adx_strength_norm = trend_analysis_df.get('adx_strength_signal', pd.Series(0.0, index=data.index)).fillna(0.0)
        vwap_dev_norm = trend_analysis_df.get('vwap_deviation_signal', pd.Series(0, index=data.index)).fillna(0)
        # volume_spike_signal 现在直接从 volume_adjusted_results_df 获取列名
        # 修正列名获取方式，使用 vc_params 中的 tf
        vc_tf_col = vc_params.get('tf', self.focus_timeframe)
        if isinstance(vc_tf_col, list): vc_tf_col = vc_tf_col[0] # 如果是列表，取第一个
        volume_spike_signal_col_name = f'VOL_SPIKE_SIGNAL_{vc_tf_col}'
        volume_spike_signal = volume_adjusted_results_df.get(volume_spike_signal_col_name, pd.Series(0, index=data.index)).fillna(0) # 确保列名正确

        # 计算加权贡献总和
        total_weighted_contribution = pd.Series(0.0, index=data.index)
        weights = self.rule_signal_weights

        total_weighted_contribution += base_score_norm * weights.get('base_score', 0)
        total_weighted_contribution += alignment_norm * weights.get('alignment', 0)
        total_weighted_contribution += long_context_norm * weights.get('long_context', 0)
        total_weighted_contribution += momentum_norm * weights.get('momentum', 0)
        total_weighted_contribution += ema_cross_norm * weights.get('ema_cross', 0)
        total_weighted_contribution += boll_breakout_norm * weights.get('boll_breakout', 0)
        total_weighted_contribution += vwap_dev_norm * weights.get('vwap_deviation', 0)

        # 将总贡献映射回 0-100 范围的基础信号 (在应用 ADX/背离等调整之前)
        base_rule_signal_before_adjust = 50.0 + total_weighted_contribution * 50.0

        # --- 应用 ADX 增强调整 ---
        # 使用 _apply_adx_boost 方法
        final_rule_signal = self._apply_adx_boost(
            base_rule_signal_before_adjust,
            adx_strength_norm,
            (base_rule_signal_before_adjust - 50.0) / 50.0 # 基础信号方向归一化
        )

        # --- 应用背离惩罚 ---
        # 调用 _apply_divergence_penalty
        final_rule_signal = self._apply_divergence_penalty(final_rule_signal, divergence_signals_df, dd_params)

        # --- 应用趋势确认过滤 ---
        final_rule_signal = self._apply_trend_confirmation(final_rule_signal)

        # 确保最终规则信号在 0-100 范围内并四舍五入
        final_rule_signal = final_rule_signal.clip(0, 100).round(2)

        # 返回最终规则信号和中间结果
        intermediate_results = {
            'base_score_raw': base_score_raw,
            'base_score_volume_adjusted': base_score_volume_adjusted,
            # 'indicator_scores': indicator_scores_df, # 评分DataFrame较大，不放入中间结果字典返回，但会添加到 processed_data
            'volume_analysis': volume_adjusted_results_df,
            'trend_analysis': trend_analysis_df,
            'divergence_signals': divergence_signals_df
        }

        # 将所有单项指标评分也添加到中间结果的DataFrame中
        # 避免在 intermediate_results 字典中嵌套大的DataFrame
        # 这些会在 generate_signals 中合并到 processed_data 里

        return final_rule_signal, intermediate_results

    def _perform_trend_analysis(self, data: pd.DataFrame, base_score_series: pd.Series) -> pd.DataFrame:
        """
        增强趋势分析，加入 ADX, STOCH, VWAP, BOLL 等辅助判断。
        """
        # 这个方法的逻辑基本不变，只需要确保引用的列名与 IndicatorService 实际生成的列名一致
        analysis_df = pd.DataFrame(index=base_score_series.index)
        ta_params = self.params.get('trend_analysis', {})
        bs_params = self.params.get('base_scoring', {})
        ia_params = self.params.get('indicator_analysis_params', {})
        focus_tf = self.focus_timeframe

        if base_score_series.isnull().all():
            logger.warning("基础分数全为 NaN，无法执行趋势分析。")
            return analysis_df

        score_series = base_score_series

        # --- 1. 计算分数 EMA ---
        all_ema_periods = ta_params.get('ema_periods', [5, 10, 20, 60])
        for period in all_ema_periods:
            try:
                analysis_df[f'ema_score_{period}'] = ta.ema(score_series.fillna(50.0), length=period)
            except Exception as e:
                logger.error(f"计算 EMA Score {period} 时出错: {e}")
                analysis_df[f'ema_score_{period}'] = np.nan

        # --- 2. 计算 EMA 排列信号 ---
        ema_periods_align = all_ema_periods[:4]
        ema_cols_align = [f'ema_score_{p}' for p in ema_periods_align]
        if len(ema_cols_align) == 4 and all(col in analysis_df.columns for col in ema_cols_align):
            short_ema_col, mid1_ema_col, mid2_ema_col, long_ema_col = ema_cols_align
            temp_df = analysis_df[ema_cols_align].dropna()
            if not temp_df.empty:
                signal_s_m1 = pd.Series(0, index=temp_df.index)
                signal_s_m1 = signal_s_m1.where(temp_df[short_ema_col] <= temp_df[mid1_ema_col], 1)
                signal_s_m1 = signal_s_m1.where(temp_df[short_ema_col] >= temp_df[mid1_ema_col], -1)
                signal_m1_m2 = pd.Series(0, index=temp_df.index)
                signal_m1_m2 = signal_m1_m2.where(temp_df[mid1_ema_col] <= temp_df[mid2_ema_col], 1)
                signal_m1_m2 = signal_m1_m2.where(temp_df[mid1_ema_col] >= temp_df[mid2_ema_col], -1)
                signal_m2_l = pd.Series(0, index=temp_df.index)
                signal_m2_l = signal_m2_l.where(temp_df[mid2_ema_col] <= temp_df[long_ema_col], 1)
                signal_m2_l = signal_m2_l.where(temp_df[mid2_ema_col] >= temp_df[long_ema_col], -1)

                analysis_df['alignment_signal'] = pd.Series(0, index=analysis_df.index)
                analysis_df.loc[temp_df.index, 'alignment_signal'] = signal_s_m1 + signal_m1_m2 + signal_m2_l
            else:
                 logger.warning(f"用于计算 EMA 排列信号的 EMA Score 数据全为 NaN。")
                 analysis_df['alignment_signal'] = 0
        else:
            logger.warning(f"无法计算 EMA 排列信号，所需的 EMA 列不足 ({len(ema_cols_align)} < 4) 或缺失: {ema_cols_align}")
            analysis_df['alignment_signal'] = 0

        # --- 3. 计算 EMA 交叉信号 ---
        if len(all_ema_periods) >= 2:
            short_ema_col = f'ema_score_{all_ema_periods[0]}'
            mid_ema_col = f'ema_score_{all_ema_periods[1]}'
            if short_ema_col in analysis_df.columns and mid_ema_col in analysis_df.columns:
                short_ema = analysis_df[short_ema_col]
                mid_ema = analysis_df[mid_ema_col]
                temp_short_ema = short_ema.fillna(50.0)
                temp_mid_ema = mid_ema.fillna(50.0)

                short_ema_shift = temp_short_ema.shift(1)
                mid_ema_shift = temp_mid_ema.shift(1)

                golden_cross = (temp_short_ema > temp_mid_ema) & (short_ema_shift <= mid_ema_shift)
                death_cross = (temp_short_ema < temp_mid_ema) & (short_ema_shift >= mid_ema_shift)

                analysis_df['ema_cross_signal'] = pd.Series(0, index=analysis_df.index)
                analysis_df.loc[golden_cross[golden_cross].index, 'ema_cross_signal'] = 1
                analysis_df.loc[death_cross[death_cross].index, 'ema_cross_signal'] = -1
            else:
                analysis_df['ema_cross_signal'] = 0
        else:
            analysis_df['ema_cross_signal'] = 0

        # --- 4. 计算 EMA 强度 ---
        if len(all_ema_periods) >= 2:
            short_ema_col = f'ema_score_{all_ema_periods[0]}'
            long_term_ema_period_for_strength = all_ema_periods[-1]
            long_ema_col = f'ema_score_{long_term_ema_period_for_strength}'
            if short_ema_col in analysis_df.columns and long_ema_col in analysis_df.columns:
                analysis_df['ema_strength'] = (analysis_df[short_ema_col].fillna(50.0) - analysis_df[long_ema_col].fillna(50.0)).fillna(0.0)
            else:
                analysis_df['ema_strength'] = np.nan
        else:
            analysis_df['ema_strength'] = np.nan

        # --- 5. 计算得分动量及动量加速 ---
        score_series_filled = score_series.fillna(50.0)
        analysis_df['score_momentum'] = score_series_filled.diff().fillna(0.0)
        analysis_df['score_momentum_acceleration'] = analysis_df['score_momentum'].diff().fillna(0.0)

        # --- 6. 计算得分波动率 ---
        volatility_window = ta_params.get('volatility_window', 10)
        analysis_df['score_volatility'] = score_series_filled.rolling(window=volatility_window, min_periods=max(1, volatility_window//2)).std().fillna(0.0)
        analysis_df['volatility_signal'] = pd.Series(0, index=analysis_df.index)
        analysis_df.loc[analysis_df['score_volatility'] > self.volatility_threshold_high, 'volatility_signal'] = -1
        analysis_df.loc[analysis_df['score_volatility'] < self.volatility_threshold_low, 'volatility_signal'] = 1

        # 7. 长期趋势背景 (基于分数与长期 EMA)
        long_term_ema_period_context = ta_params.get('long_term_ema_period', all_ema_periods[-1] if all_ema_periods else 60)
        long_term_ema_col_context = f'ema_score_{long_term_ema_period_context}'
        if long_term_ema_col_context in analysis_df.columns:
            score_series_filled = score_series.fillna(50.0)
            long_term_ema_filled = analysis_df[long_term_ema_col_context].fillna(50.0)

            analysis_df['long_term_context'] = pd.Series(0, index=analysis_df.index)
            analysis_df.loc[score_series_filled > long_term_ema_filled, 'long_term_context'] = 1
            analysis_df.loc[score_series_filled < long_term_ema_filled, 'long_term_context'] = -1
        else:
            logger.warning(f"[{self.strategy_name}] 缺少长期 EMA Score 列 ({long_term_ema_col_context})，无法计算长期趋势背景。")
            analysis_df['long_term_context'] = 0

        # --- 8. ADX 趋势强度判断 (使用 focus_timeframe) ---
        dmi_period = bs_params.get("dmi_period", 14)
        # 修正列名以匹配 pandas_ta 的输出
        pdi_col = f'PLUS_DI_{dmi_period}_{focus_tf}'
        ndi_col = f'MINUS_DI_{dmi_period}_{focus_tf}'
        adx_col = f'ADX_{dmi_period}_{focus_tf}'

        if adx_col in data.columns and pdi_col in data.columns and ndi_col in data.columns:
            adx = data[adx_col].fillna(0.0)
            pdi = data[pdi_col].fillna(0.0)
            mdi = data[ndi_col].fillna(0.0)

            analysis_df['adx_strength_signal'] = pd.Series(0.0, index=analysis_df.index)
            strong_trend = adx >= self.adx_strong_threshold
            moderate_trend = (adx >= self.adx_moderate_threshold) & (adx < self.adx_strong_threshold)

            bullish_dmi = pdi > mdi
            bearish_dmi = mdi > pdi

            analysis_df.loc[strong_trend & bullish_dmi, 'adx_strength_signal'] = 1.0
            analysis_df.loc[strong_trend & bearish_dmi, 'adx_strength_signal'] = -1.0
            analysis_df.loc[moderate_trend & bullish_dmi, 'adx_strength_signal'] = 0.5
            analysis_df.loc[moderate_trend & bearish_dmi, 'adx_strength_signal'] = -0.5
            analysis_df.loc[adx < self.adx_moderate_threshold, 'adx_strength_signal'] = 0.0
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 ADX/PDI/NDI 列 ({adx_col}, {pdi_col}, {ndi_col})，无法计算 ADX 强度信号。")
            analysis_df['adx_strength_signal'] = 0.0

        # --- 9. STOCH 超买超卖判断 (使用 focus_timeframe) ---
        stoch_k_p = ia_params.get('stoch_k', bs_params.get('kdj_period_k', 9)) # 使用 KDJ 默认值
        stoch_d_p = ia_params.get('stoch_d', bs_params.get('kdj_period_d', 3)) # 使用 KDJ 默认值
        stoch_smooth_k_p = ia_params.get('stoch_smooth_k', bs_params.get('kdj_period_j', 3)) # 使用 KDJ 默认值

        k_col = f'STOCHk_{stoch_k_p}_{stoch_d_p}_{stoch_smooth_k_p}_{focus_tf}'
        d_col = f'STOCHd_{stoch_k_p}_{stoch_d_p}_{stoch_smooth_k_p}_{focus_tf}'

        if k_col in data.columns and d_col in data.columns:
            k_val = data[k_col].fillna(50.0)
            d_val = data[d_col].fillna(50.0)
            is_oversold = (k_val < self.stoch_oversold_threshold) & (d_val < self.stoch_oversold_threshold)
            is_overbought = (k_val > self.stoch_overbought_threshold) & (d_val > self.stoch_overbought_threshold)
            turning_up = (k_val > d_val) & (k_val.shift(1).fillna(50.0) <= d_val.shift(1).fillna(50.0))
            turning_down = (k_val < d_val) & (k_val.shift(1).fillna(50.0) >= d_val.shift(1).fillna(50.0))

            analysis_df['stoch_signal'] = pd.Series(0.0, index=analysis_df.index)
            analysis_df.loc[is_oversold & turning_up, 'stoch_signal'] = 1.0
            analysis_df.loc[is_overbought & turning_down, 'stoch_signal'] = -1.0
            analysis_df.loc[is_oversold & ~(is_oversold & turning_up), 'stoch_signal'] = 0.5
            analysis_df.loc[is_overbought & ~(is_overbought & turning_down), 'stoch_signal'] = -0.5
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 STOCH K/D 列 ({k_col}, {d_col})，无法计算 STOCH 信号。")
            analysis_df['stoch_signal'] = 0.0

        # --- 10. VWAP 偏离判断 (使用 focus_timeframe) ---
        vwap_anchor = ia_params.get('vwap_anchor', None)
        vwap_col = f'VWAP_{focus_tf}' if vwap_anchor is None else f'VWAP_{vwap_anchor}_{focus_tf}'
        close_col = f'close_{focus_tf}'
        if vwap_col in data.columns and close_col in data.columns:
            vwap = data[vwap_col]
            close_price = data[close_col]
            vwap_safe = vwap.replace(0, np.nan)
            deviation = ((close_price - vwap_safe) / vwap_safe)
            analysis_df['vwap_deviation_signal'] = pd.Series(0, index=analysis_df.index)
            analysis_df.loc[deviation > self.vwap_deviation_threshold, 'vwap_deviation_signal'] = 1
            analysis_df.loc[deviation < -self.vwap_deviation_threshold, 'vwap_deviation_signal'] = -1
            analysis_df['vwap_deviation_percent'] = deviation.fillna(0) * 100
            analysis_df.loc[close_price.isna() | vwap.isna(), 'vwap_deviation_signal'] = 0
            analysis_df['vwap_deviation_percent'].fillna(0.0, inplace=True)
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 VWAP 或收盘价列 ({vwap_col}, {close_col})，无法计算 VWAP 偏离信号。")
            analysis_df['vwap_deviation_signal'] = 0
            analysis_df['vwap_deviation_percent'] = 0.0

        # --- 11. BOLL 突破判断 (使用 focus_timeframe) ---
        boll_period = bs_params.get("boll_period", 20)
        boll_std_dev = bs_params.get("boll_std_dev", 2.0)
        std_str = f"{boll_std_dev:.1f}"
        upper_col = f'BBU_{boll_period}_{std_str}_{focus_tf}'
        lower_col = f'BBL_{boll_period}_{std_str}_{focus_tf}'
        middle_col = f'BBM_{boll_period}_{std_str}_{focus_tf}'
        close_col = f'close_{focus_tf}'
        if upper_col in data.columns and lower_col in data.columns and close_col in data.columns:
            upper_band = data[upper_col]
            lower_band = data[lower_col]
            middle_band = data.get(middle_col)
            close_price = data[close_col]
            analysis_df['boll_breakout_signal'] = pd.Series(0, index=analysis_df.index)
            analysis_df.loc[close_price > upper_band, 'boll_breakout_signal'] = 1
            analysis_df.loc[close_price < lower_band, 'boll_breakout_signal'] = -1
            if middle_band is not None:
                band_width = upper_band - lower_band
                analysis_df['boll_percent_b'] = pd.Series(50.0, index=analysis_df.index)
                valid_band_width = band_width > 1e-9
                analysis_df.loc[valid_band_width, 'boll_percent_b'] = ((close_price.loc[valid_band_width] - lower_band.loc[valid_band_width]) / band_width.loc[valid_band_width]) * 100
                analysis_df['boll_percent_b'].clip(0, 100, inplace=True)
                analysis_df['boll_percent_b'].fillna(50.0, inplace=True)
            analysis_df.loc[close_price.isna() | upper_band.isna() | lower_band.isna(), 'boll_breakout_signal'] = 0
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 BOLL 上轨/下轨或收盘价列 ({upper_col}, {lower_col}, {close_col})，无法计算 BOLL 突破信号。")
            analysis_df['boll_breakout_signal'] = 0
            analysis_df['boll_percent_b'] = 50.0

        # --- 12. 计算综合趋势强度 ---
        alignment_norm = analysis_df.get('alignment_signal', pd.Series(0, index=analysis_df.index)).fillna(0) / 3
        long_context_norm = analysis_df.get('long_term_context', pd.Series(0, index=analysis_df.index)).fillna(0)
        score_momentum_series = analysis_df.get('score_momentum', pd.Series(0, index=analysis_df.index)).fillna(0)
        momentum_norm = np.sign(score_momentum_series)
        adx_strength_norm = analysis_df.get('adx_strength_signal', pd.Series(0.0, index=analysis_df.index)).fillna(0.0)
        volatility_signal_norm = analysis_df.get('volatility_signal', pd.Series(0, index=analysis_df.index)).fillna(0)

        trend_strength = pd.Series(0.0, index=analysis_df.index)
        w_align = 0.3
        w_momentum = 0.2
        w_adx = 0.3
        w_context = 0.1
        w_volatility = 0.1

        trend_strength += alignment_norm * w_align
        trend_strength += momentum_norm * w_momentum
        trend_strength += adx_strength_norm * w_adx
        trend_strength += long_context_norm * w_context
        trend_strength += volatility_signal_norm * w_volatility

        analysis_df['trend_strength_score'] = trend_strength.clip(-3, 3).fillna(0.0)

        logger.debug(f"趋势分析完成，最新趋势强度分: {analysis_df['trend_strength_score'].iloc[-1] if not analysis_df.empty else 'N/A'}")
        return analysis_df

    def _adjust_volatility_parameters(self, data: pd.DataFrame):
        """
        根据股票波动率动态调整参数，如波动率阈值。
        """
        # 逻辑不变
        focus_tf = self.focus_timeframe
        close_col = f'close_{focus_tf}'
        if close_col in data.columns and not data[close_col].isnull().all():
            volatility_window = self.params.get('trend_analysis', {}).get('volatility_window', 10)
            price_volatility = data[close_col].rolling(window=volatility_window, min_periods=max(1, volatility_window//2)).std()
            if not price_volatility.isnull().all():
                latest_volatility = price_volatility.iloc[-1] if not price_volatility.empty else 0
                base_volatility_benchmark = self.tf_params.get('volatility_benchmark', 5.0)
                if base_volatility_benchmark <= 0: base_volatility_benchmark = 5.0

                self.volatility_adjust_factor = max(0.5, latest_volatility / base_volatility_benchmark)
                original_high = self.tf_params.get('volatility_threshold_high', 10.0)
                original_low = self.tf_params.get('volatility_threshold_low', 5.0)
                self.volatility_threshold_high = original_high * self.volatility_adjust_factor
                self.volatility_threshold_low = original_low * self.volatility_adjust_factor

                self.volatility_threshold_high = np.clip(self.volatility_threshold_high, original_high * 0.5, original_high * 2.0)
                self.volatility_threshold_low = np.clip(self.volatility_threshold_low, original_low * 0.5, original_low * 2.0)

                logger.debug(f"动态调整波动率阈值: high={self.volatility_threshold_high:.2f}, low={self.volatility_threshold_low:.2f}, factor={self.volatility_adjust_factor:.2f}")
            else:
                logger.warning(f"[{self.strategy_name}] 波动率数据不可用，无法动态调整参数。")
        else:
            logger.warning(f"[{self.strategy_name}] 缺少收盘价列 {close_col}，无法动态调整参数。")

    def _apply_adx_boost(self, final_signal: pd.Series, adx_strength_norm: pd.Series, base_signal_direction_norm: pd.Series) -> pd.Series:
        """
        模块化调整逻辑：使用 ADX 强度增强信号。
        """
        # 逻辑不变
        final_signal_filled = final_signal.fillna(50.0)
        adx_strength_norm_filled = adx_strength_norm.fillna(0.0)
        base_signal_direction_norm_filled = base_signal_direction_norm.fillna(0.0)

        adx_adjustment_factor = self.tf_params.get('adx_adjustment_factor', 10.0)
        adjustment = pd.Series(0.0, index=final_signal.index)
        effective_mask = (np.abs(adx_strength_norm_filled) > 1e-9) & (np.abs(base_signal_direction_norm_filled) > 1e-9)

        adjustment.loc[effective_mask] = np.sign(base_signal_direction_norm_filled.loc[effective_mask]) * \
                                         np.abs(adx_strength_norm_filled.loc[effective_mask]) * adx_adjustment_factor

        logger.debug(f"ADX 增强调整: {adjustment.iloc[-1] if not adjustment.empty else 'N/A'}")
        return final_signal_filled + adjustment

    def _apply_divergence_penalty(self, final_signal: pd.Series, divergence_signals: pd.DataFrame, dd_params: Dict) -> pd.Series:
        """
        模块化调整逻辑：应用背离惩罚。
        """
        # 逻辑不变
        final_signal_filled = final_signal.fillna(50.0)
        divergence_signals_filled = divergence_signals.fillna(0)

        if divergence_signals_filled.empty:
            logger.debug("背离信号 DataFrame 为空，跳过背离惩罚。")
            return final_signal_filled

        penalty_factor = dd_params.get('divergence_penalty_factor', 0.45)

        has_bearish_div_col = 'HAS_BEARISH_DIVERGENCE'
        has_bullish_div_col = 'HAS_BULLISH_DIVERGENCE'

        if has_bearish_div_col not in divergence_signals_filled.columns or has_bullish_div_col not in divergence_signals_filled.columns:
             logger.warning(f"背离信号 DataFrame 缺少聚合信号列 ({has_bearish_div_col}, {has_bullish_div_col})，跳过背离惩罚。")
             return final_signal_filled

        has_bearish_div = divergence_signals_filled[has_bearish_div_col].astype(bool)
        has_bullish_div = divergence_signals_filled[has_bullish_div_col].astype(bool)

        is_bullish_signal = final_signal_filled > 50
        is_bearish_signal = final_signal_filled < 50

        mask_bullish_signal_bearish_div = is_bullish_signal & has_bearish_div
        mask_bearish_signal_bullish_div = is_bearish_signal & has_bullish_div

        adjusted_signal = final_signal_filled.copy()

        adjusted_signal.loc[mask_bullish_signal_bearish_div] = 50 + (adjusted_signal.loc[mask_bullish_signal_bearish_div] - 50) * (1 - penalty_factor)
        adjusted_signal.loc[mask_bearish_signal_bullish_div] = 50 + (adjusted_signal.loc[mask_bearish_signal_bullish_div] - 50) * (1 - penalty_factor)

        logger.debug(f"背离惩罚调整后信号: {adjusted_signal.iloc[-1] if not adjusted_signal.empty else 'N/A'}")
        return adjusted_signal

    def _apply_trend_confirmation(self, final_signal: pd.Series) -> pd.Series:
        """
        增强假信号过滤：要求信号突破阈值后持续若干周期才视为有效。
        """
        # 逻辑不变
        if final_signal.empty:
            return final_signal

        trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55)
        trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45)
        confirmation_periods = self.trend_confirmation_periods

        final_signal_filled = final_signal.fillna(50.0)

        is_consistently_bullish = final_signal_filled.rolling(window=confirmation_periods, min_periods=confirmation_periods).apply(lambda x: (x >= trend_threshold_upper).all(), raw=True).fillna(False).astype(bool)
        is_consistently_bearish = final_signal_filled.rolling(window=confirmation_periods, min_periods=confirmation_periods).apply(lambda x: (x <= trend_threshold_lower).all(), raw=True).fillna(False).astype(bool)

        is_confirmed = is_consistently_bullish | is_consistently_bearish

        filtered_signal = pd.Series(50.0, index=final_signal.index)
        filtered_signal.loc[is_confirmed] = final_signal_filled.loc[is_confirmed]

        logger.debug(f"趋势确认过滤后信号: {filtered_signal.iloc[-1] if not filtered_signal.empty else 'N/A'}")
        return filtered_signal


    def generate_signals(self, data: pd.DataFrame, stock_code: str, indicator_configs: List[Dict]) -> pd.DataFrame:
        """
        生成趋势跟踪信号，整合规则信号和Transformer模型预测。
        返回包含原始数据、中间计算结果和最终信号的 DataFrame。

        Args:
            data (pd.DataFrame): 包含所有已计算指标的原始DataFrame (由 IndicatorService 提供)。
            stock_code (str): 股票代码。
            indicator_configs (List[Dict]): IndicatorService 生成的指标配置列表。

        Returns:
            pd.DataFrame: 包含原始数据、中间计算结果和最终信号的 DataFrame。
        """
        logger.info(f"开始执行策略: {self.strategy_name} (Focus: {self.focus_timeframe})，股票代码: {stock_code}")

        # 1. 计算规则基础信号 (final_rule_signal) 和中间结果
        final_rule_signal, intermediate_results = self._calculate_rule_based_signal(data=data, stock_code=stock_code, indicator_configs=indicator_configs)

        if not isinstance(final_rule_signal, pd.Series) or final_rule_signal.index.tolist() != data.index.tolist():
             logger.error(f"[{stock_code}] _calculate_rule_based_signal 方法未返回正确的 final_rule_signal Series 或索引不匹配。")
             raise TypeError("_calculate_rule_based_signal 方法返回的 final_rule_signal 格式错误。")

        processed_data = data.copy()
        processed_data['final_rule_signal'] = final_rule_signal # 将规则信号作为 Transformer 的训练目标

        # 将规则计算的中间数据添加到 processed_data 中
        try:
            # 将单项指标评分也添加到 processed_data 中
            indicator_scores_df = intermediate_results.pop('indicator_scores', None) # 从 results 中取出评分 DataFrame
            if indicator_scores_df is not None and not indicator_scores_df.empty:
                 processed_data = processed_data.join(indicator_scores_df, how='left', rsuffix='_score') # 合并评分列

            for key, val in intermediate_results.items(): # 遍历剩余的中间结果
                if isinstance(val, pd.DataFrame) and not val.empty:
                    processed_data = processed_data.join(val, how='left', rsuffix=f'_{key}')
                elif isinstance(val, pd.Series) and not val.empty:
                    processed_data[key] = val.reindex(processed_data.index)
                elif isinstance(val, pd.DataFrame) and val.empty:
                     logger.debug(f"中间结果 '{key}' 为空DataFrame，跳过合并。")
                elif isinstance(val, pd.Series) and val.empty:
                     logger.debug(f"中间结果 '{key}' 为空Series，跳过合并。")


        except Exception as e:
            logger.warning(f"[{stock_code}] 合并规则计算中间数据时出错: {e}，可能导致 processed_data 不完整。", exc_info=True)

        # 2. Transformer模型预测
        # 初始化 Transformer 信号为中性分
        processed_data['transformer_signal'] = pd.Series(50.0, index=processed_data.index) # 列名改为 transformer_signal

        # 设置股票特定的模型路径
        self.set_model_paths(stock_code) # 确保路径已设置

        # --- 检查模型和 scaler 是否存在，如果不存在则跳过预测 ---
        # 尝试加载已存在的模型和 scaler (加载会设置 self.transformer_model, self.feature_scaler, self.target_scaler, self.selected_feature_names_for_transformer)
        self.load_lstm_model(stock_code) # 方法名保持 load_lstm_model 兼容外部调用，但内部逻辑已变更为加载 Transformer

        # 如果模型和 scaler 成功加载或训练 (由外部任务完成加载/训练并设置了self属性)，则进行预测
        # 需要同时检查模型对象、scaler对象和选中特征列表是否有效
        if self.transformer_model is not None and self.feature_scaler is not None and self.target_scaler is not None and self.selected_feature_names_for_transformer:
            try:
                logger.info(f"[{stock_code}] Transformer 模型已加载，准备进行预测...")

                # 调用 predict_with_transformer_model 函数进行预测
                # 只对最新一个时间点进行预测，因此输入数据只需包含最后一个窗口所需的全部数据
                # predict_with_transformer_model 函数内部会处理窗口构建和缩放

                # 将 processed_data 传递给预测函数，它会从中选择正确的特征列
                predicted_signal_score = predict_with_transformer_model(
                    model=self.transformer_model,
                    data=processed_data, # 传入包含所有特征的 DataFrame
                    feature_scaler=self.feature_scaler,
                    target_scaler=self.target_scaler,
                    selected_feature_names=self.selected_feature_names_for_transformer, # 传入选中特征列表
                    window_size=self.transformer_window_size, # 传入窗口大小
                    device=self.device # 传入设备
                )

                # 将预测结果赋值给 processed_data 中最新一个时间点的 'transformer_signal' 列
                if not processed_data.empty:
                    last_idx = processed_data.index[-1]
                    processed_data.loc[last_idx, 'transformer_signal'] = predicted_signal_score # predict_with_transformer_model 已返回 0-100 范围的值
                    logger.info(f"股票 {stock_code} Transformer 模型预测完成，最新预测信号: {processed_data.loc[last_idx, 'transformer_signal']:.2f}")
                else:
                     logger.warning(f"股票 {stock_code} processed_data 为空，无法将 Transformer 预测结果写入。")


            except Exception as e:
                logger.error(f"股票 {stock_code} Transformer 模型预测出错: {e}", exc_info=True)
                # 预测出错时，transformer_signal 保持默认的 50.0
                # processed_data.loc[processed_data.index[-1], 'transformer_signal'] = np.nan # 也可以设置为 NaN

        else:
            logger.warning(f"股票 {stock_code} 的 Transformer 模型或 Scaler 或选中特征列表未成功加载/可用，跳过 Transformer 预测。")


        # 3. 结合规则信号和 Transformer 信号
        processed_data['combined_signal'] = pd.Series(50.0, index=processed_data.index)

        # 只有当 final_rule_signal 和 transformer_signal 都有值时才进行组合
        # transformer_signal 目前只在最后一个点有预测值 (如果成功预测)
        if not processed_data.empty:
            last_idx = processed_data.index[-1]
            latest_rule_signal = processed_data.loc[last_idx, 'final_rule_signal'] if pd.notna(processed_data.loc[last_idx, 'final_rule_signal']) else 50.0
            latest_transformer_signal = processed_data.loc[last_idx, 'transformer_signal'] if pd.notna(processed_data.loc[last_idx, 'transformer_signal']) else 50.0

            combination_weights = self.tf_params.get('signal_combination_weights', {'rule_weight': 0.7, 'lstm_weight': 0.3})
            rule_weight = combination_weights.get('rule_weight', 0.7)
            # 使用 lstm_weight 的值作为 transformer 的权重，保持参数文件结构一致
            transformer_weight = combination_weights.get('lstm_weight', 0.3)

            # 确保权重总和为 1 (已在 _validate_params 中归一化，这里再检查一次保险)
            total_weight = rule_weight + transformer_weight
            if total_weight > 0 and not np.isclose(total_weight, 1.0):
                 rule_weight /= total_weight
                 transformer_weight /= total_weight
                 # logger.debug(f"重新归一化信号组合权重: rule={rule_weight:.2f}, transformer={transformer_weight:.2f}")

            latest_combined_signal = rule_weight * latest_rule_signal + transformer_weight * latest_transformer_signal
            processed_data.loc[last_idx, 'combined_signal'] = np.clip(latest_combined_signal, 0, 100).round(2)

            # 对于历史数据，combined_signal 等于 final_rule_signal
            if len(processed_data) > 1:
                 processed_data.loc[processed_data.index[:-1], 'combined_signal'] = processed_data.loc[processed_data.index[:-1], 'final_rule_signal']

        else:
             logger.warning(f"[{stock_code}] processed_data 为空，无法计算组合信号。")


        # --- 存储中间数据 ---
        self.intermediate_data = processed_data.copy()

        logger.info(f"{self.strategy_name}: 信号生成完毕，股票代码: {stock_code}，最新组合信号: {self.intermediate_data['combined_signal'].iloc[-1] if not self.intermediate_data.empty else 'N/A'}")

        # 调用 analyze_signals 方法进行分析
        self.analyze_signals(stock_code)

        # 方法最终返回包含所有信号和中间结果的 DataFrame
        return self.intermediate_data


    # load_lstm_model 方法重写为加载 Transformer 模型权重和 Scaler
    def load_lstm_model(self, stock_code: str):
        """
        为特定股票加载 Transformer 模型权重和 scaler。
        方法名保持 load_lstm_model 兼容外部调用，但内部逻辑已更改。
        """
        self.set_model_paths(stock_code) # 设置股票特定的路径

        # 检查模型权重文件、特征 scaler 文件、目标 scaler 文件和选中特征文件是否存在
        required_files_exist = all([
            os.path.exists(self.model_path),
            os.path.exists(self.feature_scaler_path),
            os.path.exists(self.target_scaler_path),
            os.path.exists(self.selected_features_path)
        ])

        if not required_files_exist:
            logger.warning(f"股票 {stock_code} 缺失必需的 Transformer 模型权重/Scaler/特征文件，无法加载。")
            self.transformer_model = None
            self.feature_scaler = None
            self.target_scaler = None
            self.selected_feature_names_for_transformer = []
            return

        try:
            # 加载选中特征名列表 (需要在构建模型前知道特征维度)
            with open(self.selected_features_path, 'r', encoding='utf-8') as f:
                 self.selected_feature_names_for_transformer = json.load(f)
            logger.debug(f"股票 {stock_code} 选中特征名列表从 {self.selected_features_path} 加载成功 ({len(self.selected_feature_names_for_transformer)} 个)。")

            # 根据加载的特征数量构建 Transformer 模型结构
            num_features = len(self.selected_feature_names_for_transformer)
            if num_features == 0:
                 logger.error(f"股票 {stock_code} 加载的选中特征列表为空，无法构建模型。")
                 self.transformer_model = None
                 self.feature_scaler = None
                 self.target_scaler = None
                 self.selected_feature_names_for_transformer = []
                 return

            # 构建模型架构
            self.transformer_model = build_transformer_model(
                num_features=num_features,
                model_config=self.transformer_model_config,
                summary=False, # 加载时不打印结构概览
                window_size=self.transformer_window_size
            )
            self.transformer_model.to(self.device) # 将模型移动到指定设备

            # 加载模型权重
            # torch.load 可以直接加载状态字典
            self.transformer_model.load_state_dict(torch.load(self.model_path, map_location=self.device)) # map_location 确保加载到正确设备
            logger.info(f"股票 {stock_code} Transformer 模型权重从 {self.model_path} 加载成功。")

            # 加载特征 Scaler
            self.feature_scaler = joblib.load(self.feature_scaler_path)
            logger.info(f"股票 {stock_code} 特征 Scaler 从 {self.feature_scaler_path} 加载成功。")

            # 加载目标 Scaler
            self.target_scaler = joblib.load(self.target_scaler_path)
            logger.info(f"股票 {stock_code} 目标 Scaler 从 {self.target_scaler_path} 加载成功。")


            # 最终检查是否成功加载了所有必需对象
            if self.transformer_model is None or self.feature_scaler is None or self.target_scaler is None or not self.selected_feature_names_for_transformer:
                 logger.warning(f"股票 {stock_code} 加载 Transformer 模型或 Scaler 或特征列表失败，部分对象为 None/空。")
                 self.transformer_model = None
                 self.feature_scaler = None
                 self.target_scaler = None
                 self.selected_feature_names_for_transformer = []
            else:
                 logger.info(f"股票 {stock_code} 的 Transformer 模型及相关组件加载完成。")


        except Exception as e:
            logger.error(f"股票 {stock_code} 加载 Transformer 模型或 Scaler 或特征列表出错: {e}", exc_info=True)
            self.transformer_model = None
            self.feature_scaler = None
            self.target_scaler = None
            self.selected_feature_names_for_transformer = []

    # 方法1: 将机器学习训练所需的最终数据存入文件 (修改为保存 Transformer 数据)
    def save_prepared_data(self, stock_code: str,
                        features_scaled_train: np.ndarray, targets_scaled_train: np.ndarray,
                        features_scaled_val: np.ndarray, targets_scaled_val: np.ndarray,
                        features_scaled_test: np.ndarray, targets_scaled_test: np.ndarray,
                        feature_scaler: Union[MinMaxScaler, StandardScaler], target_scaler: Union[MinMaxScaler, StandardScaler],
                        selected_feature_names: List[str]):
        """
        保存准备好的 Transformer 训练数据 (NumPy 数组) 和 Scaler (joblib) 到股票特定目录。
        所有数据整体保存为一个 .npz 文件，压缩存储。
        同时保存最终用于训练的特征名列表。
        """
        self.set_model_paths(stock_code) # 确保路径已设置

        if not self.all_prepared_data_npz_path or not self.feature_scaler_path or not self.target_scaler_path or not self.selected_features_path:
            logger.error(f"[{stock_code}] 保存准备好的数据或 Scaler 的部分或全部路径未在 set_model_paths 中设置。")
            raise RuntimeError("保存路径未正确初始化。")

        try:
            # 保存所有 NumPy 数组为一个 .npz 文件（压缩）
            # 确保数据是 float32 类型，这是 PyTorch 和大多数模型常用的类型
            np.savez_compressed(
                self.all_prepared_data_npz_path,
                features_scaled_train=features_scaled_train.astype(np.float32),
                targets_scaled_train=targets_scaled_train.astype(np.float32),
                features_scaled_val=features_scaled_val.astype(np.float32),
                targets_scaled_val=targets_scaled_val.astype(np.float32),
                features_scaled_test=features_scaled_test.astype(np.float32),
                targets_scaled_test=targets_scaled_test.astype(np.float32)
            )
            logger.debug(f"[{stock_code}] 所有数据已整体保存为 {self.all_prepared_data_npz_path}。")

            # 保存 Scaler
            joblib.dump(feature_scaler, self.feature_scaler_path)
            joblib.dump(target_scaler, self.target_scaler_path)
            logger.debug(f"[{stock_code}] Scaler 已保存。")

            # 保存选中特征名列表
            with open(self.selected_features_path, 'w', encoding='utf-8') as f:
                 json.dump(selected_feature_names, f, ensure_ascii=False, indent=4)
            logger.debug(f"[{stock_code}] 选中特征名列表已保存到 {self.selected_features_path}。")


        except Exception as e:
            logger.error(f"[{stock_code}] 保存准备好的数据、Scaler 或特征列表时出错: {e}", exc_info=True)
            raise e

    # 方法2: 读取文件进行训练 (修改为只加载数据，不进行训练)
    def load_prepared_data(self, stock_code: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler, None], Union[MinMaxScaler, StandardScaler, None]]:
        """
        从文件加载特定股票准备好的 Transformer 训练数据 (NumPy 数组) 和 Scaler (joblib)。
        同时加载最终用于训练的特征名列表 (并赋值给 self.selected_feature_names_for_transformer)。
        返回加载的数据数组和 Scaler 对象。如果文件不存在或加载失败，返回空数组和 None Scaler。
        """
        self.set_model_paths(stock_code) # 确保路径已设置

        # 检查必需的文件是否存在
        required_files_exist = all([
            os.path.exists(self.all_prepared_data_npz_path),
            os.path.exists(self.feature_scaler_path),
            os.path.exists(self.target_scaler_path),
            os.path.exists(self.selected_features_path)
        ])

        if not required_files_exist:
            logger.warning(f"股票 {stock_code} 缺失必需的准备数据/Scaler/特征文件，无法加载。")
            self.selected_feature_names_for_transformer = [] # 加载失败时清空
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None

        try:
            # 加载所有数据
            data_npz = np.load(self.all_prepared_data_npz_path)
            features_scaled_train = data_npz['features_scaled_train']
            targets_scaled_train = data_npz['targets_scaled_train']
            features_scaled_val = data_npz['features_scaled_val']
            targets_scaled_val = data_npz['targets_scaled_val']
            features_scaled_test = data_npz['features_scaled_test']
            targets_scaled_test = data_npz['targets_scaled_test']
            data_npz.close()

            # 加载 Scaler
            feature_scaler = joblib.load(self.feature_scaler_path)
            target_scaler = joblib.load(self.target_scaler_path)

            # 加载选中特征名列表 (赋值给实例属性)
            with open(self.selected_features_path, 'r', encoding='utf-8') as f:
                 self.selected_feature_names_for_transformer = json.load(f)
            logger.debug(f"股票 {stock_code} 选中特征名列表从 {self.selected_features_path} 加载成功 ({len(self.selected_feature_names_for_transformer)} 个)。")


            logger.info(f"股票 {stock_code} 准备好的数据和 Scaler 已成功加载。")
            # 返回 NumPy 数组和 scaler 对象
            return features_scaled_train, targets_scaled_train, \
                features_scaled_val, targets_scaled_val, \
                features_scaled_test, targets_scaled_test, \
                feature_scaler, target_scaler

        except Exception as e:
            logger.error(f"股票 {stock_code} 加载准备好的数据、Scaler 或特征列表时出错: {e}", exc_info=True)
            self.selected_feature_names_for_transformer = [] # 加载失败时清空
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None


    # 其他方法如 get_intermediate_data, _calculate_trend_duration, analyze_signals, save_analysis_results 基本保持不变
    # 它们主要使用 processed_data (即 self.intermediate_data) 中的最终信号和中间结果列
    # 只需要注意 analyze_signals 中对 LSTM 的引用需要改为 Transformer
    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回中间计算结果"""
        return self.intermediate_data

    def _calculate_trend_duration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算趋势的持续时间和强度，并根据 focus_timeframe 转换为具体时间单位。
        使用参数文件中定义的持续时间阈值。
        基于 'final_rule_signal' 列计算。
        """
        # 逻辑不变，基于规则信号计算持续时间
        trend_duration_info = {
            'bullish_duration': 0,
            'bearish_duration': 0,
            'bullish_duration_text': '0分钟',
            'bearish_duration_text': '0分钟',
            'current_trend': 'neutral',
            'trend_strength': 'weak',
            'duration_status': 'short'
        }

        if 'final_rule_signal' not in data.columns or data['final_rule_signal'].isnull().all():
            logger.warning("规则信号列 'final_rule_signal' 不存在或全为 NaN，无法计算趋势持续时间。")
            return trend_duration_info

        final_signal = data['final_rule_signal'].dropna()
        if final_signal.empty:
            return trend_duration_info

        current_bullish_streak = 0
        current_bearish_streak = 0
        trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55)
        trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45)
        strong_bullish_threshold = self.tf_params.get('strong_bullish_threshold', 75)
        strong_bearish_threshold = self.tf_params.get('strong_bearish_threshold', 25)
        moderate_bullish_threshold = self.tf_params.get('moderate_bullish_threshold', 60)
        moderate_bearish_threshold = self.tf_params.get('moderate_bearish_threshold', 40)


        for signal in final_signal.iloc[::-1]:
            if signal >= trend_threshold_upper:
                current_bullish_streak += 1
                current_bearish_streak = 0
            elif signal <= trend_threshold_lower:
                current_bearish_streak += 1
                current_bullish_streak = 0
            else:
                 break

        trend_duration_info['bullish_duration'] = current_bullish_streak
        trend_duration_info['bearish_duration'] = current_bearish_streak

        try:
            timeframe_value = int(self.focus_timeframe)
            bullish_duration_minutes = current_bullish_streak * timeframe_value
            bearish_duration_minutes = current_bearish_streak * timeframe_value

            def format_duration(minutes):
                if minutes < 60:
                    return f"{minutes}分钟"
                elif minutes < 1440:
                    hours = minutes // 60
                    remaining_minutes = minutes % 60
                    return f"{hours}小时{remaining_minutes}分钟" if remaining_minutes > 0 else f"{hours}小时"
                else:
                    days = minutes // 1440
                    remaining_hours = (minutes % 1440) // 60
                    return f"{days}天{remaining_hours}小时" if remaining_hours > 0 else f"{days}天"

            trend_duration_info['bullish_duration_text'] = format_duration(bullish_duration_minutes) if bullish_duration_minutes > 0 else '0分钟'
            trend_duration_info['bearish_duration_text'] = format_duration(bearish_duration_minutes) if bearish_duration_minutes > 0 else '0分钟'
        except ValueError:
            logger.warning(f"无法将 focus_timeframe '{self.focus_timeframe}' 转换为分钟数，持续时间将以周期数显示")
            trend_duration_info['bullish_duration_text'] = f"{current_bullish_streak}个周期" if current_bullish_streak > 0 else '0个周期'
            trend_duration_info['bearish_duration_text'] = f"{current_bearish_streak}个周期" if current_bearish_streak > 0 else '0个周期'

        latest_rule_signal = final_signal.iloc[-1]
        if latest_rule_signal >= strong_bullish_threshold:
            trend_duration_info['current_trend'] = '看涨↑'
            trend_duration_info['trend_strength'] = '非常强烈'
        elif latest_rule_signal >= moderate_bullish_threshold:
            trend_duration_info['current_trend'] = '看涨↑'
            trend_duration_info['trend_strength'] = '强'
        elif latest_rule_signal >= trend_threshold_upper:
            trend_duration_info['current_trend'] = '看涨↑'
            trend_duration_info['trend_strength'] = '温和'
        elif latest_rule_signal <= strong_bearish_threshold:
            trend_duration_info['current_trend'] = '看跌↓'
            trend_duration_info['trend_strength'] = '非常强烈'
        elif latest_rule_signal <= moderate_bearish_threshold:
            trend_duration_info['current_trend'] = '看跌↓'
            trend_duration_info['trend_strength'] = '强'
        elif latest_rule_signal <= trend_threshold_lower:
            trend_duration_info['current_trend'] = '看跌↓'
            trend_duration_info['trend_strength'] = '温和'
        else:
            trend_duration_info['current_trend'] = '中性'
            trend_duration_info['trend_strength'] = '不明'

        current_duration = max(current_bullish_streak, current_bearish_streak)
        if current_duration >= self.trend_duration_threshold_strong:
             trend_duration_info['duration_status'] = '长'
        elif current_duration >= self.trend_duration_threshold_moderate:
             trend_duration_info['duration_status'] = '中'
        else:
             trend_duration_info['duration_status'] = '短'
        return trend_duration_info

    def analyze_signals(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        分析趋势策略信号，增加对新指标、趋势增强/枯竭、背离的解读，优化操作建议。
        适应A股 T+1 交易制度，增加止损止盈建议。
        分析基于 self.intermediate_data 中的 combined_signal 和其他中间结果。
        """
        # 逻辑不变，只需要确保引用的是 transformer_signal 而不是 lstm_signal
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning("中间数据为空，无法进行信号分析。")
            return None
        if not stock_code:
            logger.error("股票代码为空，无法进行信号分析。")
            return None

        analysis_results = {}
        data = self.intermediate_data
        latest_data = data.iloc[-1] if not data.empty else pd.Series(dtype=object)

        # --- 统计分析 (基于 combined_signal) ---
        if 'combined_signal' in data.columns:
            combined_signal = data['combined_signal'].dropna()
            if not combined_signal.empty:
                analysis_results['combined_signal_mean'] = combined_signal.mean()
                analysis_results['combined_signal_median'] = combined_signal.median()
                analysis_results['combined_signal_std'] = combined_signal.std()
                trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55)
                trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45)
                strong_bullish_threshold = self.tf_params.get('strong_bullish_threshold', 75)
                strong_bearish_threshold = self.tf_params.get('strong_bearish_threshold', 25)

                analysis_results['combined_signal_bullish_ratio'] = (combined_signal >= trend_threshold_upper).mean()
                analysis_results['combined_signal_bearish_ratio'] = (combined_signal <= trend_threshold_lower).mean()
                analysis_results['combined_signal_strong_bullish_ratio'] = (combined_signal >= strong_bullish_threshold).mean()
                analysis_results['combined_signal_strong_bearish_ratio'] = (combined_signal <= strong_bearish_threshold).mean()

        # 统计分析其他中间信号
        if 'alignment_signal' in data.columns:
            alignment = data['alignment_signal'].dropna()
            if not alignment.empty:
                analysis_results['alignment_fully_bullish_ratio'] = (alignment == 3).mean()
                analysis_results['alignment_fully_bearish_ratio'] = (alignment == -3).mean()
                analysis_results['alignment_bullish_ratio'] = (alignment > 0).mean()
                analysis_results['alignment_bearish_ratio'] = (alignment < 0).mean()
        if 'long_term_context' in data.columns:
            context = data['long_term_context'].dropna()
            if not context.empty:
                analysis_results['long_term_bullish_ratio'] = (context == 1).mean()
                analysis_results['long_term_bearish_ratio'] = (context == -1).mean()
        if 'trend_strength_score' in data.columns:
            trend_strength_score_series = data['trend_strength_score'].dropna()
            if not trend_strength_score_series.empty:
                analysis_results['trend_strength_mean'] = trend_strength_score_series.mean()
                analysis_results['trend_strength_strong_bull_ratio'] = (trend_strength_score_series >= 1.5).mean()
                analysis_results['trend_strength_strong_bear_ratio'] = (trend_strength_score_series <= -1.5).mean()


        # --- 计算趋势持续时间 (基于 final_rule_signal) ---
        trend_duration_info = self._calculate_trend_duration(data)
        analysis_results.update(trend_duration_info)

        # --- 最新信号判断和细化操作建议 (基于 combined_signal) ---
        signal_judgment = {}
        operation_advice = "中性观望"
        risk_warning = ""
        t_plus_1_note = "（受 T+1 限制，建议次日操作）"
        stop_loss_profit_advice = ""

        final_score = latest_data.get('combined_signal', 50.0)
        current_trend_rule = trend_duration_info['current_trend']
        trend_strength_rule = trend_duration_info['trend_strength']
        duration_status_rule = trend_duration_info['duration_status']

        moderate_bullish_threshold = self.tf_params.get('moderate_bullish_threshold', 60)
        strong_bullish_threshold = self.tf_params.get('strong_bullish_threshold', 75)
        moderate_bearish_threshold = self.tf_params.get('moderate_bearish_threshold', 40)
        strong_bearish_threshold = self.tf_params.get('strong_bearish_threshold', 25)


        if final_score >= moderate_bullish_threshold:
             signal_judgment['overall_signal'] = "看涨信号"
             if final_score >= strong_bullish_threshold:
                  signal_judgment['overall_signal'] += " (强)"
                  if duration_status_rule == '长':
                      operation_advice = f"持有或逢低加仓 (信号强劲且趋势持续) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止盈（接近近期高点）"
                  elif duration_status_rule == '中':
                      operation_advice = f"持有或试探加仓 (信号强劲但需确认趋势持续) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止损（近期低点下方）"
                  else:
                      operation_advice = f"关注买入信号 (信号强劲但趋势刚启动) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止损（入场价下方3-5%）"
             else:
                  signal_judgment['overall_signal'] += " (温和)"
                  if duration_status_rule == '长':
                      operation_advice = f"谨慎持有 (信号温和但趋势持续较长) {t_plus_1_note}"
                  elif duration_status_rule == '中':
                      operation_advice = f"持有观察 (信号温和且趋势持续中) {t_plus_1_note}"
                  else:
                      operation_advice = f"观望或轻仓试多 (信号温和启动) {t_plus_1_note}"

        elif final_score <= moderate_bearish_threshold:
             signal_judgment['overall_signal'] = "看跌信号"
             if final_score <= strong_bearish_threshold:
                  signal_judgment['overall_signal'] += " (强)"
                  if duration_status_rule == '长':
                      operation_advice = f"卖出或逢高减仓 (信号强劲且趋势持续) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止盈（接近近期低点）"
                  elif duration_status_rule == '中':
                      operation_advice = f"减仓或准备卖出 (信号强劲但需确认趋势持续) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止损（近期高点上方）"
                  else:
                      operation_advice = f"关注卖出信号 (信号强劲但趋势刚启动) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止损（入场价上方3-5%）"
             else:
                  signal_judgment['overall_signal'] += " (温和)"
                  if duration_status_rule == '长':
                      operation_advice = f"谨慎持有空头或卖出 (信号温和但趋势持续较长) {t_plus_1_note}"
                  elif duration_status_rule == '中':
                      operation_advice = f"持有空头或观望 (信号温和且趋势持续中) {t_plus_1_note}"
                  else:
                      operation_advice = f"观望或轻仓试空 (信号温和启动) {t_plus_1_note}"
        else:
            signal_judgment['overall_signal'] = "中性信号"
            operation_advice = "中性观望，等待信号明朗"

        # --- 结合其他指标细化判断和建议 (使用 latest_data 中的中间结果列) ---
        alignment = latest_data.get('alignment_signal', 0)
        if alignment == 3:
            signal_judgment['alignment_status'] = "完全多头排列"
            if final_score > 50: operation_advice += " - EMA确认"
        elif alignment > 0:
            signal_judgment['alignment_status'] = "偏多头排列"
        elif alignment == -3:
            signal_judgment['alignment_status'] = "完全空头排列"
            if final_score < 50: operation_advice += " - EMA确认"
        elif alignment < 0:
            signal_judgment['alignment_status'] = "偏空头排列"
        else:
            signal_judgment['alignment_status'] = "排列混乱"

        long_context = latest_data.get('long_term_context', 0)
        if long_context == 1:
            signal_judgment['long_term_view'] = "长期看涨"
        elif long_context == -1:
            signal_judgment['long_term_view'] = "长期看跌"
        else:
            signal_judgment['long_term_view'] = "长期不明"

        adx_signal = latest_data.get('adx_strength_signal', 0)
        if adx_signal >= 0.5:
            signal_judgment['adx_status'] = f"趋势明确 (上升)"
        elif adx_signal == 0:
            signal_judgment['adx_status'] = "无明显趋势"
        else:
            signal_judgment['adx_status'] = f"趋势明确 (下降)"
        if abs(final_score - 50) > 20 and abs(adx_signal) < 0.5:
            risk_warning += "ADX显示趋势强度不足，注意假突破风险。 "

        stoch_signal = latest_data.get('stoch_signal', 0)
        if stoch_signal == 1:
            signal_judgment['stoch_status'] = "超卖区金叉"
        elif stoch_signal == -1:
            signal_judgment['stoch_status'] = "超买区死叉"
        elif stoch_signal == 0.5:
            signal_judgment['stoch_status'] = "超卖区域"
        elif stoch_signal == -0.5:
            signal_judgment['stoch_status'] = "超买区域"
        else:
            signal_judgment['stoch_status'] = "中间区域"
        if final_score > 50 and stoch_signal <= -0.5:
            risk_warning += "STOCH进入超买区，警惕回调。 "
            if "止盈" not in stop_loss_profit_advice:
                 stop_loss_profit_advice = "建议设置止盈（当前价上方5-8%）"
        if final_score < 50 and stoch_signal >= 0.5:
            risk_warning += "STOCH进入超卖区，警惕反弹。 "
            if "止盈" not in stop_loss_profit_advice:
                 stop_loss_profit_advice = "建议设置止盈（当前价下方5-8%）"

        boll_signal = latest_data.get('boll_breakout_signal', 0)
        if boll_signal == 1:
            signal_judgment['boll_status'] = "向上突破布林带"
            if final_score > 50: operation_advice += " - BOLL突破确认"
        elif boll_signal == -1:
            signal_judgment['boll_status'] = "向下突破布林带"
            if final_score < 50: operation_advice += " - BOLL突破确认"
        else:
            signal_judgment['boll_status'] = "布林带轨道内运行"

        volume_confirm = latest_data.get('volume_confirmation_signal', 0)
        volume_spike = latest_data.get('volume_spike_signal', 0)
        if volume_confirm == 1:
            signal_judgment['volume_status'] = "量能配合趋势"
        elif volume_confirm == -1:
            signal_judgment['volume_status'] = "量能不支持趋势"
        else:
            signal_judgment['volume_status'] = "量能中性"
        if volume_spike == 1:
            signal_judgment['volume_spike'] = "出现显著放量"
            if abs(final_score - 50) > 10:
                operation_advice += " (放量)"
            else:
                operation_advice += " (放量关注突破)"

        has_bearish_div = latest_data.get('HAS_BEARISH_DIVERGENCE', False)
        has_bullish_div = latest_data.get('HAS_BULLISH_DIVERGENCE', False)

        if has_bearish_div and final_score > 50:
            signal_judgment['divergence_status'] = "检测到顶背离"
            risk_warning += "检测到顶背离，趋势可能衰竭或反转！ "
            operation_advice = operation_advice.replace("加仓", "观望").replace("买入", "谨慎买入").replace("持有", "谨慎持有")
            if "止损" not in stop_loss_profit_advice:
                 stop_loss_profit_advice = "建议设置止损（当前价下方3-5%）"
        elif has_bullish_div and final_score < 50:
            signal_judgment['divergence_status'] = "检测到底背离"
            risk_warning += "检测到底背离，趋势可能衰竭或反转！ "
            operation_advice = operation_advice.replace("减仓", "观望").replace("卖出", "谨慎卖出").replace("持有空头", "谨慎持有空头")
            if "止损" not in stop_loss_profit_advice:
                 stop_loss_profit_advice = "建议设置止损（当前价上方3-5%）"
        else:
            signal_judgment['divergence_status'] = "无明显背离"

        # --- 生成中文解读 ---
        bullish_duration_text = trend_duration_info['bullish_duration_text']
        bearish_duration_text = trend_duration_info['bearish_duration_text']
        duration_text = f"看涨持续 {bullish_duration_text}" if trend_duration_info['current_trend'] == '看涨↑' and trend_duration_info['bullish_duration'] > 0 else \
                        f"看跌持续 {bearish_duration_text}" if trend_duration_info['current_trend'] == '看跌↓' and trend_duration_info['bearish_duration'] > 0 else \
                        "趋势持续时间不足"

        now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        volume_spike_str = f" ({signal_judgment['volume_spike']})" if 'volume_spike' in signal_judgment else ''
        chinese_interpretation = (
            f"【趋势跟踪策略分析 - {stock_code} - {now_str}】\n"
            f"最新组合信号分: {final_score:.2f} (规则信号: {latest_data.get('final_rule_signal', 50.0):.2f}, Transformer预测: {latest_data.get('transformer_signal', 50.0):.2f})\n" # 修改这里显示 Transformer 预测
            f"当前趋势状态: {signal_judgment.get('overall_signal', '中性')}\n"
            f"规则趋势判断: {trend_duration_info['current_trend']} ({trend_duration_info['trend_strength'].capitalize()})\n"
            f"趋势持续: {duration_text} ({trend_duration_info['duration_status'].capitalize()})\n"
            f"EMA排列: {signal_judgment.get('alignment_status', '未知')}\n"
            f"长期背景: {signal_judgment.get('long_term_view', '未知')}\n"
            f"ADX强度: {signal_judgment.get('adx_status', '未知')}\n"
            f"STOCH状态: {signal_judgment.get('stoch_status', '未知')}\n"
            f"BOLL状态: {signal_judgment.get('boll_status', '未知')}\n"
            f"量能状态: {signal_judgment.get('volume_status', '未知')}{volume_spike_str}\n"
            f"背离状态: {signal_judgment.get('divergence_status', '未知')}\n"
            f"操作建议: {operation_advice}\n"
            f"风险提示: {risk_warning if risk_warning else '无明显风险提示。'}\n"
            f"止损止盈建议: {stop_loss_profit_advice if stop_loss_profit_advice else '根据自身风险偏好设置。'}"
        )

        analysis_results['signal_judgment'] = signal_judgment
        analysis_results['operation_advice'] = operation_advice
        analysis_results['risk_warning'] = risk_warning
        analysis_results['stop_loss_profit_advice'] = stop_loss_profit_advice
        analysis_results['chinese_interpretation'] = chinese_interpretation

        self.analysis_results = analysis_results
        logger.info(f"[{stock_code}] 信号分析完成。")
        logger.info(chinese_interpretation)

        return analysis_results

    def get_analysis_results(self) -> Optional[Dict[str, Any]]:
        """返回信号分析结果"""
        return self.analysis_results

    def save_analysis_results(self, stock_code: str, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        保存趋势跟踪策略的分析结果到数据库
        """
        # 逻辑不变，只需要确保从 intermediate_data 中获取的列名是正确的 (transformer_signal 而不是 lstm_signal)
        from stock_models.stock_analytics import StockScoreAnalysis
        from stock_models.stock_basic import StockInfo
        import logging

        logger = logging.getLogger("strategy_trend_following")

        try:
            stock = StockInfo.objects.get(stock_code=stock_code)
            analysis_data = self.analysis_results if self.analysis_results is not None else {}

            intermediate_data_latest = self.intermediate_data.iloc[-1].to_dict() if self.intermediate_data is not None and not self.intermediate_data.empty else {}
            latest_data_point = data.iloc[-1].to_dict() if data is not None and not data.empty else {}

            def convert_nan_to_none(value):
                if isinstance(value, (int, float)) and (np.isinf(value) or np.isnan(value)):
                     return None
                return None if pd.isna(value) else value

            defaults_dict = {
                'score': convert_nan_to_none(intermediate_data_latest.get('combined_signal', None)),
                'rule_signal': convert_nan_to_none(intermediate_data_latest.get('final_rule_signal', None)),
                'lstm_signal': convert_nan_to_none(intermediate_data_latest.get('transformer_signal', None)), # 保存 transformer_signal 到 lstm_signal 字段（数据库兼容）

                'base_score_raw': convert_nan_to_none(intermediate_data_latest.get('base_score_raw', None)),
                'base_score_volume_adjusted': convert_nan_to_none(intermediate_data_latest.get('base_score_volume_adjusted', None)),
                'alignment_signal': convert_nan_to_none(intermediate_data_latest.get('alignment_signal', None)),
                'long_term_context': convert_nan_to_none(intermediate_data_latest.get('long_term_context', None)),
                'trend_strength_score': convert_nan_to_none(intermediate_data_latest.get('trend_strength_score', None)),
                'score_momentum': convert_nan_to_none(intermediate_data_latest.get('score_momentum', None)),
                'score_volatility': convert_nan_to_none(intermediate_data_latest.get('score_volatility', None)),
                'adx_strength_signal': convert_nan_to_none(intermediate_data_latest.get('adx_strength_signal', None)),
                'stoch_signal': convert_nan_to_none(intermediate_data_latest.get('stoch_signal', None)),
                'vwap_deviation_signal': convert_nan_to_none(intermediate_data_latest.get('vwap_deviation_signal', None)),
                'boll_breakout_signal': convert_nan_to_none(intermediate_data_latest.get('boll_breakout_signal', None)),
                'ema_cross_signal': convert_nan_to_none(intermediate_data_latest.get('ema_cross_signal', None)),
                'volume_confirmation_signal': convert_nan_to_none(intermediate_data_latest.get('volume_confirmation_signal', None)),
                'volume_spike_signal': convert_nan_to_none(intermediate_data_latest.get('volume_spike_signal', None)),
                'div_has_bearish_divergence': analysis_data.get('signal_judgment', {}).get('divergence_status', '') == '检测到顶背离',
                'div_has_bullish_divergence': analysis_data.get('signal_judgment', {}).get('divergence_status', '') == '检测到底背离',

                'close_price': convert_nan_to_none(latest_data_point.get(f'close_{self.focus_timeframe}', None)),

                'current_trend': analysis_data.get('current_trend', None),
                'trend_strength': analysis_data.get('trend_strength', None),
                'bullish_duration': convert_nan_to_none(analysis_data.get('bullish_duration', None)),
                'bearish_duration': convert_nan_to_none(analysis_data.get('bearish_duration', None)),
                'operation_advice': analysis_data.get('operation_advice', None),
                'risk_warning': analysis_data.get('risk_warning', None),
                'chinese_interpretation': analysis_data.get('chinese_interpretation', None),

                'params_snapshot': self.params,
            }

            defaults_cleaned = {}
            for k, v in defaults_dict.items():
                 if isinstance(v, (bool, str, dict, list)) or v is None:
                      defaults_cleaned[k] = v
                 else:
                      defaults_cleaned[k] = convert_nan_to_none(v)

            obj, created = StockScoreAnalysis.objects.update_or_create(
                stock=stock,
                strategy_name=self.strategy_name,
                timestamp=timestamp,
                time_level=self.focus_timeframe,
                defaults=defaults_cleaned
            )

            if created:
                logger.info(f"创建新的 {stock_code} 的趋势跟踪策略分析结果记录，时间戳: {timestamp}")
            else:
                logger.info(f"更新 {stock_code} 的趋势跟踪策略分析结果记录，时间戳: {timestamp}")

        except StockInfo.DoesNotExist:
            logger.error(f"股票 {stock_code} 未找到，无法保存分析结果")
        except Exception as e:
            logger.error(f"保存 {stock_code} 的趋势跟踪策略分析结果时出错: {e}", exc_info=True)
            # logger.error(f"尝试保存的数据: {defaults_cleaned}")

