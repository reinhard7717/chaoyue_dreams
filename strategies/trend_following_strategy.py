# 此策略侧重于识别和跟随趋势，主要使用 EMA 排列、DMI、SAR 等指标，并以 30 分钟级别为主要权重。
# strategies/trend_following_strategy.py
import pandas as pd
import numpy as np
import json
import os
import logging
from django.conf import settings
import joblib # 用于加载/保存 scaler
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas_ta as ta

# 假设 BaseStrategy 在 .base 模块中定义 (根据您的文件结构)
from .base import BaseStrategy
from .utils import strategy_utils
from .utils.deep_learning_utils import (
    build_transformer_model,
    evaluate_transformer_model,
    train_transformer_model,
    predict_with_transformer_model,
    TimeSeriesDataset,
    prepare_data_for_transformer
)

logger = logging.getLogger("strategy_trend_following") # 策略特定的 logger

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
    # strategy_name 将从参数文件中加载，这里提供一个默认值以防万一
    strategy_name = "TrendFollowingStrategy_DefaultName"
    default_focus_timeframe = '30' # 默认主要关注的时间框架

    def __init__(self, params_file: str = "strategies/indicator_parameters.json", base_data_dir: str = settings.STRATEGY_DATA_DIR):
        """
        初始化趋势跟踪策略。

        Args:
            params_file (str): 策略参数JSON文件的路径。
            base_data_dir (str): 存储策略相关数据（如模型、scalers）的基础目录。
                                 默认为 Django settings中的 STRATEGY_DATA_DIR。
        """
        # 步骤1: 加载参数文件到 loaded_params 字典
        loaded_params: Dict[str, Any] = {} # 初始化为空字典
        resolved_params_file_path = params_file # 用于记录最终使用的路径
        
        # 尝试解析 params_file 路径
        if not os.path.isabs(params_file):
            if hasattr(settings, 'BASE_DIR') and settings.BASE_DIR:
                path_based_on_base_dir = os.path.join(settings.BASE_DIR, params_file)
                if os.path.exists(path_based_on_base_dir) and os.path.isfile(path_based_on_base_dir):
                    resolved_params_file_path = path_based_on_base_dir
                else:
                    # 如果在 BASE_DIR 找不到，尝试基于当前工作目录
                    path_based_on_cwd = os.path.abspath(params_file)
                    if os.path.exists(path_based_on_cwd) and os.path.isfile(path_based_on_cwd):
                        resolved_params_file_path = path_based_on_cwd
                        logger.warning(f"[{self.strategy_name}] 参数文件 '{params_file}' 在项目根目录未找到，但在CWD '{os.getcwd()}' 找到: '{resolved_params_file_path}'. 建议使用项目相对路径。")
                    else:
                        logger.error(f"[{self.strategy_name}] 参数文件 '{params_file}' 在任何已知位置均未找到。尝试的路径包括基于BASE_DIR和CWD。")
            else: # 没有 BASE_DIR，只能依赖 CWD
                path_based_on_cwd = os.path.abspath(params_file)
                if os.path.exists(path_based_on_cwd) and os.path.isfile(path_based_on_cwd):
                    resolved_params_file_path = path_based_on_cwd
                    logger.warning(f"[{self.strategy_name}] Django settings.BASE_DIR 未定义。参数文件 '{params_file}' 在CWD '{os.getcwd()}' 找到: '{resolved_params_file_path}'.")
                else:
                    logger.error(f"[{self.strategy_name}] Django settings.BASE_DIR 未定义，且参数文件 '{params_file}' 在CWD '{os.getcwd()}' (解析为 '{path_based_on_cwd}') 也未找到。")
        
        # 实际加载文件
        if os.path.exists(resolved_params_file_path) and os.path.isfile(resolved_params_file_path):
            try:
                with open(resolved_params_file_path, 'r', encoding='utf-8') as f:
                    loaded_params = json.load(f)
                logger.info(f"[{self.strategy_name}] 策略参数已从 '{resolved_params_file_path}' 加载。")
            except json.JSONDecodeError as e:
                logger.error(f"[{self.strategy_name}] 解析参数文件 '{resolved_params_file_path}' 时发生JSON解码错误: {e}")
            except Exception as e:
                logger.error(f"[{self.strategy_name}] 加载参数文件 '{resolved_params_file_path}' 时发生未知错误: {e}", exc_info=True)
        else:
            # 如果在 __init__ 开头路径解析失败，这里会再次确认
            if not (os.path.exists(resolved_params_file_path) and os.path.isfile(resolved_params_file_path)): # 避免重复日志
                 logger.error(f"[{self.strategy_name}] 最终确认：参数文件 '{resolved_params_file_path}' (原始输入: '{params_file}') 不存在或不是文件。将使用空参数。")

        # 步骤2: 调用父类的 __init__ 方法，并传入加载的参数字典
        # BaseStrategy 的 __init__ 会将 loaded_params 赋值给 self.params
        super().__init__(params=loaded_params)

        # 步骤3: 初始化 TrendFollowingStrategy 特有的属性
        # self.params 现在应该已经被 BaseStrategy 设置好了 (可能是 loaded_params，也可能是空字典)
        
        # 从 self.params 中获取策略名，如果参数文件中有定义的话
        # 注意：BaseStrategy 的 self.strategy_name 默认是 "Base Strategy"
        # 子类应该在 super().__init__() 之后，用加载的参数覆盖它
        self.strategy_name = self.params.get('trend_following_strategy_name', TrendFollowingStrategy.strategy_name)
        
        # 记录参数加载状态
        if not self.params: # 如果 self.params 仍然是空字典 (加载失败)
            logger.error(f"[{self.strategy_name}] 策略参数 (self.params) 为空。这意味着从 '{resolved_params_file_path}' 加载参数失败或文件为空。")
        
        self.base_data_dir = base_data_dir

        # --- 加载趋势跟踪特定参数 (trend_following_params) ---
        # 如果 self.params 为空，self.tf_params 将为 {}
        self.tf_params: Dict[str, Any] = self.params.get('trend_following_params', {})
        if 'trend_following_params' not in self.params and self.params: # 仅当 self.params 非空但缺少键时警告
             logger.warning(f"[{self.strategy_name}] 策略参数 (self.params) 已加载，但顶层缺少 'trend_following_params' 键。 "
                            f"请检查参数文件 '{resolved_params_file_path}' 的内容结构。 "
                            f"已加载的顶层参数键: {list(self.params.keys())}")
        elif not self.params: # 如果 self.params 本身就是空的
             logger.warning(f"[{self.strategy_name}] 由于策略参数 (self.params) 未能加载，'trend_following_params' 将会缺失。")


        # 策略主要关注的时间框架
        self.focus_timeframe: str = str(self.tf_params.get('focus_timeframe', self.default_focus_timeframe))
        self.timeframe_weights: Optional[Dict[str, float]] = self.tf_params.get('timeframe_weights', None)
        self.trend_indicators: List[str] = self.tf_params.get('trend_indicators', ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi'])
        self.rule_signal_weights: Dict[str, float] = self.tf_params.get('rule_signal_weights', {
            'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15,
            'ema_cross': 0.15, 'boll_breakout': 0.1, 'adx_strength': 0.1, 'vwap_deviation': 0.05,
            'volume_spike': 0.05
        })

        vc_global_params = self.params.get('volume_confirmation', {})
        self.volume_boost_factor: float = self.tf_params.get('volume_boost_factor', vc_global_params.get('boost_factor', 1.2))
        self.volume_penalty_factor: float = self.tf_params.get('volume_penalty_factor', vc_global_params.get('penalty_factor', 0.8))
        self.volume_spike_threshold: float = self.tf_params.get('volume_spike_threshold', 2.0)

        self.volatility_threshold_high: float = self.tf_params.get('volatility_threshold_high', 10.0)
        self.volatility_threshold_low: float = self.tf_params.get('volatility_threshold_low', 5.0)
        self.volatility_adjust_factor: float = 1.0

        self.adx_strong_threshold: int = self.tf_params.get('adx_strong_threshold', 30)
        self.adx_moderate_threshold: int = self.tf_params.get('adx_moderate_threshold', 20)
        # 注意：JSON 文件中的 trend_duration_threshold_strong 是 5, moderate 是 10
        # 代码中的默认值是 3 和 5。如果参数文件加载成功，会使用文件中的值。
        self.trend_duration_threshold_strong: int = self.tf_params.get('trend_duration_threshold_strong', 3)
        self.trend_duration_threshold_moderate: int = self.tf_params.get('trend_duration_threshold_moderate', 5)
        self.stoch_oversold_threshold: int = self.tf_params.get('stoch_oversold_threshold', 20)
        self.stoch_overbought_threshold: int = self.tf_params.get('stoch_overbought_threshold', 80)
        self.vwap_deviation_threshold: float = self.tf_params.get('vwap_deviation_threshold', 0.01)
        self.trend_confirmation_periods: int = self.tf_params.get('trend_confirmation_periods', 3)

        # --- Transformer 模型相关配置 ---
        self.transformer_window_size: int = self.tf_params.get('transformer_window_size', 60)
        self.transformer_batch_size: int = self.tf_params.get('transformer_batch_size', 128) # 实际会被 training_config 中的覆盖
        self.transformer_target_column: str = self.tf_params.get('transformer_target_column', 'final_rule_signal')

        self.transformer_model_config: Dict[str, Any] = self.tf_params.get('transformer_model_config', {
            'd_model': 128, 'nhead': 8, 'dim_feedforward': 512, 'nlayers': 4, 'dropout': 0.2, 'activation': 'relu',
        })
        self.transformer_training_config: Dict[str, Any] = self.tf_params.get('transformer_training_config', {
            'epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001, 'weight_decay': 0.004,
            'optimizer': 'adamw', 'loss': 'mse', 'early_stopping_patience': 30, 'reduce_lr_patience': 10,
            'reduce_lr_factor': 0.5, 'monitor_metric': 'val_loss', 'verbose': 1,
            'tensorboard_log_dir': None, 'clip_grad_norm': 1.0
        })

        # 确保训练配置中的参数优先
        if 'learning_rate' in self.transformer_training_config:
            self.transformer_model_config['learning_rate'] = self.transformer_training_config['learning_rate']
        if 'weight_decay' in self.transformer_training_config:
            self.transformer_model_config['weight_decay'] = self.transformer_training_config['weight_decay']
        if 'batch_size' in self.transformer_training_config:
             self.transformer_batch_size = self.transformer_training_config['batch_size']

        self.transformer_data_prep_config: Dict[str, Any] = self.tf_params.get('transformer_data_prep_config', {
            'scaler_type': 'standard', 'train_split': 0.7, 'val_split': 0.15,
            'apply_variance_threshold': False, 'variance_threshold_value': 0.01,
            'use_pca': False, 'pca_n_components': 0.99, 'pca_solver': 'auto',
            'use_feature_selection': True, 'feature_selector_model_type': 'rf',
            'fs_model_n_estimators': 100, 'fs_model_max_depth': None, 'fs_max_features': 50,
            'fs_selection_threshold': 'median', 'target_scaler_type': 'minmax'
        })

        # --- 初始化模型和scaler相关属性 ---
        self.transformer_model: Optional[nn.Module] = None
        self.feature_scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
        self.target_scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
        self.selected_feature_names_for_transformer: List[str] = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[{self.strategy_name}] PyTorch 使用设备: {self.device}")

        self.model_path: Optional[str] = None
        self.feature_scaler_path: Optional[str] = None
        self.target_scaler_path: Optional[str] = None
        self.selected_features_path: Optional[str] = None
        self.all_prepared_data_npz_path: Optional[str] = None # 确保这个也被初始化

        self.intermediate_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[Dict[str, Any]] = None # 初始化分析结果属性

        if ta is None:
             logger.error(f"[{self.strategy_name}] pandas_ta 未成功加载，策略部分功能可能不可用。")

        # _validate_params 会在 BaseStrategy 的 __init__ 中被调用
        # 如果需要在此之后再次验证（例如基于 TrendFollowingStrategy 特有的属性），可以再次调用
        # self._validate_params() # 或者在 BaseStrategy 中确保它在最后被调用
        # 但根据您提供的 BaseStrategy，它是在 self.params 赋值后调用的，所以应该已经执行过了。
        # 我们这里可以再调用一次，以确保 TrendFollowingStrategy 特有的参数也被验证。
        # 或者，将 TrendFollowingStrategy 的验证逻辑放入其自己的 _validate_specific_params 方法中。
        # 为简单起见，我们依赖 BaseStrategy 调用 _validate_params，
        # 并且 TrendFollowingStrategy 的 _validate_params 会调用 super()._validate_params()。

        logger.info(f"策略 '{self.strategy_name}' 初始化完成。主要关注时间框架: {self.focus_timeframe}。参数加载自: '{resolved_params_file_path if self.params else '无或加载失败'}'")


    def _normalize_weights(self, weights: Dict[str, float]):
        """归一化权重字典，使其总和为1。"""
        total_weight = sum(weights.values())
        if total_weight > 0 and not np.isclose(total_weight, 1.0):
            for key in weights:
                weights[key] /= total_weight
        elif total_weight == 0:
             logger.warning(f"[{self.strategy_name}] 权重总和为零，无法归一化。权重字典: {weights}")

    def _validate_params(self):
        """
        验证策略特定参数的有效性。
        这个方法会由 BaseStrategy 的 __init__ 调用。
        """
        super()._validate_params() # 调用父类的验证 (在您提供的 BaseStrategy 中是 pass)

        # 核心检查：'trend_following_params' 是否存在于已加载的 self.params 中
        if 'trend_following_params' not in self.params:
            logger.warning(f"[{self.strategy_name}] _validate_params: 策略参数 (self.params) 中缺少 'trend_following_params' 部分，将使用大量默认值。 "
                           f"请检查参数文件是否正确加载且包含此顶级键。已加载的顶层键: {list(self.params.keys()) if self.params else 'None'}")
        elif not self.tf_params: # 如果 'trend_following_params' 键存在，但其对应的值是空字典
            logger.warning(f"[{self.strategy_name}] _validate_params: 'trend_following_params' 部分在参数中存在，但其内容为空。将使用大量默认值。")

        bs_params = self.params.get('base_scoring', {})
        if not bs_params.get('timeframes'):
            logger.error(f"[{self.strategy_name}] 参数 'base_scoring.timeframes' 未定义或为空，无法确定操作时间级别。")
        elif self.focus_timeframe not in bs_params.get('timeframes', []): # 添加空列表作为 get 的默认值
            logger.warning(f"[{self.strategy_name}] 主要关注时间框架 '{self.focus_timeframe}' 不在 'base_scoring.timeframes' ({bs_params.get('timeframes', [])}) 中。可能导致错误。")

        if self.timeframe_weights:
            if not isinstance(self.timeframe_weights, dict):
                logger.error(f"[{self.strategy_name}] 'trend_following_params.timeframe_weights' 必须是一个字典，但得到的是: {type(self.timeframe_weights)}")

        if not self.trend_indicators:
            logger.warning(f"[{self.strategy_name}] 'trend_following_params.trend_indicators' 为空，策略可能无法有效识别趋势。")

        model_conf = self.transformer_model_config
        required_model_keys = ['d_model', 'nhead', 'dim_feedforward', 'nlayers']
        if not all(key in model_conf for key in required_model_keys):
             logger.warning(f"[{self.strategy_name}] Transformer模型结构配置 'transformer_model_config' 缺少关键参数: {required_model_keys}。将使用默认值。当前配置: {model_conf}")

        train_conf = self.transformer_training_config
        required_train_keys = ['epochs', 'batch_size', 'learning_rate', 'loss']
        if not all(key in train_conf for key in required_train_keys):
             logger.warning(f"[{self.strategy_name}] Transformer训练配置 'transformer_training_config' 缺少关键参数: {required_train_keys}。将使用默认值。当前配置: {train_conf}")

        if not isinstance(self.rule_signal_weights, dict) or not self.rule_signal_weights:
             logger.warning(f"[{self.strategy_name}] 'rule_signal_weights' 参数无效或为空，将使用默认权重。当前值: {self.rule_signal_weights}")
             self.rule_signal_weights = { # 确保使用代码中定义的默认值
                'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15,
                'ema_cross': 0.15, 'boll_breakout': 0.1, 'adx_strength': 0.1, 'vwap_deviation': 0.05,
                'volume_spike': 0.05
            }
        self._normalize_weights(self.rule_signal_weights)

        # signal_combination_weights 来自 self.tf_params
        # 如果 tf_params 为空 (因为 trend_following_params 未加载或为空), lstm_combination_weights 会是 {}
        lstm_combination_weights = self.tf_params.get('signal_combination_weights', {})
        if not isinstance(lstm_combination_weights, dict) or not lstm_combination_weights:
             logger.warning(f"[{self.strategy_name}] 'signal_combination_weights' 参数无效或为空，将使用默认权重 0.6/0.4 (来自JSON) 或 0.7/0.3 (代码默认)。当前值: {lstm_combination_weights}")
             # 使用一个明确的默认值进行归一化，如果参数缺失
             default_combo_weights = {'rule_weight': 0.6, 'lstm_weight': 0.4} # 与JSON中默认值保持一致
             self._normalize_weights(default_combo_weights)
             # 如果希望在 self.tf_params 中也反映这个默认值（如果它可写且代表实际配置）
             # self.tf_params['signal_combination_weights'] = default_combo_weights # 谨慎操作，取决于 tf_params 的来源
        else:
            self._normalize_weights(lstm_combination_weights)
            # 如果 self.tf_params['signal_combination_weights'] 是同一个对象，它已经被归一化
            # 否则，如果它存在且不同，也归一化它
            if 'signal_combination_weights' in self.tf_params and self.tf_params['signal_combination_weights'] is not lstm_combination_weights:
                self._normalize_weights(self.tf_params['signal_combination_weights'])


        logger.debug(f"[{self.strategy_name}] 特定参数验证完成。")

    def set_model_paths(self, stock_code: str):
        """
        为特定股票设置模型、scaler 和准备好的数据的保存/加载路径。
        """
        stock_root_dir =  os.path.join(self.base_data_dir, stock_code)
        os.makedirs(stock_root_dir, exist_ok=True) # 确保股票根目录存在

        prepared_data_dir = os.path.join(stock_root_dir, "prepared_data")
        os.makedirs(prepared_data_dir, exist_ok=True)
        
        trained_model_dir = os.path.join(stock_root_dir, "trained_model")
        os.makedirs(trained_model_dir, exist_ok=True)

        self.model_path = os.path.join(trained_model_dir, "trend_following_transformer_weights.pth")
        self.feature_scaler_path = os.path.join(prepared_data_dir, "trend_following_transformer_feature_scaler.save")
        self.target_scaler_path = os.path.join(prepared_data_dir, "trend_following_transformer_target_scaler.save")
        self.selected_features_path = os.path.join(prepared_data_dir, "trend_following_transformer_selected_features.json")
        self.all_prepared_data_npz_path = os.path.join(prepared_data_dir, "all_prepared_data_transformer.npz")

        logger.debug(f"[{self.strategy_name}] 为股票 {stock_code} 设置文件路径:")
        logger.debug(f"  模型权重: {self.model_path}")
        logger.debug(f"  特征Scaler: {self.feature_scaler_path}")
        logger.debug(f"  目标Scaler: {self.target_scaler_path}")
        logger.debug(f"  准备数据NPZ: {self.all_prepared_data_npz_path}")
        logger.debug(f"  选中特征: {self.selected_features_path}")

    # ... (train_transformer_model_from_prepared_data 方法保持不变) ...
    def train_transformer_model_from_prepared_data(self, stock_code: str):
        """
        为特定股票加载已准备好的数据，构建并训练 Transformer 模型，然后保存模型权重。
        """
        self.set_model_paths(stock_code)
        logger.info(f"[{self.strategy_name}] 开始为股票 {stock_code} 训练 Transformer 模型 (从已准备数据加载)...")

        features_scaled_train_np, targets_scaled_train_np, \
        features_scaled_val_np, targets_scaled_val_np, \
        features_scaled_test_np, targets_scaled_test_np, \
        feature_scaler, target_scaler = self.load_prepared_data(stock_code)

        if features_scaled_train_np.shape[0] == 0 or targets_scaled_train_np.shape[0] == 0 or \
           feature_scaler is None or target_scaler is None or not self.selected_feature_names_for_transformer:
            logger.error(f"[{self.strategy_name}] 股票 {stock_code} 加载已准备好的数据/Scaler/特征列表失败或数据无效，无法继续训练。")
            self.transformer_model = None # 明确置空
            self.feature_scaler = None
            self.target_scaler = None
            self.selected_feature_names_for_transformer = []
            return

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        num_features = features_scaled_train_np.shape[1]
        logger.info(f"[{self.strategy_name}][{stock_code}] 最终用于训练的平坦数据集 shape: train_features={features_scaled_train_np.shape}, train_targets={targets_scaled_train_np.shape}, "
                    f"val_features={features_scaled_val_np.shape}, val_targets={targets_scaled_val_np.shape}, "
                    f"test_features={features_scaled_test_np.shape}, test_targets={targets_scaled_test_np.shape}")
        logger.info(f"[{self.strategy_name}][{stock_code}] 实际用于训练的特征维度: {num_features}")

        try:
            train_dataset = TimeSeriesDataset(features_scaled_train_np, targets_scaled_train_np, self.transformer_window_size)
            val_loader = None
            if features_scaled_val_np.shape[0] > 0 and targets_scaled_val_np.shape[0] > 0:
                val_dataset = TimeSeriesDataset(features_scaled_val_np, targets_scaled_val_np, self.transformer_window_size)
                if len(val_dataset) > 0:
                    val_loader = DataLoader(val_dataset, batch_size=self.transformer_batch_size, shuffle=False)
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 验证集 Dataset 为空。验证阶段将跳过。")
            
            test_loader = None
            if features_scaled_test_np.shape[0] > 0 and targets_scaled_test_np.shape[0] > 0:
                test_dataset = TimeSeriesDataset(features_scaled_test_np, targets_scaled_test_np, self.transformer_window_size)
                if len(test_dataset) > 0:
                    test_loader = DataLoader(test_dataset, batch_size=self.transformer_batch_size, shuffle=False)
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 测试集 Dataset 为空。测试评估将跳过。")

            if len(train_dataset) == 0:
                logger.error(f"[{self.strategy_name}][{stock_code}] 训练集 Dataset 为空。停止训练。")
                return
            train_loader = DataLoader(train_dataset, batch_size=self.transformer_batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 创建 PyTorch Dataset/DataLoader 出错: {e}", exc_info=True)
            return

        try:
            model = build_transformer_model(
                num_features=num_features,
                model_config=self.transformer_model_config,
                summary=True,
                window_size=self.transformer_window_size
            )
            self.transformer_model = model
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 构建 Transformer 模型出错: {e}", exc_info=True)
            self.transformer_model = None
            return

        try:
            # 确保 checkpoint_dir 存在
            checkpoint_dir = os.path.dirname(self.model_path) if self.model_path else None
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                logger.info(f"[{self.strategy_name}][{stock_code}] 创建模型保存目录: {checkpoint_dir}")

            self.transformer_model, history_df = train_transformer_model(
                model=self.transformer_model,
                train_loader=train_loader,
                val_loader=val_loader,
                target_scaler=self.target_scaler,
                training_config=self.transformer_training_config,
                checkpoint_dir=checkpoint_dir, # 使用 self.model_path 的目录
                stock_code=stock_code,
                plot_training_history=self.tf_params.get('transformer_plot_history', False),
            )
            if self.model_path: # 只有当 model_path 有效时才记录
                 logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型训练完成，最佳模型权重已保存到 {self.model_path}")
            else:
                 logger.warning(f"[{self.strategy_name}][{stock_code}] Transformer 模型训练完成，但 model_path 未设置，模型权重可能未按预期保存。")


            if test_loader is not None and len(test_loader) > 0 and self.transformer_model is not None:
                logger.info(f"[{self.strategy_name}] 开始在测试集上评估股票 {stock_code} 的 Transformer 模型...")
                # 确保损失函数和评估指标与训练时一致或兼容
                loss_fn_name = self.transformer_training_config.get('loss', 'mse').lower()
                criterion_eval = nn.MSELoss() if loss_fn_name == 'mse' else \
                                 nn.L1Loss() if loss_fn_name == 'mae' else \
                                 nn.HuberLoss() if loss_fn_name == 'huber' else nn.MSELoss() # 默认MSE
                mae_metric_eval = nn.L1Loss()

                test_metrics = evaluate_transformer_model(
                    model=self.transformer_model,
                    test_loader=test_loader,
                    criterion=criterion_eval,
                    mae_metric=mae_metric_eval,
                    target_scaler=self.target_scaler,
                    device=self.device
                )
                logger.info(f"[{self.strategy_name}][{stock_code}] 测试集评估结果: {test_metrics}")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 训练 Transformer 模型出错: {e}", exc_info=True)
            self.transformer_model = None


    def get_required_columns(self) -> List[str]:
        """
        根据策略参数，动态生成并返回 IndicatorService 需要准备的所有数据列名。
        """
        required = set()
        # 确保 self.params 存在且非空，否则 bs_params 将是空字典，可能导致后续逻辑问题
        if not self.params:
            logger.error(f"[{self.strategy_name}] get_required_columns: 策略参数 (self.params) 为空，无法确定所需列。")
            return []
            
        bs_params = self.params.get('base_scoring', {})
        timeframes = bs_params.get('timeframes', [])
        if not timeframes:
            logger.error(f"[{self.strategy_name}] 无法获取所需列，因为 'base_scoring.timeframes' 未定义或为空。")
            return []

        # 1. 基础OHLCV
        for tf_str in timeframes:
            for col_base in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                required.add(f"{col_base}_{tf_str}")

        # 2. 基础评分指标
        score_indicators_config = bs_params.get('score_indicators', [])
        macd_fast = bs_params.get('macd_fast', 12)
        macd_slow = bs_params.get('macd_slow', 26)
        macd_sig = bs_params.get('macd_signal', 9)
        rsi_period = bs_params.get('rsi_period', 14)
        kdj_k = bs_params.get('kdj_period_k', 9)
        kdj_d = bs_params.get('kdj_period_d', 3)
        kdj_j = bs_params.get('kdj_period_j', 3)
        boll_period = bs_params.get('boll_period', 20)
        boll_std_dev = bs_params.get('boll_std_dev', 2.0)
        boll_std_str = f"{boll_std_dev:.1f}"
        cci_period = bs_params.get('cci_period', 14)
        mfi_period = bs_params.get('mfi_period', 14)
        roc_period = bs_params.get('roc_period', 12)
        dmi_period = bs_params.get('dmi_period', 14)
        sar_step = bs_params.get('sar_step', 0.02)
        sar_max_af = bs_params.get('sar_max', 0.2)
        ema_p_base = bs_params.get('ema_params',{}).get('period', 20) # 基础EMA参数
        sma_p_base = bs_params.get('sma_params',{}).get('period', 20) # 基础SMA参数

        for indi_key in score_indicators_config:
            for tf_str in timeframes:
                if indi_key == 'macd':
                    required.add(f'MACD_{macd_fast}_{macd_slow}_{macd_sig}_{tf_str}')
                    required.add(f'MACDh_{macd_fast}_{macd_slow}_{macd_sig}_{tf_str}')
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
                    required.add(f'PLUS_DI_{dmi_period}_{tf_str}')
                    required.add(f'MINUS_DI_{dmi_period}_{tf_str}')
                    required.add(f'ADX_{dmi_period}_{tf_str}')
                elif indi_key == 'sar':
                    sar_step_str = f"{sar_step:.2f}".rstrip('0').rstrip('.')
                    sar_max_af_str = f"{sar_max_af:.2f}".rstrip('0').rstrip('.')
                    required.add(f'SAR_{sar_step_str}_{sar_max_af_str}_{tf_str}')
                elif indi_key == 'ema': # 基础评分中的EMA
                    required.add(f'EMA_{ema_p_base}_{tf_str}')
                elif indi_key == 'sma': # 基础评分中的SMA
                    required.add(f'SMA_{sma_p_base}_{tf_str}')
        
        # 3. 量能确认指标
        vc_params = self.params.get('volume_confirmation', {})
        if vc_params.get('enabled', False) or vc_params.get('volume_analysis_enabled', False): # 兼容旧参数名
            vc_tf_list_raw = vc_params.get('tf', self.focus_timeframe)
            vc_tf_list = [vc_tf_list_raw] if isinstance(vc_tf_list_raw, str) else vc_tf_list_raw
            
            amount_ma_period = vc_params.get('amount_ma_period',10)
            cmf_period = vc_params.get('cmf_period',20)
            for vc_tf_str in vc_tf_list:
                if vc_tf_str not in timeframes:
                    logger.warning(f"[{self.strategy_name}] 量能确认时间框架 '{vc_tf_str}' 未在 'base_scoring.timeframes' 中定义。")
                    continue
                required.add(f"AMT_MA_{amount_ma_period}_{vc_tf_str}")
                required.add(f"CMF_{cmf_period}_{vc_tf_str}")
        
        # 4. 其他分析指标
        ia_params = self.params.get('indicator_analysis_params', {})
        ia_timeframes = bs_params.get('timeframes', []) # 通常与基础时间框架一致

        stoch_k_ia = ia_params.get('stoch_k', 9) # JSON 中是 14
        stoch_d_ia = ia_params.get('stoch_d', 3)
        stoch_smooth_k_ia = ia_params.get('stoch_smooth_k', 3)
        vol_ma_period_ia = ia_params.get('volume_ma_period', 20)
        vwap_anchor_ia = ia_params.get('vwap_anchor', None) # 假设 IndicatorService 会处理 None
        calculate_adl_ia = ia_params.get('calculate_adl', True)
        ichimoku_enabled_ia = ia_params.get('calculate_ichimoku', True)
        ichimoku_tenkan_ia = ia_params.get('ichimoku_tenkan', 9)
        ichimoku_kijun_ia = ia_params.get('ichimoku_kijun', 26)
        ichimoku_senkou_ia = ia_params.get('ichimoku_senkou', 52)
        calculate_pivot_points_ia = ia_params.get('calculate_pivot_points', True)

        for tf_str_ia in ia_timeframes:
            required.add(f"STOCHk_{stoch_k_ia}_{stoch_d_ia}_{stoch_smooth_k_ia}_{tf_str_ia}")
            required.add(f"STOCHd_{stoch_k_ia}_{stoch_d_ia}_{stoch_smooth_k_ia}_{tf_str_ia}")
            required.add(f"VOL_MA_{vol_ma_period_ia}_{tf_str_ia}")
            vwap_col_name_ia = 'VWAP' if vwap_anchor_ia is None else f'VWAP_{vwap_anchor_ia}'
            required.add(f"{vwap_col_name_ia}_{tf_str_ia}")
            if calculate_adl_ia:
                required.add(f"ADL_{tf_str_ia}") # ADL 通常无参数
            if ichimoku_enabled_ia:
                required.add(f"TENKAN_{ichimoku_tenkan_ia}_{tf_str_ia}")
                required.add(f"KIJUN_{ichimoku_kijun_ia}_{tf_str_ia}")
                required.add(f"CHIKOU_{ichimoku_kijun_ia}_{tf_str_ia}") # CHIKOU 通常用 kijun 周期或 senkou_b 周期
                required.add(f"SENKOU_A_{ichimoku_tenkan_ia}_{ichimoku_kijun_ia}_{tf_str_ia}")
                required.add(f"SENKOU_B_{ichimoku_senkou_ia}_{tf_str_ia}")
            if calculate_pivot_points_ia and tf_str_ia == 'D': # Pivot Points 通常基于日线
                required.add("PP_D") # 假设 IndicatorService 知道如何处理这个
                for i in range(1, 5): required.add(f"S{i}_D"); required.add(f"R{i}_D")
                for i in range(1, 4): required.add(f"F_S{i}_D"); required.add(f"F_R{i}_D")

        # 5. 特征工程产生的指标
        fe_params = self.params.get('feature_engineering_params', {})
        fe_tf_list = fe_params.get('apply_on_timeframes', timeframes)
        atr_period_fe = fe_params.get('atr_params',{}).get('period',14)
        hv_period_fe = fe_params.get('hv_params',{}).get('period',20)
        kc_ema_period_fe = fe_params.get('kc_params',{}).get('ema_period',20)
        kc_atr_period_fe = fe_params.get('kc_params',{}).get('atr_period',10)
        mom_period_fe = fe_params.get('mom_params',{}).get('period',10)
        willr_period_fe = fe_params.get('willr_params',{}).get('period',14)
        vroc_period_fe = fe_params.get('vroc_params',{}).get('period',10)
        aroc_period_fe = fe_params.get('aroc_params',{}).get('period',10)
        ema_periods_fe = fe_params.get('ema_periods', []) # 例如 [5, 10, 20, 60]
        sma_periods_fe = fe_params.get('sma_periods', [])
        ema_periods_rel_fe = fe_params.get('ema_periods_for_relation', [])
        sma_periods_rel_fe = fe_params.get('sma_periods_for_relation', [])
        indicators_to_diff_fe = fe_params.get('indicators_for_difference', [])
        diff_periods_fe = fe_params.get('difference_periods', [1])

        for tf_str_fe in fe_tf_list:
            if tf_str_fe not in timeframes: continue
            if fe_params.get('calculate_atr', True): required.add(f"ATR_{atr_period_fe}_{tf_str_fe}")
            if fe_params.get('calculate_hv', True): required.add(f"HV_{hv_period_fe}_{tf_str_fe}")
            if fe_params.get('calculate_kc', True):
                required.add(f"KCL_{kc_ema_period_fe}_{kc_atr_period_fe}_{tf_str_fe}")
                required.add(f"KCM_{kc_ema_period_fe}_{kc_atr_period_fe}_{tf_str_fe}")
                required.add(f"KCU_{kc_ema_period_fe}_{kc_atr_period_fe}_{tf_str_fe}")
            if fe_params.get('calculate_mom', True): required.add(f"MOM_{mom_period_fe}_{tf_str_fe}")
            if fe_params.get('calculate_willr', True): required.add(f"WILLR_{willr_period_fe}_{tf_str_fe}")
            if fe_params.get('calculate_vroc', True): required.add(f"VROC_{vroc_period_fe}_{tf_str_fe}")
            if fe_params.get('calculate_aroc', True): required.add(f"AROC_{aroc_period_fe}_{tf_str_fe}")

            for p_fe in ema_periods_fe: required.add(f"EMA_{p_fe}_{tf_str_fe}")
            for p_fe in sma_periods_fe: required.add(f"SMA_{p_fe}_{tf_str_fe}")

            for ma_type_deriv_fe, periods_list in [('EMA', ema_periods_rel_fe), ('SMA', sma_periods_rel_fe)]:
                 for p_deriv_fe in periods_list:
                     required.add(f'CLOSE_{ma_type_deriv_fe}_RATIO_{p_deriv_fe}_{tf_str_fe}')
                     required.add(f'CLOSE_{ma_type_deriv_fe}_NDIFF_{p_deriv_fe}_{tf_str_fe}')
            
            # 指标差分 (这里逻辑比较复杂，需要精确匹配 IndicatorService 的命名)
            # 假设 IndicatorService 会根据参数生成原始指标列，然后我们在这里请求差分列
            for indi_diff_conf in indicators_to_diff_fe:
                base_name = indi_diff_conf['base_name'].upper() # 如 RSI, MACD, K, D, J, ADX
                param_keys = indi_diff_conf.get('params_key', [])
                default_periods = indi_diff_conf.get('default_period', [])
                
                param_values_for_col = []
                # 从 bs_params, ia_params, fe_params 中查找参数值
                # 注意：K,D,J 的参数键 kdj_period_k, kdj_period_d, kdj_period_j 在 bs_params 中
                # RSI 的 rsi_period 在 bs_params 中
                # MACD 的 macd_fast, macd_slow, macd_signal 在 bs_params 中
                # ADX 的 dmi_period 在 bs_params 中
                param_source_map = {**bs_params, **ia_params, **fe_params}

                valid_params = True
                for i, p_key in enumerate(param_keys):
                    val = param_source_map.get(p_key, default_periods[i] if i < len(default_periods) else None)
                    if val is None:
                        logger.warning(f"[{self.strategy_name}] 指标差分 '{base_name}' 的参数 '{p_key}' 未找到且无默认值。跳过此差分。")
                        valid_params = False
                        break
                    param_values_for_col.append(str(val).replace('.0','')) # 简化浮点数表示
                
                if not valid_params: continue

                param_suffix = ("_" + "_".join(param_values_for_col)) if param_values_for_col else ""
                
                # 特殊处理多输出指标的差分列名
                # 例如 KDJ(k,d,j) 会生成 K_params, D_params, J_params
                # MACD 会生成 MACD_params, MACDh_params, MACDs_params
                # 我们需要为每个原始输出列生成差分列请求
                original_output_prefixes = []
                if base_name == "KDJ": # JSON 中配置的是 K, D, J 分开的
                    original_output_prefixes = [base_name] # 因为 JSON 中 K,D,J 是分开的 base_name
                elif base_name == "MACD":
                    original_output_prefixes = ["MACD", "MACDh", "MACDs"]
                elif base_name == "RSI":
                    original_output_prefixes = ["RSI"]
                elif base_name == "ADX":
                    original_output_prefixes = ["ADX"] # DMI 还会产生 PDI, MDI，但这里只差分 ADX
                else: # 其他单输出指标
                    original_output_prefixes = [base_name]

                for prefix in original_output_prefixes:
                    for diff_p in diff_periods_fe:
                        # 假设 IndicatorService 生成的差分列名格式为: PREFIX_PARAMS_DIFF_PERIOD_TF
                        # 例如: K_7_3_3_DIFF1_30
                        # 或 MACD_8_26_7_DIFF1_30
                        required.add(f"{prefix}{param_suffix}_DIFF{diff_p}_{tf_str_fe}")
            
            # 价格在通道中位置
            # boll_period, boll_std_str 来自基础评分参数
            # kc_ema_period_fe, kc_atr_period_fe 来自特征工程参数
            required.add(f'CLOSE_BB_POS_{boll_period}_{boll_std_str}_{tf_str_fe}')
            required.add(f'CLOSE_KC_POS_{kc_ema_period_fe}_{kc_atr_period_fe}_{tf_str_fe}')

        # 6. K线形态 (如果启用，IndicatorService 应负责计算所有可能的形态列)
        # 这里不需要列出具体形态名，依赖后续特征选择
        kp_params = self.params.get('kline_pattern_detection', {})
        if kp_params.get('enabled', False):
            # 可以添加一个通用标记或不添加，因为特征选择会处理
            pass 

        # 确保 OBV (无参数) 被包含 (通常在所有时间级别计算)
        for tf_str_basic_vol in timeframes:
            required.add(f"OBV_{tf_str_basic_vol}")
            # ADL 已在 IA 部分处理

        final_columns = sorted(list(required))
        logger.info(f"[{self.strategy_name}] 策略共需要 {len(final_columns)} 个数据列。")
        logger.debug(f"[{self.strategy_name}] 所需列名 (部分): {final_columns[:20]}...")
        return final_columns

    def _calculate_rule_based_signal(self, data: pd.DataFrame, stock_code: str, indicator_configs: List[Dict]) -> Tuple[pd.Series, Dict]:
        """
        计算基于规则的信号。
        """
        if data is None or data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 输入数据为空，无法生成规则信号。")
            return pd.Series(dtype=float), {}

        # --- 检查必需列 ---
        # 确保 self.params 存在且非空
        if not self.params:
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算：策略参数 (self.params) 为空。")
            return pd.Series(50.0, index=data.index), {} # 返回中性信号

        bs_params = self.params.get('base_scoring', {})
        vc_params = self.params.get('volume_confirmation', {})
        ia_params = self.params.get('indicator_analysis_params', {}) # 虽然未使用，但保持结构
        dd_params = self.params.get('divergence_detection', {})

        focus_tf = self.focus_timeframe # 已在 __init__ 中设置
        critical_cols = [f'close_{focus_tf}', f'volume_{focus_tf}']
        # 简单检查是否有评分指标列 (更鲁棒的检查应基于 indicator_configs)
        if bs_params.get('score_indicators'):
            first_score_indi_key = bs_params['score_indicators'][0]
            # 尝试找到一个与该指标和焦点时间框架相关的列
            # 这只是一个启发式检查，实际列名可能更复杂
            potential_score_col_pattern = f"{first_score_indi_key.upper()}" # 例如 "MACD"
            found_critical_indi_col = False
            for col_name_data in data.columns:
                if potential_score_col_pattern in col_name_data and f"_{focus_tf}" in col_name_data:
                    critical_cols.append(col_name_data)
                    found_critical_indi_col = True
                    break
            if not found_critical_indi_col:
                 logger.warning(f"[{self.strategy_name}][{stock_code}] 未能在数据中找到与首个评分指标 '{first_score_indi_key}' 和焦点时间框架 '{focus_tf}' 相关的列。")


        missing_critical_cols = [col for col in critical_cols if col not in data.columns or data[col].isnull().all()]
        if missing_critical_cols:
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算缺少关键输入列或数据全为 NaN: {missing_critical_cols}。")
            return pd.Series(50.0, index=data.index), {}

        self._adjust_volatility_parameters(data) # 动态调整波动率参数

        # --- 计算所有配置的指标评分 ---
        # strategy_utils.calculate_all_indicator_scores 需要 data, bs_params, indicator_configs
        # 确保 indicator_configs 被正确传递和使用
        indicator_scores_df = strategy_utils.calculate_all_indicator_scores(data, bs_params, indicator_configs)

        # --- 计算多时间框架加权的基础评分 ---
        current_weights: Dict[str, float]
        timeframes_from_config = bs_params.get('timeframes', [])
        if not timeframes_from_config: # 防御空列表
            logger.error(f"[{self.strategy_name}][{stock_code}] 'base_scoring.timeframes' 为空，无法计算基础评分。")
            return pd.Series(50.0, index=data.index), {}

        if self.timeframe_weights: # 使用 trend_following_params 中的 timeframe_weights
            current_weights = self.timeframe_weights.copy()
            # 清理权重中不在配置时间框架内的条目，并为缺失的配置时间框架补零权重
            defined_tfs_set = set(timeframes_from_config)
            for tf_w in list(current_weights.keys()): # 使用 list 转换以允许在迭代时删除
                if tf_w not in defined_tfs_set:
                    del current_weights[tf_w]
            for tf_d in defined_tfs_set:
                if tf_d not in current_weights:
                    current_weights[tf_d] = 0.0 # 确保所有在 config 中的 timeframe 都有权重
        else: # 如果 trend_following_params.timeframe_weights 未提供，则使用 focus_weight 计算
            focus_weight_val = self.tf_params.get('focus_weight', 0.45) # 来自 trend_following_params
            num_other_tfs = len(timeframes_from_config) - 1
            if num_other_tfs > 0:
                base_weight_val = (1.0 - focus_weight_val) / num_other_tfs
            elif len(timeframes_from_config) == 1: # 只有一个时间框架
                base_weight_val = 0.0
                focus_weight_val = 1.0 # 焦点权重应为1
            else: # 没有时间框架 (理论上不应发生，因为上面有检查)
                base_weight_val = 0.0
                focus_weight_val = 0.0
            
            current_weights = {tf: base_weight_val for tf in timeframes_from_config if tf != self.focus_timeframe}
            current_weights[self.focus_timeframe] = focus_weight_val
        
        self._normalize_weights(current_weights) # 归一化权重

        base_score_raw = pd.Series(0.0, index=data.index)
        total_effective_weight = 0.0

        for tf_s in timeframes_from_config:
            tf_weight = current_weights.get(tf_s, 0)
            if tf_weight == 0: continue

            # 查找该时间框架下的所有 SCORE_INDICATOR_TF 列
            tf_score_cols = [col for col in indicator_scores_df.columns if col.endswith(f'_{tf_s}') and col.startswith('SCORE_')]
            if tf_score_cols:
                tf_average_score = indicator_scores_df[tf_score_cols].mean(axis=1).fillna(50.0)
                base_score_raw = base_score_raw.add(tf_average_score * tf_weight, fill_value=0.0) # fill_value for add
                total_effective_weight += tf_weight
            else:
                logger.debug(f"[{self.strategy_name}][{stock_code}] 时间框架 '{tf_s}' (权重 {tf_weight:.2f}) 没有找到任何指标评分列。将使用中性分50参与加权。")
                # 如果某个时间框架没有评分列但有权重，可以认为其贡献是中性的 (50分)
                base_score_raw = base_score_raw.add(pd.Series(50.0, index=data.index) * tf_weight, fill_value=0.0)
                total_effective_weight += tf_weight
        
        # 如果总有效权重不为1 (例如某些时间框架无数据或无评分列)，进行调整
        # 但由于上面已经对 current_weights 归一化，且为无评分列的 tf 也加了中性分，
        # total_effective_weight 理论上应该接近1 (除非 current_weights 本身有问题)
        # 为保险起见，如果 total_effective_weight 显著偏离1，可以重新调整 base_score_raw
        # 但更可能的情况是，如果 total_effective_weight 为0 (所有权重都为0)，则 base_score_raw 应为中性
        if total_effective_weight == 0:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 所有时间框架的有效权重总和为零，基础评分将为中性50。")
            base_score_raw = pd.Series(50.0, index=data.index)
        elif not np.isclose(total_effective_weight, sum(current_weights.values())) and sum(current_weights.values()) > 0:
            # 这种情况理论上不应发生，因为我们为每个有权重的tf都加了分数（真实或中性）
            logger.warning(f"[{self.strategy_name}][{stock_code}] 有效权重总和 {total_effective_weight:.4f} 与预期权重总和 {sum(current_weights.values()):.4f} 不符。可能存在计算问题。")
            # 可以选择是否重新标准化 base_score_raw，例如除以 total_effective_weight 再乘以 sum(current_weights.values())
            # base_score_raw = (base_score_raw / total_effective_weight) * sum(current_weights.values())

        base_score_raw = base_score_raw.clip(0, 100).fillna(50.0)

        # --- 应用量能调整 ---
        vc_params_adjusted = vc_params.copy() # 使用全局的 volume_confirmation 参数
        # 但 trend_following_params 中也定义了 volume_boost_factor 等，应优先使用策略特定的
        vc_params_adjusted['boost_factor'] = self.volume_boost_factor # 来自 self.tf_params
        vc_params_adjusted['penalty_factor'] = self.volume_penalty_factor # 来自 self.tf_params
        vc_params_adjusted['volume_spike_threshold'] = self.volume_spike_threshold # 来自 self.tf_params
        # vc_params_adjusted['tf'] 应该来自全局 vc_params，因为这是量能分析的时间框架
        
        volume_adjusted_results_df = strategy_utils.adjust_score_with_volume(
            base_score_raw, data, vc_params_adjusted # vc_params_adjusted 包含了正确的 tf
        )
        base_score_volume_adjusted = volume_adjusted_results_df['ADJUSTED_SCORE']

        # --- 执行趋势分析 (基于量能调整后的分数) ---
        trend_analysis_df = self._perform_trend_analysis(data, base_score_volume_adjusted)

        # --- 检测背离信号 ---
        divergence_signals_df = pd.DataFrame(index=data.index)
        if dd_params.get('enabled', True): # 使用全局的 divergence_detection 参数
            try:
                # detect_divergence 需要 data, dd_params, indicator_configs
                divergence_signals_df = strategy_utils.detect_divergence(data, dd_params, indicator_configs)
                if not divergence_signals_df.empty:
                     logger.debug(f"[{self.strategy_name}][{stock_code}] 背离检测完成，最新信号: {divergence_signals_df.iloc[-1].to_dict() if not divergence_signals_df.empty else '无'}")
            except Exception as e:
                logger.error(f"[{self.strategy_name}][{stock_code}] 执行背离检测时出错: {e}", exc_info=True)

        # --- 组合最终规则信号 ---
        # 使用 self.rule_signal_weights (来自 trend_following_params)
        weights = self.rule_signal_weights 
        # 确保权重已归一化 (在 _validate_params 中已做)

        base_score_norm = (base_score_volume_adjusted.fillna(50.0) - 50) / 50
        alignment_norm = trend_analysis_df.get('alignment_signal', pd.Series(0.0, index=data.index)).fillna(0.0) / 3.0 # 范围是-3到3，归一化到-1到1
        long_context_norm = trend_analysis_df.get('long_term_context', pd.Series(0.0, index=data.index)).fillna(0.0) # 范围是-1,0,1
        score_momentum_series = trend_analysis_df.get('score_momentum', pd.Series(0.0, index=data.index)).fillna(0.0)
        momentum_norm = np.sign(score_momentum_series) # 取方向 -1,0,1
        ema_cross_norm = trend_analysis_df.get('ema_cross_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # -1,0,1
        boll_breakout_norm = trend_analysis_df.get('boll_breakout_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # -1,0,1
        adx_strength_norm = trend_analysis_df.get('adx_strength_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # -1到1
        vwap_dev_norm = trend_analysis_df.get('vwap_deviation_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # -1,0,1
        
        # volume_spike_signal 来自 volume_adjusted_results_df
        # 其时间框架由 vc_params['tf'] 决定
        vc_tf_col_name = vc_params.get('tf', self.focus_timeframe) # 获取量能分析的时间框架
        if isinstance(vc_tf_col_name, list): vc_tf_col_name = vc_tf_col_name[0] # 如果是列表取第一个
        volume_spike_signal_col = f'VOL_SPIKE_SIGNAL_{vc_tf_col_name}' # 构建列名
        volume_spike_norm = volume_adjusted_results_df.get(volume_spike_signal_col, pd.Series(0.0, index=data.index)).fillna(0.0) # -1,0,1

        total_weighted_contribution = pd.Series(0.0, index=data.index)
        total_weighted_contribution += base_score_norm * weights.get('base_score', 0)
        total_weighted_contribution += alignment_norm * weights.get('alignment', 0)
        total_weighted_contribution += long_context_norm * weights.get('long_context', 0)
        total_weighted_contribution += momentum_norm * weights.get('momentum', 0)
        total_weighted_contribution += ema_cross_norm * weights.get('ema_cross', 0)
        total_weighted_contribution += boll_breakout_norm * weights.get('boll_breakout', 0)
        total_weighted_contribution += adx_strength_norm * weights.get('adx_strength', 0) # ADX 强度直接参与加权
        total_weighted_contribution += vwap_dev_norm * weights.get('vwap_deviation', 0)
        total_weighted_contribution += volume_spike_norm * weights.get('volume_spike', 0)

        # 将总贡献映射回 0-100 范围
        # total_weighted_contribution 的范围理论上是 -sum(abs(weights)) 到 +sum(abs(weights))
        # 但由于权重已归一化且各项因子归一化到 -1 到 1，其范围大致在 -1 到 1 之间
        # 所以乘以 50 再加 50 可以映射到 0-100
        base_rule_signal_before_adjust = 50.0 + total_weighted_contribution * 50.0
        base_rule_signal_before_adjust = base_rule_signal_before_adjust.clip(0, 100) # 确保在0-100

        # --- 应用 ADX 增强调整 (如果不在加权中，则在这里调整) ---
        # 在当前版本中，adx_strength_norm 已经通过权重参与了 base_rule_signal_before_adjust 的计算
        # 如果希望 ADX 有额外的、非线性的增强/减弱作用，可以在这里添加
        # final_rule_signal = self._apply_adx_boost(base_rule_signal_before_adjust, adx_strength_norm, (base_rule_signal_before_adjust - 50.0) / 50.0)
        # 但参数文件中 adx_adjustment_factor 存在，暗示有独立调整步骤
        # 我们将使用 _apply_adx_boost 方法，但需要注意不要重复计算 ADX 的影响
        # 也许 rule_signal_weights 中的 adx_strength 权重应该为0，或者这里的调整是额外的
        # 假设这里的调整是额外的，基于 adx_adjustment_factor
        final_rule_signal = self._apply_adx_boost(
            base_rule_signal_before_adjust, # 传入已经包含ADX基础权重的信号
            adx_strength_norm, # ADX信号本身
            (base_rule_signal_before_adjust - 50.0) / 50.0 # 基础信号方向
        )


        # --- 应用背离惩罚 ---
        # dd_params 来自全局配置，包含 divergence_penalty_factor
        final_rule_signal = self._apply_divergence_penalty(final_rule_signal, divergence_signals_df, dd_params)

        # --- 应用趋势确认过滤 ---
        # 使用 self.trend_confirmation_periods (来自 trend_following_params)
        final_rule_signal = self._apply_trend_confirmation(final_rule_signal)

        final_rule_signal = final_rule_signal.clip(0, 100).round(2)

        intermediate_results = {
            'base_score_raw': base_score_raw,
            'base_score_volume_adjusted': base_score_volume_adjusted,
            'indicator_scores_df': indicator_scores_df, # 将评分DataFrame放入，后续处理
            'volume_analysis_df': volume_adjusted_results_df, # 重命名以示区分
            'trend_analysis_df': trend_analysis_df,
            'divergence_signals_df': divergence_signals_df
        }
        return final_rule_signal, intermediate_results

    def _perform_trend_analysis(self, data: pd.DataFrame, base_score_series: pd.Series) -> pd.DataFrame:
        """
        增强趋势分析，加入 ADX, STOCH, VWAP, BOLL 等辅助判断。
        """
        analysis_df = pd.DataFrame(index=base_score_series.index)
        # 参数来自全局配置或 trend_following_params
        # trend_analysis 参数在全局配置中
        ta_params_global = self.params.get('trend_analysis', {})
        # base_scoring 参数在全局配置中
        bs_params_global = self.params.get('base_scoring', {})
        # indicator_analysis_params 在全局配置中
        ia_params_global = self.params.get('indicator_analysis_params', {})
        
        focus_tf = self.focus_timeframe # 来自 trend_following_params

        if base_score_series.isnull().all():
            logger.warning(f"[{self.strategy_name}] 基础分数全为 NaN，无法执行趋势分析。")
            return analysis_df

        score_series_filled = base_score_series.fillna(50.0)

        # 1. 计算分数 EMA (使用全局 trend_analysis.ema_periods)
        all_ema_periods = ta_params_global.get('ema_periods', [5, 10, 20, 60])
        for period in all_ema_periods:
            try:
                analysis_df[f'ema_score_{period}'] = ta.ema(score_series_filled, length=period)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] 计算 EMA Score {period} 时出错: {e}")
                analysis_df[f'ema_score_{period}'] = np.nan
        
        # 2. 计算 EMA 排列信号 (基于上面计算的 ema_score)
        ema_periods_align = all_ema_periods[:4] # 取前4个周期用于排列
        if len(ema_periods_align) == 4 and all(f'ema_score_{p}' in analysis_df.columns for p in ema_periods_align):
            s_ema, m1_ema, m2_ema, l_ema = (analysis_df[f'ema_score_{p}'] for p in ema_periods_align)
            # 计算排列信号 (-3 到 3)
            alignment = pd.Series(0, index=analysis_df.index)
            alignment += np.sign(s_ema - m1_ema).fillna(0)
            alignment += np.sign(m1_ema - m2_ema).fillna(0)
            alignment += np.sign(m2_ema - l_ema).fillna(0)
            analysis_df['alignment_signal'] = alignment
        else:
            logger.warning(f"[{self.strategy_name}] 无法计算 EMA 排列信号，所需 EMA Score 列不足或缺失。")
            analysis_df['alignment_signal'] = 0.0

        # 3. 计算 EMA 交叉信号 (使用 all_ema_periods 的前两个)
        if len(all_ema_periods) >= 2:
            short_ema_col = f'ema_score_{all_ema_periods[0]}'
            mid_ema_col = f'ema_score_{all_ema_periods[1]}'
            if short_ema_col in analysis_df.columns and mid_ema_col in analysis_df.columns:
                short_ema = analysis_df[short_ema_col].fillna(50.0)
                mid_ema = analysis_df[mid_ema_col].fillna(50.0)
                golden_cross = (short_ema > mid_ema) & (short_ema.shift(1) <= mid_ema.shift(1))
                death_cross = (short_ema < mid_ema) & (short_ema.shift(1) >= mid_ema.shift(1))
                analysis_df['ema_cross_signal'] = 0.0
                analysis_df.loc[golden_cross, 'ema_cross_signal'] = 1.0
                analysis_df.loc[death_cross, 'ema_cross_signal'] = -1.0
            else:
                analysis_df['ema_cross_signal'] = 0.0
        else:
            analysis_df['ema_cross_signal'] = 0.0
            
        # 4. 计算 EMA 强度 (短期EMA与长期EMA之差)
        if len(all_ema_periods) >= 2:
            short_ema_col = f'ema_score_{all_ema_periods[0]}'
            # 使用全局 trend_analysis.long_term_ema_period
            long_term_ema_period_for_strength = ta_params_global.get('long_term_ema_period', all_ema_periods[-1])
            long_ema_col = f'ema_score_{long_term_ema_period_for_strength}'
            if short_ema_col in analysis_df.columns and long_ema_col in analysis_df.columns:
                analysis_df['ema_strength'] = (analysis_df[short_ema_col].fillna(50.0) - analysis_df[long_ema_col].fillna(50.0)).fillna(0.0)
            else:
                analysis_df['ema_strength'] = 0.0
        else:
            analysis_df['ema_strength'] = 0.0

        # 5. 计算得分动量及动量加速
        analysis_df['score_momentum'] = score_series_filled.diff().fillna(0.0)
        analysis_df['score_momentum_acceleration'] = analysis_df['score_momentum'].diff().fillna(0.0)

        # 6. 计算得分波动率 (使用全局 trend_analysis.volatility_window)
        volatility_window = ta_params_global.get('volatility_window', 10)
        analysis_df['score_volatility'] = score_series_filled.rolling(window=volatility_window, min_periods=max(1, volatility_window//2)).std().fillna(0.0)
        # volatility_signal 使用 trend_following_params 中的阈值 (self.volatility_threshold_high/low)
        analysis_df['volatility_signal'] = 0.0
        analysis_df.loc[analysis_df['score_volatility'] > self.volatility_threshold_high, 'volatility_signal'] = -1.0 # 高波动可能不利于趋势
        analysis_df.loc[analysis_df['score_volatility'] < self.volatility_threshold_low, 'volatility_signal'] = 1.0 # 低波动可能酝酿趋势

        # 7. 长期趋势背景 (基于分数与长期EMA，使用全局 trend_analysis.long_term_ema_period)
        long_term_ema_period_context = ta_params_global.get('long_term_ema_period', all_ema_periods[-1] if all_ema_periods else 60)
        long_term_ema_col_context = f'ema_score_{long_term_ema_period_context}'
        if long_term_ema_col_context in analysis_df.columns:
            long_term_ema_filled = analysis_df[long_term_ema_col_context].fillna(50.0)
            analysis_df['long_term_context'] = 0.0
            analysis_df.loc[score_series_filled > long_term_ema_filled, 'long_term_context'] = 1.0
            analysis_df.loc[score_series_filled < long_term_ema_filled, 'long_term_context'] = -1.0
        else:
            logger.warning(f"[{self.strategy_name}] 缺少长期 EMA Score 列 ({long_term_ema_col_context})，无法计算长期趋势背景。")
            analysis_df['long_term_context'] = 0.0

        # 8. ADX 趋势强度判断 (使用 focus_timeframe 的 DMI 数据)
        # DMI 周期来自全局 base_scoring.dmi_period
        dmi_period_bs = bs_params_global.get("dmi_period", 14)
        pdi_col = f'PLUS_DI_{dmi_period_bs}_{focus_tf}'
        ndi_col = f'MINUS_DI_{dmi_period_bs}_{focus_tf}'
        adx_col = f'ADX_{dmi_period_bs}_{focus_tf}'

        if adx_col in data.columns and pdi_col in data.columns and ndi_col in data.columns:
            adx = data[adx_col].fillna(0.0)
            pdi = data[pdi_col].fillna(0.0)
            mdi = data[ndi_col].fillna(0.0)
            # ADX 阈值来自 trend_following_params (self.adx_strong_threshold / self.adx_moderate_threshold)
            analysis_df['adx_strength_signal'] = 0.0
            strong_trend = adx >= self.adx_strong_threshold
            moderate_trend = (adx >= self.adx_moderate_threshold) & (adx < self.adx_strong_threshold)
            bullish_dmi = pdi > mdi
            bearish_dmi = mdi > pdi
            analysis_df.loc[strong_trend & bullish_dmi, 'adx_strength_signal'] = 1.0
            analysis_df.loc[strong_trend & bearish_dmi, 'adx_strength_signal'] = -1.0
            analysis_df.loc[moderate_trend & bullish_dmi, 'adx_strength_signal'] = 0.5
            analysis_df.loc[moderate_trend & bearish_dmi, 'adx_strength_signal'] = -0.5
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 ADX/PDI/NDI 列 ({adx_col}, {pdi_col}, {ndi_col}) for focus_tf='{focus_tf}'，无法计算 ADX 强度信号。")
            analysis_df['adx_strength_signal'] = 0.0

        # 9. STOCH 超买超卖判断 (使用 focus_timeframe 的 STOCH 数据)
        # STOCH 参数来自全局 indicator_analysis_params
        stoch_k_p_ia = ia_params_global.get('stoch_k', 14) # JSON 中是 14
        stoch_d_p_ia = ia_params_global.get('stoch_d', 3)
        stoch_smooth_k_p_ia = ia_params_global.get('stoch_smooth_k', 3)
        k_col = f'STOCHk_{stoch_k_p_ia}_{stoch_d_p_ia}_{stoch_smooth_k_p_ia}_{focus_tf}'
        d_col = f'STOCHd_{stoch_k_p_ia}_{stoch_d_p_ia}_{stoch_smooth_k_p_ia}_{focus_tf}'
        # STOCH 超买超卖阈值来自 trend_following_params (self.stoch_oversold_threshold / self.stoch_overbought_threshold)
        if k_col in data.columns and d_col in data.columns:
            k_val = data[k_col].fillna(50.0)
            d_val = data[d_col].fillna(50.0)
            is_oversold = (k_val < self.stoch_oversold_threshold) & (d_val < self.stoch_oversold_threshold)
            is_overbought = (k_val > self.stoch_overbought_threshold) & (d_val > self.stoch_overbought_threshold)
            turning_up = (k_val > d_val) & (k_val.shift(1).fillna(50.0) <= d_val.shift(1).fillna(50.0))
            turning_down = (k_val < d_val) & (k_val.shift(1).fillna(50.0) >= d_val.shift(1).fillna(50.0))
            analysis_df['stoch_signal'] = 0.0
            analysis_df.loc[is_oversold & turning_up, 'stoch_signal'] = 1.0   # 超卖区金叉 - 看涨
            analysis_df.loc[is_overbought & turning_down, 'stoch_signal'] = -1.0 # 超买区死叉 - 看跌
            analysis_df.loc[is_oversold & ~(turning_up), 'stoch_signal'] = 0.5 # 仍在超卖区但未金叉 - 潜在看涨
            analysis_df.loc[is_overbought & ~(turning_down), 'stoch_signal'] = -0.5# 仍在超买区但未死叉 - 潜在看跌
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 STOCH K/D 列 ({k_col}, {d_col}) for focus_tf='{focus_tf}'，无法计算 STOCH 信号。")
            analysis_df['stoch_signal'] = 0.0

        # 10. VWAP 偏离判断 (使用 focus_timeframe 的 VWAP 数据)
        # VWAP anchor 来自全局 indicator_analysis_params
        vwap_anchor_ia = ia_params_global.get('vwap_anchor', None)
        vwap_col_name_ia = 'VWAP' if vwap_anchor_ia is None else f'VWAP_{vwap_anchor_ia}'
        vwap_col = f"{vwap_col_name_ia}_{focus_tf}"
        close_col = f'close_{focus_tf}'
        # VWAP 偏离阈值来自 trend_following_params (self.vwap_deviation_threshold)
        if vwap_col in data.columns and close_col in data.columns:
            vwap = data[vwap_col]
            close_price = data[close_col]
            vwap_safe = vwap.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill') # 避免除零，并填充NaN
            deviation = ((close_price - vwap_safe) / vwap_safe).fillna(0.0)
            analysis_df['vwap_deviation_signal'] = 0.0
            analysis_df.loc[deviation > self.vwap_deviation_threshold, 'vwap_deviation_signal'] = 1.0
            analysis_df.loc[deviation < -self.vwap_deviation_threshold, 'vwap_deviation_signal'] = -1.0
            analysis_df['vwap_deviation_percent'] = deviation * 100
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 VWAP 或收盘价列 ({vwap_col}, {close_col}) for focus_tf='{focus_tf}'，无法计算 VWAP 偏离信号。")
            analysis_df['vwap_deviation_signal'] = 0.0
            analysis_df['vwap_deviation_percent'] = 0.0

        # 11. BOLL 突破判断 (使用 focus_timeframe 的 BOLL 数据)
        # BOLL 参数来自全局 base_scoring
        boll_period_bs = bs_params_global.get("boll_period", 20)
        boll_std_dev_bs = bs_params_global.get("boll_std_dev", 2.0)
        std_str_bs = f"{boll_std_dev_bs:.1f}"
        upper_col = f'BBU_{boll_period_bs}_{std_str_bs}_{focus_tf}'
        lower_col = f'BBL_{boll_period_bs}_{std_str_bs}_{focus_tf}'
        # middle_col = f'BBM_{boll_period_bs}_{std_str_bs}_{focus_tf}' # 可选
        # close_col 已定义
        if upper_col in data.columns and lower_col in data.columns and close_col in data.columns:
            upper_band = data[upper_col]
            lower_band = data[lower_col]
            close_price = data[close_col]
            analysis_df['boll_breakout_signal'] = 0.0
            analysis_df.loc[close_price > upper_band, 'boll_breakout_signal'] = 1.0
            analysis_df.loc[close_price < lower_band, 'boll_breakout_signal'] = -1.0
            # 可选：计算 %B
            # band_width = (upper_band - lower_band).replace(0, np.nan)
            # analysis_df['boll_percent_b'] = ((close_price - lower_band) / band_width * 100).clip(0,100).fillna(50.0)
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 BOLL 上轨/下轨或收盘价列 ({upper_col}, {lower_col}, {close_col}) for focus_tf='{focus_tf}'，无法计算 BOLL 突破信号。")
            analysis_df['boll_breakout_signal'] = 0.0
            # analysis_df['boll_percent_b'] = 50.0

        # 12. 计算综合趋势强度 (可选，作为辅助分析)
        # 这里可以定义一些权重来组合上面计算的各项辅助信号
        # 例如: w_align_trend = 0.3, w_momentum_trend = 0.2, w_adx_trend = 0.3, w_context_trend = 0.1, w_volatility_trend = 0.1
        # trend_strength_score = (analysis_df['alignment_signal']/3 * w_align_trend +
        #                         np.sign(analysis_df['score_momentum']) * w_momentum_trend +
        #                         analysis_df['adx_strength_signal'] * w_adx_trend +
        #                         analysis_df['long_term_context'] * w_context_trend +
        #                         analysis_df['volatility_signal'] * w_volatility_trend) # volatility_signal 本身是 -1,0,1
        # analysis_df['trend_strength_score'] = trend_strength_score.clip(-1, 1).fillna(0.0) # 归一化到 -1 到 1

        logger.debug(f"[{self.strategy_name}] 趋势分析完成，最新分析信号 (部分): "
                     f"Alignment: {analysis_df['alignment_signal'].iloc[-1] if not analysis_df.empty else 'N/A'}, "
                     f"ADX Strength: {analysis_df['adx_strength_signal'].iloc[-1] if not analysis_df.empty else 'N/A'}")
        return analysis_df

    def _adjust_volatility_parameters(self, data: pd.DataFrame):
        """
        根据股票波动率动态调整参数，如波动率阈值。
        """
        focus_tf = self.focus_timeframe
        close_col = f'close_{focus_tf}'
        if close_col not in data.columns or data[close_col].isnull().all():
            logger.warning(f"[{self.strategy_name}] 动态调整波动率：缺少收盘价列 {close_col} 或数据全为空。")
            return

        # volatility_window 来自全局 trend_analysis 参数
        volatility_window = self.params.get('trend_analysis', {}).get('volatility_window', 10)
        price_volatility = data[close_col].rolling(window=volatility_window, min_periods=max(1, volatility_window//2)).std()
        
        if price_volatility.isnull().all() or price_volatility.empty:
            logger.warning(f"[{self.strategy_name}] 动态调整波动率：价格波动率数据不可用。")
            return

        latest_volatility = price_volatility.iloc[-1]
        # volatility_benchmark 来自 trend_following_params
        base_volatility_benchmark = self.tf_params.get('volatility_benchmark', 5.0)
        if pd.isna(latest_volatility) or base_volatility_benchmark <= 0:
            logger.warning(f"[{self.strategy_name}] 动态调整波动率：最新波动率 ({latest_volatility}) 或基准 ({base_volatility_benchmark}) 无效。")
            return
            
        self.volatility_adjust_factor = max(0.5, min(2.0, latest_volatility / base_volatility_benchmark)) # 限制调整因子范围
        
        # 原始阈值来自 trend_following_params (在 __init__ 中已用 tf_params.get 设置到 self 属性)
        original_high = self.tf_params.get('volatility_threshold_high', 10.0) # 再次获取以防 self 属性被意外修改
        original_low = self.tf_params.get('volatility_threshold_low', 5.0)
        
        self.volatility_threshold_high = original_high * self.volatility_adjust_factor
        self.volatility_threshold_low = original_low * self.volatility_adjust_factor

        # 对调整后的阈值也进行合理性限制
        self.volatility_threshold_high = np.clip(self.volatility_threshold_high, original_high * 0.5, original_high * 2.0)
        self.volatility_threshold_low = np.clip(self.volatility_threshold_low, original_low * 0.5, original_low * 2.0)
        if self.volatility_threshold_low >= self.volatility_threshold_high: # 确保低阈值小于高阈值
            self.volatility_threshold_low = self.volatility_threshold_high * 0.5 

        logger.debug(f"[{self.strategy_name}] 动态调整波动率阈值: high={self.volatility_threshold_high:.2f}, low={self.volatility_threshold_low:.2f}, factor={self.volatility_adjust_factor:.2f} (based on latest_vol={latest_volatility:.2f})")

    def _apply_adx_boost(self, final_signal: pd.Series, adx_strength_norm: pd.Series, base_signal_direction_norm: pd.Series) -> pd.Series:
        """
        模块化调整逻辑：使用 ADX 强度增强信号。
        adx_strength_norm: -1 到 1 的 ADX 信号 (已包含方向)
        base_signal_direction_norm: -1 到 1 的基础信号方向
        """
        final_signal_filled = final_signal.fillna(50.0)
        adx_strength_norm_filled = adx_strength_norm.fillna(0.0)
        base_signal_direction_norm_filled = base_signal_direction_norm.fillna(0.0)

        # adx_adjustment_factor 来自 trend_following_params
        adx_adjustment_factor = self.tf_params.get('adx_adjustment_factor', 10.0) 
        adjustment = pd.Series(0.0, index=final_signal.index)

        # 只有当 ADX 信号方向与基础信号方向一致时才增强
        # adx_strength_norm 已经包含了趋势方向 (+/-) 和强度 (大小)
        # 所以，如果 adx_strength_norm 和 base_signal_direction_norm 同号，则增强
        # 增强幅度与 adx_strength_norm 的绝对值（强度）成正比
        effective_mask = (np.sign(adx_strength_norm_filled) == np.sign(base_signal_direction_norm_filled)) & \
                         (base_signal_direction_norm_filled != 0) # 基础信号有明确方向

        adjustment.loc[effective_mask] = np.sign(base_signal_direction_norm_filled.loc[effective_mask]) * \
                                         np.abs(adx_strength_norm_filled.loc[effective_mask]) * \
                                         adx_adjustment_factor
        
        # 如果 ADX 指示无趋势 (adx_strength_norm 接近0)，或者与基础信号反向，则可能有惩罚或不调整
        # 当前逻辑只做同向增强。如果反向，则不调整。
        # 如果希望反向时有惩罚，可以增加逻辑：
        # reverse_mask = (np.sign(adx_strength_norm_filled) == -np.sign(base_signal_direction_norm_filled)) & \
        #                (base_signal_direction_norm_filled != 0)
        # adjustment.loc[reverse_mask] = -np.sign(base_signal_direction_norm_filled.loc[reverse_mask]) * \
        #                                np.abs(adx_strength_norm_filled.loc[reverse_mask]) * \
        #                                adx_adjustment_factor * 0.5 # 例如惩罚因子为0.5

        if not adjustment.empty:
             logger.debug(f"[{self.strategy_name}] ADX 增强调整 (最新值): {adjustment.iloc[-1] if not adjustment.empty else 'N/A'}")
        return (final_signal_filled + adjustment).clip(0, 100) # 确保在0-100

    def _apply_divergence_penalty(self, final_signal: pd.Series, divergence_signals_df: pd.DataFrame, dd_params: Dict) -> pd.Series:
        """
        模块化调整逻辑：应用背离惩罚。
        dd_params: 全局的 divergence_detection 参数
        """
        final_signal_filled = final_signal.fillna(50.0)
        if divergence_signals_df.empty:
            logger.debug(f"[{self.strategy_name}] 背离信号 DataFrame 为空，跳过背离惩罚。")
            return final_signal_filled

        # divergence_penalty_factor 来自全局 dd_params
        penalty_factor = dd_params.get('divergence_penalty_factor', 0.45)
        has_bearish_div_col = 'HAS_BEARISH_DIVERGENCE' # strategy_utils.detect_divergence 应生成此列
        has_bullish_div_col = 'HAS_BULLISH_DIVERGENCE' # strategy_utils.detect_divergence 应生成此列

        if has_bearish_div_col not in divergence_signals_df.columns or \
           has_bullish_div_col not in divergence_signals_df.columns:
            logger.warning(f"[{self.strategy_name}] 背离信号 DataFrame 缺少聚合信号列，跳过背离惩罚。")
            return final_signal_filled

        # 确保索引一致，以便安全合并或对齐
        if not final_signal_filled.index.equals(divergence_signals_df.index):
            logger.warning(f"[{self.strategy_name}] 背离惩罚：信号和背离数据索引不一致，尝试重新对齐。")
            # 尝试将 divergence_signals_df 的索引对齐到 final_signal_filled
            # 这假设 divergence_signals_df 的索引是 final_signal_filled 索引的子集或可以安全对齐
            try:
                divergence_signals_aligned = divergence_signals_df.reindex(final_signal_filled.index, fill_value=False) # 用 False 填充缺失的背离信号
                has_bearish_div = divergence_signals_aligned[has_bearish_div_col].astype(bool)
                has_bullish_div = divergence_signals_aligned[has_bullish_div_col].astype(bool)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] 背离数据重新对齐失败: {e}。跳过背离惩罚。")
                return final_signal_filled
        else:
            has_bearish_div = divergence_signals_df[has_bearish_div_col].astype(bool)
            has_bullish_div = divergence_signals_df[has_bullish_div_col].astype(bool)


        is_bullish_signal = final_signal_filled > 50
        is_bearish_signal = final_signal_filled < 50
        adjusted_signal = final_signal_filled.copy()

        # 当看涨信号遇到顶背离时，减弱看涨信号
        mask_bullish_penalty = is_bullish_signal & has_bearish_div
        adjusted_signal.loc[mask_bullish_penalty] = 50 + (adjusted_signal.loc[mask_bullish_penalty] - 50) * (1 - penalty_factor)

        # 当看跌信号遇到低背离时，减弱看跌信号
        mask_bearish_penalty = is_bearish_signal & has_bullish_div
        adjusted_signal.loc[mask_bearish_penalty] = 50 + (adjusted_signal.loc[mask_bearish_penalty] - 50) * (1 - penalty_factor)
        
        if not adjusted_signal.empty:
             logger.debug(f"[{self.strategy_name}] 背离惩罚调整后信号 (最新值): {adjusted_signal.iloc[-1] if not adjusted_signal.empty else 'N/A'}")
        return adjusted_signal.clip(0, 100) # 确保在0-100

    def _apply_trend_confirmation(self, final_signal: pd.Series) -> pd.Series:
        """
        增强假信号过滤：要求信号突破阈值后持续若干周期才视为有效。
        """
        if final_signal.empty:
            return final_signal

        # 阈值和周期来自 trend_following_params
        trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55) # JSON 中无此特定参数，使用默认或强信号阈值
        trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45) # JSON 中无此特定参数
        confirmation_periods = self.trend_confirmation_periods # 来自 trend_following_params (JSON 中为 3)

        final_signal_filled = final_signal.fillna(50.0)
        filtered_signal = pd.Series(50.0, index=final_signal.index) # 默认中性

        # 标记信号是否持续在阈值之上/之下
        above_upper = final_signal_filled >= trend_threshold_upper
        below_lower = final_signal_filled <= trend_threshold_lower

        # 使用 rolling.sum() 来计算连续满足条件的周期数是否达到 confirmation_periods
        # 如果连续 N 个周期都为 True (1), sum 就是 N
        confirmed_bullish_streak = above_upper.rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() == confirmation_periods
        confirmed_bearish_streak = below_lower.rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() == confirmation_periods
        
        # 只有当趋势被确认后，才采用原始信号，否则保持中性
        # 注意：这里如果一个看涨趋势刚被确认，那么 confirmed_bullish_streak.iloc[t] 为 True
        # 此时 filtered_signal.iloc[t] 应取 final_signal_filled.iloc[t]
        # 如果之后信号回落到阈值内，confirmed_bullish_streak 会变回 False，filtered_signal 又回到50
        # 这可能不是期望的行为。期望的行为可能是：一旦趋势确认，信号保持，直到反向趋势被确认或明确的中性信号出现。
        # 当前实现是：只有在严格满足连续 N 周期条件的时间点，信号才有效。
        
        # 一个更常见的做法是：
        # 1. 识别趋势开始点（例如，第一次满足连续N周期）
        # 2. 趋势持续，直到出现反转信号或趋势结束信号
        # 为了简化，我们暂时保持当前逻辑：只有在当前点满足N周期确认时，信号才有效。
        
        current_signal_state = pd.Series(0, index=final_signal.index) # 0 中性, 1 看涨, -1 看跌
        
        # 遍历，一旦趋势确认，保持该趋势直到反向趋势确认或信号回到中性区
        # 这个逻辑比较复杂，暂时使用简单的N周期确认
        
        # 简单N周期确认：
        is_confirmed_bullish = confirmed_bullish_streak
        is_confirmed_bearish = confirmed_bearish_streak

        # 应用确认后的信号
        # 如果同时满足看涨和看跌确认（理论上不太可能，除非阈值设置不当），需要处理冲突
        # 这里假设它们不冲突
        filtered_signal.loc[is_confirmed_bullish] = final_signal_filled.loc[is_confirmed_bullish]
        filtered_signal.loc[is_confirmed_bearish] = final_signal_filled.loc[is_confirmed_bearish]
        
        # 如果一个点既不是确认的看涨也不是确认的看跌，则为50 (已默认设置)

        if not filtered_signal.empty:
             logger.debug(f"[{self.strategy_name}] 趋势确认过滤后信号 (最新值): {filtered_signal.iloc[-1] if not filtered_signal.empty else 'N/A'}")
        return filtered_signal.clip(0, 100)


    def generate_signals(self, data: pd.DataFrame, stock_code: Optional[str] = None, indicator_configs: Optional[List[Dict]] = None) -> pd.Series:
        """
        生成趋势跟踪信号，整合规则信号和Transformer模型预测。
        返回最终的信号 Series (通常是 combined_signal)。

        Args:
            data (pd.DataFrame): 包含所有已计算指标的原始DataFrame。
            stock_code (Optional[str]): 股票代码，主要用于日志和模型路径。
            indicator_configs (Optional[List[Dict]]): IndicatorService 生成的指标配置列表，用于规则信号计算。

        Returns:
            pd.Series: 最终的信号 Series (combined_signal)。
        """
        # 确保 stock_code 和 indicator_configs 提供了有效值
        if stock_code is None:
            logger.error(f"[{self.strategy_name}] generate_signals: 必须提供 stock_code。")
            return pd.Series(50.0, index=data.index, name="combined_signal") # 返回中性信号
        if indicator_configs is None:
            logger.error(f"[{self.strategy_name}][{stock_code}] generate_signals: 必须提供 indicator_configs。")
            return pd.Series(50.0, index=data.index, name="combined_signal")

        logger.info(f"[{self.strategy_name}] 开始为股票 {stock_code} 生成信号 (Focus: {self.focus_timeframe})...")

        # 1. 计算规则基础信号 (final_rule_signal) 和中间结果
        final_rule_signal, intermediate_results_dict = self._calculate_rule_based_signal(
            data=data, stock_code=stock_code, indicator_configs=indicator_configs
        )

        # 创建一个 DataFrame 来存储所有中间计算结果和信号
        # 初始时包含原始数据和规则信号
        # 注意：data 已经是包含所有指标的 DataFrame
        processed_data = data.copy() # 复制以避免修改原始输入
        processed_data['final_rule_signal'] = final_rule_signal

        # 合并 _calculate_rule_based_signal 返回的中间 DataFrame
        # 例如 indicator_scores_df, volume_analysis_df, trend_analysis_df, divergence_signals_df
        for key, df_to_join in intermediate_results_dict.items():
            if isinstance(df_to_join, pd.DataFrame) and not df_to_join.empty:
                # 为避免列名冲突，可以添加后缀，但如果列名设计良好则不需要
                # processed_data = processed_data.join(df_to_join, how='left', rsuffix=f'_{key.replace("_df","")}')
                # 简单合并，假设列名不冲突或冲突是预期的覆盖
                for col_join in df_to_join.columns:
                    if col_join not in processed_data.columns: # 只合并新列
                        processed_data[col_join] = df_to_join[col_join]
                    else: # 如果列已存在，可以选择不合并或记录警告
                        logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果列 '{col_join}' 已存在于 processed_data，跳过合并来自 '{key}' 的同名列。")
            elif isinstance(df_to_join, pd.Series) and not df_to_join.empty: # 如果返回的是 Series
                 if df_to_join.name and df_to_join.name not in processed_data.columns:
                     processed_data[df_to_join.name] = df_to_join
                 elif not df_to_join.name and key not in processed_data.columns: # 如果 Series 没名字，用字典的 key
                     processed_data[key] = df_to_join
                 else:
                     logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果Series '{df_to_join.name or key}' 已存在或无名，跳过合并。")


        # 2. Transformer模型预测
        processed_data['transformer_signal'] = pd.Series(50.0, index=processed_data.index) # 初始化为中性
        self.set_model_paths(stock_code)
        self.load_lstm_model(stock_code) # 方法名兼容，内部加载 Transformer

        if self.transformer_model and self.feature_scaler and self.target_scaler and self.selected_feature_names_for_transformer:
            try:
                logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型已加载，准备进行预测...")
                # predict_with_transformer_model 需要处理数据窗口和特征选择
                # 它应该返回一个 Series，索引与输入数据最后部分对齐
                predicted_signal_series = predict_with_transformer_model(
                    model=self.transformer_model,
                    data=processed_data, # 传入包含所有潜在特征的 DataFrame
                    feature_scaler=self.feature_scaler,
                    target_scaler=self.target_scaler,
                    selected_feature_names=self.selected_feature_names_for_transformer,
                    window_size=self.transformer_window_size,
                    device=self.device
                )
                # 将预测结果（通常是最后一个点的预测）更新到 processed_data
                # predict_with_transformer_model 返回的 Series 可能只包含最新的一个或几个预测值
                if not predicted_signal_series.empty:
                    # 更新 'transformer_signal' 列中与 predicted_signal_series 索引匹配的部分
                    processed_data.loc[predicted_signal_series.index, 'transformer_signal'] = predicted_signal_series
                    latest_pred_idx = predicted_signal_series.index[-1]
                    latest_pred_val = predicted_signal_series.iloc[-1]
                    logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型预测完成，最新预测信号 ({latest_pred_idx}): {latest_pred_val:.2f}")
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] Transformer 模型预测返回空 Series。")

            except Exception as e:
                logger.error(f"[{self.strategy_name}][{stock_code}] Transformer 模型预测出错: {e}", exc_info=True)
        else:
            logger.warning(f"[{self.strategy_name}][{stock_code}] Transformer 模型/Scaler/特征列表未加载，跳过 Transformer 预测。")

        # 3. 结合规则信号和 Transformer 信号
        # signal_combination_weights 来自 trend_following_params
        combination_weights = self.tf_params.get('signal_combination_weights', {'rule_weight': 0.6, 'lstm_weight': 0.4})
        rule_weight = combination_weights.get('rule_weight', 0.6)
        transformer_weight = combination_weights.get('lstm_weight', 0.4) # 使用 lstm_weight 作为 transformer 的权重

        # 确保权重归一化 (在 _validate_params 中已做，这里可以再检查一次)
        total_combo_weight = rule_weight + transformer_weight
        if total_combo_weight > 0 and not np.isclose(total_combo_weight, 1.0):
            rule_weight /= total_combo_weight
            transformer_weight /= total_combo_weight
        
        # 组合信号：transformer_signal 可能只在最后几个点有值，其他点为初始的50.0
        # final_rule_signal 在所有点都有值
        # 我们希望 combined_signal 在 transformer 有预测时使用加权，否则主要依赖规则信号
        
        # 默认情况下，combined_signal 等于 final_rule_signal
        processed_data['combined_signal'] = processed_data['final_rule_signal']
        
        # 找到 transformer_signal 不是初始值 (50.0) 或 NaN 的索引
        # (更准确地说，是 transformer 实际做出预测的索引)
        # 假设 predict_with_transformer_model 返回的 Series 的索引就是这些点
        if 'predicted_signal_series' in locals() and not predicted_signal_series.empty:
            valid_transformer_indices = predicted_signal_series.index
            if not valid_transformer_indices.empty:
                logger.info(f"[{self.strategy_name}][{stock_code}] 在 {len(valid_transformer_indices)} 个点上应用 Transformer 加权组合。")
                
                # 在这些点上，重新计算 combined_signal
                rule_comp = processed_data.loc[valid_transformer_indices, 'final_rule_signal'] * rule_weight
                trans_comp = processed_data.loc[valid_transformer_indices, 'transformer_signal'] * transformer_weight
                processed_data.loc[valid_transformer_indices, 'combined_signal'] = (rule_comp + trans_comp).clip(0, 100).round(2)
        else:
            logger.info(f"[{self.strategy_name}][{stock_code}] 未进行 Transformer 预测或预测结果为空，Combined_signal 将等于 Final_rule_signal。")


        # --- 存储中间数据 ---
        self.intermediate_data = processed_data.copy() # 保存包含所有计算列的DataFrame

        if not self.intermediate_data.empty:
            # 确保 combined_signal 列存在
            latest_combined_signal_val = self.intermediate_data['combined_signal'].iloc[-1] if 'combined_signal' in self.intermediate_data.columns else np.nan
            logger.info(f"[{self.strategy_name}][{stock_code}] 信号生成完毕，最新组合信号: {latest_combined_signal_val:.2f}")
        else:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 信号生成过程未产生有效数据 (intermediate_data 为空)。")

        # 调用 analyze_signals 方法进行分析 (可选，如果需要立即分析)
        # self.analyze_signals(stock_code) # analyze_signals 会使用 self.intermediate_data

        # 返回包含所有计算结果的 DataFrame
        return self.intermediate_data # 或者 processed_data，取决于 self.intermediate_data 是否是最终版本


    def load_lstm_model(self, stock_code: str): # 方法名保持兼容，但加载 Transformer
        """
        为特定股票加载 Transformer 模型权重和 scaler。
        """
        # self.set_model_paths(stock_code) 已在 generate_signals 中调用
        if not all([self.model_path, self.feature_scaler_path, self.target_scaler_path, self.selected_features_path]):
            logger.error(f"[{self.strategy_name}][{stock_code}] 模型或Scaler路径未正确设置，无法加载。")
            self._reset_model_components()
            return

        required_files_exist = all([
            os.path.exists(self.model_path),
            os.path.exists(self.feature_scaler_path),
            os.path.exists(self.target_scaler_path),
            os.path.exists(self.selected_features_path)
        ])

        if not required_files_exist:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 缺失必需的 Transformer 模型/Scaler/特征文件，无法加载。")
            self._reset_model_components()
            return
        
        try:
            with open(self.selected_features_path, 'r', encoding='utf-8') as f:
                 self.selected_feature_names_for_transformer = json.load(f)
            logger.debug(f"[{self.strategy_name}][{stock_code}] 选中特征名列表 ({len(self.selected_feature_names_for_transformer)}个) 从 {self.selected_features_path} 加载。")

            num_features = len(self.selected_feature_names_for_transformer)
            if num_features == 0:
                 logger.error(f"[{self.strategy_name}][{stock_code}] 加载的选中特征列表为空，无法构建模型。")
                 self._reset_model_components()
                 return

            self.transformer_model = build_transformer_model(
                num_features=num_features,
                model_config=self.transformer_model_config,
                summary=False,
                window_size=self.transformer_window_size
            )
            self.transformer_model.to(self.device)
            self.transformer_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型权重从 {self.model_path} 加载成功。")

            self.feature_scaler = joblib.load(self.feature_scaler_path)
            logger.info(f"[{self.strategy_name}][{stock_code}] 特征 Scaler 从 {self.feature_scaler_path} 加载成功。")

            self.target_scaler = joblib.load(self.target_scaler_path)
            logger.info(f"[{self.strategy_name}][{stock_code}] 目标 Scaler 从 {self.target_scaler_path} 加载成功。")

            if not all([self.transformer_model, self.feature_scaler, self.target_scaler, self.selected_feature_names_for_transformer]):
                 logger.warning(f"[{self.strategy_name}][{stock_code}] 加载 Transformer 模型或相关组件失败，部分对象为 None/空。")
                 self._reset_model_components() # 确保状态一致
            else:
                 logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型及相关组件加载完成。")

        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 加载 Transformer 模型或Scaler或特征列表出错: {e}", exc_info=True)
            self._reset_model_components()

    def _reset_model_components(self):
        """辅助函数，用于重置模型相关组件状态。"""
        self.transformer_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.selected_feature_names_for_transformer = []


    def save_prepared_data(self, stock_code: str,
                        features_scaled_train: np.ndarray, targets_scaled_train: np.ndarray,
                        features_scaled_val: np.ndarray, targets_scaled_val: np.ndarray,
                        features_scaled_test: np.ndarray, targets_scaled_test: np.ndarray,
                        feature_scaler: Union[MinMaxScaler, StandardScaler], 
                        target_scaler: Union[MinMaxScaler, StandardScaler],
                        selected_feature_names: List[str]):
        """
        保存准备好的 Transformer 训练数据和 Scaler。
        """
        self.set_model_paths(stock_code) # 确保路径已设置
        if not all([self.all_prepared_data_npz_path, self.feature_scaler_path, self.target_scaler_path, self.selected_features_path]):
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存准备数据：部分或全部路径未设置。")
            raise RuntimeError("保存路径未正确初始化。")

        try:
            np.savez_compressed(
                self.all_prepared_data_npz_path,
                features_scaled_train=features_scaled_train.astype(np.float32),
                targets_scaled_train=targets_scaled_train.astype(np.float32),
                features_scaled_val=features_scaled_val.astype(np.float32),
                targets_scaled_val=targets_scaled_val.astype(np.float32),
                features_scaled_test=features_scaled_test.astype(np.float32),
                targets_scaled_test=targets_scaled_test.astype(np.float32)
            )
            logger.debug(f"[{self.strategy_name}][{stock_code}] 所有准备数据已保存到 {self.all_prepared_data_npz_path}。")

            joblib.dump(feature_scaler, self.feature_scaler_path)
            joblib.dump(target_scaler, self.target_scaler_path)
            logger.debug(f"[{self.strategy_name}][{stock_code}] Scaler 已保存。")

            with open(self.selected_features_path, 'w', encoding='utf-8') as f:
                 json.dump(selected_feature_names, f, ensure_ascii=False, indent=4)
            logger.debug(f"[{self.strategy_name}][{stock_code}] 选中特征名列表已保存到 {self.selected_features_path}。")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存准备好的数据、Scaler或特征列表时出错: {e}", exc_info=True)
            raise e

    def load_prepared_data(self, stock_code: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Union[MinMaxScaler, StandardScaler]], Optional[Union[MinMaxScaler, StandardScaler]]]:
        """
        从文件加载特定股票准备好的 Transformer 训练数据和 Scaler。
        """
        self.set_model_paths(stock_code) # 确保路径已设置
        empty_array = np.array([])
        
        if not all([self.all_prepared_data_npz_path, self.feature_scaler_path, self.target_scaler_path, self.selected_features_path]):
            logger.warning(f"[{self.strategy_name}][{stock_code}] 加载准备数据：部分或全部路径未设置。")
            self.selected_feature_names_for_transformer = []
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None

        required_files_exist = all([
            os.path.exists(self.all_prepared_data_npz_path),
            os.path.exists(self.feature_scaler_path),
            os.path.exists(self.target_scaler_path),
            os.path.exists(self.selected_features_path)
        ])

        if not required_files_exist:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 缺失必需的准备数据/Scaler/特征文件，无法加载。")
            self.selected_feature_names_for_transformer = []
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None
        
        try:
            data_npz = np.load(self.all_prepared_data_npz_path)
            features_train = data_npz['features_scaled_train']
            targets_train = data_npz['targets_scaled_train']
            features_val = data_npz['features_scaled_val']
            targets_val = data_npz['targets_scaled_val']
            features_test = data_npz['features_scaled_test']
            targets_test = data_npz['targets_scaled_test']
            data_npz.close()

            feature_scaler = joblib.load(self.feature_scaler_path)
            target_scaler = joblib.load(self.target_scaler_path)

            with open(self.selected_features_path, 'r', encoding='utf-8') as f:
                 self.selected_feature_names_for_transformer = json.load(f)
            logger.debug(f"[{self.strategy_name}][{stock_code}] 选中特征名列表 ({len(self.selected_feature_names_for_transformer)}个) 从 {self.selected_features_path} 加载。")
            
            logger.info(f"[{self.strategy_name}][{stock_code}] 准备好的数据和 Scaler 已成功加载。")
            return features_train, targets_train, features_val, targets_val, features_test, targets_test, feature_scaler, target_scaler
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 加载准备好的数据、Scaler或特征列表时出错: {e}", exc_info=True)
            self.selected_feature_names_for_transformer = []
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None

    def _calculate_trend_duration(self, data_with_signals: pd.DataFrame) -> Dict[str, Any]:
        """
        计算趋势的持续时间和强度，基于 'final_rule_signal' 列。
        """
        trend_duration_info = {
            'bullish_duration': 0, 'bearish_duration': 0,
            'bullish_duration_text': '0分钟', 'bearish_duration_text': '0分钟',
            'current_trend': '中性', 'trend_strength': '不明', 'duration_status': '短'
        }
        if 'final_rule_signal' not in data_with_signals.columns or data_with_signals['final_rule_signal'].isnull().all():
            logger.warning(f"[{self.strategy_name}] 规则信号列 'final_rule_signal' 不存在或全为空，无法计算趋势持续时间。")
            return trend_duration_info

        final_signal_series = data_with_signals['final_rule_signal'].dropna()
        if final_signal_series.empty: return trend_duration_info

        # 阈值来自 trend_following_params
        trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55)
        trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45)
        strong_bullish_threshold = self.tf_params.get('strong_bullish_threshold', 75) # 来自 tf_params
        strong_bearish_threshold = self.tf_params.get('strong_bearish_threshold', 25) # 来自 tf_params
        moderate_bullish_threshold = self.tf_params.get('moderate_bullish_threshold', 60) # 来自 tf_params
        moderate_bearish_threshold = self.tf_params.get('moderate_bearish_threshold', 40) # 来自 tf_params

        current_bullish_streak = 0
        current_bearish_streak = 0
        for signal_val in final_signal_series.iloc[::-1]: # 从最近的开始往前看
            if signal_val >= trend_threshold_upper:
                current_bullish_streak += 1
                current_bearish_streak = 0 # 中断看跌连胜
            elif signal_val <= trend_threshold_lower:
                current_bearish_streak += 1
                current_bullish_streak = 0 # 中断看涨连胜
            else: # 信号回到中性区域，中断连胜
                break 
        
        trend_duration_info['bullish_duration'] = current_bullish_streak
        trend_duration_info['bearish_duration'] = current_bearish_streak

        try: # 转换 focus_timeframe 为分钟数
            timeframe_minutes = int(self.focus_timeframe) # 假设 focus_timeframe 是分钟字符串
            bullish_total_minutes = current_bullish_streak * timeframe_minutes
            bearish_total_minutes = current_bearish_streak * timeframe_minutes
            def format_duration(minutes):
                if minutes == 0: return "0分钟"
                if minutes < 60: return f"{minutes}分钟"
                hours, rem_minutes = divmod(minutes, 60)
                if hours < 24: return f"{hours}小时{rem_minutes}分钟" if rem_minutes else f"{hours}小时"
                days, rem_hours = divmod(hours, 24)
                return f"{days}天{rem_hours}小时" if rem_hours else f"{days}天"
            trend_duration_info['bullish_duration_text'] = format_duration(bullish_total_minutes)
            trend_duration_info['bearish_duration_text'] = format_duration(bearish_total_minutes)
        except ValueError:
            logger.warning(f"[{self.strategy_name}] 无法将 focus_timeframe '{self.focus_timeframe}' 转换为分钟数。持续时间将以周期数显示。")
            trend_duration_info['bullish_duration_text'] = f"{current_bullish_streak}个周期"
            trend_duration_info['bearish_duration_text'] = f"{current_bearish_streak}个周期"

        latest_rule_signal_val = final_signal_series.iloc[-1]
        if latest_rule_signal_val >= strong_bullish_threshold:
            trend_duration_info.update({'current_trend': '看涨↑', 'trend_strength': '非常强烈'})
        elif latest_rule_signal_val >= moderate_bullish_threshold:
            trend_duration_info.update({'current_trend': '看涨↑', 'trend_strength': '强'})
        elif latest_rule_signal_val >= trend_threshold_upper:
            trend_duration_info.update({'current_trend': '看涨↑', 'trend_strength': '温和'})
        elif latest_rule_signal_val <= strong_bearish_threshold:
            trend_duration_info.update({'current_trend': '看跌↓', 'trend_strength': '非常强烈'})
        elif latest_rule_signal_val <= moderate_bearish_threshold:
            trend_duration_info.update({'current_trend': '看跌↓', 'trend_strength': '强'})
        elif latest_rule_signal_val <= trend_threshold_lower:
            trend_duration_info.update({'current_trend': '看跌↓', 'trend_strength': '温和'})
        
        current_duration_periods = max(current_bullish_streak, current_bearish_streak)
        # trend_duration_threshold_strong/moderate 来自 trend_following_params
        if current_duration_periods >= self.trend_duration_threshold_strong:
             trend_duration_info['duration_status'] = '长'
        elif current_duration_periods >= self.trend_duration_threshold_moderate:
             trend_duration_info['duration_status'] = '中'
        
        return trend_duration_info

    def analyze_signals(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        分析趋势策略信号，生成解读和建议。
        """
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 中间数据为空，无法进行信号分析。")
            return None
        
        analysis_results_dict = {}
        latest_data_row = self.intermediate_data.iloc[-1]

        # --- 统计分析 (基于 combined_signal) ---
        if 'combined_signal' in self.intermediate_data.columns:
            combined_signal_series = self.intermediate_data['combined_signal'].dropna()
            if not combined_signal_series.empty:
                analysis_results_dict['combined_signal_mean'] = combined_signal_series.mean()
                # ... (其他统计可以添加)
        
        trend_duration_info_dict = self._calculate_trend_duration(self.intermediate_data)
        analysis_results_dict.update(trend_duration_info_dict)

        # --- 最新信号判断和操作建议 (基于 combined_signal) ---
        signal_judgment_dict = {}
        operation_advice_str = "中性观望"
        risk_warning_str = ""
        t_plus_1_note_str = "（受 T+1 限制，建议次日操作）"
        stop_loss_profit_advice_str = ""

        final_score_val = latest_data_row.get('combined_signal', 50.0)
        # 阈值来自 trend_following_params
        moderate_bullish_thresh = self.tf_params.get('moderate_bullish_threshold', 60)
        strong_bullish_thresh = self.tf_params.get('strong_bullish_threshold', 75)
        moderate_bearish_thresh = self.tf_params.get('moderate_bearish_threshold', 40)
        strong_bearish_thresh = self.tf_params.get('strong_bearish_threshold', 25)
        duration_status_rule_str = trend_duration_info_dict['duration_status']

        if final_score_val >= moderate_bullish_thresh:
             signal_judgment_dict['overall_signal'] = "看涨信号"
             if final_score_val >= strong_bullish_thresh:
                  signal_judgment_dict['overall_signal'] += " (强)"
                  operation_advice_str = f"持有或逢低加仓 (信号强劲) {t_plus_1_note_str}"
                  if duration_status_rule_str == '长': operation_advice_str = f"坚定持有或加仓 (信号强劲且趋势持续) {t_plus_1_note_str}"
             else: # 温和看涨
                  signal_judgment_dict['overall_signal'] += " (温和)"
                  operation_advice_str = f"观望或轻仓试多 (信号温和) {t_plus_1_note_str}"
                  if duration_status_rule_str == '长': operation_advice_str = f"谨慎持有 (信号温和但趋势已久) {t_plus_1_note_str}"
        elif final_score_val <= moderate_bearish_thresh:
             signal_judgment_dict['overall_signal'] = "看跌信号"
             if final_score_val <= strong_bearish_thresh:
                  signal_judgment_dict['overall_signal'] += " (强)"
                  operation_advice_str = f"卖出或逢高减仓 (信号强劲) {t_plus_1_note_str}"
                  if duration_status_rule_str == '长': operation_advice_str = f"坚定空仓或减仓 (信号强劲且趋势持续) {t_plus_1_note_str}"
             else: # 温和看跌
                  signal_judgment_dict['overall_signal'] += " (温和)"
                  operation_advice_str = f"观望或轻仓试空 (信号温和) {t_plus_1_note_str}"
                  if duration_status_rule_str == '长': operation_advice_str = f"谨慎空仓或观望 (信号温和但趋势已久) {t_plus_1_note_str}"
        else: # 中性
            signal_judgment_dict['overall_signal'] = "中性信号"
            operation_advice_str = f"中性观望，等待信号明朗 {t_plus_1_note_str}"
        
        # --- 结合其他指标细化 ---
        alignment_val = latest_data_row.get('alignment_signal', 0) # 来自 _perform_trend_analysis
        if alignment_val == 3: signal_judgment_dict['alignment_status'] = "完全多头排列"
        elif alignment_val == -3: signal_judgment_dict['alignment_status'] = "完全空头排列"
        # ... (更多 alignment 状态)

        adx_signal_val = latest_data_row.get('adx_strength_signal', 0) # 来自 _perform_trend_analysis
        if abs(adx_signal_val) < 0.5 and abs(final_score_val - 50) > 20 : # ADX弱但信号强
            risk_warning_str += "ADX显示趋势强度不足，注意假信号风险。 "

        # 背离判断 (HAS_BEARISH_DIVERGENCE 等列应由 _calculate_rule_based_signal 中的 detect_divergence 产生并合并到 intermediate_data)
        has_bearish_div_val = latest_data_row.get('HAS_BEARISH_DIVERGENCE', False)
        has_bullish_div_val = latest_data_row.get('HAS_BULLISH_DIVERGENCE', False)
        if has_bearish_div_val and final_score_val > 50:
            signal_judgment_dict['divergence_status'] = "检测到顶背离"
            risk_warning_str += "检测到顶背离，趋势可能衰竭或反转！ "
        elif has_bullish_div_val and final_score_val < 50:
            signal_judgment_dict['divergence_status'] = "检测到底背离"
            risk_warning_str += "检测到底背离，趋势可能衰竭或反转！ "
        else:
            signal_judgment_dict['divergence_status'] = "无明显背离"

        # --- 生成中文解读 ---
        now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        chinese_interpretation_str = (
            f"【趋势跟踪策略分析 - {stock_code} - {now_str}】\n"
            f"最新组合信号分: {final_score_val:.2f} (规则: {latest_data_row.get('final_rule_signal', 50.0):.2f}, Transformer: {latest_data_row.get('transformer_signal', 50.0):.2f})\n"
            f"当前趋势状态: {signal_judgment_dict.get('overall_signal', '中性')}\n"
            f"规则趋势判断: {trend_duration_info_dict['current_trend']} ({trend_duration_info_dict['trend_strength']})\n"
            f"趋势持续: {trend_duration_info_dict.get('bullish_duration_text' if trend_duration_info_dict.get('current_trend','').startswith('看涨') else 'bearish_duration_text','未知')} ({trend_duration_info_dict['duration_status']})\n"
            f"EMA排列: {signal_judgment_dict.get('alignment_status', '未知')}\n"
            f"ADX强度: {latest_data_row.get('adx_strength_signal', 0.0):.2f}\n" # 直接显示数值
            f"背离状态: {signal_judgment_dict.get('divergence_status', '未知')}\n"
            f"操作建议: {operation_advice_str}\n"
            f"风险提示: {risk_warning_str if risk_warning_str else '无明显风险提示。'}\n"
            # f"止损止盈建议: {stop_loss_profit_advice_str if stop_loss_profit_advice_str else '根据自身风险偏好设置。'}" # 暂时移除，简化
        )
        analysis_results_dict['signal_judgment'] = signal_judgment_dict
        analysis_results_dict['operation_advice'] = operation_advice_str
        analysis_results_dict['risk_warning'] = risk_warning_str
        analysis_results_dict['chinese_interpretation'] = chinese_interpretation_str

        self.analysis_results = analysis_results_dict # 存储到实例属性
        logger.info(f"[{self.strategy_name}][{stock_code}] 信号分析完成。")
        logger.info(chinese_interpretation_str) # 打印解读

        return analysis_results_dict

    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回策略计算过程中的中间数据DataFrame。"""
        return self.intermediate_data

    def get_analysis_results(self) -> Optional[Dict[str, Any]]:
        """返回信号分析结果字典。"""
        return self.analysis_results

    def save_analysis_results(self, stock_code: str, timestamp: pd.Timestamp, data: Optional[pd.DataFrame]=None):
        """
        保存趋势跟踪策略的分析结果到数据库。
        'data' 参数在这里通常不需要，因为分析结果基于 self.intermediate_data 和 self.analysis_results。
        """
        # 导入模型应在函数内部，以避免循环导入或在类加载时就尝试导入Django模型
        from stock_models.stock_analytics import StockScoreAnalysis
        from stock_models.stock_basic import StockInfo
        
        if self.analysis_results is None:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 无分析结果可保存。请先运行 analyze_signals。")
            return
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 中间数据为空，部分保存字段可能缺失。")
            latest_intermediate_row = pd.Series(dtype=object) # 空Series
        else:
            latest_intermediate_row = self.intermediate_data.iloc[-1]

        try:
            stock_obj = StockInfo.objects.get(stock_code=stock_code)
            
            def convert_nan_to_none(value): # 辅助函数处理 NaN 和 infinity
                if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
                    return None
                return value if pd.notna(value) else None # 处理 pandas的NA类型

            defaults_payload = {
                'score': convert_nan_to_none(latest_intermediate_row.get('combined_signal')),
                'rule_signal': convert_nan_to_none(latest_intermediate_row.get('final_rule_signal')),
                'lstm_signal': convert_nan_to_none(latest_intermediate_row.get('transformer_signal')), # 保存到 lstm_signal 字段
                'base_score_raw': convert_nan_to_none(latest_intermediate_row.get('base_score_raw')),
                'base_score_volume_adjusted': convert_nan_to_none(latest_intermediate_row.get('base_score_volume_adjusted')),
                'alignment_signal': convert_nan_to_none(latest_intermediate_row.get('alignment_signal')),
                'long_term_context': convert_nan_to_none(latest_intermediate_row.get('long_term_context')),
                # 'trend_strength_score': convert_nan_to_none(latest_intermediate_row.get('trend_strength_score')), # 如果计算了
                'adx_strength_signal': convert_nan_to_none(latest_intermediate_row.get('adx_strength_signal')),
                'stoch_signal': convert_nan_to_none(latest_intermediate_row.get('stoch_signal')),
                'div_has_bearish_divergence': latest_intermediate_row.get('HAS_BEARISH_DIVERGENCE', False), # 来自 detect_divergence
                'div_has_bullish_divergence': latest_intermediate_row.get('HAS_BULLISH_DIVERGENCE', False),
                'close_price': convert_nan_to_none(latest_intermediate_row.get(f'close_{self.focus_timeframe}')),
                'current_trend': self.analysis_results.get('current_trend'),
                'trend_strength': self.analysis_results.get('trend_strength'),
                'bullish_duration': convert_nan_to_none(self.analysis_results.get('bullish_duration')),
                'bearish_duration': convert_nan_to_none(self.analysis_results.get('bearish_duration')),
                'operation_advice': self.analysis_results.get('operation_advice'),
                'risk_warning': self.analysis_results.get('risk_warning'),
                'chinese_interpretation': self.analysis_results.get('chinese_interpretation'),
                'params_snapshot': self.params, # 保存当前策略使用的参数
            }
            
            # 清理 NaN 和确保类型正确
            for key, value in defaults_payload.items():
                if key != 'params_snapshot': # params_snapshot 是字典，不需要转换
                     defaults_payload[key] = convert_nan_to_none(value)


            obj, created = StockScoreAnalysis.objects.update_or_create(
                stock=stock_obj,
                strategy_name=self.strategy_name, # 使用从参数加载的策略名
                timestamp=timestamp, # 传入的时间戳
                time_level=self.focus_timeframe, # 策略关注的时间级别
                defaults=defaults_payload
            )
            status_msg = "创建新的" if created else "更新"
            logger.info(f"[{self.strategy_name}][{stock_code}] {status_msg} 策略分析结果记录，时间戳: {timestamp}")

        except StockInfo.DoesNotExist:
            logger.error(f"[{self.strategy_name}] 股票 {stock_code} 未在数据库中找到，无法保存分析结果。")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存策略分析结果时出错: {e}", exc_info=True)
            # logger.error(f"尝试保存的数据 (部分): {{k: v for k, v in defaults_payload.items() if k != 'params_snapshot'}}")

