# 此策略侧重于识别和跟随趋势，主要使用 EMA 排列、DMI、SAR 等指标，并以 30 分钟级别为主要权重。
# strategies/trend_following_strategy.py
import pandas as pd
import numpy as np
import json
import os
import logging
from pathlib import Path
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

# 读取指标规范命名json
def load_naming_config():
    with open(settings.INDICATOR_NAMING_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_params_config():
    with open(settings.INDICATOR_PARAMETERS_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

NAMING_CONFIG = load_naming_config()

logger = logging.getLogger("strategy_trend_following") # 策略特定的 logger

class TrendFollowingStrategy:
    """
    趋势跟踪策略：
    - 基于多时间框架指标评分，并根据参数侧重特定时间框架 (`focus_timeframe`)。
    - 主要关注趋势指标 (DMI, SAR, MACD, EMA排列, OBV趋势) 和趋势强度 (ADX)。
    - 结合量能确认、波动率、STOCH、VWAP、BOLL等辅助判断趋势的持续性、增强或衰竭。
    - 增加量价背离检测作为潜在反转的警示信号。
    - 适应A股 T+1 交易制度，增强假信号过滤，动态调整参数。
    - 集成 Transformer 模型进行信号预测增强。
    """
    # 类属性，作为最终的备用默认值，如果JSON和实例都未能提供名称
    strategy_name_class_default = "TrendFollowingStrategy_ClassDefault" # 改个名字以区分
    default_focus_timeframe = '30' # 默认主要关注的时间框架

    def __init__(self, params_file: str = None, base_data_dir: str = None):
        """
        初始化趋势跟踪策略。

        Args:
            params_file (str): 策略参数JSON文件的路径。
                               可以是绝对路径，也可以是相对于项目根目录或当前工作目录的相对路径。
            base_data_dir (str): 存储策略相关数据（如模型、scalers）的基础目录。
                                 默认为 Django settings 中的 STRATEGY_DATA_DIR。
        """
        if params_file is None:
            params_file = settings.INDICATOR_PARAMETERS_CONFIG_PATH
        if base_data_dir is None:
            # 确保 settings 中定义了 STRATEGY_DATA_DIR
            if not hasattr(settings, 'STRATEGY_DATA_DIR') or not settings.STRATEGY_DATA_DIR:
                 logger.error("CRITICAL: Django settings.STRATEGY_DATA_DIR 未配置!")
                 # 设置一个备用默认值，但强烈建议在 settings.py 中配置
                 base_data_dir = os.path.join(Path.home(), ".stock_quant_data", "strategy_data")
                 logger.warning(f"使用默认备用策略数据目录: {base_data_dir}")
            else:
                base_data_dir = settings.STRATEGY_DATA_DIR

        self.params_file = params_file
        self.base_data_dir = base_data_dir
        # --- 阶段 0: 初始化局部变量 ---
        loaded_params: Dict[str, Any] = {} # 用于存储从文件加载的参数
        resolved_params_file_path = params_file # 初始化解析后的路径为传入的路径
        file_load_success = False # 标记参数文件是否成功加载并解析
        
        # 使用类名作为临时的日志前缀，直到实例的 strategy_name 被最终确定
        temp_log_prefix = f"[{TrendFollowingStrategy.strategy_name_class_default}-init]"

        # --- 阶段 1: 解析参数文件的绝对路径 ---
        logger.debug(f"{temp_log_prefix} 接收到参数文件路径: '{params_file}'")
        # 检查 params_file 是否为空字符串或 None
        if not params_file:
            logger.error(f"{temp_log_prefix} 警告: params_file 参数为空或 None。无法加载配置。")
            resolved_params_file_path = None # 标记为无效路径
        elif not os.path.isabs(params_file): # 如果不是绝对路径
            logger.debug(f"{temp_log_prefix} 参数文件路径是相对路径，开始解析...")
            if hasattr(settings, 'BASE_DIR') and settings.BASE_DIR:
                path_based_on_base_dir = os.path.join(settings.BASE_DIR, params_file)
                if os.path.exists(path_based_on_base_dir) and os.path.isfile(path_based_on_base_dir):
                    resolved_params_file_path = path_based_on_base_dir
                    logger.debug(f"{temp_log_prefix} 参数文件解析为基于 BASE_DIR 的路径: '{resolved_params_file_path}'")
                else:
                    logger.debug(f"{temp_log_prefix} 未在 BASE_DIR ('{settings.BASE_DIR}') 下找到 '{params_file}'。尝试基于 CWD...")
                    path_based_on_cwd = os.path.abspath(params_file) 
                    if os.path.exists(path_based_on_cwd) and os.path.isfile(path_based_on_cwd):
                        resolved_params_file_path = path_based_on_cwd
                        # 修改: 将 CWD 路径解析成功的日志级别降低，并强调建议使用相对 BASE_DIR 的路径
                        logger.info(f"{temp_log_prefix} 参数文件在当前工作目录 '{os.getcwd()}' 找到: '{resolved_params_file_path}'. 建议使用相对于项目根目录 (BASE_DIR) 的路径以提高健壮性。")
                    else:
                        # 修改: 对于相对路径在 BASE_DIR 和 CWD 都找不到的情况，提升日志级别到 ERROR 或 CRITICAL
                        logger.error(f"{temp_log_prefix} CRITICAL: 相对参数文件 '{params_file}' 在 BASE_DIR 和 CWD (解析为 '{path_based_on_cwd}') 中均未找到。无法加载参数。")
                        resolved_params_file_path = None # 标记为无效路径
            else: 
                logger.warning(f"{temp_log_prefix} Django settings.BASE_DIR 未定义。尝试基于 CWD 解析相对路径 '{params_file}'...")
                path_based_on_cwd = os.path.abspath(params_file)
                if os.path.exists(path_based_on_cwd) and os.path.isfile(path_based_on_cwd):
                    resolved_params_file_path = path_based_on_cwd
                    # 修改: 将 CWD 路径解析成功的日志级别降低，并强调建议定义 BASE_DIR
                    logger.info(f"{temp_log_prefix} Django settings.BASE_DIR 未定义。相对参数文件在当前工作目录 '{os.getcwd()}' 找到: '{resolved_params_file_path}'. 强烈建议定义 settings.BASE_DIR。")
                else:
                    # 修改: 对于 BASE_DIR 未定义且相对路径在 CWD 找不到的情况，提升日志级别到 ERROR 或 CRITICAL
                    logger.error(f"{temp_log_prefix} CRITICAL: Django settings.BASE_DIR 未定义，且相对参数文件在 CWD (解析为 '{path_based_on_cwd}') 也未找到。无法加载参数。")
                    resolved_params_file_path = None # 标记为无效路径
        else: 
            logger.debug(f"{temp_log_prefix} 参数文件路径 '{params_file}' 是绝对路径，直接使用。")
            resolved_params_file_path = params_file

        # --- 阶段 2: 从解析后的路径加载参数文件 ---
        # 修改: 在尝试加载前，先检查 resolved_params_file_path 是否有效
        if resolved_params_file_path is not None:
            logger.info(f"{temp_log_prefix} 尝试从最终路径 '{resolved_params_file_path}' 加载参数...")
            if os.path.exists(resolved_params_file_path) and os.path.isfile(resolved_params_file_path):
                try:
                    with open(resolved_params_file_path, 'r', encoding='utf-8') as f:
                        loaded_params = json.load(f)
                    if loaded_params and isinstance(loaded_params, dict):
                        file_load_success = True
                        logger.info(f"{temp_log_prefix} 策略参数已成功从 '{resolved_params_file_path}' 解析。顶层键数量: {len(loaded_params)}. 顶层键 (部分): {list(loaded_params.keys())[:5]}")
                    else:
                        # 修改: 文件存在但内容为空或无效 JSON，提升日志级别到 ERROR 或 CRITICAL
                        logger.error(f"{temp_log_prefix} CRITICAL: 参数文件 '{resolved_params_file_path}' 内容为空或不是有效的JSON对象 (解析后类型: {type(loaded_params)}).")
                        loaded_params = {} # 确保 loaded_params 是一个空字典
                except FileNotFoundError: 
                    # 修改: 尽管之前检查存在，但打开时未找到，这非常异常，提升日志级别
                    logger.error(f"{temp_log_prefix} CRITICAL: 文件 '{resolved_params_file_path}' 在尝试打开时未找到 (despite previous check).")
                except PermissionError:
                    # 修改: 权限错误，提升日志级别
                    logger.error(f"{temp_log_prefix} CRITICAL: 没有权限读取参数文件 '{resolved_params_file_path}'。")
                except json.JSONDecodeError as e_json:
                    # 修改: JSON 解析错误，提升日志级别
                    logger.error(f"{temp_log_prefix} CRITICAL: 解析参数文件 '{resolved_params_file_path}' 时发生JSON解码错误: {e_json}")
                except Exception as e_load: 
                    # 修改: 其他未知加载错误，提升日志级别
                    logger.error(f"{temp_log_prefix} CRITICAL: 加载参数文件 '{resolved_params_file_path}' 时发生未知错误: {e_load}", exc_info=True)
            else:
                # 修改: 最终确认文件不存在或不是文件，提升日志级别
                logger.error(f"{temp_log_prefix} CRITICAL: 最终确认参数文件 '{resolved_params_file_path}' (原始输入: '{params_file}') 不存在或不是文件。无法加载参数。")
        else:
            # 如果 resolved_params_file_path 在解析阶段就无效，直接记录无法加载
             logger.error(f"{temp_log_prefix} CRITICAL: 参数文件路径解析失败。无法加载参数文件 '{params_file}'.")


        if not file_load_success:
            # 修改: 如果文件加载失败，明确记录策略将使用默认参数运行
            logger.warning(f"{temp_log_prefix} 参数文件加载失败或内容无效，策略将使用默认参数运行。")
            loaded_params = {} # 确保 loaded_params 是一个空字典

        # --- 阶段 3: 设置 self.params (之前由 BaseStrategy 完成) ---
        self.params: Dict[str, Any] = loaded_params # 修改：直接将加载的参数赋值给 self.params
        # 修改: 增加对 self.params 是否为空的日志判断
        logger.debug(f"{temp_log_prefix} self.params 已设置。是否为空: {not bool(self.params)}. 顶层键 (部分): {list(self.params.keys())[:5] if self.params else 'None'}")
        
        # --- 阶段 4: 设置 TrendFollowingStrategy 实例的最终 strategy_name ---
        # 修改: 优化 strategy_name 的设置逻辑，确保总能得到一个字符串名称
        strategy_name_from_params = self.params.get('trend_following_strategy_name')
        if isinstance(strategy_name_from_params, str) and strategy_name_from_params:
            self.strategy_name = strategy_name_from_params
            logger.info(f"[{self.strategy_name}-init-阶段4] 实例策略名从参数文件成功设置为: '{self.strategy_name}'")
        else:
            self.strategy_name = TrendFollowingStrategy.strategy_name_class_default
            if strategy_name_from_params is None:
                 logger.warning(f"[{self.strategy_name}-init-阶段4] 参数中未找到 'trend_following_strategy_name' 键，实例策略名使用类默认值: '{self.strategy_name}'")
            elif not isinstance(strategy_name_from_params, str) or not strategy_name_from_params:
                 logger.warning(f"[{self.strategy_name}-init-阶段4] 参数中的 'trend_following_strategy_name' 无效 (非字符串或为空): '{strategy_name_from_params}'，实例策略名使用类默认值: '{self.strategy_name}'")
        
        log_prefix = f"[{self.strategy_name}]"

        # --- 阶段 5: 初始化 TrendFollowingStrategy 特有的其他属性 ---
        self.base_data_dir = base_data_dir
        logger.debug(f"{log_prefix} base_data_dir 设置为: '{self.base_data_dir}'")

        # 修改: 安全获取 trend_following_params，即使 self.params 为空
        self.tf_params: Dict[str, Any] = self.params.get('trend_following_params', {})

        if not self.params: 
            logger.error(f"{log_prefix} CRITICAL INIT (最终属性设置前): 策略参数 (self.params) 仍为空！后续属性将完全依赖代码默认值。")
        elif not self.tf_params: 
            logger.error(f"{log_prefix} CRITICAL INIT (最终属性设置前): 'trend_following_params' 块在已加载的参数中缺失或为空！后续特定参数将依赖代码默认值。")
        
        # 使用 .get() 方法安全获取参数，提供默认值
        self.focus_timeframe: str = str(self.tf_params.get('focus_timeframe', self.default_focus_timeframe))
        self.timeframe_weights: Optional[Dict[str, float]] = self.tf_params.get('timeframe_weights', None)
        # 修改: 确保 trend_indicators 是列表，即使参数中不是
        self.trend_indicators: List[str] = self.tf_params.get('trend_indicators', ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi'])
        if not isinstance(self.trend_indicators, list):
            logger.warning(f"{log_prefix} 参数 'trend_indicators' 不是一个列表，使用默认值。")
            self.trend_indicators = ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi']

        # 修改: 确保 rule_signal_weights 是字典，即使参数中不是
        self.rule_signal_weights: Dict[str, float] = self.tf_params.get('rule_signal_weights', {
            'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15,
            'ema_cross': 0.15, 'boll_breakout': 0.1, 'adx_strength': 0.1, 'vwap_deviation': 0.05,
            'volume_spike': 0.05 # 注意这里有 volume_spike
        })
        if not isinstance(self.rule_signal_weights, dict) or not self.rule_signal_weights:
             logger.warning(f"{log_prefix} 参数 'rule_signal_weights' 无效或为空，使用代码默认值。")
             self.rule_signal_weights = {
                'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15,
                'ema_cross': 0.15, 'boll_breakout': 0.1, 'adx_strength': 0.1, 'vwap_deviation': 0.05,
                'volume_spike': 0.05
            }

        vc_global_params = self.params.get('volume_confirmation', {})
        self.volume_boost_factor: float = self.tf_params.get('volume_boost_factor', vc_global_params.get('boost_factor', 1.2))
        self.volume_penalty_factor: float = self.tf_params.get('volume_penalty_factor', vc_global_params.get('penalty_factor', 0.8))
        self.volume_spike_threshold: float = self.tf_params.get('volume_spike_threshold', 2.0)

        self.volatility_threshold_high: float = self.tf_params.get('volatility_threshold_high', 10.0)
        self.volatility_threshold_low: float = self.tf_params.get('volatility_threshold_low', 5.0)
        self.volatility_adjust_factor: float = 1.0 # 这个值似乎是内部计算的，不是参数

        self.adx_strong_threshold: int = self.tf_params.get('adx_strong_threshold', 30)
        self.adx_moderate_threshold: int = self.tf_params.get('adx_moderate_threshold', 20)
        self.trend_duration_threshold_strong: int = self.tf_params.get('trend_duration_threshold_strong', 5) 
        self.trend_duration_threshold_moderate: int = self.tf_params.get('trend_duration_threshold_moderate', 10)
        self.stoch_oversold_threshold: int = self.tf_params.get('stoch_oversold_threshold', 20)
        self.stoch_overbought_threshold: int = self.tf_params.get('stoch_overbought_threshold', 80)
        self.vwap_deviation_threshold: float = self.tf_params.get('vwap_deviation_threshold', 0.01)
        self.trend_confirmation_periods: int = self.tf_params.get('trend_confirmation_periods', 3)

        self.transformer_window_size: int = self.tf_params.get('transformer_window_size', 60)
        self.transformer_batch_size: int = self.tf_params.get('transformer_batch_size', 128) 
        self.transformer_target_column: str = self.tf_params.get('transformer_target_column', 'final_rule_signal')

        # 修改: 安全获取 transformer_model_config 和 transformer_training_config
        self.transformer_model_config: Dict[str, Any] = self.tf_params.get('transformer_model_config', {})
        self.transformer_training_config: Dict[str, Any] = self.tf_params.get('transformer_training_config', {})

        # 确保训练配置中的 batch_size, learning_rate, weight_decay 覆盖模型配置中的值（如果存在）
        if 'batch_size' in self.transformer_training_config:
             self.transformer_batch_size = self.transformer_training_config['batch_size']
        # 修改: learning_rate 和 weight_decay 通常是训练过程的参数，不是模型结构参数
        # if 'learning_rate' in self.transformer_training_config: 
        #     self.transformer_model_config['learning_rate'] = self.transformer_training_config['learning_rate']
        # if 'weight_decay' in self.transformer_training_config: 
        #     self.transformer_model_config['weight_decay'] = self.transformer_training_config['weight_decay']

        # 修改: 安全获取 transformer_data_prep_config
        self.transformer_data_prep_config: Dict[str, Any] = self.tf_params.get('transformer_data_prep_config', {})


        self.transformer_model: Optional[nn.Module] = None
        self.feature_scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
        self.target_scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
        self.selected_feature_names_for_transformer: List[str] = []
        # 检查CUDA是否可用，但训练函数中会再次检查并使用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"{log_prefix} 使用设备: {self.device}")

        self.model_path: Optional[str] = None
        self.feature_scaler_path: Optional[str] = None
        self.target_scaler_path: Optional[str] = None
        self.selected_features_path: Optional[str] = None
        self.all_prepared_data_npz_path: Optional[str] = None
        self.intermediate_data: Optional[pd.DataFrame] = None # 用于存储中间数据，可能用于调试或特征工程
        self.analysis_results: Optional[Dict[str, Any]] = None # 用于存储分析结果

        if ta is None:
             logger.error(f"{log_prefix} pandas_ta 未成功加载，策略部分功能可能不可用。")

        # --- 阶段 6: 执行参数验证 ---
        logger.debug(f"{log_prefix} 即将调用 self._validate_params()...")
        try:
            self._validate_params() 
            logger.debug(f"{log_prefix} self._validate_params() 调用完成。")
        except Exception as e_validate:
            # 修改: 在验证过程中发生错误，提升日志级别
            logger.error(f"{log_prefix} CRITICAL: 在执行 _validate_params 时发生错误: {e_validate}", exc_info=True)

        # --- 阶段 7: 初始化完成最终日志 ---
        logger.info(f"策略 '{self.strategy_name}' 初始化流程完成。")
        logger.info(f"{log_prefix} 最终确定的主要关注时间框架: {self.focus_timeframe}.")
        logger.info(f"{log_prefix} 参数最终来源: '{resolved_params_file_path if file_load_success else '无或加载失败'}'.")
        logger.info(f"{log_prefix} self.params 最终是否为空: {not bool(self.params)} (True表示空).")
        logger.info(f"{log_prefix} self.tf_params 最终是否为空: {not bool(self.tf_params)} (True表示空).")
        logger.info(f"{log_prefix} 最终使用的 transformer_window_size: {self.transformer_window_size}")
        logger.info(f"{log_prefix} 最终使用的 transformer_target_column: '{self.transformer_target_column}'")
        # 修改: 安全访问 rule_signal_weights 中的键
        logger.info(f"{log_prefix} 最终使用的 rule_signal_weights (部分): base_score={self.rule_signal_weights.get('base_score') if isinstance(self.rule_signal_weights, dict) else 'N/A'}")
        
        if self.params: 
            logger.debug(f"{log_prefix} 最终已加载参数的顶层键: {list(self.params.keys())}")
            if self.tf_params: 
                 logger.debug(f"{log_prefix} 最终 trend_following_params 内容 (部分): "
                              f"focus_timeframe='{self.tf_params.get('focus_timeframe')}', "
                              f"transformer_model_config.d_model='{self.tf_params.get('transformer_model_config', {}).get('d_model')}'")
            elif 'trend_following_params' in self.params: 
                 logger.warning(f"{log_prefix} 最终 'trend_following_params' 键存在于参数中，但其内容为空。")
        
        logger.info(f"{log_prefix} TrendFollowingStrategy __init__ 执行完毕。")

    @staticmethod # 设为静态方法
    def _format_indicator_name(template_or_list: Union[str, List[str]], **kwargs) -> List[str]:
        """
        格式化指标名称模板或模板列表。
        总是返回一个列表。
        Args:
            template_or_list: 单个模板字符串或模板字符串列表。
            **kwargs: 用于填充模板的参数。
        Returns:
            格式化后的指标名称列表。如果格式化失败，对应的名称会从结果中移除。
        """
        # 清理kwargs中的None值，避免 .format 失败
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        templates = [template_or_list] if isinstance(template_or_list, str) else template_or_list

        formatted_list = []
        for t in templates:
            try:
                # 使用str.format_map()可以避免KeyError，如果字典中缺少key，它会保留原始占位符
                # 但这里我们期望所有占位符都有对应的值，缺少值表示参数配置有问题，所以还是用format
                # 改进：检查模板中是否包含 kwargs 中没有的 key
                # import re
                # keys_in_template = set(re.findall(r'{(\w+)(?::[^}]*)?}', t)) # 查找所有 {key} 或 {key:format}
                # if not keys_in_template.issubset(clean_kwargs.keys()):
                #     missing_keys = keys_in_template - clean_kwargs.keys()
                #     logger.warning(f"格式化指标名模板 '{t}' 缺少参数: {missing_keys}. Kwargs: {clean_kwargs}")
                #     continue # 跳过这个模板

                formatted_list.append(t.format(**clean_kwargs))
            except KeyError as e:
                # 格式化失败（通常是因为缺少参数），记录警告并跳过
                # logger.warning(f"格式化指标名模板 '{t}' 失败，缺少参数: {e}。Kwargs: {clean_kwargs}") # 日志记录可能过于频繁
                pass # 暂时忽略格式化失败的模板

        return formatted_list

    def _normalize_weights(self, weights: Dict[str, float]):
        """归一化权重字典，使其总和为1。"""
        # 修改：增加对 weights 是否为字典的检查
        if not isinstance(weights, dict):
            logger.warning(f"[{self.strategy_name}] 尝试归一化的权重对象不是字典: {type(weights)}")
            return # 不是字典则直接返回，不处理

        total_weight = sum(weights.values())
        # 修改：使用一个小的容差值进行浮点数比较，避免精确比较问题
        if total_weight > 0 and not np.isclose(total_weight, 1.0, atol=1e-9): # atol=1e-9 是一个合理的默认容差
            for key in weights:
                if total_weight != 0: # 避免除以零
                     weights[key] /= total_weight
        elif total_weight == 0:
             logger.warning(f"[{self.strategy_name}] 权重总和为零，无法归一化。权重字典: {weights}")

    def _validate_params(self):
        """
        验证 TrendFollowingStrategy 特定的参数以及从基类继承的参数。
        此方法由 __init__ 在 self.params 被赋值后调用。
        """
        log_prefix = f"[{self.strategy_name}]" 
        if not self.params:
            logger.warning(f"{log_prefix} _validate_params: 由于 self.params 为空 (参数文件可能未加载或无效)，无法进行详细参数验证。策略将依赖代码中的默认值。")
            # 后续验证会继续，但会基于默认值。
            return # 如果 params 为空，后续验证也无意义，直接返回

        # 验证 trend_following_params 块
        if 'trend_following_params' not in self.params:
            logger.error(f"{log_prefix} CRITICAL VALIDATION: 'trend_following_params' 块在参数 (self.params) 中缺失！将完全依赖代码默认值。")
            self.tf_params = {} # 确保 tf_params 是空字典
        elif not self.tf_params: 
            logger.error(f"{log_prefix} CRITICAL VALIDATION: 'trend_following_params' 块在参数中存在，但其内容为空！将依赖代码默认值。")
        
        # 验证 base_scoring 块和 timeframes
        bs_params = self.params.get('base_scoring', {}) 
        if not bs_params:
            logger.warning(f"{log_prefix} VALIDATION: 'base_scoring' 参数块缺失或为空。基础评分相关功能可能受影响。")
        elif not bs_params.get('timeframes'): 
            logger.error(f"{log_prefix} VALIDATION: 'base_scoring.timeframes' 未定义或为空。无法确定操作时间级别。基础评分将无法计算。")

        # 验证 focus_timeframe 是否在 timeframes 中
        timeframes_list = bs_params.get('timeframes', [])
        if self.focus_timeframe not in timeframes_list: 
             logger.warning(f"{log_prefix} VALIDATION: 主要关注时间框架 '{self.focus_timeframe}' 不在 'base_scoring.timeframes' ({timeframes_list}) 中。请检查配置。策略可能无法按预期工作或导致错误。")

        # 验证 timeframe_weights (如果存在)
        if self.timeframe_weights is not None: 
            if not isinstance(self.timeframe_weights, dict):
                logger.error(f"{log_prefix} VALIDATION: 'trend_following_params.timeframe_weights' 必须是一个字典，但得到的是: {type(self.timeframe_weights)}。将忽略此配置。")
                self.timeframe_weights = None 
            else:
                # 检查 timeframe_weights 中的时间框架是否都在 base_scoring.timeframes 中
                for tf_weight in self.timeframe_weights.keys():
                    if tf_weight not in timeframes_list:
                        logger.warning(f"{log_prefix} VALIDATION: timeframe_weights 中包含未在 base_scoring.timeframes 中定义的时间框架: '{tf_weight}'。对应的权重可能不会生效。")
                self._normalize_weights(self.timeframe_weights) # 归一化权重

        # 验证 trend_indicators
        if not self.trend_indicators: 
            logger.warning(f"{log_prefix} VALIDATION: 'trend_following_params.trend_indicators' 为空列表。策略可能无法有效识别趋势。")

        # 验证 Transformer 模型和训练配置
        model_conf = self.transformer_model_config
        required_model_keys = ['d_model', 'nhead', 'dim_feedforward', 'nlayers']
        if not all(key in model_conf for key in required_model_keys):
             logger.warning(f"{log_prefix} VALIDATION: Transformer模型结构配置 'transformer_model_config' 缺少关键参数: {required_model_keys}。可能使用默认值。当前配置: {model_conf}")
        
        train_conf = self.transformer_training_config
        required_train_keys = ['epochs', 'batch_size', 'learning_rate', 'loss']
        if not all(key in train_conf for key in required_train_keys):
             logger.warning(f"{log_prefix} VALIDATION: Transformer训练配置 'transformer_training_config' 缺少关键参数: {required_train_keys}。可能使用默认值。当前配置: {train_conf}")

        # 验证 rule_signal_weights 并归一化
        if not isinstance(self.rule_signal_weights, dict) or not self.rule_signal_weights:
             logger.warning(f"{log_prefix} VALIDATION: 'rule_signal_weights' 参数无效或为空。将使用代码中定义的默认权重并归一化。当前值: {self.rule_signal_weights}")
             self.rule_signal_weights = {
                'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15,
                'ema_cross': 0.15, 'boll_breakout': 0.1, 'adx_strength': 0.1, 'vwap_deviation': 0.05,
                'volume_spike': 0.05
            }
        self._normalize_weights(self.rule_signal_weights) 

        # 验证 signal_combination_weights 并归一化
        # 修改: 从 tf_params 中安全获取 signal_combination_weights
        lstm_combination_weights = self.tf_params.get('signal_combination_weights', {}) 
        if not isinstance(lstm_combination_weights, dict) or not lstm_combination_weights:
             logger.warning(f"{log_prefix} VALIDATION: 'signal_combination_weights' (在 trend_following_params 中) 参数无效或为空。将使用代码中定义的默认组合权重 (0.6/0.4) 并归一化。当前值: {lstm_combination_weights}")
             default_combo_weights = {'rule_weight': 0.6, 'transformer_weight': 0.4} # 修改键名以反映是 Transformer
             # 修改: 直接赋值给 tf_params，确保这个默认值被保存
             if isinstance(self.tf_params, dict):
                 self.tf_params['signal_combination_weights'] = default_combo_weights
             lstm_combination_weights = default_combo_weights # 使用默认值进行归一化
        self._normalize_weights(lstm_combination_weights)
        # 确保 self.tf_params 中的 signal_combination_weights 也是归一化后的值
        if isinstance(self.tf_params, dict) and 'signal_combination_weights' in self.tf_params:
             self.tf_params['signal_combination_weights'] = lstm_combination_weights


        # 检查其他可能需要的参数，例如用于 BOLL 突破、OBV_MA 的周期等
        # 这些参数通常会在 get_required_columns 或其他信号计算部分使用，
        # 可以在这里增加检查，确保关键参数存在于 tf_params 或其他相关参数块中。
        boll_breakout_params = self.tf_params.get('boll_breakout_params', {})
        if self.rule_signal_weights.get('boll_breakout', 0) > 0 and (not boll_breakout_params or 'period' not in boll_breakout_params or 'std_dev' not in boll_breakout_params):
             logger.warning(f"{log_prefix} VALIDATION: 'boll_breakout' 信号权重 > 0，但 'boll_breakout_params' (包含 period, std_dev) 未在 trend_following_params 中找到或无效。将使用默认值或可能导致错误。")

        volume_conf = self.params.get('volume_confirmation', {})
        if volume_conf.get('enabled', False) or volume_conf.get('volume_analysis_enabled', False):
             obv_ma_period = volume_conf.get('obv_ma_period')
             if obv_ma_period is None:
                  logger.warning(f"{log_prefix} VALIDATION: 量能确认启用，但 'volume_confirmation.obv_ma_period' 未定义。OBV MA 相关功能可能受影响。")
             vc_tfs = volume_conf.get('tf')
             if not vc_tfs:
                  logger.warning(f"{log_prefix} VALIDATION: 量能确认启用，但 'volume_confirmation.tf' 未定义或为空。将使用 focus_timeframe ({self.focus_timeframe})。")
             elif isinstance(vc_tfs, str):
                  volume_conf['tf'] = [vc_tfs] # 统一为列表
             elif not isinstance(vc_tfs, list):
                  logger.error(f"{log_prefix} VALIDATION: 'volume_confirmation.tf' 必须是字符串或列表，但得到的是: {type(vc_tfs)}。量能确认可能无法按预期工作。")


        logger.debug(f"{log_prefix} TrendFollowingStrategy 特定参数验证完成。")


    def set_model_paths(self, stock_code: str):
        """
        为特定股票设置模型、scaler 和准备好的数据的保存/加载路径。
        """
        stock_root_dir =  os.path.join(self.base_data_dir, stock_code)
        os.makedirs(stock_root_dir, exist_ok=True) 
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

    def train_transformer_model_from_prepared_data(self, stock_code: str):
        """
        为特定股票加载已准备好的数据，构建并训练 Transformer 模型，然后保存模型权重。
        """
        self.set_model_paths(stock_code)
        logger.info(f"[{self.strategy_name}] 开始为股票 {stock_code} 训练 Transformer 模型 (从已准备数据加载)...")

        # 修改: 增加错误处理，确保 load_prepared_data 返回有效数据
        try:
            features_scaled_train_np, targets_scaled_train_np, \
            features_scaled_val_np, targets_scaled_val_np, \
            features_scaled_test_np, targets_scaled_test_np, \
            feature_scaler, target_scaler = self.load_prepared_data(stock_code)

            if features_scaled_train_np is None or targets_scaled_train_np is None or \
               features_scaled_val_np is None or targets_scaled_val_np is None or \
               features_scaled_test_np is None or targets_scaled_test_np is None or \
               feature_scaler is None or target_scaler is None or not self.selected_feature_names_for_transformer:
                logger.error(f"[{self.strategy_name}] 股票 {stock_code} 加载已准备好的数据/Scaler/特征列表失败或数据无效，无法继续训练。")
                self.transformer_model = None 
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

        except Exception as e:
             logger.error(f"[{self.strategy_name}][{stock_code}] 加载已准备好的数据时发生错误: {e}", exc_info=True)
             self.transformer_model = None
             self.feature_scaler = None
             self.target_scaler = None
             self.selected_feature_names_for_transformer = []
             return

        try:
            train_dataset = TimeSeriesDataset(features_scaled_train_np, targets_scaled_train_np, self.transformer_window_size)
            
            val_loader = None
            # 修改: 检查验证集数据量是否足够创建 Dataset 和 DataLoader
            if features_scaled_val_np.shape[0] > self.transformer_window_size and targets_scaled_val_np.shape[0] > self.transformer_window_size:
                val_dataset = TimeSeriesDataset(features_scaled_val_np, targets_scaled_val_np, self.transformer_window_size)
                if len(val_dataset) > 0:
                    # 修改: 验证集 DataLoader 不需要 shuffle
                    val_loader = DataLoader(val_dataset, batch_size=self.transformer_batch_size, shuffle=False)
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 验证集 Dataset 为空 (数据量不足 {self.transformer_window_size} 或其他原因)。验证阶段将跳过。")
            else:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 验证集数据量不足 {self.transformer_window_size}。验证阶段将跳过。")


            test_loader = None
            # 修改: 检查测试集数据量是否足够创建 Dataset 和 DataLoader
            if features_scaled_test_np.shape[0] > self.transformer_window_size and targets_scaled_test_np.shape[0] > self.transformer_window_size:
                test_dataset = TimeSeriesDataset(features_scaled_test_np, targets_scaled_test_np, self.transformer_window_size)
                if len(test_dataset) > 0:
                    # 修改: 测试集 DataLoader 不需要 shuffle
                    test_loader = DataLoader(test_dataset, batch_size=self.transformer_batch_size, shuffle=False)
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 测试集 Dataset 为空 (数据量不足 {self.transformer_window_size} 或其他原因)。测试评估将跳过。")
            else:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 测试集数据量不足 {self.transformer_window_size}。测试评估将跳过。")


            # 修改: 检查训练集数据量是否足够创建 Dataset 和 DataLoader
            if len(train_dataset) == 0:
                logger.error(f"[{self.strategy_name}][{stock_code}] 训练集 Dataset 为空 (数据量不足 {self.transformer_window_size})。停止训练。")
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
                window_size=self.transformer_window_size # 修改: 传递 window_size 给 build_transformer_model
            )
            self.transformer_model = model
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 构建 Transformer 模型出错: {e}", exc_info=True)
            self.transformer_model = None
            return

        try:
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
                checkpoint_dir=checkpoint_dir, 
                stock_code=stock_code,
                plot_training_history=self.tf_params.get('transformer_plot_history', False),
            )
            # 训练完成后，最佳模型权重应该已经加载到 self.transformer_model
            if self.model_path: 
                 logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型训练完成，最佳模型权重已加载。")
            else:
                 logger.warning(f"[{self.strategy_name}][{stock_code}] Transformer 模型训练完成，但 model_path 未设置，模型权重可能未按预期保存或加载。")


            if test_loader is not None and len(test_loader) > 0 and self.transformer_model is not None:
                logger.info(f"[{self.strategy_name}] 开始在测试集上评估股票 {stock_code} 的 Transformer 模型...")
                # 评估时使用训练配置中指定的损失函数名称来创建损失函数对象
                loss_fn_name = self.transformer_training_config.get('loss', 'mse').lower()
                criterion_eval = nn.MSELoss() if loss_fn_name == 'mse' else \
                                 nn.L1Loss() if loss_fn_name == 'mae' else \
                                 nn.HuberLoss() if loss_fn_name == 'huber' else nn.MSELoss() 
                mae_metric_eval = nn.L1Loss() # 通常评估也会关注 MAE

                test_metrics = evaluate_transformer_model(
                    model=self.transformer_model,
                    test_loader=test_loader,
                    criterion=criterion_eval,
                    mae_metric=mae_metric_eval, # 传递 MAE metric
                    target_scaler=self.target_scaler,
                    device=self.device # 传递 device
                )
                logger.info(f"[{self.strategy_name}][{stock_code}] 测试集评估结果: {test_metrics}")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 训练 Transformer 模型出错: {e}", exc_info=True)
            self.transformer_model = None

    def get_required_columns(self) -> List[str]: # 使用JSON配置
        """
        根据策略参数和指标命名规范，动态生成并返回 IndicatorService 需要准备的所有数据列名。
        """
        required = set()
        log_prefix = f"[{self.strategy_name}]"
        
        # 检查 NAMING_CONFIG 是否已加载
        if not NAMING_CONFIG:
            logger.error(f"{log_prefix} get_required_columns: NAMING_CONFIG 未加载或为空，无法确定所需列。")
            return []

        # 检查策略参数 self.params 是否已加载
        if not self.params:
            logger.error(f"{log_prefix} get_required_columns: 策略参数 (self.params) 为空，无法确定所需列。")
            return []
        
        # 确保 base_scoring 参数块存在且包含 timeframes
        bs_params = self.params.get('base_scoring', {})
        timeframes = bs_params.get('timeframes', [])
        if not timeframes:
            logger.error(f"{log_prefix} 无法获取所需列，因为 'base_scoring.timeframes' 未定义或为空。")
            return []

        # 提取参数的辅助函数
        def _get_param_val(sources: List[Dict], key: str, default: Any = None) -> Any:
            """从参数源列表中按顺序查找键的值，返回第一个找到的值或默认值。"""
            for source_dict in sources:
                # 检查 source_dict 是否是有效的字典
                if isinstance(source_dict, dict) and key in source_dict:
                    return source_dict[key]
            return default

        # 准备参数源列表，优先级从高到低 (策略特定参数通常优先级高)
        param_sources = [
            self.tf_params, # trend_following_params (策略特定参数)
            self.params.get('volume_confirmation', {}), # volume_confirmation
            self.params.get('indicator_analysis_params', {}), # indicator_analysis_params
            self.params.get('feature_engineering_params', {}), # feature_engineering_params
            bs_params # base_scoring
        ]
        
        # 1. 基础OHLCV (来自NAMING_CONFIG.ohlcv_naming_convention)
        ohlcv_config = NAMING_CONFIG.get('ohlcv_naming_convention', {}).get('output_columns', [])
        for tf_str in timeframes:
            for col_conf in ohlcv_config:
                # 确保 column_conf 是字典且包含 'name_pattern'
                if isinstance(col_conf, dict) and 'name_pattern' in col_conf:
                    required.add(f"{col_conf['name_pattern']}_{tf_str}")

        # 2. 基础评分指标 (来自NAMING_CONFIG.indicator_naming_conventions)
        score_indicators_config_keys = bs_params.get('score_indicators', [])
        # 获取指标参数，这些值将用于填充命名模板
        # 使用 _get_param_val 从 param_sources 中获取参数
        # MACD参数
        macd_fast = _get_param_val(param_sources, 'macd_fast', 12)
        macd_slow = _get_param_val(param_sources, 'macd_slow', 26)
        # JSON中是 signal_period, 策略bs_params中可能是 macd_signal_period，这里统一用 signal_period 并在 param_sources 中查找
        macd_sig = _get_param_val(param_sources, 'signal_period', 9) 
        # RSI参数
        rsi_period = _get_param_val(param_sources, 'rsi_period', 14)
        # KDJ参数 (注意JSON和策略中参数名的对应)
        # JSON KDJ: period, signal_period, smooth_k_period
        # 策略 bs_params: kdj_period_k, kdj_period_d, kdj_period_j
        kdj_k_period = _get_param_val(param_sources, 'kdj_period_k', 9)
        kdj_d_period = _get_param_val(param_sources, 'kdj_period_d', 3)
        kdj_j_smooth_k = _get_param_val(param_sources, 'kdj_period_j', 3) # 对应 smooth_k_period
        # BOLL参数 (基础评分用的BOLL)
        boll_period = _get_param_val(param_sources, 'boll_period', 20)
        boll_std_dev = _get_param_val(param_sources, 'boll_std_dev', 2.0)
        # CCI参数
        cci_period = _get_param_val(param_sources, 'cci_period', 14)
        # MFI参数
        mfi_period = _get_param_val(param_sources, 'mfi_period', 14)
        # ROC参数
        roc_period = _get_param_val(param_sources, 'roc_period', 12)
        # DMI参数
        dmi_period = _get_param_val(param_sources, 'dmi_period', 14)
        # SAR参数
        sar_step = _get_param_val(param_sources, 'sar_step', 0.02)
        sar_max_af = _get_param_val(param_sources, 'sar_max', 0.2)
        # EMA/SMA 基础参数 (如果 score_indicators 中包含它们，通常是某个默认周期)
        # 注意：这里的基础周期可能与 feature_engineering 中定义的 EMA/SMA periods 列表不同
        ema_base_period = _get_param_val(param_sources, 'ema_base_period', 20) # 添加一个默认的 base_period 参数查找
        sma_base_period = _get_param_val(param_sources, 'sma_base_period', 20) # 添加一个默认的 base_period 参数查找


        indicator_naming_conv = NAMING_CONFIG.get('indicator_naming_conventions', {})

        for indi_key_upper in [str(key).upper() for key in score_indicators_config_keys]: # 转为大写以匹配JSON键
            if indi_key_upper not in indicator_naming_conv:
                logger.warning(f"{log_prefix} 评分指标 '{indi_key_upper}' 在命名规范中未定义或加载失败，跳过。")
                continue
            
            indi_naming_conf = indicator_naming_conv[indi_key_upper]
            # 确保 indi_naming_conf 是字典且包含 output_columns
            if not isinstance(indi_naming_conf, dict) or 'output_columns' not in indi_naming_conf:
                logger.warning(f"{log_prefix} 评分指标 '{indi_key_upper}' 的命名配置无效，跳过。")
                continue

            params_for_format = {}
            # 根据指标类型填充格式化参数字典
            if indi_key_upper == 'MACD':
                params_for_format = {'period_fast': macd_fast, 'period_slow': macd_slow, 'signal_period': macd_sig}
            elif indi_key_upper == 'RSI':
                params_for_format = {'period': rsi_period}
            elif indi_key_upper == 'KDJ': # KDJ的参数名在JSON和策略中不完全一致，这里做映射
                params_for_format = {'period': kdj_k_period, 'signal_period': kdj_d_period, 'smooth_k_period': kdj_j_smooth_k}
            elif indi_key_upper == 'BOLL':
                params_for_format = {'period': boll_period, 'std_dev': boll_std_dev}
            elif indi_key_upper == 'CCI':
                params_for_format = {'period': cci_period}
            elif indi_key_upper == 'MFI':
                params_for_format = {'period': mfi_period}
            elif indi_key_upper == 'ROC':
                params_for_format = {'period': roc_period}
            elif indi_key_upper == 'DMI':
                params_for_format = {'period': dmi_period}
            elif indi_key_upper == 'SAR':
                params_for_format = {'af_step': sar_step, 'max_af': sar_max_af}
            elif indi_key_upper == 'EMA':
                params_for_format = {'period': ema_base_period} # 使用基础EMA周期
            elif indi_key_upper == 'SMA':
                params_for_format = {'period': sma_base_period} # 使用基础SMA周期
            # 其他无参数或固定参数的指标如 OBV, ADL 不需要特别处理 params_for_format

            for tf_str in timeframes:
                for col_conf in indi_naming_conf['output_columns']:
                     # 确保 col_conf 是字典且包含 'name_pattern'
                     if isinstance(col_conf, dict) and 'name_pattern' in col_conf:
                        base_names = TrendFollowingStrategy._format_indicator_name(col_conf['name_pattern'], **params_for_format)
                        for base_name in base_names:
                            required.add(f"{base_name}_{tf_str}")
        
        # 3. 量能确认指标 (AMT_MA, CMF) 和 **OBV_MA (衍生特征)**
        vc_params = self.params.get('volume_confirmation', {})
        # 修改: 检查量能确认是否启用
        if vc_params.get('enabled', False) or vc_params.get('volume_analysis_enabled', False):
            # 修改: 获取量能确认应用的时间框架列表，如果未指定则使用所有 timeframes
            vc_tf_list_raw = vc_params.get('tf', timeframes) # 如果未指定tf，则对所有timeframes计算
            vc_tf_list = [vc_tf_list_raw] if isinstance(vc_tf_list_raw, str) else vc_tf_list_raw

            # 获取量能指标参数
            amt_ma_period_vc = _get_param_val(param_sources, 'amount_ma_period', 15) # 日志显示是 15
            cmf_period_vc = _get_param_val(param_sources, 'cmf_period', 20)
            # 修改: 获取 OBV MA 周期，日志显示是 10
            obv_ma_period_vc = _get_param_val(param_sources, 'obv_ma_period', 10) # 默认值设为日志中的 10

            for vc_tf_str in vc_tf_list:
                # 检查时间框架是否在 base_scoring.timeframes 中定义
                if vc_tf_str not in timeframes:
                    # logger.warning(f"{log_prefix} 量能确认时间框架 '{vc_tf_str}' 未在 'base_scoring.timeframes' 中定义，跳过此时间框架的量能指标请求。") # 日志记录可能过于频繁
                    continue

                # AMT_MA
                if 'AMT_MA' in indicator_naming_conv:
                    amt_ma_naming_conf = indicator_naming_conv['AMT_MA']
                    if isinstance(amt_ma_naming_conf, dict) and 'output_columns' in amt_ma_naming_conf:
                        amt_ma_name_patterns = [c['name_pattern'] for c in amt_ma_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                        for name in TrendFollowingStrategy._format_indicator_name(amt_ma_name_patterns, period=amt_ma_period_vc):
                            required.add(f"{name}_{vc_tf_str}")

                # CMF
                if 'CMF' in indicator_naming_conv:
                    cmf_naming_conf = indicator_naming_conv['CMF']
                    if isinstance(cmf_naming_conf, dict) and 'output_columns' in cmf_naming_conf:
                        cmf_name_patterns = [c['name_pattern'] for c in cmf_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                        for name in TrendFollowingStrategy._format_indicator_name(cmf_name_patterns, period=cmf_period_vc):
                            required.add(f"{name}_{vc_tf_str}")
                
                # 添加 OBV 作为基础指标 (OBV MA 依赖 OBV)
                if 'OBV' in indicator_naming_conv:
                    obv_naming_conf = indicator_naming_conv['OBV']
                    if isinstance(obv_naming_conf, dict) and 'output_columns' in obv_naming_conf:
                         obv_name_patterns = [c['name_pattern'] for c in obv_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                         for name in TrendFollowingStrategy._format_indicator_name(obv_name_patterns): # OBV 无参数
                            required.add(f"{name}_{vc_tf_str}")

                # OBV_MA (衍生特征) - 需要 OBV 作为基础
                derivative_naming_conv = NAMING_CONFIG.get('derivative_feature_naming_conventions', {})
                if 'OBV_MA' in derivative_naming_conv:
                    obv_ma_deriv_conf = derivative_naming_conv['OBV_MA']
                    if isinstance(obv_ma_deriv_conf, dict) and 'output_column_pattern' in obv_ma_deriv_conf and obv_ma_period_vc is not None:
                        obv_ma_pattern = obv_ma_deriv_conf['output_column_pattern']
                        for name in TrendFollowingStrategy._format_indicator_name(obv_ma_pattern, period=obv_ma_period_vc):
                            required.add(f"{name}_{vc_tf_str}")


        # 4. 其他分析指标 (STOCH, VOL_MA, VWAP, ADL, ICHIMOKU, PIVOT_POINTS, ATR, HV, KC, MOM, WILLR, VROC, AROC)
        # 这些指标可能用于基础评分以外的分析或信号计算
        ia_params = self.params.get('indicator_analysis_params', {})
        fe_params = self.params.get('feature_engineering_params', {})
        
        # 获取分析和特征工程应用的时间框架列表，如果未指定则使用 base_scoring 的 timeframes
        ia_tf_list = ia_params.get('apply_on_timeframes', timeframes)
        if isinstance(ia_tf_list, str): ia_tf_list = [ia_tf_list]
        fe_tf_list = fe_params.get('apply_on_timeframes', timeframes)
        if isinstance(fe_tf_list, str): fe_tf_list = [fe_tf_list]

        # 合并去重时间框架列表
        analysis_timeframes = list(set(ia_tf_list + fe_tf_list))


        for analyze_tf_str in analysis_timeframes:
            # 检查时间框架是否在 base_scoring.timeframes 中定义
            if analyze_tf_str not in timeframes:
                 # logger.warning(f"{log_prefix} 分析/特征工程时间框架 '{analyze_tf_str}' 未在 'base_scoring.timeframes' 中定义，跳过此时间框架的指标请求。")
                 continue # 跳过未在基础时间框架中定义的级别

            # STOCH (JSON: STOCHk, STOCHd)
            stoch_k_ia = _get_param_val(param_sources, 'stoch_k', 14) # 对应 k_period, 日志显示 14
            stoch_d_ia = _get_param_val(param_sources, 'stoch_d', 3)   # 对应 d_period, 日志显示 3
            stoch_smooth_k_ia = _get_param_val(param_sources, 'stoch_smooth_k', 3) # 对应 smooth_k_period, 日志显示 3
            if 'STOCH' in indicator_naming_conv:
                stoch_naming_conf = indicator_naming_conv['STOCH']
                if isinstance(stoch_naming_conf, dict) and 'output_columns' in stoch_naming_conf:
                    stoch_name_patterns = [c['name_pattern'] for c in stoch_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(stoch_name_patterns, k_period=stoch_k_ia, d_period=stoch_d_ia, smooth_k_period=stoch_smooth_k_ia):
                        required.add(f"{name}_{analyze_tf_str}")
            
            # VOL_MA
            vol_ma_period_ia = _get_param_val(param_sources, 'volume_ma_period', 20)
            if 'VOL_MA' in indicator_naming_conv:
                vol_ma_naming_conf = indicator_naming_conv['VOL_MA']
                if isinstance(vol_ma_naming_conf, dict) and 'output_columns' in vol_ma_naming_conf:
                    vol_ma_name_patterns = [c['name_pattern'] for c in vol_ma_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(vol_ma_name_patterns, period=vol_ma_period_ia):
                        required.add(f"{name}_{analyze_tf_str}")

            # VWAP
            # VWAP 参数可以在 indicator_analysis_params 或 feature_engineering_params 中配置
            vwap_anchor_ia = _get_param_val(param_sources, 'vwap_anchor', None)
            if 'VWAP' in indicator_naming_conv:
                vwap_configs = indicator_naming_conv['VWAP']['output_columns']
                # 确保 vwap_configs 是列表且包含字典
                if isinstance(vwap_configs, list):
                    if vwap_anchor_ia is None: # 无锚点
                        # 查找不包含 {anchor} 占位符的模式
                        pattern = next((c['name_pattern'] for c in vwap_configs if isinstance(c, dict) and 'name_pattern' in c and "{anchor}" not in c['name_pattern']), None)
                        if pattern:
                            # 无锚点 VWAP 通常只需要 VWAP 列
                            # 检查 VWAP 是否已在基础指标中为该时间框架请求 (不太可能，VWAP通常是分析指标)
                            # required.add(f"VWAP_{analyze_tf_str}") # 直接添加 VWAP_{tf}
                            for name in TrendFollowingStrategy._format_indicator_name(pattern): # 无参数
                                required.add(f"{name}_{analyze_tf_str}")
                        else:
                             logger.warning(f"{log_prefix} VWAP 命名规范中未找到无锚点的模式。无法为 {analyze_tf_str} 请求无锚点VWAP。")
                    else: # 有锚点
                        # 查找包含 {anchor} 占位符的模式
                        pattern = next((c['name_pattern'] for c in vwap_configs if isinstance(c, dict) and 'name_pattern' in c and "{anchor}" in c['name_pattern']), None)
                        if pattern:
                             for name in TrendFollowingStrategy._format_indicator_name(pattern, anchor=vwap_anchor_ia):
                                required.add(f"{name}_{analyze_tf_str}")
                        else:
                             logger.warning(f"{log_prefix} VWAP 命名规范中未找到带锚点的模式。无法为 {analyze_tf_str} 请求带锚点VWAP (锚点: {vwap_anchor_ia})。")

            # ADL (无参数)
            if ia_params.get('calculate_adl', False) and 'ADL' in indicator_naming_conv: # 默认为 False
                 adl_naming_conf = indicator_naming_conv['ADL']
                 if isinstance(adl_naming_conf, dict) and 'output_columns' in adl_naming_conf:
                    adl_name_patterns = [c['name_pattern'] for c in adl_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(adl_name_patterns): # 无参数
                        required.add(f"{name}_{analyze_tf_str}")

            # ICHIMOKU
            if ia_params.get('calculate_ichimoku', False) and 'ICHIMOKU' in indicator_naming_conv: # 默认为 False
                ichimoku_tenkan_ia = _get_param_val(param_sources, 'ichimoku_tenkan', 9)
                ichimoku_kijun_ia = _get_param_val(param_sources, 'ichimoku_kijun', 26)
                ichimoku_senkou_ia = _get_param_val(param_sources, 'ichimoku_senkou', 52)
                ichimoku_conf = indicator_naming_conv['ICHIMOKU']['output_columns']
                 # 确保 ichimoku_conf 是列表且包含字典
                if isinstance(ichimoku_conf, list):
                    for col_conf in ichimoku_conf:
                        if isinstance(col_conf, dict) and 'name_pattern' in col_conf:
                            params_for_ichi = { # 根据模式中的占位符提供参数
                                'tenkan_period': ichimoku_tenkan_ia,
                                'kijun_period': ichimoku_kijun_ia,
                                'senkou_period': ichimoku_senkou_ia
                            }
                            for name in TrendFollowingStrategy._format_indicator_name(col_conf['name_pattern'], **params_for_ichi):
                                required.add(f"{name}_{analyze_tf_str}")
            
            # PIVOT_POINTS (通常基于日线, 无可变参数)
            # 检查 ia_params 中是否明确配置了计算 PIVOT_POINTS，且当前时间框架是日线 'D'
            if ia_params.get('calculate_pivot_points', False) and analyze_tf_str == 'D' and 'PIVOT_POINTS' in indicator_naming_conv:
                pivot_conf = indicator_naming_conv['PIVOT_POINTS']['output_columns']
                # 确保 pivot_conf 是列表且包含字典
                if isinstance(pivot_conf, list):
                    for col_conf in pivot_conf:
                        if isinstance(col_conf, dict) and 'name_pattern' in col_conf:
                            # Pivot point 名称是固定的，不需要 format
                            required.add(f"{col_conf['name_pattern']}_{analyze_tf_str}")

            # ATR
            atr_period_fe = _get_param_val(param_sources, 'atr_period', 14)
            if fe_params.get('calculate_atr', True) and 'ATR' in indicator_naming_conv:
                 atr_naming_conf = indicator_naming_conv['ATR']
                 if isinstance(atr_naming_conf, dict) and 'output_columns' in atr_naming_conf:
                    atr_patterns = [c['name_pattern'] for c in atr_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(atr_patterns, period=atr_period_fe):
                        required.add(f"{name}_{analyze_tf_str}")
            
            # HV (Historical Volatility)
            hv_period_fe = _get_param_val(param_sources, 'hv_period', 20)
            if fe_params.get('calculate_hv', True) and 'HV' in indicator_naming_conv:
                 hv_naming_conf = indicator_naming_conv['HV']
                 if isinstance(hv_naming_conf, dict) and 'output_columns' in hv_naming_conf:
                    hv_patterns = [c['name_pattern'] for c in hv_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(hv_patterns, period=hv_period_fe):
                        required.add(f"{name}_{analyze_tf_str}")

            # KC (Keltner Channels)
            kc_ema_period_fe = _get_param_val(param_sources, 'kc_ema_period', 20)
            kc_atr_period_fe = _get_param_val(param_sources, 'kc_atr_period', 10)
            if fe_params.get('calculate_kc', True) and 'KC' in indicator_naming_conv:
                 kc_naming_conf = indicator_naming_conv['KC']
                 if isinstance(kc_naming_conf, dict) and 'output_columns' in kc_naming_conf:
                    kc_patterns = [c['name_pattern'] for c in kc_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(kc_patterns, ema_period=kc_ema_period_fe, atr_period=kc_atr_period_fe):
                        required.add(f"{name}_{analyze_tf_str}")
            
            # MOM
            mom_period_fe = _get_param_val(param_sources, 'mom_period', 10)
            if fe_params.get('calculate_mom', True) and 'MOM' in indicator_naming_conv:
                 mom_naming_conf = indicator_naming_conv['MOM']
                 if isinstance(mom_naming_conf, dict) and 'output_columns' in mom_naming_conf:
                    mom_patterns = [c['name_pattern'] for c in mom_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(mom_patterns, period=mom_period_fe):
                        required.add(f"{name}_{analyze_tf_str}")

            # WILLR
            willr_period_fe = _get_param_val(param_sources, 'willr_period', 14)
            if fe_params.get('calculate_willr', True) and 'WILLR' in indicator_naming_conv:
                 willr_naming_conf = indicator_naming_conv['WILLR']
                 if isinstance(willr_naming_conf, dict) and 'output_columns' in willr_naming_conf:
                    willr_patterns = [c['name_pattern'] for c in willr_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(willr_patterns, period=willr_period_fe):
                        required.add(f"{name}_{analyze_tf_str}")
            
            # VROC
            vroc_period_fe = _get_param_val(param_sources, 'vroc_period', 10)
            if fe_params.get('calculate_vroc', True) and 'VROC' in indicator_naming_conv:
                 vroc_naming_conf = indicator_naming_conv['VROC']
                 if isinstance(vroc_naming_conf, dict) and 'output_columns' in vroc_naming_conf:
                    vroc_patterns = [c['name_pattern'] for c in vroc_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(vroc_patterns, period=vroc_period_fe):
                        required.add(f"{name}_{analyze_tf_str}")

            # AROC
            aroc_period_fe = _get_param_val(param_sources, 'aroc_period', 10)
            if fe_params.get('calculate_aroc', True) and 'AROC' in indicator_naming_conv:
                 aroc_naming_conf = indicator_naming_conv['AROC']
                 if isinstance(aroc_naming_conf, dict) and 'output_columns' in aroc_naming_conf:
                    aroc_patterns = [c['name_pattern'] for c in aroc_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(aroc_patterns, period=aroc_period_fe):
                        required.add(f"{name}_{analyze_tf_str}")

            # EMA 列表, SMA 列表 (来自 feature_engineering_params)
            for ma_type_fe_upper in ['EMA', 'SMA']:
                ma_key_lower = ma_type_fe_upper.lower()
                ma_periods_list_fe = fe_params.get(f'{ma_key_lower}_periods', [])
                # 确保 ma_periods_list_fe 是列表
                if not isinstance(ma_periods_list_fe, list):
                     logger.warning(f"{log_prefix} feature_engineering_params 中的 '{ma_key_lower}_periods' 参数不是列表，跳过。")
                     continue

                if ma_periods_list_fe and ma_type_fe_upper in indicator_naming_conv:
                     ma_naming_conf = indicator_naming_conv[ma_type_fe_upper]
                     if isinstance(ma_naming_conf, dict) and 'output_columns' in ma_naming_conf:
                        ma_patterns = [c['name_pattern'] for c in ma_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                        for period in ma_periods_list_fe:
                             # 确保 period 是有效数字
                             if isinstance(period, (int, float)) and period > 0:
                                 for name in TrendFollowingStrategy._format_indicator_name(ma_patterns, period=period):
                                     required.add(f"{name}_{analyze_tf_str}")


        # 5. 衍生特征 (Derivative Features)
        # 遍历 derivative_feature_naming_conventions，根据 feature_engineering_params 中的配置决定是否请求
        derivative_naming_conv = NAMING_CONFIG.get('derivative_feature_naming_conventions', {})
        fe_derivative_features_params = fe_params.get('derivative_features', {}) # 例如 {"CLOSE_MA_RATIO": {"enabled": true, "ma_type": ["ema", "sma"], "periods": [10, 20]}, ...}

        if isinstance(fe_derivative_features_params, dict): # 确保是字典格式
            for deriv_feature_key, deriv_feature_config in fe_derivative_features_params.items():
                 # 确保配置是字典且已启用
                if isinstance(deriv_feature_config, dict) and deriv_feature_config.get('enabled', False):
                    deriv_feature_key_upper = deriv_feature_key.upper()
                    if deriv_feature_key_upper not in derivative_naming_conv:
                         logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' 在命名规范中未定义，跳过。")
                         continue

                    deriv_naming_conf = derivative_naming_conv[deriv_feature_key_upper]
                    # 确保命名配置有效
                    if not isinstance(deriv_naming_conf, dict) or 'output_column_pattern' not in deriv_naming_conf:
                         logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' 的命名配置无效，跳过。")
                         continue

                    pattern = deriv_naming_conf['output_column_pattern']
                    apply_tfs_deriv = deriv_feature_config.get('apply_on_timeframes', fe_tf_list) # 可以在衍生特征层面指定应用时间框架
                    if isinstance(apply_tfs_deriv, str): apply_tfs_deriv = [apply_tfs_deriv]

                    for deriv_tf_str in apply_tfs_deriv:
                        if deriv_tf_str not in timeframes:
                            # logger.warning(f"{log_prefix} 衍生特征时间框架 '{deriv_tf_str}' 未在 'base_scoring.timeframes' 中定义，跳过。")
                            continue # 跳过未在基础时间框架中定义的级别

                        # 根据不同的衍生特征类型，从配置中获取参数并格式化列名
                        if deriv_feature_key_upper == 'CLOSE_MA_RATIO' or deriv_feature_key_upper == 'CLOSE_MA_NDIFF':
                            ma_types = deriv_feature_config.get('ma_type', [])
                            periods = deriv_feature_config.get('periods', [])
                            if isinstance(ma_types, str): ma_types = [ma_types]
                            if isinstance(periods, (int, float)): periods = [periods]
                            if isinstance(ma_types, list) and isinstance(periods, list):
                                for ma_type in ma_types:
                                    for period in periods:
                                         # 确保 period 是有效数字
                                         if isinstance(period, (int, float)) and period > 0:
                                            for name in TrendFollowingStrategy._format_indicator_name(pattern, ma_type=ma_type.upper(), period=period): # MA_TYPE usually upper
                                                required.add(f"{name}_{deriv_tf_str}")
                        
                        elif deriv_feature_key_upper == 'INDICATOR_DIFF':
                            # 这类衍生特征需要知道原始指标名和差分周期
                            # 配置示例: {"INDICATOR_DIFF": {"enabled": true, "indicators": [{"name": "RSI_14", "diff_periods": [1, 3]}]}}
                            indicator_configs = deriv_feature_config.get('indicators', [])
                            if isinstance(indicator_configs, list):
                                for indi_cfg in indicator_configs:
                                    if isinstance(indi_cfg, dict) and 'name' in indi_cfg and 'diff_periods' in indi_cfg:
                                        original_name_pattern = indi_cfg['name'] # 这里的 name 应该是原始指标的模式，例如 "RSI_{period}"
                                        diff_periods = indi_cfg['diff_periods']
                                        if isinstance(diff_periods, (int, float)): diff_periods = [diff_periods]

                                        # 需要先根据原始指标模式和参数生成原始指标名
                                        # 这是一个简化的处理，假设这里的 name 就是最终的不带时间框架的列名模式的一部分
                                        # 例如，如果 original_name_pattern 是 "RSI_{period}"，并且需要RSI_14的diff1，则需要获取到RSI_14
                                        # 更精确的做法是根据原始指标类型和参数来生成，但这里简化处理
                                        original_indicator_base_name = original_name_pattern # 假设这里的 name 就是基础指标名模式，不带时间框架
                                        
                                        if isinstance(diff_periods, list):
                                            for diff_period in diff_periods:
                                                # 确保 diff_period 是有效数字
                                                if isinstance(diff_period, (int, float)) and diff_period > 0:
                                                    # 格式化衍生特征列名模式
                                                    for name in TrendFollowingStrategy._format_indicator_name(pattern, original_indicator_name_with_params=original_indicator_base_name, diff_period=diff_period):
                                                        # 注意：INDICATOR_DIFF 的 original_indicator_name_with_params 应该是原始指标带参数的部分，例如 RSI_14
                                                        # 需要根据原始指标的配置重新构建带参数的名称
                                                        # 例如，如果原始指标是 RSI，参数是 14，那么 original_indicator_name_with_params 就是 "RSI_14"
                                                        # 这是一个复杂的问题，简化处理，假设配置中的 "name" 已经是带参数的基名，例如 "RSI_14"
                                                        # 那么格式化模式就是 "{original_indicator_name_with_params}_DIFF{diff_period}"
                                                        # pattern = "{original_indicator_name_with_params}_DIFF{diff_period}" # 假设模式是这样
                                                        # 检查命名规范中 INDICATOR_DIFF 的模式是否匹配
                                                        deriv_pattern_template = derivative_naming_conv.get('INDICATOR_DIFF', {}).get('output_column_pattern')
                                                        if deriv_pattern_template:
                                                            formatted_name = TrendFollowingStrategy._format_indicator_name(deriv_pattern_template, original_indicator_name_with_params=original_indicator_base_name, diff_period=diff_period)
                                                            required.update([f"{n}_{deriv_tf_str}" for n in formatted_name])
                                                        else:
                                                            logger.warning(f"{log_prefix} 衍生特征 INDICATOR_DIFF 的命名模式未找到。")


                        elif deriv_feature_key_upper == 'CLOSE_BB_POS':
                            # 需要 BOLL 的周期和标准差
                            boll_params_deriv = deriv_feature_config.get('boll_params', {})
                            if isinstance(boll_params_deriv, dict):
                                boll_period_deriv = boll_params_deriv.get('period')
                                boll_std_dev_deriv = boll_params_deriv.get('std_dev')
                                if boll_period_deriv is not None and boll_std_dev_deriv is not None:
                                     # 格式化模式 "{period}_{std_dev:.1f}"
                                     for name in TrendFollowingStrategy._format_indicator_name(pattern, period=boll_period_deriv, std_dev=boll_std_dev_deriv):
                                         required.add(f"{name}_{deriv_tf_str}")
                                else:
                                     logger.warning(f"{log_prefix} 衍生特征 CLOSE_BB_POS 配置缺少 boll_params (period, std_dev)。")
                            else:
                                logger.warning(f"{log_prefix} 衍生特征 CLOSE_BB_POS 配置中的 boll_params 不是字典。")

                        elif deriv_feature_key_upper == 'CLOSE_KC_POS':
                             # 需要 KC 的 EMA 周期和 ATR 周期
                            kc_params_deriv = deriv_feature_config.get('kc_params', {})
                            if isinstance(kc_params_deriv, dict):
                                kc_ema_period_deriv = kc_params_deriv.get('ema_period')
                                kc_atr_period_deriv = kc_params_deriv.get('atr_period')
                                if kc_ema_period_deriv is not None and kc_atr_period_deriv is not None:
                                     for name in TrendFollowingStrategy._format_indicator_name(pattern, ema_period=kc_ema_period_deriv, atr_period=kc_atr_period_deriv):
                                         required.add(f"{name}_{deriv_tf_str}")
                                else:
                                     logger.warning(f"{log_prefix} 衍生特征 CLOSE_KC_POS 配置缺少 kc_params (ema_period, atr_period)。")
                            else:
                                logger.warning(f"{log_prefix} 衍生特征 CLOSE_KC_POS 配置中的 kc_params 不是字典。")

                        # OBV_MA 已经在量能确认部分处理，这里可以跳过或确保一致性
                        # elif deriv_feature_key_upper == 'OBV_MA':
                        #     obv_ma_period_deriv = deriv_feature_config.get('period')
                        #     if obv_ma_period_deriv is not None:
                        #          for name in TrendFollowingStrategy._format_indicator_name(pattern, period=obv_ma_period_deriv):
                        #              required.add(f"{name}_{deriv_tf_str}")
                        #     else:
                        #          logger.warning(f"{log_prefix} 衍生特征 OBV_MA 配置缺少 period。")
                        
                        # 添加其他可能需要的衍生特征处理逻辑

        # 6. 添加策略内部计算所需的其他基础指标列 (如果它们没有在评分/分析/特征工程中包含)
        # 例如，计算 EMA 排列需要多个周期的 EMA。这些周期的列表可以在参数中配置。
        ema_alignment_periods = self.tf_params.get('ema_alignment_periods', [])
        if isinstance(ema_alignment_periods, (int, float)): ema_alignment_periods = [ema_alignment_periods]
        if isinstance(ema_alignment_periods, list) and 'EMA' in indicator_naming_conv:
             ema_naming_conf = indicator_naming_conv['EMA']
             if isinstance(ema_naming_conf, dict) and 'output_columns' in ema_naming_conf:
                ema_patterns = [c['name_pattern'] for c in ema_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                for period in ema_alignment_periods:
                     if isinstance(period, (int, float)) and period > 0:
                        for tf_str in timeframes: # 通常EMA排列会看多个时间框架
                             for name in TrendFollowingStrategy._format_indicator_name(ema_patterns, period=period):
                                required.add(f"{name}_{tf_str}")

        # 添加计算量价背离可能需要的基础指标（如 OBV, ADL, VROC, AROC 等，确保它们已被包含）
        # 背离检测通常也依赖于 OHLCV 数据，确保包含。
        # 添加计算趋势强度信号可能需要的 DMI 指标 (PDI, NDI, ADX)，确保它们已被包含。
        # 添加计算 STOCH 信号可能需要的 STOCHk, STOCHd，确保它们已被包含。
        # 添加计算 VWAP 偏离可能需要的 VWAP 和 close，确保已被包含。
        # 添加计算 BOLL 突破可能需要的 BBU, BBL 和 close，确保已被包含。
        # 特别是日志中警告缺失的 BOLL (15, 2.2)，如果它不是基础评分用的，它可能需要单独在 tf_params 或 fe_params 中配置
        boll_breakout_params_tf = self.tf_params.get('boll_breakout_params', {})
        if isinstance(boll_breakout_params_tf, dict):
            boll_breakout_period = boll_breakout_params_tf.get('period', 15) # 默认值设为日志中的 15
            boll_breakout_std_dev = boll_breakout_params_tf.get('std_dev', 2.2) # 默认值设为日志中的 2.2
            if boll_breakout_period is not None and boll_breakout_std_dev is not None and 'BOLL' in indicator_naming_conv:
                 boll_naming_conf = indicator_naming_conv['BOLL']
                 if isinstance(boll_naming_conf, dict) and 'output_columns' in boll_naming_conf:
                    boll_patterns_breakout = [c['name_pattern'] for c in boll_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c and ('BBL' in c['name_pattern'] or 'BBU' in c['name_pattern'])]
                    # 日志显示是 focus_timeframe '30'
                    breakout_tf_str = self.focus_timeframe # 假设 BOLL 突破信号只在 focus_timeframe 计算
                    if breakout_tf_str in timeframes:
                         for name in TrendFollowingStrategy._format_indicator_name(boll_patterns_breakout, period=boll_breakout_period, std_dev=boll_breakout_std_dev):
                            required.add(f"{name}_{breakout_tf_str}")
                    else:
                         logger.warning(f"{log_prefix} BOLL 突破信号的 focus_timeframe '{breakout_tf_str}' 未在 base_scoring.timeframes 中定义，无法请求 BOLL 突破指标。")

        # 确保 close 列也被请求，因为很多信号依赖收盘价
        if 'ohlcv_naming_convention' in NAMING_CONFIG:
            ohlcv_cols = NAMING_CONFIG['ohlcv_naming_convention'].get('output_columns', [])
            close_pattern = next((c['name_pattern'] for c in ohlcv_cols if isinstance(c, dict) and c.get('name_pattern') == 'close'), None)
            if close_pattern:
                for tf_str in timeframes:
                     required.add(f"{close_pattern}_{tf_str}")

        # 确保 volume 和 amount 也被请求，因为量能指标和特征需要
        if 'ohlcv_naming_convention' in NAMING_CONFIG:
             ohlcv_cols = NAMING_CONFIG['ohlcv_naming_convention'].get('output_columns', [])
             volume_pattern = next((c['name_pattern'] for c in ohlcv_cols if isinstance(c, dict) and c.get('name_pattern') == 'volume'), None)
             amount_pattern = next((c['name_pattern'] for c in ohlcv_cols if isinstance(c, dict) and c.get('name_pattern') == 'amount'), None)
             if volume_pattern:
                for tf_str in timeframes:
                     required.add(f"{volume_pattern}_{tf_str}")
             if amount_pattern:
                for tf_str in timeframes:
                     required.add(f"{amount_pattern}_{tf_str}")


        logger.info(f"{log_prefix} get_required_columns 将请求 {len(required)} 列。")
        # logger.debug(f"{log_prefix} 请求的列包括: {sorted(list(required))}") # 可能非常冗长

        return list(sorted(list(required))) # 返回排序后的列表

    def _calculate_rule_based_signal(self, data: pd.DataFrame, stock_code: str, indicator_configs: List[Dict]) -> Tuple[pd.Series, Dict]:
        """
        计算基于规则的信号。
        修改：使用JSON配置获取指标列名。
        """
        if data is None or data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 输入数据为空，无法生成规则信号。")
            return pd.Series(dtype=float), {}

        if not self.params:
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算：策略参数 (self.params) 为空。")
            return pd.Series(50.0, index=data.index), {} 

        bs_params = self.params.get('base_scoring', {})
        vc_params = self.params.get('volume_confirmation', {})
        ia_params = self.params.get('indicator_analysis_params', {}) 
        dd_params = self.params.get('divergence_detection', {})

        focus_tf = self.focus_timeframe 
        
        # 检查关键OHLCV列
        ohlcv_base_names = [c['name_pattern'] for c in NAMING_CONFIG.get('ohlcv_naming_convention', {}).get('output_columns', [])]
        critical_ohlcv_cols = [f"{base_col}_{focus_tf}" for base_col in ohlcv_base_names if base_col in ['close', 'volume']]
        
        missing_critical_cols = [col for col in critical_ohlcv_cols if col not in data.columns or data[col].isnull().all()]
        if missing_critical_cols:
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算缺少关键OHLCV列或数据全为 NaN: {missing_critical_cols}。")
            return pd.Series(50.0, index=data.index), {}

        self._adjust_volatility_parameters(data) 

        indicator_scores_df = strategy_utils.calculate_all_indicator_scores(data, bs_params, indicator_configs) # indicator_configs 由 IndicatorService 准备

        current_weights: Dict[str, float]
        timeframes_from_config = bs_params.get('timeframes', [])
        if not timeframes_from_config: 
            logger.error(f"[{self.strategy_name}][{stock_code}] 'base_scoring.timeframes' 为空，无法计算基础评分。")
            return pd.Series(50.0, index=data.index), {}

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
            if tf_weight == 0: continue

            # SCORE_ 列名由 strategy_utils.calculate_all_indicator_scores 内部逻辑决定
            # 它们通常是 SCORE_{INDICATOR_NAME}_{PARAMS}_{TF}
            # 这里假设 indicator_scores_df 包含了这些列
            tf_score_cols = [col for col in indicator_scores_df.columns if col.endswith(f'_{tf_s}') and col.startswith('SCORE_')]
            if tf_score_cols:
                tf_average_score = indicator_scores_df[tf_score_cols].mean(axis=1).fillna(50.0)
                base_score_raw = base_score_raw.add(tf_average_score * tf_weight, fill_value=0.0) 
                total_effective_weight += tf_weight
            else:
                logger.debug(f"[{self.strategy_name}][{stock_code}] 时间框架 '{tf_s}' (权重 {tf_weight:.2f}) 没有找到任何指标评分列。将使用中性分50参与加权。")
                base_score_raw = base_score_raw.add(pd.Series(50.0, index=data.index) * tf_weight, fill_value=0.0)
                total_effective_weight += tf_weight
        
        if total_effective_weight == 0:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 所有时间框架的有效权重总和为零，基础评分将为中性50。")
            base_score_raw = pd.Series(50.0, index=data.index)
        elif not np.isclose(total_effective_weight, sum(current_weights.values())) and sum(current_weights.values()) > 0:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 有效权重总和 {total_effective_weight:.4f} 与预期权重总和 {sum(current_weights.values()):.4f} 不符。可能存在计算问题。")

        base_score_raw = base_score_raw.clip(0, 100).fillna(50.0)

        vc_params_adjusted = vc_params.copy() 
        vc_params_adjusted['boost_factor'] = self.volume_boost_factor 
        vc_params_adjusted['penalty_factor'] = self.volume_penalty_factor 
        vc_params_adjusted['volume_spike_threshold'] = self.volume_spike_threshold 
        
        volume_adjusted_results_df = strategy_utils.adjust_score_with_volume(
            base_score_raw, data, vc_params_adjusted 
        )
        base_score_volume_adjusted = volume_adjusted_results_df['ADJUSTED_SCORE']

        trend_analysis_df = self._perform_trend_analysis(data, base_score_volume_adjusted)

        divergence_signals_df = pd.DataFrame(index=data.index)
        if dd_params.get('enabled', True): 
            try:
                divergence_signals_df = strategy_utils.detect_divergence(data, dd_params, indicator_configs)
                if not divergence_signals_df.empty:
                     logger.debug(f"[{self.strategy_name}][{stock_code}] 背离检测完成，最新信号: {divergence_signals_df.iloc[-1].to_dict() if not divergence_signals_df.empty else '无'}")
            except Exception as e:
                logger.error(f"[{self.strategy_name}][{stock_code}] 执行背离检测时出错: {e}", exc_info=True)

        weights = self.rule_signal_weights 

        base_score_norm = (base_score_volume_adjusted.fillna(50.0) - 50) / 50
        alignment_norm = trend_analysis_df.get('alignment_signal', pd.Series(0.0, index=data.index)).fillna(0.0) / 3.0 
        long_context_norm = trend_analysis_df.get('long_term_context', pd.Series(0.0, index=data.index)).fillna(0.0) 
        score_momentum_series = trend_analysis_df.get('score_momentum', pd.Series(0.0, index=data.index)).fillna(0.0)
        momentum_norm = np.sign(score_momentum_series) 
        ema_cross_norm = trend_analysis_df.get('ema_cross_signal', pd.Series(0.0, index=data.index)).fillna(0.0) 
        boll_breakout_norm = trend_analysis_df.get('boll_breakout_signal', pd.Series(0.0, index=data.index)).fillna(0.0) 
        adx_strength_norm = trend_analysis_df.get('adx_strength_signal', pd.Series(0.0, index=data.index)).fillna(0.0) 
        vwap_dev_norm = trend_analysis_df.get('vwap_deviation_signal', pd.Series(0.0, index=data.index)).fillna(0.0) 
        
        vc_tf_col_name = vc_params.get('tf', self.focus_timeframe) 
        if isinstance(vc_tf_col_name, list): vc_tf_col_name = vc_tf_col_name[0] 
        
        # 使用JSON配置获取 VOL_SPIKE_SIGNAL 列名
        # 假设 VOL_SPIKE_SIGNAL 在 strategy_internal_columns 中定义
        vol_spike_pattern = "VOL_SPIKE_SIGNAL_{timeframe}" # 默认模式
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        for item in internal_cols_conf:
            if item['name_pattern'].startswith("VOL_SPIKE_SIGNAL"):
                vol_spike_pattern = item['name_pattern']
                break
        
        volume_spike_signal_col = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=vc_tf_col_name)[0]
        volume_spike_norm = volume_adjusted_results_df.get(volume_spike_signal_col, pd.Series(0.0, index=data.index)).fillna(0.0) 

        total_weighted_contribution = pd.Series(0.0, index=data.index)
        total_weighted_contribution += base_score_norm * weights.get('base_score', 0)
        total_weighted_contribution += alignment_norm * weights.get('alignment', 0)
        total_weighted_contribution += long_context_norm * weights.get('long_context', 0)
        total_weighted_contribution += momentum_norm * weights.get('momentum', 0)
        total_weighted_contribution += ema_cross_norm * weights.get('ema_cross', 0)
        total_weighted_contribution += boll_breakout_norm * weights.get('boll_breakout', 0)
        total_weighted_contribution += adx_strength_norm * weights.get('adx_strength', 0) 
        total_weighted_contribution += vwap_dev_norm * weights.get('vwap_deviation', 0)
        total_weighted_contribution += volume_spike_norm * weights.get('volume_spike', 0)

        base_rule_signal_before_adjust = 50.0 + total_weighted_contribution * 50.0
        base_rule_signal_before_adjust = base_rule_signal_before_adjust.clip(0, 100) 

        final_rule_signal = self._apply_adx_boost(
            base_rule_signal_before_adjust, 
            adx_strength_norm, 
            (base_rule_signal_before_adjust - 50.0) / 50.0 
        )

        final_rule_signal = self._apply_divergence_penalty(final_rule_signal, divergence_signals_df, dd_params)
        final_rule_signal = self._apply_trend_confirmation(final_rule_signal)
        final_rule_signal = final_rule_signal.clip(0, 100).round(2)

        intermediate_results = {
            'base_score_raw': base_score_raw,
            'base_score_volume_adjusted': base_score_volume_adjusted,
            'indicator_scores_df': indicator_scores_df, 
            'volume_analysis_df': volume_adjusted_results_df, 
            'trend_analysis_df': trend_analysis_df,
            'divergence_signals_df': divergence_signals_df
        }
        return final_rule_signal, intermediate_results

    def _calculate_rule_based_signal(self, data: pd.DataFrame, stock_code: str, indicator_configs: List[Dict]) -> Tuple[pd.Series, Dict]:
        """
        计算基于规则的信号。
        修改：使用JSON配置获取指标列名。
        """
        if data is None or data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 输入数据为空，无法生成规则信号。")
            return pd.Series(dtype=float), {}

        if not self.params:
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算：策略参数 (self.params) 为空。")
            return pd.Series(50.0, index=data.index), {} 

        bs_params = self.params.get('base_scoring', {})
        vc_params = self.params.get('volume_confirmation', {})
        ia_params = self.params.get('indicator_analysis_params', {}) 
        dd_params = self.params.get('divergence_detection', {})

        focus_tf = self.focus_timeframe 
        
        # 检查关键OHLCV列
        ohlcv_base_names = [c['name_pattern'] for c in NAMING_CONFIG.get('ohlcv_naming_convention', {}).get('output_columns', [])]
        critical_ohlcv_cols = [f"{base_col}_{focus_tf}" for base_col in ohlcv_base_names if base_col in ['close', 'volume']]
        
        missing_critical_cols = [col for col in critical_ohlcv_cols if col not in data.columns or data[col].isnull().all()]
        if missing_critical_cols:
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算缺少关键OHLCV列或数据全为 NaN: {missing_critical_cols}。")
            return pd.Series(50.0, index=data.index), {}

        self._adjust_volatility_parameters(data) 

        indicator_scores_df = strategy_utils.calculate_all_indicator_scores(data, bs_params, indicator_configs) # indicator_configs 由 IndicatorService 准备

        current_weights: Dict[str, float]
        timeframes_from_config = bs_params.get('timeframes', [])
        if not timeframes_from_config: 
            logger.error(f"[{self.strategy_name}][{stock_code}] 'base_scoring.timeframes' 为空，无法计算基础评分。")
            return pd.Series(50.0, index=data.index), {}

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
            if tf_weight == 0: continue

            # SCORE_ 列名由 strategy_utils.calculate_all_indicator_scores 内部逻辑决定
            # 它们通常是 SCORE_{INDICATOR_NAME}_{PARAMS}_{TF}
            # 这里假设 indicator_scores_df 包含了这些列
            tf_score_cols = [col for col in indicator_scores_df.columns if col.endswith(f'_{tf_s}') and col.startswith('SCORE_')]
            if tf_score_cols:
                tf_average_score = indicator_scores_df[tf_score_cols].mean(axis=1).fillna(50.0)
                base_score_raw = base_score_raw.add(tf_average_score * tf_weight, fill_value=0.0) 
                total_effective_weight += tf_weight
            else:
                logger.debug(f"[{self.strategy_name}][{stock_code}] 时间框架 '{tf_s}' (权重 {tf_weight:.2f}) 没有找到任何指标评分列。将使用中性分50参与加权。")
                base_score_raw = base_score_raw.add(pd.Series(50.0, index=data.index) * tf_weight, fill_value=0.0)
                total_effective_weight += tf_weight
        
        if total_effective_weight == 0:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 所有时间框架的有效权重总和为零，基础评分将为中性50。")
            base_score_raw = pd.Series(50.0, index=data.index)
        elif not np.isclose(total_effective_weight, sum(current_weights.values())) and sum(current_weights.values()) > 0:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 有效权重总和 {total_effective_weight:.4f} 与预期权重总和 {sum(current_weights.values()):.4f} 不符。可能存在计算问题。")

        base_score_raw = base_score_raw.clip(0, 100).fillna(50.0)

        vc_params_adjusted = vc_params.copy() 
        vc_params_adjusted['boost_factor'] = self.volume_boost_factor 
        vc_params_adjusted['penalty_factor'] = self.volume_penalty_factor 
        vc_params_adjusted['volume_spike_threshold'] = self.volume_spike_threshold 
        
        volume_adjusted_results_df = strategy_utils.adjust_score_with_volume(
            base_score_raw, data, vc_params_adjusted 
        )
        base_score_volume_adjusted = volume_adjusted_results_df['ADJUSTED_SCORE']

        trend_analysis_df = self._perform_trend_analysis(data, base_score_volume_adjusted)

        divergence_signals_df = pd.DataFrame(index=data.index)
        if dd_params.get('enabled', True): 
            try:
                divergence_signals_df = strategy_utils.detect_divergence(data, dd_params, indicator_configs)
                if not divergence_signals_df.empty:
                     logger.debug(f"[{self.strategy_name}][{stock_code}] 背离检测完成，最新信号: {divergence_signals_df.iloc[-1].to_dict() if not divergence_signals_df.empty else '无'}")
            except Exception as e:
                logger.error(f"[{self.strategy_name}][{stock_code}] 执行背离检测时出错: {e}", exc_info=True)

        weights = self.rule_signal_weights 

        base_score_norm = (base_score_volume_adjusted.fillna(50.0) - 50) / 50
        alignment_norm = trend_analysis_df.get('alignment_signal', pd.Series(0.0, index=data.index)).fillna(0.0) / 3.0 
        long_context_norm = trend_analysis_df.get('long_term_context', pd.Series(0.0, index=data.index)).fillna(0.0) 
        score_momentum_series = trend_analysis_df.get('score_momentum', pd.Series(0.0, index=data.index)).fillna(0.0)
        momentum_norm = np.sign(score_momentum_series) 
        ema_cross_norm = trend_analysis_df.get('ema_cross_signal', pd.Series(0.0, index=data.index)).fillna(0.0) 
        boll_breakout_norm = trend_analysis_df.get('boll_breakout_signal', pd.Series(0.0, index=data.index)).fillna(0.0) 
        adx_strength_norm = trend_analysis_df.get('adx_strength_signal', pd.Series(0.0, index=data.index)).fillna(0.0) 
        vwap_dev_norm = trend_analysis_df.get('vwap_deviation_signal', pd.Series(0.0, index=data.index)).fillna(0.0) 
        
        vc_tf_col_name = vc_params.get('tf', self.focus_timeframe) 
        if isinstance(vc_tf_col_name, list): vc_tf_col_name = vc_tf_col_name[0] 
        
        # 使用JSON配置获取 VOL_SPIKE_SIGNAL 列名
        # 假设 VOL_SPIKE_SIGNAL 在 strategy_internal_columns 中定义
        vol_spike_pattern = "VOL_SPIKE_SIGNAL_{timeframe}" # 默认模式
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        for item in internal_cols_conf:
            if item['name_pattern'].startswith("VOL_SPIKE_SIGNAL"):
                vol_spike_pattern = item['name_pattern']
                break
        
        volume_spike_signal_col = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=vc_tf_col_name)[0]
        volume_spike_norm = volume_adjusted_results_df.get(volume_spike_signal_col, pd.Series(0.0, index=data.index)).fillna(0.0) 

        total_weighted_contribution = pd.Series(0.0, index=data.index)
        total_weighted_contribution += base_score_norm * weights.get('base_score', 0)
        total_weighted_contribution += alignment_norm * weights.get('alignment', 0)
        total_weighted_contribution += long_context_norm * weights.get('long_context', 0)
        total_weighted_contribution += momentum_norm * weights.get('momentum', 0)
        total_weighted_contribution += ema_cross_norm * weights.get('ema_cross', 0)
        total_weighted_contribution += boll_breakout_norm * weights.get('boll_breakout', 0)
        total_weighted_contribution += adx_strength_norm * weights.get('adx_strength', 0) 
        total_weighted_contribution += vwap_dev_norm * weights.get('vwap_deviation', 0)
        total_weighted_contribution += volume_spike_norm * weights.get('volume_spike', 0)

        base_rule_signal_before_adjust = 50.0 + total_weighted_contribution * 50.0
        base_rule_signal_before_adjust = base_rule_signal_before_adjust.clip(0, 100) 

        final_rule_signal = self._apply_adx_boost(
            base_rule_signal_before_adjust, 
            adx_strength_norm, 
            (base_rule_signal_before_adjust - 50.0) / 50.0 
        )

        final_rule_signal = self._apply_divergence_penalty(final_rule_signal, divergence_signals_df, dd_params)
        final_rule_signal = self._apply_trend_confirmation(final_rule_signal)
        final_rule_signal = final_rule_signal.clip(0, 100).round(2)

        intermediate_results = {
            'base_score_raw': base_score_raw,
            'base_score_volume_adjusted': base_score_volume_adjusted,
            'indicator_scores_df': indicator_scores_df, 
            'volume_analysis_df': volume_adjusted_results_df, 
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
        pdi_col = f'PDI_{dmi_period_bs}_{focus_tf}'
        ndi_col = f'NDI_{dmi_period_bs}_{focus_tf}'
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
        修改：使用JSON配置获取close列名。
        """
        focus_tf = self.focus_timeframe
        # 使用JSON配置获取close列名
        close_base_name = TrendFollowingStrategy._format_indicator_name(NAMING_CONFIG['ohlcv_naming_convention']['output_columns'][3]['name_pattern'])[0] # 'close'
        close_col = f'{close_base_name}_{focus_tf}'

        if close_col not in data.columns or data[close_col].isnull().all():
            logger.warning(f"[{self.strategy_name}] 动态调整波动率：缺少收盘价列 {close_col} 或数据全为空。")
            return

        volatility_window = self.params.get('trend_analysis', {}).get('volatility_window', 10)
        price_volatility = data[close_col].rolling(window=volatility_window, min_periods=max(1, volatility_window//2)).std()
        
        if price_volatility.isnull().all() or price_volatility.empty:
            logger.warning(f"[{self.strategy_name}] 动态调整波动率：价格波动率数据不可用。")
            return

        latest_volatility = price_volatility.iloc[-1]
        base_volatility_benchmark = self.tf_params.get('volatility_benchmark', 5.0)
        if pd.isna(latest_volatility) or base_volatility_benchmark <= 0:
            logger.warning(f"[{self.strategy_name}] 动态调整波动率：最新波动率 ({latest_volatility}) 或基准 ({base_volatility_benchmark}) 无效。")
            return
            
        self.volatility_adjust_factor = max(0.5, min(2.0, latest_volatility / base_volatility_benchmark)) 
        
        original_high = self.tf_params.get('volatility_threshold_high', 10.0) 
        original_low = self.tf_params.get('volatility_threshold_low', 5.0)
        
        self.volatility_threshold_high = original_high * self.volatility_adjust_factor
        self.volatility_threshold_low = original_low * self.volatility_adjust_factor

        self.volatility_threshold_high = np.clip(self.volatility_threshold_high, original_high * 0.5, original_high * 2.0)
        self.volatility_threshold_low = np.clip(self.volatility_threshold_low, original_low * 0.5, original_low * 2.0)
        if self.volatility_threshold_low >= self.volatility_threshold_high: 
            self.volatility_threshold_low = self.volatility_threshold_high * 0.5 

        logger.debug(f"[{self.strategy_name}] 动态调整波动率阈值: high={self.volatility_threshold_high:.2f}, low={self.volatility_threshold_low:.2f}, factor={self.volatility_adjust_factor:.2f} (based on latest_vol={latest_volatility:.2f})")

    def _apply_adx_boost(self, final_signal: pd.Series, adx_strength_norm: pd.Series, base_signal_direction_norm: pd.Series) -> pd.Series:
        """
        模块化调整逻辑：使用 ADX 强度增强信号。
        """
        final_signal_filled = final_signal.fillna(50.0)
        adx_strength_norm_filled = adx_strength_norm.fillna(0.0)
        base_signal_direction_norm_filled = base_signal_direction_norm.fillna(0.0)

        adx_adjustment_factor = self.tf_params.get('adx_adjustment_factor', 10.0) 
        adjustment = pd.Series(0.0, index=final_signal.index)

        effective_mask = (np.sign(adx_strength_norm_filled) == np.sign(base_signal_direction_norm_filled)) & \
                         (base_signal_direction_norm_filled != 0) 

        adjustment.loc[effective_mask] = np.sign(base_signal_direction_norm_filled.loc[effective_mask]) * \
                                         np.abs(adx_strength_norm_filled.loc[effective_mask]) * \
                                         adx_adjustment_factor
        
        if not adjustment.empty:
             logger.debug(f"[{self.strategy_name}] ADX 增强调整 (最新值): {adjustment.iloc[-1] if not adjustment.empty else 'N/A'}")
        return (final_signal_filled + adjustment).clip(0, 100)

    def _apply_divergence_penalty(self, final_signal: pd.Series, divergence_signals_df: pd.DataFrame, dd_params: Dict) -> pd.Series:
        """
        模块化调整逻辑：应用背离惩罚。
        修改：使用JSON配置获取背离信号列名。
        """
        final_signal_filled = final_signal.fillna(50.0)
        if divergence_signals_df.empty:
            logger.debug(f"[{self.strategy_name}] 背离信号 DataFrame 为空，跳过背离惩罚。")
            return final_signal_filled

        penalty_factor = dd_params.get('divergence_penalty_factor', 0.45)
        
        # 从JSON获取背离信号列名 (这些是策略内部列)
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        has_bearish_div_col = "HAS_BEARISH_DIVERGENCE" # 默认
        has_bullish_div_col = "HAS_BULLISH_DIVERGENCE" # 默认
        for item in internal_cols_conf:
            if item['name_pattern'] == "HAS_BEARISH_DIVERGENCE": has_bearish_div_col = item['name_pattern']
            if item['name_pattern'] == "HAS_BULLISH_DIVERGENCE": has_bullish_div_col = item['name_pattern']

        if has_bearish_div_col not in divergence_signals_df.columns or \
           has_bullish_div_col not in divergence_signals_df.columns:
            logger.warning(f"[{self.strategy_name}] 背离信号 DataFrame 缺少聚合信号列 ('{has_bearish_div_col}', '{has_bullish_div_col}')，跳过背离惩罚。")
            return final_signal_filled

        if not final_signal_filled.index.equals(divergence_signals_df.index):
            logger.warning(f"[{self.strategy_name}] 背离惩罚：信号和背离数据索引不一致，尝试重新对齐。")
            try:
                divergence_signals_aligned = divergence_signals_df.reindex(final_signal_filled.index, fill_value=False) 
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

        mask_bullish_penalty = is_bullish_signal & has_bearish_div
        adjusted_signal.loc[mask_bullish_penalty] = 50 + (adjusted_signal.loc[mask_bullish_penalty] - 50) * (1 - penalty_factor)

        mask_bearish_penalty = is_bearish_signal & has_bullish_div
        adjusted_signal.loc[mask_bearish_penalty] = 50 + (adjusted_signal.loc[mask_bearish_penalty] - 50) * (1 - penalty_factor)
        
        if not adjusted_signal.empty:
             logger.debug(f"[{self.strategy_name}] 背离惩罚调整后信号 (最新值): {adjusted_signal.iloc[-1] if not adjusted_signal.empty else 'N/A'}")
        return adjusted_signal.clip(0, 100) 

    def _apply_trend_confirmation(self, final_signal: pd.Series) -> pd.Series:
        """
        增强假信号过滤：要求信号突破阈值后持续若干周期才视为有效。
        """
        if final_signal.empty:
            return final_signal

        trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55) 
        trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45) 
        confirmation_periods = self.trend_confirmation_periods 

        final_signal_filled = final_signal.fillna(50.0)
        filtered_signal = pd.Series(50.0, index=final_signal.index) 

        above_upper = final_signal_filled >= trend_threshold_upper
        below_lower = final_signal_filled <= trend_threshold_lower

        confirmed_bullish_streak = above_upper.rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() == confirmation_periods
        confirmed_bearish_streak = below_lower.rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() == confirmation_periods
        
        is_confirmed_bullish = confirmed_bullish_streak
        is_confirmed_bearish = confirmed_bearish_streak
        
        filtered_signal.loc[is_confirmed_bullish] = final_signal_filled.loc[is_confirmed_bullish]
        filtered_signal.loc[is_confirmed_bearish] = final_signal_filled.loc[is_confirmed_bearish]
        
        if not filtered_signal.empty:
             logger.debug(f"[{self.strategy_name}] 趋势确认过滤后信号 (最新值): {filtered_signal.iloc[-1] if not filtered_signal.empty else 'N/A'}")
        return filtered_signal.clip(0, 100)

    def generate_signals(self, data: pd.DataFrame, stock_code: Optional[str] = None, indicator_configs: Optional[List[Dict]] = None) -> pd.Series:
        """
        生成趋势跟踪信号，整合规则信号和Transformer模型预测。
        返回最终的信号 Series (通常是 combined_signal)。
        """
        if stock_code is None:
            logger.error(f"[{self.strategy_name}] generate_signals: 必须提供 stock_code。")
            return pd.Series(50.0, index=data.index, name="combined_signal") 
        if indicator_configs is None:
            logger.error(f"[{self.strategy_name}][{stock_code}] generate_signals: 必须提供 indicator_configs。")
            return pd.Series(50.0, index=data.index, name="combined_signal")

        logger.info(f"[{self.strategy_name}] 开始为股票 {stock_code} 生成信号 (Focus: {self.focus_timeframe})...")

        final_rule_signal, intermediate_results_dict = self._calculate_rule_based_signal(
            data=data, stock_code=stock_code, indicator_configs=indicator_configs
        )

        processed_data = data.copy() 
        processed_data['final_rule_signal'] = final_rule_signal # final_rule_signal 是内部列

        for key, df_to_join in intermediate_results_dict.items():
            if isinstance(df_to_join, pd.DataFrame) and not df_to_join.empty:
                for col_join in df_to_join.columns:
                    if col_join not in processed_data.columns: 
                        processed_data[col_join] = df_to_join[col_join]
                    else: 
                        logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果列 '{col_join}' 已存在于 processed_data，跳过合并来自 '{key}' 的同名列。")
            elif isinstance(df_to_join, pd.Series) and not df_to_join.empty: 
                 if df_to_join.name and df_to_join.name not in processed_data.columns:
                     processed_data[df_to_join.name] = df_to_join
                 elif not df_to_join.name and key not in processed_data.columns: 
                     processed_data[key] = df_to_join
                 else:
                     logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果Series '{df_to_join.name or key}' 已存在或无名，跳过合并。")

        processed_data['transformer_signal'] = pd.Series(50.0, index=processed_data.index) # transformer_signal 是内部列
        self.set_model_paths(stock_code)
        self.load_lstm_model(stock_code) 

        if self.transformer_model and self.feature_scaler and self.target_scaler and self.selected_feature_names_for_transformer:
            try:
                logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型已加载，准备进行预测...")
                predicted_signal_series = predict_with_transformer_model(
                    model=self.transformer_model,
                    data=processed_data, 
                    feature_scaler=self.feature_scaler,
                    target_scaler=self.target_scaler,
                    selected_feature_names=self.selected_feature_names_for_transformer,
                    window_size=self.transformer_window_size,
                    device=self.device
                )
                if not predicted_signal_series.empty:
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

        combination_weights = self.tf_params.get('signal_combination_weights', {'rule_weight': 0.6, 'lstm_weight': 0.4})
        rule_weight = combination_weights.get('rule_weight', 0.6)
        transformer_weight = combination_weights.get('lstm_weight', 0.4) 

        total_combo_weight = rule_weight + transformer_weight
        if total_combo_weight > 0 and not np.isclose(total_combo_weight, 1.0):
            rule_weight /= total_combo_weight
            transformer_weight /= total_combo_weight
        
        processed_data['combined_signal'] = processed_data['final_rule_signal'] # combined_signal 是内部列
        
        if 'predicted_signal_series' in locals() and not predicted_signal_series.empty:
            valid_transformer_indices = predicted_signal_series.index
            if not valid_transformer_indices.empty:
                logger.info(f"[{self.strategy_name}][{stock_code}] 在 {len(valid_transformer_indices)} 个点上应用 Transformer 加权组合。")
                
                rule_comp = processed_data.loc[valid_transformer_indices, 'final_rule_signal'] * rule_weight
                trans_comp = processed_data.loc[valid_transformer_indices, 'transformer_signal'] * transformer_weight
                processed_data.loc[valid_transformer_indices, 'combined_signal'] = (rule_comp + trans_comp).clip(0, 100).round(2)
        else:
            logger.info(f"[{self.strategy_name}][{stock_code}] 未进行 Transformer 预测或预测结果为空，Combined_signal 将等于 Final_rule_signal。")

        self.intermediate_data = processed_data.copy() 

        if not self.intermediate_data.empty:
            latest_combined_signal_val = self.intermediate_data['combined_signal'].iloc[-1] if 'combined_signal' in self.intermediate_data.columns else np.nan
            logger.info(f"[{self.strategy_name}][{stock_code}] 信号生成完毕，最新组合信号: {latest_combined_signal_val:.2f}")
        else:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 信号生成过程未产生有效数据 (intermediate_data 为空)。")

        return self.intermediate_data 

    def load_lstm_model(self, stock_code: str): 
        """
        为特定股票加载 Transformer 模型权重和 scaler。
        """
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
                 self._reset_model_components() 
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
        self.set_model_paths(stock_code) 
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
        self.set_model_paths(stock_code) 
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
        # final_rule_signal 是内部列
        if 'final_rule_signal' not in data_with_signals.columns or data_with_signals['final_rule_signal'].isnull().all():
            logger.warning(f"[{self.strategy_name}] 规则信号列 'final_rule_signal' 不存在或全为空，无法计算趋势持续时间。")
            return trend_duration_info

        final_signal_series = data_with_signals['final_rule_signal'].dropna()
        if final_signal_series.empty: return trend_duration_info

        trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55)
        trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45)
        strong_bullish_threshold = self.tf_params.get('strong_bullish_threshold', 75) 
        strong_bearish_threshold = self.tf_params.get('strong_bearish_threshold', 25) 
        moderate_bullish_threshold = self.tf_params.get('moderate_bullish_threshold', 60) 
        moderate_bearish_threshold = self.tf_params.get('moderate_bearish_threshold', 40) 

        current_bullish_streak = 0
        current_bearish_streak = 0
        for signal_val in final_signal_series.iloc[::-1]: 
            if signal_val >= trend_threshold_upper:
                current_bullish_streak += 1
                current_bearish_streak = 0 
            elif signal_val <= trend_threshold_lower:
                current_bearish_streak += 1
                current_bullish_streak = 0 
            else: 
                break 
        
        trend_duration_info['bullish_duration'] = current_bullish_streak
        trend_duration_info['bearish_duration'] = current_bearish_streak

        try: 
            timeframe_minutes = int(self.focus_timeframe) 
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
        if current_duration_periods >= self.trend_duration_threshold_strong:
             trend_duration_info['duration_status'] = '长'
        elif current_duration_periods >= self.trend_duration_threshold_moderate:
             trend_duration_info['duration_status'] = '中'
        
        return trend_duration_info

    def analyze_signals(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        分析趋势策略信号，生成解读和建议。
        修改：使用JSON配置获取内部列名。
        """
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 中间数据为空，无法进行信号分析。")
            return None
        
        analysis_results_dict = {}
        latest_data_row = self.intermediate_data.iloc[-1]

        # combined_signal 是内部列
        if 'combined_signal' in self.intermediate_data.columns:
            combined_signal_series = self.intermediate_data['combined_signal'].dropna()
            if not combined_signal_series.empty:
                analysis_results_dict['combined_signal_mean'] = combined_signal_series.mean()
        
        trend_duration_info_dict = self._calculate_trend_duration(self.intermediate_data)
        analysis_results_dict.update(trend_duration_info_dict)

        signal_judgment_dict = {}
        operation_advice_str = "中性观望"
        risk_warning_str = ""
        t_plus_1_note_str = "（受 T+1 限制，建议次日操作）"
        
        # final_rule_signal, transformer_signal, combined_signal 都是内部列
        final_score_val = latest_data_row.get('combined_signal', 50.0)
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
             else: 
                  signal_judgment_dict['overall_signal'] += " (温和)"
                  operation_advice_str = f"观望或轻仓试多 (信号温和) {t_plus_1_note_str}"
                  if duration_status_rule_str == '长': operation_advice_str = f"谨慎持有 (信号温和但趋势已久) {t_plus_1_note_str}"
        elif final_score_val <= moderate_bearish_thresh:
             signal_judgment_dict['overall_signal'] = "看跌信号"
             if final_score_val <= strong_bearish_thresh:
                  signal_judgment_dict['overall_signal'] += " (强)"
                  operation_advice_str = f"卖出或逢高减仓 (信号强劲) {t_plus_1_note_str}"
                  if duration_status_rule_str == '长': operation_advice_str = f"坚定空仓或减仓 (信号强劲且趋势持续) {t_plus_1_note_str}"
             else: 
                  signal_judgment_dict['overall_signal'] += " (温和)"
                  operation_advice_str = f"观望或轻仓试空 (信号温和) {t_plus_1_note_str}"
                  if duration_status_rule_str == '长': operation_advice_str = f"谨慎空仓或观望 (信号温和但趋势已久) {t_plus_1_note_str}"
        else: 
            signal_judgment_dict['overall_signal'] = "中性信号"
            operation_advice_str = f"中性观望，等待信号明朗 {t_plus_1_note_str}"
        
        # alignment_signal, adx_strength_signal 是内部列
        alignment_val = latest_data_row.get('alignment_signal', 0) 
        if alignment_val == 3: signal_judgment_dict['alignment_status'] = "完全多头排列"
        elif alignment_val == -3: signal_judgment_dict['alignment_status'] = "完全空头排列"

        adx_signal_val = latest_data_row.get('adx_strength_signal', 0) 
        if abs(adx_signal_val) < 0.5 and abs(final_score_val - 50) > 20 : 
            risk_warning_str += "ADX显示趋势强度不足，注意假信号风险。 "

        # HAS_BEARISH_DIVERGENCE, HAS_BULLISH_DIVERGENCE 是内部列
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

        now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        chinese_interpretation_str = (
            f"【趋势跟踪策略分析 - {stock_code} - {now_str}】\n"
            f"最新组合信号分: {final_score_val:.2f} (规则: {latest_data_row.get('final_rule_signal', 50.0):.2f}, Transformer: {latest_data_row.get('transformer_signal', 50.0):.2f})\n"
            f"当前趋势状态: {signal_judgment_dict.get('overall_signal', '中性')}\n"
            f"规则趋势判断: {trend_duration_info_dict['current_trend']} ({trend_duration_info_dict['trend_strength']})\n"
            f"趋势持续: {trend_duration_info_dict.get('bullish_duration_text' if trend_duration_info_dict.get('current_trend','').startswith('看涨') else 'bearish_duration_text','未知')} ({trend_duration_info_dict['duration_status']})\n"
            f"EMA排列: {signal_judgment_dict.get('alignment_status', '未知')}\n"
            f"ADX强度: {latest_data_row.get('adx_strength_signal', 0.0):.2f}\n" 
            f"背离状态: {signal_judgment_dict.get('divergence_status', '未知')}\n"
            f"操作建议: {operation_advice_str}\n"
            f"风险提示: {risk_warning_str if risk_warning_str else '无明显风险提示。'}\n"
        )
        analysis_results_dict['signal_judgment'] = signal_judgment_dict
        analysis_results_dict['operation_advice'] = operation_advice_str
        analysis_results_dict['risk_warning'] = risk_warning_str
        analysis_results_dict['chinese_interpretation'] = chinese_interpretation_str

        self.analysis_results = analysis_results_dict 
        logger.info(f"[{self.strategy_name}][{stock_code}] 信号分析完成。")
        logger.info(chinese_interpretation_str) 

        return analysis_results_dict

    def get_analysis_results(self) -> Optional[Dict[str, Any]]:
        """返回信号分析结果字典。"""
        return self.analysis_results

    def save_analysis_results(self, stock_code: str, timestamp: pd.Timestamp, data: Optional[pd.DataFrame]=None):
        """
        保存趋势跟踪策略的分析结果到数据库。
        修改：使用JSON配置获取内部列名和OHLCV列名。
        """
        from stock_models.stock_analytics import StockScoreAnalysis
        from stock_models.stock_basic import StockInfo
        
        if self.analysis_results is None:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 无分析结果可保存。请先运行 analyze_signals。")
            return
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 中间数据为空，部分保存字段可能缺失。")
            latest_intermediate_row = pd.Series(dtype=object) 
        else:
            latest_intermediate_row = self.intermediate_data.iloc[-1]

        try:
            stock_obj = StockInfo.objects.get(stock_code=stock_code)
            
            def convert_nan_to_none(value): 
                if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
                    return None
                return value if pd.notna(value) else None 

            # 获取 close 列名
            close_base_name = TrendFollowingStrategy._format_indicator_name(NAMING_CONFIG['ohlcv_naming_convention']['output_columns'][3]['name_pattern'])[0]
            close_price_col_name = f'{close_base_name}_{self.focus_timeframe}'


            defaults_payload = {
                'score': convert_nan_to_none(latest_intermediate_row.get('combined_signal')), # 内部列
                'rule_signal': convert_nan_to_none(latest_intermediate_row.get('final_rule_signal')), # 内部列
                'lstm_signal': convert_nan_to_none(latest_intermediate_row.get('transformer_signal')), # 内部列
                'base_score_raw': convert_nan_to_none(latest_intermediate_row.get('base_score_raw')), # 内部列
                'base_score_volume_adjusted': convert_nan_to_none(latest_intermediate_row.get('base_score_volume_adjusted')), # 内部列
                'alignment_signal': convert_nan_to_none(latest_intermediate_row.get('alignment_signal')), # 内部列
                'long_term_context': convert_nan_to_none(latest_intermediate_row.get('long_term_context')), # 内部列
                'adx_strength_signal': convert_nan_to_none(latest_intermediate_row.get('adx_strength_signal')), # 内部列
                'stoch_signal': convert_nan_to_none(latest_intermediate_row.get('stoch_signal')), # 内部列
                'div_has_bearish_divergence': latest_intermediate_row.get('HAS_BEARISH_DIVERGENCE', False), # 内部列
                'div_has_bullish_divergence': latest_intermediate_row.get('HAS_BULLISH_DIVERGENCE', False), # 内部列
                'close_price': convert_nan_to_none(latest_intermediate_row.get(close_price_col_name)), # 来自JSON配置
                'current_trend': self.analysis_results.get('current_trend'),
                'trend_strength': self.analysis_results.get('trend_strength'),
                'bullish_duration': convert_nan_to_none(self.analysis_results.get('bullish_duration')),
                'bearish_duration': convert_nan_to_none(self.analysis_results.get('bearish_duration')),
                'operation_advice': self.analysis_results.get('operation_advice'),
                'risk_warning': self.analysis_results.get('risk_warning'),
                'chinese_interpretation': self.analysis_results.get('chinese_interpretation'),
                'params_snapshot': self.params, 
            }
            
            for key, value in defaults_payload.items():
                if key != 'params_snapshot': 
                     defaults_payload[key] = convert_nan_to_none(value)


            obj, created = StockScoreAnalysis.objects.update_or_create(
                stock=stock_obj,
                strategy_name=self.strategy_name, 
                timestamp=timestamp, 
                time_level=self.focus_timeframe, 
                defaults=defaults_payload
            )
            status_msg = "创建新的" if created else "更新"
            logger.info(f"[{self.strategy_name}][{stock_code}] {status_msg} 策略分析结果记录，时间戳: {timestamp}")

        except StockInfo.DoesNotExist:
            logger.error(f"[{self.strategy_name}] 股票 {stock_code} 未在数据库中找到，无法保存分析结果。")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存策略分析结果时出错: {e}", exc_info=True)

    def get_analysis_results(self) -> Optional[Dict[str, Any]]:
        """返回信号分析结果字典。"""
        return self.analysis_results





