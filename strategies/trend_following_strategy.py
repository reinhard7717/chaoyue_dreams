# 此策略侧重于识别和跟随趋势，主要使用 EMA 排列、DMI、SAR 等指标，并以 30 分钟级别为主要权重。
# strategies/trend_following_strategy.py
import asyncio
import pickle
import re
import optuna
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import logging
from pathlib import Path
from django.conf import settings
import joblib # 用于加载/保存 scaler
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from services.indicator_services import IndicatorService # 确保导入 IndicatorService
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas_ta as ta
from dao_manager.tushare_daos.industry_dao import IndustryDao
from stock_models.stock_analytics import StockAnalysisResultTrendFollowing
from stock_models.stock_basic import StockInfo
from utils.cache_set import StrategyCacheSet
from .utils import strategy_utils
from .utils.deep_learning_utils import (
    build_transformer_model,
    evaluate_transformer_model,
    predict_with_transformer_model,
    TimeSeriesDataset,
    prepare_data_for_transformer # prepare_data_for_transformer 仍然从这里导入
)
from strategies.utils import deep_learning_utils

# 创建一个专门用于evaluation_results的logger
evaluation_logger = logging.getLogger("evaluation_results")
evaluation_logger.setLevel(logging.INFO)

# 创建文件处理器，写入evaluation_results.log
file_handler = logging.FileHandler("evaluation_results.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# 创建日志格式器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger = logging.getLogger("strategy_trend_following") # 策略特定的 logger

torch.backends.mkl.enabled = True       # 启用 MKL（Intel Math Kernel Library）加速
torch.backends.openmp.enabled = True    # 启用 OpenMP 多线程加速

# 读取指标规范命名json
def load_naming_config():
    # 检查文件是否存在再尝试打开，更健壮
    if not hasattr(settings, 'INDICATOR_NAMING_CONFIG_PATH') or not settings.INDICATOR_NAMING_CONFIG_PATH:
        logger.error("CRITICAL: Django settings.INDICATOR_NAMING_CONFIG_PATH 未配置!")
        return {} # 返回空字典避免后续错误
    if not os.path.exists(settings.INDICATOR_NAMING_CONFIG_PATH):
        logger.error(f"命名规范文件未找到: {settings.INDICATOR_NAMING_CONFIG_PATH}")
        return {} # 返回空字典避免后续错误
    try:
        with open(settings.INDICATOR_NAMING_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载命名规范文件出错: {e}", exc_info=True)
        return {} # 加载失败返回空字典

NAMING_CONFIG = load_naming_config()

def load_indicator_parameters():
    # 检查文件是否存在再尝试打开，更健壮
    if not hasattr(settings, 'INDICATOR_PARAMETERS_CONFIG_PATH') or not settings.INDICATOR_PARAMETERS_CONFIG_PATH:
        logger.error("CRITICAL: Django settings.INDICATOR_PARAMETERS_CONFIG_PATH 未配置!")
        return {} # 返回空字典避免后续错误
    if not os.path.exists(settings.INDICATOR_PARAMETERS_CONFIG_PATH):
        logger.error(f"策略指标配置文件未找到: {settings.INDICATOR_PARAMETERS_CONFIG_PATH}")
        return {} # 返回空字典避免后续错误
    try:
        with open(settings.INDICATOR_PARAMETERS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载策略指标配置文件出错: {e}", exc_info=True)
        return {} # 加载失败返回空字典

INDICATOR_PARAMETERS = load_indicator_parameters()

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
    strategy_name_class_default = "TrendFollowingStrategy_ClassDefault" # 改个名字以区分
    default_focus_timeframe = '30' # 默认主要关注的时间框架

    torch.backends.mkl.enabled = True       # 启用 MKL（Intel Math Kernel Library）加速
    torch.backends.openmp.enabled = True    # 启用 OpenMP 多线程加速

    def __init__(self, params_file: str = None, base_data_dir: str = None):
        """
        初始化趋势跟踪策略。
        Args:
            params_file (str): 策略参数JSON文件的路径。
            base_data_dir (str): 存储策略相关数据的基础目录。
        """
        # print(f"TrendFollowingStrategy __init__ called with params_file: {params_file}, base_data_dir: {base_data_dir}") # DEBUG: 初始调用信息

        # --- 阶段 0: 基础依赖和路径初始化 ---
        params_file, base_data_dir = self._initialize_base_paths_and_dependencies(params_file=params_file, base_data_dir=base_data_dir) # 调用辅助方法
        self.base_data_dir = base_data_dir # 确保 self.base_data_dir 设置
        self.fe_params: Dict[str, Any] = {} # 初始化 fe_params
        # 新增属性初始化
        self.pca_model_path = None # 初始化 pca_model_path
        self.scaler_for_pca_path = None # 初始化 scaler_for_pca_path
        self.feature_selector_model_path = None # 初始化 feature_selector_model_path
        self.jit_traced = False
        temp_log_prefix = f"[{TrendFollowingStrategy.strategy_name_class_default}-init]" # 临时日志前缀

        # --- 阶段 1 & 2: 解析和加载参数文件 ---
        self.params_file_path, loaded_params, file_load_success = self._resolve_and_load_params_file(params_file_input=params_file, temp_log_prefix=temp_log_prefix) # 调用辅助方法

        # --- 阶段 3 & 4: 设置策略名称和核心参数 ---
        log_prefix = self._setup_strategy_name_and_core_params(loaded_params=loaded_params, temp_log_prefix=temp_log_prefix) # 调用辅助方法

        # --- 阶段 5: 初始化策略特定属性 ---
        self._initialize_strategy_attributes(log_prefix=log_prefix) # 调用辅助方法
        self._initialize_model_related_attributes(log_prefix=log_prefix) # 调用辅助方法，拆分模型相关属性

        # --- 阶段 6 & 7: 参数验证和最终日志 ---
        self._perform_initial_validation_and_logging(log_prefix=log_prefix, resolved_params_file_path=self.params_file_path, file_load_success=file_load_success, params_loaded=loaded_params is not None and bool(loaded_params)) # 调用辅助方法

        logger.info(f"{log_prefix} TrendFollowingStrategy __init__ 执行完毕。")
        # print(f"{log_prefix} TrendFollowingStrategy __init__ 执行完毕。") # DEBUG: 结束信息

    #辅助方法 - 初始化基础路径和依赖
    def _initialize_base_paths_and_dependencies(self, params_file: Optional[str], base_data_dir: Optional[str]) -> Tuple[str, str]:
        """处理 params_file 和 base_data_dir 的初始设置及基础依赖。"""
        self.industry_dao = IndustryDao()
        self.indicator_service = IndicatorService()
        if params_file is None:
            if not hasattr(settings, 'INDICATOR_PARAMETERS_CONFIG_PATH') or not settings.INDICATOR_PARAMETERS_CONFIG_PATH:
                logger.error("CRITICAL: Django settings.INDICATOR_PARAMETERS_CONFIG_PATH 未配置!")
                params_file = "" # 使用空字符串标记错误或缺失
            else:
                params_file = settings.INDICATOR_PARAMETERS_CONFIG_PATH
        if base_data_dir is None:
            if not hasattr(settings, 'STRATEGY_DATA_DIR') or not settings.STRATEGY_DATA_DIR:
                logger.error("CRITICAL: Django settings.STRATEGY_DATA_DIR 未配置!")
                base_data_dir = os.path.join(Path.home(), ".stock_quant_data", "strategy_data")
                logger.warning(f"使用默认备用策略数据目录: {base_data_dir}")
            else:
                base_data_dir = settings.STRATEGY_DATA_DIR
        logger.info(f"初始化基础路径和依赖 -  params_file: {params_file}, base_data_dir: {base_data_dir}")
        return params_file, base_data_dir

    #辅助方法 - 解析参数文件路径
    def _resolve_config_file_path(self, file_path_input: str, log_prefix_temp: str) -> Optional[str]:
        """解析配置文件（如参数文件）的绝对路径。"""
        logger.debug(f"{log_prefix_temp} 接收到文件路径: '{file_path_input}'")
        if not file_path_input: # 空字符串或 None
            logger.error(f"{log_prefix_temp} 警告: file_path_input 参数为空或 None。无法加载。")
            return None
        if os.path.isabs(file_path_input):
            logger.debug(f"{log_prefix_temp} 文件路径 '{file_path_input}' 是绝对路径。")
            return file_path_input
        logger.debug(f"{log_prefix_temp} 文件路径是相对路径，开始解析...")
        # 尝试基于 Django BASE_DIR
        if hasattr(settings, 'BASE_DIR') and settings.BASE_DIR:
            path_based_on_base_dir = os.path.join(settings.BASE_DIR, file_path_input)
            if os.path.exists(path_based_on_base_dir) and os.path.isfile(path_based_on_base_dir):
                logger.debug(f"{log_prefix_temp} 文件解析为基于 BASE_DIR 的路径: '{path_based_on_base_dir}'")
                return path_based_on_base_dir
            else:
                logger.debug(f"{log_prefix_temp} 未在 BASE_DIR ('{settings.BASE_DIR}') 下找到 '{file_path_input}'。尝试基于 CWD...")
        else:
            logger.warning(f"{log_prefix_temp} Django settings.BASE_DIR 未定义。将仅基于 CWD 解析相对路径 '{file_path_input}'。")
        # 尝试基于当前工作目录 (CWD)
        path_based_on_cwd = os.path.abspath(file_path_input)
        if os.path.exists(path_based_on_cwd) and os.path.isfile(path_based_on_cwd):
            log_message_level = logger.info if hasattr(settings, 'BASE_DIR') else logger.warning
            log_message_level(f"{log_prefix_temp} 文件在当前工作目录 '{os.getcwd()}' 找到: '{path_based_on_cwd}'. "
                              f"{'建议使用相对于项目根目录 (BASE_DIR) 的路径。' if hasattr(settings, 'BASE_DIR') else '强烈建议定义 settings.BASE_DIR。'}")
            return path_based_on_cwd
        logger.error(f"{log_prefix_temp} CRITICAL: 相对文件 '{file_path_input}' 在 BASE_DIR (如果已定义) 和 CWD (解析为 '{path_based_on_cwd}') 中均未找到。")
        return None

    #辅助方法 - 从文件加载JSON参数
    def _load_json_from_file(self, resolved_file_path: Optional[str], log_prefix_temp: str, file_description: str = "参数") -> Tuple[Dict[str, Any], bool]:
        """从解析后的路径加载JSON文件。"""
        loaded_data = {}
        load_success = False
        if resolved_file_path is None:
            logger.error(f"{log_prefix_temp} CRITICAL: {file_description}文件路径解析失败。无法加载{file_description}。")
            return loaded_data, load_success
        # logger.info(f"{log_prefix_temp} 尝试从最终路径 '{resolved_file_path}' 加载{file_description}...")
        if not (os.path.exists(resolved_file_path) and os.path.isfile(resolved_file_path)):
            logger.error(f"{log_prefix_temp} CRITICAL: 最终确认{file_description}文件 '{resolved_file_path}' 不存在或不是文件。")
            return loaded_data, load_success
        try:
            with open(resolved_file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if loaded_data and isinstance(loaded_data, dict):
                load_success = True
                logger.info(f"{log_prefix_temp} {file_description}文件已成功从 '{resolved_file_path}' 解析。")
            else:
                logger.error(f"{log_prefix_temp} CRITICAL: {file_description}文件 '{resolved_file_path}' 内容为空或不是有效的JSON对象。")
                loaded_data = {} # 确保是空字典
        except FileNotFoundError: # 理论上不太可能，因为前面检查过
            logger.error(f"{log_prefix_temp} CRITICAL: 文件 '{resolved_file_path}' 在尝试打开时未找到。")
        except PermissionError:
            logger.error(f"{log_prefix_temp} CRITICAL: 没有权限读取文件 '{resolved_file_path}'。")
        except json.JSONDecodeError as e_json:
            logger.error(f"{log_prefix_temp} CRITICAL: 解析文件 '{resolved_file_path}' 时发生JSON解码错误: {e_json}")
        except Exception as e_load:
            logger.error(f"{log_prefix_temp} CRITICAL: 加载文件 '{resolved_file_path}' 时发生未知错误: {e_load}", exc_info=True)
        if not load_success:
            logger.warning(f"{log_prefix_temp} {file_description}文件加载失败或内容无效，策略将使用默认{file_description}。")
            loaded_data = {}
        return loaded_data, load_success

    # 提取参数的辅助函数 (保留原有的，因为它从多个源查找参数)
    def _get_param_val(self, sources: List[Dict], key: str, default: Any = None) -> Any:
        """从参数源列表中按顺序查找键的值，返回第一个找到的值或默认值。"""
        for source_dict in sources:
            if isinstance(source_dict, dict) and key in source_dict:
                return source_dict[key]
        return default

    #辅助方法 - 整合阶段1和2
    def _resolve_and_load_params_file(self, params_file_input: str, temp_log_prefix: str) -> Tuple[Optional[str], Dict[str, Any], bool]:
        """解析参数文件路径并从该路径加载参数。"""
        resolved_path = self._resolve_config_file_path(file_path_input=params_file_input, log_prefix_temp=temp_log_prefix)
        loaded_params, file_load_success = self._load_json_from_file(resolved_file_path=resolved_path, log_prefix_temp=temp_log_prefix, file_description="策略参数")
        return resolved_path, loaded_params, file_load_success

    #辅助方法 - 设置策略名称和核心参数字典
    def _setup_strategy_name_and_core_params(self, loaded_params: Dict[str, Any], temp_log_prefix: str) -> str:
        """设置 self.params, self.fe_params, self.strategy_name，并返回最终的 log_prefix。"""
        self.params: Dict[str, Any] = loaded_params
        self.fe_params = self.params.get('feature_engineering_params', {})
        logger.debug(f"{temp_log_prefix} self.params 已设置。是否为空: {not bool(self.params)}.")
        strategy_name_from_params = self.params.get('trend_following_strategy_name')
        if isinstance(strategy_name_from_params, str) and strategy_name_from_params:
            self.strategy_name = strategy_name_from_params
            logger.info(f"[{self.strategy_name}-init] 实例策略名从参数文件成功设置为: '{self.strategy_name}'")
        else:
            self.strategy_name = TrendFollowingStrategy.strategy_name_class_default
            log_msg_detail = "参数中未找到 'trend_following_strategy_name' 键" if strategy_name_from_params is None \
                else f"参数中的 'trend_following_strategy_name' 无效: '{strategy_name_from_params}'"
            logger.warning(f"[{self.strategy_name}-init] {log_msg_detail}，实例策略名使用类默认值: '{self.strategy_name}'")
        return f"[{self.strategy_name}]" # 返回最终的 log_prefix

    #辅助方法 - 初始化策略特定属性 (除模型外)
    def _initialize_strategy_attributes(self, log_prefix: str):
        """初始化 TrendFollowingStrategy 特有的属性（不包括模型相关）。"""
        logger.debug(f"{log_prefix} base_data_dir 设置为: '{self.base_data_dir}'")
        self.tf_params: Dict[str, Any] = self.params.get('trend_following_params', {})
        if not self.params: # self.params 可能因加载失败而为空
            logger.error(f"{log_prefix} CRITICAL INIT: 策略参数 (self.params) 为空！属性将依赖代码默认值。")
        elif not self.tf_params: # self.params 不为空，但 trend_following_params 块缺失或为空
            logger.error(f"{log_prefix} CRITICAL INIT: 'trend_following_params' 块缺失或为空！特定参数将依赖代码默认值。")
        self.focus_timeframe: str = str(self.tf_params.get('focus_timeframe', self.default_focus_timeframe))
        self.timeframe_weights: Optional[Dict[str, float]] = self.tf_params.get('timeframe_weights', None)
        self.trend_indicators: List[str] = self.tf_params.get('trend_indicators', ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi'])
        if not isinstance(self.trend_indicators, list):
            logger.warning(f"{log_prefix} 参数 'trend_indicators' 不是列表，用默认值。")
            self.trend_indicators = ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi']
        default_rule_weights = {
            'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15,
            'ema_cross': 0.15, 'boll_breakout': 0.1, 'adx_strength': 0.1, 'vwap_deviation': 0.05,
            'volume_spike': 0.05
        }
        self.rule_signal_weights: Dict[str, float] = self.tf_params.get('rule_signal_weights', default_rule_weights)
        if not isinstance(self.rule_signal_weights, dict) or not self.rule_signal_weights:
            logger.warning(f"{log_prefix} 参数 'rule_signal_weights' 无效或为空，用默认值。")
            self.rule_signal_weights = default_rule_weights.copy()
        vc_global_params = self.params.get('volume_confirmation', {}) # vc_global_params可能是空字典
        self.volume_boost_factor: float = self.tf_params.get('volume_boost_factor', vc_global_params.get('boost_factor', 1.2))
        self.volume_penalty_factor: float = self.tf_params.get('volume_penalty_factor', vc_global_params.get('penalty_factor', 0.8))
        self.volume_spike_threshold: float = self.tf_params.get('volume_spike_threshold', 2.0) # 来自tf_params
        self.volatility_threshold_high: float = self.tf_params.get('volatility_threshold_high', 10.0)
        self.volatility_threshold_low: float = self.tf_params.get('volatility_threshold_low', 5.0)
        self.volatility_adjust_factor: float = 1.0 # 初始值，后续可能由 _adjust_volatility_parameters 更新
        self.adx_strong_threshold: int = self.tf_params.get('adx_strong_threshold', 30)
        self.adx_moderate_threshold: int = self.tf_params.get('adx_moderate_threshold', 20)
        self.trend_duration_threshold_strong: int = self.tf_params.get('trend_duration_threshold_strong', 5)
        self.trend_duration_threshold_moderate: int = self.tf_params.get('trend_duration_threshold_moderate', 10)
        self.stoch_oversold_threshold: int = self.tf_params.get('stoch_oversold_threshold', 20)
        self.stoch_overbought_threshold: int = self.tf_params.get('stoch_overbought_threshold', 80)
        self.vwap_deviation_threshold: float = self.tf_params.get('vwap_deviation_threshold', 0.01)
        self.trend_confirmation_periods: int = self.tf_params.get('trend_confirmation_periods', 3)
        self.intermediate_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[Dict[str, Any]] = None
        if ta is None:
            logger.error(f"{log_prefix} pandas_ta 未成功加载，策略部分功能可能不可用。")

    #辅助方法 - 初始化模型相关属性和路径
    def _initialize_model_related_attributes(self, log_prefix: str):
        """初始化 Transformer 模型相关的属性和路径设置。"""
        self.transformer_window_size: int = self.tf_params.get('transformer_window_size', 60)
        self.transformer_batch_size: int = self.tf_params.get('transformer_batch_size', 128)
        self.transformer_target_column: str = self.tf_params.get('transformer_target_column', 'final_rule_signal')
        self.transformer_model_config: Dict[str, Any] = self.tf_params.get('transformer_model_config', {})
        self.transformer_training_config: Dict[str, Any] = self.tf_params.get('transformer_training_config', {})
        # 确保训练配置中的 batch_size 覆盖策略参数中的值（如果存在）
        if 'batch_size' in self.transformer_training_config and isinstance(self.transformer_training_config['batch_size'], int):
            self.transformer_batch_size = self.transformer_training_config['batch_size']
        self.transformer_data_prep_config: Dict[str, Any] = self.tf_params.get('transformer_data_prep_config', {})
        self.transformer_model: Optional[nn.Module] = None
        self.feature_scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
        self.target_scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
        self.selected_feature_names_for_transformer: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"{log_prefix} 使用设备: {self.device}")
        # 路径初始化为 None，set_model_paths 会具体设置
        self.model_dir: Optional[str] = None
        self.model_path: Optional[str] = None
        self.feature_scaler_path: Optional[str] = None
        self.target_scaler_path: Optional[str] = None
        self.selected_features_path: Optional[str] = None
        self.all_prepared_data_npz_path: Optional[str] = None

    #辅助方法 - 执行参数验证和最终日志记录
    def _perform_initial_validation_and_logging(self, log_prefix: str, resolved_params_file_path: Optional[str], file_load_success: bool, params_loaded: bool):
        """执行参数验证并记录初始化完成的最终日志。"""
        logger.debug(f"{log_prefix} 即将调用 self._validate_params()...")
        try:
            self._validate_params() # _validate_params 内部会使用 self.params, self.tf_params 等
            logger.debug(f"{log_prefix} self._validate_params() 调用完成。")
        except Exception as e_validate:
            logger.error(f"{log_prefix} CRITICAL: 在执行 _validate_params 时发生错误: {e_validate}", exc_info=True)
        logger.debug(f"策略 '{self.strategy_name}' 初始化流程完成。")
        if self.params: # 检查 self.params 是否有效
            logger.debug(f"{log_prefix} 最终已加载参数的顶层键: {list(self.params.keys())}")
            if self.tf_params: # 检查 self.tf_params 是否有效
                logger.debug(f"{log_prefix} 最终 trend_following_params 内容 (部分): "
                            f"focus_timeframe='{self.tf_params.get('focus_timeframe')}', "
                            f"transformer_model_config.d_model='{self.tf_params.get('transformer_model_config', {}).get('d_model')}'")
            elif 'trend_following_params' in self.params: # params 中有键但内容为空
                logger.warning(f"{log_prefix} 最终 'trend_following_params' 键存在于参数中，但其内容为空。")

    async def prepare_data(self, stock_code: str, base_needed_count: int = 10000) -> Optional[Tuple[pd.DataFrame, List[Dict]]]:
        """
        使用 IndicatorService 准备包含所有时间级别数据和计算指标的 DataFrame。
        Args:
            stock_code (str): 股票代码。
        Returns:
            Optional[pd.DataFrame]: 包含所有数据的 DataFrame，如果准备失败则返回 None。
        """
        log_prefix = f"[{self.strategy_name}][{stock_code}]"
        logger.info(f"{log_prefix} 开始准备策略数据...")
        # 调用 IndicatorService 的 prepare_strategy_dataframe 方法
        # 将参数文件路径和 LSTM 窗口大小传递给服务
        # IndicatorService 会根据参数文件确定所需时间级别、指标、特征工程等
        try:
            # prepare_strategy_dataframe 返回 (DataFrame, indicator_configs)
            prepared_data_tuple = await self.indicator_service.prepare_strategy_dataframe(
                stock_code=stock_code,
                params_file=self.params_file_path, # 传递存储的参数文件路径属性
                base_needed_bars=base_needed_count + self.transformer_window_size # 传递基础所需条数（例如 LSTM 窗口大小）
            )
            if prepared_data_tuple is None:
                logger.error(f"{log_prefix} IndicatorService.prepare_strategy_dataframe 返回 None。数据准备失败。")
                return None, None
            data_df, indicator_configs_from_service = prepared_data_tuple # 解包返回的元组
            if data_df is None or data_df.empty:
                logger.error(f"{log_prefix} IndicatorService.prepare_strategy_dataframe 返回空 DataFrame。数据准备失败。")
                return None, None
            # 将 IndicatorService 返回的 DataFrame 存储到实例变量，可能用于调试或后续步骤
            self.intermediate_data = data_df # 存储 IndicatorService 返回的 DataFrame
            logger.info(f"{log_prefix} 数据准备完成。DataFrame Shape: {data_df.shape}, 列数: {len(data_df.columns)}")
            # logger.debug(f"{log_prefix} 准备好的数据列 (部分): {data_df.columns.tolist()[:30]}...") # 调试输出
            # IndicatorService 已经处理了缺失值填充，这里不再需要额外的填充步骤
            return data_df, indicator_configs_from_service # 返回 DataFrame 和 indicator_configs
        except Exception as e:
            logger.error(f"{log_prefix} 调用 IndicatorService.prepare_strategy_dataframe 时出错: {e}", exc_info=True)
            return None, None

    @staticmethod
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
            if not isinstance(t, str):
                logger.warning(f"指标名称模板不是字符串: {t}. 类型: {type(t)}")
                continue
            try:
                # 改进：检查模板中是否包含 kwargs 中没有的 key，避免 KeyError
                import re
                # 使用 {key} 或 {key:format} 捕获键名，但不对格式部分进行严格验证
                keys_in_template = set(re.findall(r'{(\w+)(?:[.:!<>=\^].*?)?}', t)) # 查找所有 {key} 或 {key:format}
                missing_keys = keys_in_template - clean_kwargs.keys()
                if missing_keys:
                    # logger.warning(f"格式化指标名模板 '{t}' 缺少参数: {missing_keys}. Kwargs: {clean_kwargs}") # 日志可能过于频繁
                    # 如果缺少关键参数，这个模板就无法格式化，直接跳过
                    continue

                formatted_list.append(t.format(**clean_kwargs))
            except KeyError as e:
                # 格式化失败（通常是因为缺少参数），记录警告并跳过
                # logger.warning(f"格式化指标名模板 '{t}' 失败，缺少参数: {e}。Kwargs: {clean_kwargs}") # 日志记录可能过于频繁
                pass # 暂时忽略格式化失败的模板
            except Exception as e:
                logger.error(f"格式化指标名模板 '{t}' 时发生未知错误: {e}", exc_info=True)
                pass # 发生其他错误也跳过

        return formatted_list

    def _normalize_weights(self, weights: Dict[str, float]):
        """归一化权重字典，使其总和为1。"""
        # 增加对 weights 是否为字典的检查
        if not isinstance(weights, dict):
            logger.warning(f"[{self.strategy_name}] 尝试归一化的权重对象不是字典: {type(weights)}")
            return # 不是字典则直接返回，不处理

        total_weight = sum(weights.values())
        # 使用一个小的容差值进行浮点数比较，避免精确比较问题
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
        elif not isinstance(self.tf_params, dict):
            logger.error(f"{log_prefix} CRITICAL VALIDATION: 'trend_following_params' 参数类型不正确 (应为字典)，但得到的是: {type(self.tf_params)}！将完全依赖代码默认值。")
            self.tf_params = {}
        elif not self.tf_params:
            logger.warning(f"{log_prefix} VALIDATION: 'trend_following_params' 块在参数中存在，但其内容为空！将依赖代码默认值。")
        # 验证 base_scoring 块和 timeframes
        bs_params = self.params.get('base_scoring', {})
        timeframes = bs_params.get('timeframes', [])
        if not isinstance(timeframes, list):
            logger.error(f"{log_prefix} 'base_scoring.timeframes' 参数类型不正确 (应为列表)，但得到的是: {type(timeframes)}！")
            timeframes = [] # 设为空列表以避免后续错误
        if not timeframes:
            logger.error(f"{log_prefix} 无法获取所需列，因为 'base_scoring.timeframes' 未定义或为空。")
            # 不返回，让后续依赖 timeframes 的验证继续，但会基于空列表

        # 验证 focus_timeframe 是否在 timeframes 中
        timeframes_list = bs_params.get('timeframes', []) # 重新获取，确保是列表
        if not isinstance(timeframes_list, list): timeframes_list = [] # 再次检查类型

        if self.focus_timeframe not in timeframes_list:
             # 如果 timeframes_list 为空，这个警告是预期内的
            if timeframes_list:
                logger.warning(f"{log_prefix} VALIDATION: 主要关注时间框架 '{self.focus_timeframe}' 不在 'base_scoring.timeframes' ({timeframes_list}) 中。请检查配置。策略可能无法按预期工作或导致错误。")
            else:
                logger.warning(f"{log_prefix} VALIDATION: 'base_scoring.timeframes' 为空，无法验证 focus_timeframe '{self.focus_timeframe}' 是否在其列表中。")
        # 验证 timeframe_weights (如果存在)
        if self.timeframe_weights is not None:
            if not isinstance(self.timeframe_weights, dict):
                logger.error(f"{log_prefix} VALIDATION: 'trend_following_params.timeframe_weights' 必须是一个字典，但得到的是: {type(self.timeframe_weights)}。将忽略此配置。")
                self.timeframe_weights = None
            else:
                # 检查 timeframe_weights 中的时间框架是否都在 base_scoring.timeframes 中
                defined_tfs_set = set(timeframes_list) # 使用已验证的 timeframes_list
                for tf_w in list(self.timeframe_weights.keys()): # 遍历 keys 的副本，因为可能删除元素
                    if tf_w not in defined_tfs_set:
                        logger.warning(f"{log_prefix} VALIDATION: timeframe_weights 中包含未在 base_scoring.timeframes 中定义的时间框架: '{tf_w}'。对应的权重将被忽略。")
                        del self.timeframe_weights[tf_w] # 移除不在定义列表中的时间框架权重
                # 检查是否所有 base_scoring timeframes 都在 timeframe_weights 中定义 (如果 weights 存在的话)
                for tf_d in defined_tfs_set:
                    if tf_d not in self.timeframe_weights:
                        logger.warning(f"{log_prefix} VALIDATION: base_scoring.timeframes 中定义的时间框架 '{tf_d}' 未在 timeframe_weights 中找到。将使用默认权重 0。")
                self._normalize_weights(self.timeframe_weights) # 归一化权重
        # else: # 如果 timeframe_weights 为 None，表示使用 focus_weight 逻辑，不需要额外验证结构
        # 验证 trend_indicators
        if not self.trend_indicators:
            logger.warning(f"{log_prefix} VALIDATION: 'trend_following_params.trend_indicators' 为空列表。策略可能无法有效识别趋势。")
        elif not isinstance(self.trend_indicators, list):
             logger.error(f"{log_prefix} VALIDATION: 'trend_following_params.trend_indicators' 参数类型不正确 (应为列表)，但得到的是: {type(self.trend_indicators)}！")
        # 验证 Transformer 模型和训练配置
        model_conf = self.transformer_model_config
        if not isinstance(model_conf, dict):
             logger.error(f"{log_prefix} VALIDATION: Transformer模型结构配置 'transformer_model_config' 参数类型不正确 (应为字典)，但得到的是: {type(model_conf)}！")
             model_conf = {} # 重置为空字典以避免后续错误
        required_model_keys = ['d_model', 'nhead', 'dim_feedforward', 'nlayers']
        if not all(key in model_conf for key in required_model_keys):
             logger.warning(f"{log_prefix} VALIDATION: Transformer模型结构配置 'transformer_model_config' 缺少关键参数: {required_model_keys}。可能使用默认值。当前配置: {model_conf}")
        train_conf = self.transformer_training_config
        if not isinstance(train_conf, dict):
             logger.error(f"{log_prefix} VALIDATION: Transformer训练配置 'transformer_training_config' 参数类型不正确 (应为字典)，但得到的是: {type(train_conf)}！")
             train_conf = {} # 重置为空字典以避免后续错误
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
        # 从 tf_params 中安全获取 signal_combination_weights
        lstm_combination_weights = self.tf_params.get('signal_combination_weights', {})
        if not isinstance(lstm_combination_weights, dict) or not lstm_combination_weights:
             logger.warning(f"{log_prefix} VALIDATION: 'signal_combination_weights' (在 trend_following_params 中) 参数无效或为空。将使用代码中定义的默认组合权重 (0.6/0.4) 并归一化。当前值: {lstm_combination_weights}")
             default_combo_weights = {'rule_weight': 0.6, 'transformer_weight': 0.4} # 修改键名以反映是 Transformer
             # 直接赋值给 tf_params，确保这个默认值被保存
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
        if self.rule_signal_weights.get('boll_breakout', 0) > 0 and (not isinstance(boll_breakout_params, dict) or 'period' not in boll_breakout_params or 'std_dev' not in boll_breakout_params):
             logger.warning(f"{log_prefix} VALIDATION: 'boll_breakout' 信号权重 > 0，但 'boll_breakout_params' (包含 period, std_dev) 未在 trend_following_params 中找到或无效。将使用默认值或可能导致错误。")
        volume_conf = self.params.get('volume_confirmation', {})
        if isinstance(volume_conf, dict) and (volume_conf.get('enabled', False) or volume_conf.get('volume_analysis_enabled', False)):
             obv_ma_period = volume_conf.get('obv_ma_period')
             if obv_ma_period is None:
                  logger.warning(f"{log_prefix} VALIDATION: 量能确认启用，但 'volume_confirmation.obv_ma_period' 未定义。OBV MA 相关功能可能受影响。")
             vc_tfs = volume_conf.get('timeframes')
             if not vc_tfs:
                  logger.warning(f"{log_prefix} VALIDATION: 量能确认启用，但 'volume_confirmation.tf' 未定义或为空。将使用 focus_timeframe ({self.focus_timeframe})。")
             elif isinstance(vc_tfs, str):
                  volume_conf['timeframes'] = [vc_tfs] # 统一为列表
             elif not isinstance(vc_tfs, list):
                  logger.error(f"{log_prefix} VALIDATION: 'volume_confirmation.tf' 必须是字符串或列表，但得到的是: {type(vc_tfs)}。量能确认可能无法按预期工作。")
        logger.debug(f"{log_prefix} TrendFollowingStrategy 特定参数验证完成。")

    def _load_selected_feature_names(self):
        """
        从 JSON 文件加载 Transformer 模型所需的特征名列表。
        此方法应在 set_model_paths 被调用并设置 self.selected_features_path 后调用。
        """
        # MODIFIED LINE: 确保在加载前设置了路径
        if not hasattr(self, 'selected_features_path') or self.selected_features_path is None:
            logger.warning(f"[{self.strategy_name}] selected_features_path 未设置，无法加载特征名。请先调用 set_model_paths。") # MODIFIED LINE: 提示用户先调用set_model_paths
            return
        try:
            with open(self.selected_features_path, 'r', encoding='utf-8') as f:
                self.selected_feature_names_for_transformer = json.load(f)
            # print(f"[{self.strategy_name}] 已从 {self.selected_features_path} 加载 {len(self.selected_feature_names_for_transformer)} 个特征名。") # ADDED LINE: 调试信息
        except FileNotFoundError:
            logger.error(f"[{self.strategy_name}] 特征名文件未找到: {self.selected_features_path}")
        except json.JSONDecodeError:
            logger.error(f"[{self.strategy_name}] 特征名文件 JSON 解析失败: {self.selected_features_path}")
        except Exception as e:
            logger.error(f"[{self.strategy_name}] 加载特征名时发生未知错误: {e}")

    def set_model_paths(self, stock_code: str):
        """
        为特定股票设置模型、scaler 和准备好的数据的保存/加载路径。
        """
        stock_root_dir =  os.path.join(self.base_data_dir, stock_code)
        os.makedirs(stock_root_dir, exist_ok=True)
        prepared_data_dir = os.path.join(stock_root_dir, "prepared_data")
        os.makedirs(prepared_data_dir, exist_ok=True)
        trained_model_dir = os.path.join(self.base_data_dir, "trained_model")
        os.makedirs(trained_model_dir, exist_ok=True)
        self.feature_scaler_path = os.path.join(prepared_data_dir, "trend_following_transformer_feature_scaler.save")
        self.target_scaler_path = os.path.join(prepared_data_dir, "trend_following_transformer_target_scaler.save")
        self.selected_features_path = os.path.join(prepared_data_dir, "trend_following_transformer_selected_features.json")
        self.all_prepared_data_npz_path = os.path.join(prepared_data_dir, "all_prepared_data_transformer.npz")
        self.model_dir = trained_model_dir
        self.model_path = os.path.join(trained_model_dir, f"best_transformer_model_{stock_code}.pth")
        # print(f"[{self.strategy_name}] 为股票 {stock_code} 设置文件路径:")
        # print(f"  模型权重: {self.model_path}")
        # print(f"  特征Scaler: {self.feature_scaler_path}")
        # logger.debug(f"  目标Scaler: {self.target_scaler_path}")
        # logger.debug(f"  准备数据NPZ: {self.all_prepared_data_npz_path}")
        # logger.debug(f"  选中特征: {self.selected_features_path}")
        # 设置 PCA 模型保存路径
        self.pca_model_path = os.path.join(prepared_data_dir, "trend_following_transformer_pca_model.joblib")
        # 设置 PCA 前使用的 Scaler 保存路径
        self.scaler_for_pca_path = os.path.join(prepared_data_dir, "trend_following_transformer_scaler_for_pca.joblib")
        # 设置特征选择模型保存路径
        self.feature_selector_model_path = os.path.join(prepared_data_dir, "trend_following_transformer_feature_selector_model.joblib")

        self._load_selected_feature_names() # 在设置路径后立即加载特征名

    def train_transformer_model_from_prepared_data(
        self,
        stock_code: str,
        transformer_hyperparams: dict = None,
        only_return_val_metric: bool = False,
        trial=None  # Optuna Trial 对象，或 None
    ):
        """
        为特定股票加载已准备好的数据，构建并训练 Transformer 模型，然后保存模型权重。
        如果提供 transformer_hyperparams，将使用这些参数来构建模型，否则使用默认参数。
        如果 only_return_val_metric=True，则只返回val_mae，不保存模型。
        """
        # 设置模型文件路径，包括模型目录和基准文件名（非 Trial 专属）
        self.set_model_paths(stock_code)
        logger.info(f"[{self.strategy_name}] 开始为股票 {stock_code} 训练 Transformer 模型 (从已准备数据加载)...")

        # 设置 PyTorch 使用的线程数
        torch.set_num_threads(16)  # 根据你的 CPU 核心数进行调整

        # 增加错误处理，确保 load_prepared_data 返回有效数据
        try:
            features_scaled_train_np, targets_scaled_train_np, \
            features_scaled_val_np, targets_scaled_val_np, \
            features_scaled_test_np, targets_scaled_test_np, \
            feature_scaler, target_scaler = self.load_prepared_data(stock_code)

            if features_scaled_train_np is None or features_scaled_train_np.shape[0] == 0 or \
            targets_scaled_train_np is None or targets_scaled_train_np.shape[0] == 0 or \
            feature_scaler is None or target_scaler is None or not self.selected_feature_names_for_transformer:
                logger.error(f"[{self.strategy_name}] 股票 {stock_code} 加载已准备好的数据/Scaler/特征列表失败或数据无效，无法继续训练。")
                self.transformer_model = None
                self.feature_scaler = None
                self.target_scaler = None
                self.selected_feature_names_for_transformer = []
                return None if only_return_val_metric else False
            
            # 保存 Scaler 和特征名称到实例属性，供后续使用
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler
            # num_features 应该是时间序列数据的特征维度，即 (num_samples, sequence_length, num_features) 中的最后一个维度
            # 或者如果是平坦的 (num_samples, num_features_flat)，则直接用第二个维度
            num_features = features_scaled_train_np.shape[-1] 
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
            return None if only_return_val_metric else False
        
        # ----------- 合并贝叶斯优化参数 -----------
        # 复制原始配置，避免污染全局
        # 这些是模型参数和训练参数的“默认值”，Optuna采样到的会覆盖它们
        model_config = self.transformer_model_config.copy()
        training_config = self.transformer_training_config.copy()

        if transformer_hyperparams:
            # 检查并更新模型参数
            if 'transformer_model_config' in transformer_hyperparams and \
               isinstance(transformer_hyperparams['transformer_model_config'], dict):
                model_config.update(transformer_hyperparams['transformer_model_config'])
                logger.info(f"[{self.strategy_name}][{stock_code}] 已从 Optuna 更新模型配置: {model_config}") # 调试日志

            # 检查并更新训练参数
            if 'transformer_training_config' in transformer_hyperparams and \
               isinstance(transformer_hyperparams['transformer_training_config'], dict):
                training_config.update(transformer_hyperparams['transformer_training_config'])
                logger.info(f"[{self.strategy_name}][{stock_code}] 已从 Optuna 更新训练配置: {training_config}") # 调试日志

                # 针对 batch_size 的特殊处理 (为了 DataLoader 的创建)
                if 'batch_size' in training_config:
                    self.transformer_batch_size = training_config['batch_size']
                    logger.info(f"[{self.strategy_name}][{stock_code}] self.transformer_batch_size 已更新为: {self.transformer_batch_size}") # 调试日志
        # ----------- 结束合并参数 -----------

        # 现在，self.transformer_batch_size 已经更新为 Optuna 采样到的值 (例如 96)
        # 所以接下来创建 DataLoader 时会使用正确的值
        try:
            # 将 NumPy 数组转换为 PyTorch Tensor
            # 确保 features 是 (num_samples, sequence_length, num_features)
            # 并且 targets 是 (num_samples, output_dim)
            # 您的原始代码使用了 TimeSeriesDataset，这表示数据需要转换为序列形式
            train_features_tensor = torch.tensor(features_scaled_train_np, dtype=torch.float32)
            train_targets_tensor = torch.tensor(targets_scaled_train_np, dtype=torch.float32)
            pin_memory = torch.cuda.is_available()
            # 确保 targets 是 2D，例如 (N, 1)
            if train_targets_tensor.ndim == 1:
                train_targets_tensor = train_targets_tensor.unsqueeze(1)

            train_dataset = TimeSeriesDataset(train_features_tensor, train_targets_tensor, self.transformer_window_size)
            train_loader = DataLoader(train_dataset, batch_size=self.transformer_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=0) # num_workers 可根据系统调整

            val_loader = None
            if features_scaled_val_np is not None and features_scaled_val_np.shape[0] >= self.transformer_window_size and \
               targets_scaled_val_np is not None and targets_scaled_val_np.shape[0] >= self.transformer_window_size:
                val_features_tensor = torch.tensor(features_scaled_val_np, dtype=torch.float32)
                val_targets_tensor = torch.tensor(targets_scaled_val_np, dtype=torch.float32)
                if val_targets_tensor.ndim == 1:
                    val_targets_tensor = val_targets_tensor.unsqueeze(1)
                val_dataset = TimeSeriesDataset(val_features_tensor, val_targets_tensor, self.transformer_window_size)
                if len(val_dataset) > 0:
                    val_loader = DataLoader(val_dataset, batch_size=self.transformer_batch_size, shuffle=False, pin_memory=pin_memory, num_workers=0)
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 验证集 Dataset 为空 (数据量不足 {self.transformer_window_size} 或其他原因)。验证阶段将跳过。")
            else:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 验证集数据量不足 {self.transformer_window_size}。验证阶段将跳过。")

            test_loader = None
            if features_scaled_test_np is not None and features_scaled_test_np.shape[0] >= self.transformer_window_size and \
               targets_scaled_test_np is not None and targets_scaled_test_np.shape[0] >= self.transformer_window_size:
                test_features_tensor = torch.tensor(features_scaled_test_np, dtype=torch.float32)
                test_targets_tensor = torch.tensor(targets_scaled_test_np, dtype=torch.float32)
                if test_targets_tensor.ndim == 1:
                    test_targets_tensor = test_targets_tensor.unsqueeze(1)
                test_dataset = TimeSeriesDataset(test_features_tensor, test_targets_tensor, self.transformer_window_size)
                if len(test_dataset) > 0:
                    test_loader = DataLoader(test_dataset, batch_size=self.transformer_batch_size, shuffle=False, num_workers=0)
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 测试集 Dataset 为空 (数据量不足 {self.transformer_window_size} 或其他原因)。测试评估将跳过。")
            else:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 测试集数据量不足 {self.transformer_window_size}。测试评估将跳过。")
            
            if len(train_dataset) == 0:
                logger.error(f"[{self.strategy_name}][{stock_code}] 训练集 Dataset 为空 (数据量不足 {self.transformer_window_size})。停止训练。")
                return None if only_return_val_metric else False
            
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 创建 PyTorch Dataset/DataLoader 出错: {e}", exc_info=True)
            return None if only_return_val_metric else False
        
        # ----------- 构建 Transformer 模型 (根据合并后的 model_config) -----------
        try:
            # 确保 num_features 和 window_size 正确传递给模型构建
            # num_features 应该是每个时间步的特征数量
            model_instance = build_transformer_model(
                num_features=num_features, # 从数据加载中获取的特征维度
                model_config=model_config, # 包含 Optuna 采样的 d_model, nhead, nlayers 等
                summary=True,
                window_size=self.transformer_window_size # 从实例属性获取序列长度
            )
            # 在这里将构建好的模型赋值给 self.transformer_model
            # 如果 only_return_val_metric 为 True，此模型只用于当前 trial，不会被保存为最终模型
            self.transformer_model = model_instance # <--- 将新实例化的模型赋值给属性
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 构建 Transformer 模型出错: {e}", exc_info=True)
            self.transformer_model = None # 确保模型状态为 None
            return None if only_return_val_metric else False

        try:
            # checkpoint_dir 是股票专属的模型保存目录
            # `self.model_dir_path` 是由 `self.set_model_paths` 设置的，它应该包含 /trained_model 部分
            checkpoint_dir_for_dl_utils = str(self.model_dir) 
            if not os.path.exists(checkpoint_dir_for_dl_utils):
                os.makedirs(checkpoint_dir_for_dl_utils)
                logger.info(f"[{self.strategy_name}][{stock_code}] 创建模型保存目录: {checkpoint_dir_for_dl_utils}")

            # ----------- 传递动态训练参数给 deep_learning_utils.train_transformer_model -----------
            # `deep_learning_utils.train_transformer_model` 会根据 `trial` 参数在 `checkpoint_dir` 下生成唯一的文件名
            print(f"DEBUG: 传递给 deep_learning_utils.train_transformer_model 的 trial: {trial}")
            
            # 调用训练函数，传入新实例化的模型，并根据 `trial` 参数处理文件路径
            self.transformer_model, history_df = deep_learning_utils.train_transformer_model(
                model=self.transformer_model, # <--- 传入刚刚构建的 TransformerModel 实例
                train_loader=train_loader,
                val_loader=val_loader,
                target_scaler=self.target_scaler,
                training_config=training_config, # <-- 传递更新后的 training_config
                checkpoint_dir=checkpoint_dir_for_dl_utils, # <--- 传递股票专属的目录
                stock_code=stock_code,
                plot_training_history=self.tf_params.get('transformer_plot_history', False),
                enable_anomaly_detection=self.tf_params.get('transformer_enable_anomaly_detection', False),
                trial=trial # <--- 保持传递 Optuna Trial 对象
            )
            print(f"[{self.strategy_name}][{stock_code}] 训练完成，返回验证指标历史，长度: {len(history_df)}")
            # ----------- 结束训练调用 -----------

            # ----------- 只返回val_mae，不保存模型和不做测试集评估 -----------
            if only_return_val_metric:
                # Optuna 模式下，直接返回验证指标
                if history_df is not None and 'val_mae' in history_df.columns and not history_df['val_mae'].isnull().all():
                    val_mae = history_df['val_mae'].dropna().values[-1] # 取最后一个非 NaN 的 val_mae
                    logger.info(f"[{self.strategy_name}][{stock_code}] Optuna Trial 训练完成，返回 val_mae={val_mae:.6f}")
                else:
                    # 如果 val_mae 历史全是 NaN，表示训练可能失败或无效，返回极大值
                    val_mae = float('inf')
                    logger.warning(f"[{self.strategy_name}][{stock_code}] Optuna Trial 训练未获得有效 val_mae，返回 inf。")
                return val_mae
            # ----------- 结束只返回val_mae -----------

            # --- 非 Optuna Trial 模式 (最终训练) ---
            # 此时 deep_learning_utils.train_transformer_model 应该已经将最佳模型保存到标准路径 (无 Trial ID)
            # 并且已经加载回 self.transformer_model (如果成功)
            if self.transformer_model: # 确保模型已加载
                 logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型训练完成，最佳模型权重已加载到实例。")
                 # 可以在这里额外保存最终模型到 self.model_path（非 Trial 专属路径），
                 # 但 train_transformer_model 函数已经处理了最终保存逻辑，所以这里不需要重复保存。
                 # 只需确保 self.model_path 在这里被正确地引用，
                 # 例如，如果后续有其他需要加载这个最终模型的地方，它会从这个路径加载。
            else:
                 logger.warning(f"[{self.strategy_name}][{stock_code}] Transformer 模型训练完成，但模型实例为 None。模型权重可能未按预期加载。")
                 return False # 训练未成功

            # ----------- 测试集评估 -----------
            if test_loader is not None and len(test_loader) > 0 and self.transformer_model is not None:
                logger.info(f"[{self.strategy_name}] 开始在测试集上评估股票 {stock_code} 的 Transformer 模型...")
                loss_fn_name = training_config.get('loss', 'mse').lower()
                criterion_eval = nn.MSELoss() if loss_fn_name == 'mse' else \
                                nn.L1Loss() if loss_fn_name == 'mae' else \
                                nn.HuberLoss() if loss_fn_name == 'huber' else nn.MSELoss()
                mae_metric_eval = nn.L1Loss()
                
                test_metrics = evaluate_transformer_model(
                    model=self.transformer_model, # 使用训练完成并加载了最佳权重的模型
                    test_loader=test_loader,
                    criterion=criterion_eval,
                    mae_metric=mae_metric_eval,
                    target_scaler=self.target_scaler,
                    device=self.device
                )
                # logger.info(f"[{self.strategy_name}][{stock_code}] 测试集评估结果: {test_metrics}")
                evaluation_logger.info(f"[{self.strategy_name}][{stock_code}] 测试集评估结果: {test_metrics}")
                return True
            else:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 测试集 DataLoader 为空或模型实例为 None，跳过测试评估。")
                return True # 即使没有测试集评估，如果训练成功也返回 True

        # === 区分 TrialPruned 和其他异常 ===
        except optuna.exceptions.TrialPruned:
            # 如果是 Optuna 触发的早停，则重新抛出此异常
            logger.info(f"[{self.strategy_name}][{stock_code}] Trial 被 Optuna 早停 (Pruned)，重新抛出异常。")
            raise # 重新抛出异常，让 run_local_optuna_multi_process.py 中的 objective 函数捕获并正确处理

        except Exception as e:
            # 捕获所有其他非 TrialPruned 的错误
            logger.error(f"[{self.strategy_name}][{stock_code}] 训练 Transformer 模型出错: {e}", exc_info=True)
            self.transformer_model = None # 确保模型状态为 None
            # 对于这些真正的错误，如果处于 only_return_val_metric 模式，返回 NaN 表示 Trial 失败
            return float('nan') if only_return_val_metric else False
        # === 结束 ===

    async def get_required_columns(self, stock_code: str) -> List[str]:
        """
        根据策略参数和指标命名规范，动态生成并返回 IndicatorService 需要准备的所有数据列名。
        这些是 IndicatorService 最终会在 DataFrame 中生成的列名。
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
        if not isinstance(timeframes, list):
            logger.error(f"{log_prefix} 'base_scoring.timeframes' 参数类型不正确 (应为列表)，但得到的是: {type(timeframes)}！无法获取所需列。")
            return []
        if not timeframes:
            logger.error(f"{log_prefix} 无法获取所需列，因为 'base_scoring.timeframes' 未定义或为空。")
            return []
        # 准备参数源列表，优先级从高到低 (策略特定参数通常优先级高)
        param_sources = [
            self.tf_params, # trend_following_params (策略特定参数)
            self.params.get('volume_confirmation', {}), # volume_confirmation
            self.params.get('indicator_analysis_params', {}), # indicator_analysis_params
            self.fe_params, # feature_engineering_params
            bs_params # base_scoring
        ]
        # 获取命名规范字典
        indicator_naming_conv = NAMING_CONFIG.get('indicator_naming_conventions', {})
        derivative_naming_conv = NAMING_CONFIG.get('derivative_feature_naming_conventions', {})
        ohlcv_naming_conv = NAMING_CONFIG.get('ohlcv_naming_convention', {})
        external_naming_conv = NAMING_CONFIG.get('external_feature_naming_conventions', {}) # 新增：外部特征命名规范

        # 增加类型检查
        if not isinstance(indicator_naming_conv, dict): indicator_naming_conv = {}
        if not isinstance(derivative_naming_conv, dict): derivative_naming_conv = {}
        if not isinstance(ohlcv_naming_conv, dict): ohlcv_naming_conv = {}
        if not isinstance(external_naming_conv, dict): external_naming_conv = {}


        # 1. 基础OHLCV (来自NAMING_CONFIG.ohlcv_naming_convention)
        ohlcv_output_cols_conf = ohlcv_naming_conv.get('output_columns', [])
        if not isinstance(ohlcv_output_cols_conf, list):
            logger.error(f"{log_prefix} OHLCV 命名规范 output_columns 类型不正确 (应为列表)。")
            ohlcv_output_cols_conf = []
        for tf_str in timeframes:
            for col_conf in ohlcv_output_cols_conf:
                if isinstance(col_conf, dict) and 'name_pattern' in col_conf:
                    # OHLCV 列名模式通常不包含参数，直接拼接时间框架后缀
                    required.add(f"{col_conf['name_pattern']}_{tf_str}")
        # 2. 基础评分指标 (来自NAMING_CONFIG.indicator_naming_conventions)
        score_indicators_config_keys = bs_params.get('score_indicators', [])
        if not isinstance(score_indicators_config_keys, list):
            logger.error(f"{log_prefix} 'base_scoring.score_indicators' 参数类型不正确 (应为列表)。")
            score_indicators_config_keys = []
        # 获取指标参数，这些值将用于填充命名模板 (使用 _get_param_val 从 param_sources 中获取参数)
        # MACD参数
        macd_fast = self._get_param_val(param_sources, 'period_fast', 12) # 优先从 tf_params, ia_params, fe_params, bs_params 查找 period_fast
        macd_slow = self._get_param_val(param_sources, 'period_slow', 26) # 优先从 tf_params, ia_params, fe_params, bs_params 查找 period_slow
        macd_sig = self._get_param_val(param_sources, 'signal_period', 9) # 优先从 tf_params, ia_params, fe_params, bs_params 查找 signal_period
        # RSI参数
        rsi_period = self._get_param_val(param_sources, 'period', 14) # 优先从 tf_params, ia_params, fe_params, bs_params 查找 period
        # KDJ参数 (注意JSON和策略中参数名的对应)
        kdj_k_period = self._get_param_val(param_sources, 'kdj_period_k', self._get_param_val(param_sources, 'period', 9)) # 优先 kdj_period_k，其次 period
        kdj_d_period = self._get_param_val(param_sources, 'kdj_period_d', self._get_param_val(param_sources, 'signal_period', 3)) # 优先 kdj_period_d，其次 signal_period
        kdj_j_smooth_k = self._get_param_val(param_sources, 'kdj_period_j', self._get_param_val(param_sources, 'smooth_k_period', 3)) # 优先 kdj_period_j，其次 smooth_k_period
        # BOLL参数 (基础评分用的BOLL)
        boll_period = self._get_param_val(param_sources, 'boll_period', self._get_param_val(param_sources, 'period', 20)) # 优先 boll_period，其次 period
        boll_std_dev = self._get_param_val(param_sources, 'boll_std_dev', self._get_param_val(param_sources, 'std_dev', 2.0)) # 优先 boll_std_dev，其次 std_dev
        # CCI参数
        cci_period = self._get_param_val(param_sources, 'period', 14)
        # MFI参数
        mfi_period = self._get_param_val(param_sources, 'period', 14)
        # ROC参数
        roc_period = self._get_param_val(param_sources, 'period', 12)
        # DMI参数
        dmi_period = self._get_param_val(param_sources, 'period', 14)
        # SAR参数
        sar_step = self._get_param_val(param_sources, 'af_step', 0.02)
        sar_max_af = self._get_param_val(param_sources, 'max_af', 0.2)
        # EMA/SMA 基础参数 (如果 score_indicators 中包含它们，通常是某个默认周期)
        ema_base_period = self._get_param_val(param_sources, 'ema_base_period', self._get_param_val(param_sources, 'period', 20)) # 优先 ema_base_period，其次 period
        sma_base_period = self._get_param_val(param_sources, 'sma_base_period', self._get_param_val(param_sources, 'period', 20)) # 优先 sma_base_period，其次 period
        for indi_key_upper in [str(key).upper() for key in score_indicators_config_keys]:
            if indi_key_upper not in indicator_naming_conv:
                logger.warning(f"{log_prefix} 评分指标 '{indi_key_upper}' 在命名规范中未定义或加载失败，跳过。")
                continue
            indi_naming_conf = indicator_naming_conv[indi_key_upper]
            if not isinstance(indi_naming_conf, dict) or 'output_columns' not in indi_naming_conf or not isinstance(indi_naming_conf['output_columns'], list):
                logger.warning(f"{log_prefix} 评分指标 '{indi_key_upper}' 的命名配置无效，跳过。")
                continue
            params_for_format = {}
            # 根据指标类型填充格式化参数字典
            if indi_key_upper == 'MACD':
                params_for_format = {'period_fast': macd_fast, 'period_slow': macd_slow, 'signal_period': macd_sig}
            elif indi_key_upper == 'RSI':
                params_for_format = {'period': rsi_period}
            elif indi_key_upper == 'KDJ':
                if isinstance(kdj_k_period, (int, float)) and isinstance(kdj_d_period, (int, float)) and isinstance(kdj_j_smooth_k, (int, float)):
                    params_for_format = {'period': kdj_k_period, 'signal_period': kdj_d_period, 'smooth_k_period': kdj_j_smooth_k}
                else:
                    logger.warning(f"{log_prefix} KDJ 评分指标参数无效: k_period={kdj_k_period}, d_period={kdj_d_period}, smooth_k_period={kdj_j_smooth_k}. 跳过 KDJ 评分列请求。")
                    continue
            elif indi_key_upper == 'BOLL':
                if isinstance(boll_period, (int, float)) and isinstance(boll_std_dev, (int, float)):
                    params_for_format = {'period': boll_period, 'std_dev': boll_std_dev}
                else:
                    logger.warning(f"{log_prefix} BOLL 评分指标参数无效: period={boll_period}, std_dev={boll_std_dev}. 跳过 BOLL 评分列请求。")
                    continue
            elif indi_key_upper == 'CCI':
                if isinstance(cci_period, (int, float)): params_for_format = {'period': cci_period}
                else: continue
            elif indi_key_upper == 'MFI':
                if isinstance(mfi_period, (int, float)): params_for_format = {'period': mfi_period}
                else: continue
            elif indi_key_upper == 'ROC':
                if isinstance(roc_period, (int, float)): params_for_format = {'period': roc_period}
                else: continue
            elif indi_key_upper == 'DMI':
                if isinstance(dmi_period, (int, float)): params_for_format = {'period': dmi_period}
                else: continue
            elif indi_key_upper == 'SAR':
                if isinstance(sar_step, (int, float)) and isinstance(sar_max_af, (int, float)): params_for_format = {'af_step': sar_step, 'max_af': sar_max_af}
                else: continue
            elif indi_key_upper == 'EMA':
                if isinstance(ema_base_period, (int, float)): params_for_format = {'period': ema_base_period}
                else: continue
            elif indi_key_upper == 'SMA':
                if isinstance(sma_base_period, (int, float)): params_for_format = {'period': sma_base_period}
                else: continue
            # 其他无参数或固定参数的指标如 OBV, ADL 不需要特别处理 params_for_format
            for tf_str in timeframes:
                for col_conf in indi_naming_conf['output_columns']:
                    if isinstance(col_conf, dict) and 'name_pattern' in col_conf:
                        base_names = TrendFollowingStrategy._format_indicator_name(col_conf['name_pattern'], **params_for_format)
                        for base_name in base_names:
                            # 基础指标列名格式: BASE_NAME_PARAMS_TF
                            required.add(f"{base_name}_{tf_str}")
        # 3. 量能确认指标 (AMT_MA, CMF) 和 OBV, OBV_MA
        vc_params = self.params.get('volume_confirmation', {})
        if isinstance(vc_params, dict) and (vc_params.get('enabled', False) or vc_params.get('volume_analysis_enabled', False)):
            vc_tf_list_raw = vc_params.get('timeframes', timeframes)
            vc_tf_list = [vc_tf_list_raw] if isinstance(vc_tf_list_raw, str) else vc_tf_list_raw
            if not isinstance(vc_tf_list, list):
                logger.error(f"{log_prefix} 'volume_confirmation.tf' 参数类型不正确 (应为字符串或列表)，但得到的是: {type(vc_tf_list)}！量能指标请求可能不完整。")
                vc_tf_list = []
            # 获取量能指标参数
            amt_ma_period_vc = self._get_param_val(param_sources, 'amount_ma_period', 15)
            cmf_period_vc = self._get_param_val(param_sources, 'cmf_period', 20)
            obv_ma_period_vc = self._get_param_val(param_sources, 'obv_ma_period', 10)
            for vc_tf_str in vc_tf_list:
                if vc_tf_str not in timeframes: continue # 只请求在基础时间框架中定义的级别
                # AMT_MA
                if 'AMT_MA' in indicator_naming_conv:
                    amt_ma_naming_conf = indicator_naming_conv['AMT_MA']
                    if isinstance(amt_ma_naming_conf, dict) and 'output_columns' in amt_ma_naming_conf and isinstance(amt_ma_naming_conf['output_columns'], list):
                        amt_ma_name_patterns = [c['name_pattern'] for c in amt_ma_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                        if isinstance(amt_ma_period_vc, (int, float)) and amt_ma_period_vc > 0:
                            for name in TrendFollowingStrategy._format_indicator_name(amt_ma_name_patterns, period=amt_ma_period_vc):
                                required.add(f"{name}_{vc_tf_str}")
                # CMF
                if 'CMF' in indicator_naming_conv:
                    cmf_naming_conf = indicator_naming_conv['CMF']
                    if isinstance(cmf_naming_conf, dict) and 'output_columns' in cmf_naming_conf and isinstance(cmf_naming_conf['output_columns'], list):
                        cmf_name_patterns = [c['name_pattern'] for c in cmf_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                        if isinstance(cmf_period_vc, (int, float)) and cmf_period_vc > 0:
                            for name in TrendFollowingStrategy._format_indicator_name(cmf_name_patterns, period=cmf_period_vc):
                                required.add(f"{name}_{vc_tf_str}")
                # OBV (基础指标)
                if 'OBV' in indicator_naming_conv:
                    obv_naming_conf = indicator_naming_conv['OBV']
                    if isinstance(obv_naming_conf, dict) and 'output_columns' in obv_naming_conf and isinstance(obv_naming_conf['output_columns'], list):
                         obv_name_patterns = [c['name_pattern'] for c in obv_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                         for name in TrendFollowingStrategy._format_indicator_name(obv_name_patterns): # OBV 无参数
                            required.add(f"{name}_{vc_tf_str}")
                # OBV_MA (衍生特征)
                if 'OBV_MA' in derivative_naming_conv:
                    obv_ma_deriv_conf = derivative_naming_conv['OBV_MA']
                    if isinstance(obv_ma_deriv_conf, dict) and 'output_column_pattern' in obv_ma_deriv_conf and obv_ma_period_vc is not None:
                        obv_ma_pattern = obv_ma_deriv_conf['output_column_pattern']
                        if isinstance(obv_ma_period_vc, (int, float)) and obv_ma_period_vc > 0:
                            for name in TrendFollowingStrategy._format_indicator_name(obv_ma_pattern, period=obv_ma_period_vc):
                                required.add(f"{name}_{vc_tf_str}")
        # 4. 其他分析指标 (STOCH, VOL_MA, VWAP, ADL, ICHIMOKU, PIVOT_POINTS, ATR, HV, KC, MOM, WILLR, VROC, AROC)
        ia_params_global = self.params.get('indicator_analysis_params', {})
        fe_params_global = self.fe_params # 使用 self.fe_params
        if not isinstance(ia_params_global, dict): ia_params_global = {}
        if not isinstance(fe_params_global, dict): fe_params_global = {}
        ia_tf_list_raw = ia_params_global.get('apply_on_timeframes', timeframes)
        ia_tf_list = [ia_tf_list_raw] if isinstance(ia_tf_list_raw, str) else ia_tf_list_raw
        if not isinstance(ia_tf_list, list): ia_tf_list = []
        fe_tf_list_raw = fe_params_global.get('apply_on_timeframes', timeframes)
        fe_tf_list = [fe_tf_list_raw] if isinstance(fe_tf_list_raw, str) else fe_tf_list_raw
        if not isinstance(fe_tf_list, list): fe_tf_list = []
        analysis_timeframes = list(set(ia_tf_list + fe_tf_list))
        for analyze_tf_str in analysis_timeframes:
            if analyze_tf_str not in timeframes: continue # 只请求在基础时间框架中定义的级别
            # STOCH
            stoch_k_ia = self._get_param_val(param_sources, 'stoch_k', self._get_param_val(param_sources, 'k_period', 14))
            stoch_d_ia = self._get_param_val(param_sources, 'stoch_d', self._get_param_val(param_sources, 'd_period', 3))
            stoch_smooth_k_ia = self._get_param_val(param_sources, 'stoch_smooth_k', self._get_param_val(param_sources, 'smooth_k_period', 3))
            if ia_params_global.get('calculate_stoch', False) and 'STOCH' in indicator_naming_conv:
                stoch_naming_conf = indicator_naming_conv['STOCH']
                if isinstance(stoch_naming_conf, dict) and 'output_columns' in stoch_naming_conf and isinstance(stoch_naming_conf['output_columns'], list):
                    stoch_name_patterns = [c['name_pattern'] for c in stoch_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(stoch_k_ia, (int, float)) and isinstance(stoch_d_ia, (int, float)) and isinstance(stoch_smooth_k_ia, (int, float)):
                        for name in TrendFollowingStrategy._format_indicator_name(stoch_name_patterns, k_period=stoch_k_ia, d_period=stoch_d_ia, smooth_k_period=stoch_smooth_k_ia):
                            required.add(f"{name}_{analyze_tf_str}")
                    else:
                        logger.warning(f"{log_prefix} STOCH 指标参数无效: k={stoch_k_ia}, d={stoch_d_ia}, smooth_k={stoch_smooth_k_ia}. 跳过 STOCH 列请求 ({analyze_tf_str})。")
            # VOL_MA
            vol_ma_period_ia = self._get_param_val(param_sources, 'volume_ma_period', self._get_param_val(param_sources, 'period', 20))
            if ia_params_global.get('calculate_vol_ma', False) and 'VOL_MA' in indicator_naming_conv:
                vol_ma_naming_conf = indicator_naming_conv['VOL_MA']
                if isinstance(vol_ma_naming_conf, dict) and 'output_columns' in vol_ma_naming_conf and isinstance(vol_ma_naming_conf['output_columns'], list):
                    vol_ma_name_patterns = [c['name_pattern'] for c in vol_ma_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(vol_ma_period_ia, (int, float)) and vol_ma_period_ia > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(vol_ma_name_patterns, period=vol_ma_period_ia):
                            required.add(f"{name}_{analyze_tf_str}")
            # VWAP
            vwap_anchor_ia = self._get_param_val(param_sources, 'vwap_anchor', None)
            if ia_params_global.get('calculate_vwap', False) and 'VWAP' in indicator_naming_conv:
                vwap_configs = indicator_naming_conv['VWAP'].get('output_columns', [])
                if isinstance(vwap_configs, list):
                    if vwap_anchor_ia is None:
                        pattern = next((c['name_pattern'] for c in vwap_configs if isinstance(c, dict) and 'name_pattern' in c and "{anchor}" not in c['name_pattern']), None)
                        if pattern:
                            for name in TrendFollowingStrategy._format_indicator_name(pattern):
                                required.add(f"{name}_{analyze_tf_str}")
                        else:
                            logger.warning(f"{log_prefix} VWAP 命名规范中未找到无锚点的模式。无法为 {analyze_tf_str} 请求无锚点VWAP。")
                    else:
                        pattern = next((c['name_pattern'] for c in vwap_configs if isinstance(c, dict) and 'name_pattern' in c and "{anchor}" in c['name_pattern']), None)
                        if pattern:
                            for name in TrendFollowingStrategy._format_indicator_name(pattern, anchor=vwap_anchor_ia):
                                required.add(f"{name}_{analyze_tf_str}")
                        else:
                            logger.warning(f"{log_prefix} VWAP 命名规范中未找到带锚点的模式。无法为 {analyze_tf_str} 请求带锚点VWAP (锚点: {vwap_anchor_ia})。")
                else:
                    logger.warning(f"{log_prefix} VWAP 命名规范 output_columns 类型不正确 (应为列表)。")
            # ADL (无参数)
            if ia_params_global.get('calculate_adl', False) and 'ADL' in indicator_naming_conv:
                adl_naming_conf = indicator_naming_conv['ADL']
                if isinstance(adl_naming_conf, dict) and 'output_columns' in adl_naming_conf and isinstance(adl_naming_conf['output_columns'], list):
                    adl_name_patterns = [c['name_pattern'] for c in adl_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    for name in TrendFollowingStrategy._format_indicator_name(adl_name_patterns):
                        required.add(f"{name}_{analyze_tf_str}")
            # ICHIMOKU
            if ia_params_global.get('calculate_ichimoku', False) and 'ICHIMOKU' in indicator_naming_conv:
                ichimoku_tenkan_ia = self._get_param_val(param_sources, 'ichimoku_tenkan', self._get_param_val(param_sources, 'tenkan_period', 9))
                ichimoku_kijun_ia = self._get_param_val(param_sources, 'ichimoku_kijun', self._get_param_val(param_sources, 'kijun_period', 26))
                ichimoku_senkou_ia = self._get_param_val(param_sources, 'ichimoku_senkou', self._get_param_val(param_sources, 'senkou_period', 52))
                ichimoku_conf = indicator_naming_conv['ICHIMOKU'].get('output_columns', [])
                if isinstance(ichimoku_conf, list):
                    if isinstance(ichimoku_tenkan_ia, (int, float)) and isinstance(ichimoku_kijun_ia, (int, float)) and isinstance(ichimoku_senkou_ia, (int, float)):
                        for col_conf in ichimoku_conf:
                            if isinstance(col_conf, dict) and 'name_pattern' in col_conf:
                                params_for_ichi = {
                                    'tenkan_period': ichimoku_tenkan_ia,
                                    'kijun_period': ichimoku_kijun_ia,
                                    'senkou_period': ichimoku_senkou_ia
                                }
                                for name in TrendFollowingStrategy._format_indicator_name(col_conf['name_pattern'], **params_for_ichi):
                                    required.add(f"{name}_{analyze_tf_str}")
                    else:
                        logger.warning(f"{log_prefix} ICHIMOKU 指标参数无效，跳过 ICHIMOKU 列请求 ({analyze_tf_str})。")
                else:
                    logger.warning(f"{log_prefix} ICHIMOKU 命名规范 output_columns 类型不正确 (应为列表)。")
            # PIVOT_POINTS (通常基于日线, 无可变参数)
            if ia_params_global.get('calculate_pivot_points', False) and analyze_tf_str == 'D' and 'PIVOT_POINTS' in indicator_naming_conv:
                pivot_conf = indicator_naming_conv['PIVOT_POINTS'].get('output_columns', [])
                if isinstance(pivot_conf, list):
                    for col_conf in pivot_conf:
                        if isinstance(col_conf, dict) and 'name_pattern' in col_conf:
                            required.add(f"{col_conf['name_pattern']}_{analyze_tf_str}")
                else:
                    logger.warning(f"{log_prefix} PIVOT_POINTS 命名规范 output_columns 类型不正确 (应为列表)。")
            # ATR
            atr_period_fe = self._get_param_val(param_sources, 'atr_period', self._get_param_val(param_sources, 'period', 14))
            if fe_params_global.get('calculate_atr', False) and 'ATR' in indicator_naming_conv:
                atr_naming_conf = indicator_naming_conv['ATR']
                if isinstance(atr_naming_conf, dict) and 'output_columns' in atr_naming_conf and isinstance(atr_naming_conf['output_columns'], list):
                    atr_patterns = [c['name_pattern'] for c in atr_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(atr_period_fe, (int, float)) and atr_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(atr_patterns, period=atr_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # HV (Historical Volatility)
            hv_period_fe = self._get_param_val(param_sources, 'hv_period', self._get_param_val(param_sources, 'period', 20))
            if fe_params_global.get('calculate_hv', False) and 'HV' in indicator_naming_conv:
                hv_naming_conf = indicator_naming_conv['HV']
                if isinstance(hv_naming_conf, dict) and 'output_columns' in hv_naming_conf and isinstance(hv_naming_conf['output_columns'], list):
                    hv_patterns = [c['name_pattern'] for c in hv_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(hv_period_fe, (int, float)) and hv_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(hv_patterns, period=hv_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # KC (Keltner Channels)
            kc_ema_period_fe = self._get_param_val(param_sources, 'kc_ema_period', self._get_param_val(param_sources, 'ema_period', 20))
            kc_atr_period_fe = self._get_param_val(param_sources, 'kc_atr_period', self._get_param_val(param_sources, 'atr_period', 10))
            if fe_params_global.get('calculate_kc', False) and 'KC' in indicator_naming_conv:
                kc_naming_conf = indicator_naming_conv['KC']
                if isinstance(kc_naming_conf, dict) and 'output_columns' in kc_naming_conf and isinstance(kc_naming_conf['output_columns'], list):
                    kc_patterns = [c['name_pattern'] for c in kc_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(kc_ema_period_fe, (int, float)) and isinstance(kc_atr_period_fe, (int, float)):
                        for name in TrendFollowingStrategy._format_indicator_name(kc_patterns, ema_period=kc_ema_period_fe, atr_period=kc_atr_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
                    else:
                        logger.warning(f"{log_prefix} KC 指标参数无效: ema_period={kc_ema_period_fe}, atr_period={kc_atr_period_fe}. 跳过 KC 列请求 ({analyze_tf_str})。")
            # MOM
            mom_period_fe = self._get_param_val(param_sources, 'mom_period', self._get_param_val(param_sources, 'period', 10))
            if fe_params_global.get('calculate_mom', False) and 'MOM' in indicator_naming_conv:
                mom_naming_conf = indicator_naming_conv['MOM']
                if isinstance(mom_naming_conf, dict) and 'output_columns' in mom_naming_conf and isinstance(mom_naming_conf['output_columns'], list):
                    mom_patterns = [c['name_pattern'] for c in mom_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(mom_period_fe, (int, float)) and mom_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(mom_patterns, period=mom_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # WILLR
            willr_period_fe = self._get_param_val(param_sources, 'willr_period', self._get_param_val(param_sources, 'period', 14))
            if fe_params_global.get('calculate_willr', False) and 'WILLR' in indicator_naming_conv:
                willr_naming_conf = indicator_naming_conv['WILLR']
                if isinstance(willr_naming_conf, dict) and 'output_columns' in willr_naming_conf and isinstance(willr_naming_conf['output_columns'], list):
                    willr_patterns = [c['name_pattern'] for c in willr_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(willr_period_fe, (int, float)) and willr_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(willr_patterns, period=willr_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # VROC
            vroc_period_fe = self._get_param_val(param_sources, 'vroc_period', self._get_param_val(param_sources, 'period', 10))
            if fe_params_global.get('calculate_vroc', False) and 'VROC' in indicator_naming_conv:
                vroc_naming_conf = indicator_naming_conv['VROC']
                if isinstance(vroc_naming_conf, dict) and 'output_columns' in vroc_naming_conf and isinstance(vroc_naming_conf['output_columns'], list):
                    vroc_patterns = [c['name_pattern'] for c in vroc_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(vroc_period_fe, (int, float)) and vroc_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(vroc_patterns, period=vroc_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # AROC
            aroc_period_fe = self._get_param_val(param_sources, 'aroc_period', self._get_param_val(param_sources, 'period', 10))
            if fe_params_global.get('calculate_aroc', False) and 'AROC' in indicator_naming_conv:
                aroc_naming_conf = indicator_naming_conv['AROC']
                if isinstance(aroc_naming_conf, dict) and 'output_columns' in aroc_naming_conf and isinstance(aroc_naming_conf['output_columns'], list):
                    aroc_patterns = [c['name_pattern'] for c in aroc_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(aroc_period_fe, (int, float)) and aroc_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(aroc_patterns, period=aroc_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # EMA 列表, SMA 列表 (来自 feature_engineering_params)
            for ma_type_fe_upper in ['EMA', 'SMA']:
                ma_key_lower = ma_type_fe_upper.lower()
                ma_periods_list_fe = fe_params_global.get(f'{ma_key_lower}_periods', [])
                if not isinstance(ma_periods_list_fe, list): continue
                if ma_periods_list_fe and ma_type_fe_upper in indicator_naming_conv:
                    ma_naming_conf = indicator_naming_conv[ma_type_fe_upper]
                    if isinstance(ma_naming_conf, dict) and 'output_columns' in ma_naming_conf and isinstance(ma_naming_conf['output_columns'], list):
                        ma_patterns = [c['name_pattern'] for c in ma_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                        for period in ma_periods_list_fe:
                            if isinstance(period, (int, float)) and period > 0:
                                for name in TrendFollowingStrategy._format_indicator_name(ma_patterns, period=period):
                                    required.add(f"{name}_{analyze_tf_str}")
        # 5. 衍生特征 (Derivative Features) - 请求这些特征本身
        derivative_naming_conv = NAMING_CONFIG.get('derivative_feature_naming_conventions', {})
        if not isinstance(derivative_naming_conv, dict): derivative_naming_conv = {}
        fe_derivative_features_params = self.fe_params.get('derivative_features', {})
        if isinstance(fe_derivative_features_params, dict):
            for deriv_feature_key, deriv_feature_config in fe_derivative_features_params.items():
                if isinstance(deriv_feature_config, dict) and deriv_feature_config.get('enabled', False):
                    deriv_feature_key_upper = deriv_feature_key.upper()
                    if deriv_feature_key_upper not in derivative_naming_conv:
                        logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' 在命名规范中未定义，跳过。")
                        continue
                    deriv_naming_conf = derivative_naming_conv[deriv_feature_key_upper]
                    if not isinstance(deriv_naming_conf, dict) or 'output_column_pattern' not in deriv_naming_conf:
                        logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' 的命名配置无效，跳过。")
                        continue
                    pattern = deriv_naming_conf['output_column_pattern']
                    apply_tfs_deriv_raw = deriv_feature_config.get('apply_on_timeframes', fe_tf_list)
                    apply_tfs_deriv = [apply_tfs_deriv_raw] if isinstance(apply_tfs_deriv_raw, str) else apply_tfs_deriv_raw
                    if not isinstance(apply_tfs_deriv, list): apply_tfs_deriv = []
                    for deriv_tf_str in apply_tfs_deriv:
                        if deriv_tf_str not in timeframes: continue # 只请求在基础时间框架中定义的级别
                        # 根据不同的衍生特征类型，从配置中获取参数并格式化列名
                        if deriv_feature_key_upper in ['CLOSE_MA_RATIO', 'CLOSE_MA_NDIFF']:
                            ma_types = deriv_feature_config.get('ma_type', [])
                            periods = deriv_feature_config.get('periods', [])
                            if isinstance(ma_types, str): ma_types = [ma_types]
                            if isinstance(periods, (int, float)): periods = [periods]
                            if not isinstance(ma_types, list) or not isinstance(periods, list):
                                logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' 的 ma_type 或 periods 参数类型不正确。跳过。")
                                continue
                            if ma_types and periods:
                                for ma_type in ma_types:
                                    for period in periods:
                                        if isinstance(period, (int, float)) and period > 0 and isinstance(ma_type, str):
                                            for name in TrendFollowingStrategy._format_indicator_name(pattern, ma_type=ma_type.upper(), period=period):
                                                required.add(f"{name}_{deriv_tf_str}")
                                        else:
                                            logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' 参数无效: ma_type={ma_type}, period={period}. 跳过。")
                        elif deriv_feature_key_upper == 'INDICATOR_DIFF':
                            indicator_configs_deriv = deriv_feature_config.get('indicators', [])
                            if not isinstance(indicator_configs_deriv, list):
                                logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' 的 indicators 参数类型不正确。跳过。")
                                continue
                            if indicator_configs_deriv:
                                for indi_cfg in indicator_configs_deriv:
                                    if isinstance(indi_cfg, dict) and 'name' in indi_cfg and 'diff_periods' in indi_cfg:
                                        original_name_pattern = indi_cfg['name']
                                        diff_periods = indi_cfg['diff_periods']
                                        if isinstance(diff_periods, (int, float)): diff_periods = [diff_periods]
                                        if not isinstance(diff_periods, list):
                                            logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' ({original_name_pattern}) 的 diff_periods 参数类型不正确。跳过。")
                                            continue

                                        # 这里的 original_name_pattern 应该是基础指标的名称模式，例如 "RSI_{period}"
                                        # 需要根据这个模式和参数，找到 IndicatorService 生成的带参数的基础指标名
                                        # 例如，如果 original_name_pattern 是 "RSI_{period}"，参数是 {"period": 14}
                                        # 那么 IndicatorService 生成的基础指标名是 "RSI_14"
                                        # 衍生特征列名模式是 "{original_indicator_name_with_params}_diff_{diff_period}"
                                        # 最终列名是 "RSI_14_diff_1_30" (如果 deriv_tf_str 是 '30')

                                        # 尝试根据 original_name_pattern 和可能的参数，构建 IndicatorService 生成的基础指标名
                                        # 这是一个简化的查找逻辑，可能需要更精确的匹配
                                        # 假设 original_name_pattern 已经是带参数的基名，例如 "RSI_14"
                                        original_indicator_base_name = original_name_pattern # 假设配置中已经是 "RSI_14" 这样的格式

                                        if isinstance(original_indicator_base_name, str) and diff_periods:
                                            deriv_pattern_template = derivative_naming_conv.get('INDICATOR_DIFF', {}).get('output_column_pattern')
                                            if deriv_pattern_template:
                                                for diff_period in diff_periods:
                                                    if isinstance(diff_period, (int, float)) and diff_period > 0:
                                                        # 格式化列名，例如 RSI_14_diff_1
                                                        base_diff_name = TrendFollowingStrategy._format_indicator_name(deriv_pattern_template, original_indicator_name_with_params=original_indicator_base_name, diff_period=diff_period)
                                                        # 添加时间级别后缀，例如 RSI_14_diff_1_30
                                                        required.update([f"{n}_{deriv_tf_str}" for n in base_diff_name])
                                                    else:
                                                        logger.warning(f"{log_prefix} 衍生特征 INDICATOR_DIFF 的 diff_period 参数无效: {diff_period}. 跳过。")
                                            else:
                                                logger.warning(f"{log_prefix} 衍生特征 INDICATOR_DIFF 的命名模式未找到。")
                                        else:
                                            logger.warning(f"{log_prefix} 衍生特征 '{deriv_feature_key}' 的 name 或 diff_periods 参数无效。跳过。")
                        elif deriv_feature_key_upper == 'CLOSE_BB_POS':
                            boll_params_deriv = deriv_feature_config.get('boll_params', {})
                            if isinstance(boll_params_deriv, dict):
                                boll_period_deriv = boll_params_deriv.get('period')
                                boll_std_dev_deriv = boll_params_deriv.get('std_dev')
                                if isinstance(boll_period_deriv, (int, float)) and isinstance(boll_std_dev_deriv, (int, float)):
                                    for name in TrendFollowingStrategy._format_indicator_name(pattern, period=boll_period_deriv, std_dev=boll_std_dev_deriv):
                                        required.add(f"{name}_{deriv_tf_str}")
                                else:
                                    logger.warning(f"{log_prefix} 衍生特征 CLOSE_BB_POS 配置缺少有效 boll_params (period, std_dev)。")
                            else:
                                logger.warning(f"{log_prefix} 衍生特征 CLOSE_BB_POS 配置中的 boll_params 不是字典。")
                        elif deriv_feature_key_upper == 'CLOSE_KC_POS':
                            kc_params_deriv = deriv_feature_config.get('kc_params', {})
                            if isinstance(kc_params_deriv, dict):
                                kc_ema_period_deriv = kc_params_deriv.get('ema_period')
                                kc_atr_period_deriv = kc_params_deriv.get('atr_period')
                                if isinstance(kc_ema_period_deriv, (int, float)) and isinstance(kc_atr_period_deriv, (int, float)):
                                    for name in TrendFollowingStrategy._format_indicator_name(pattern, ema_period=kc_ema_period_deriv, atr_period=kc_atr_period_deriv):
                                        required.add(f"{name}_{deriv_tf_str}")
                                else:
                                    logger.warning(f"{log_prefix} 衍生特征 CLOSE_KC_POS 配置缺少有效 kc_params (ema_period, atr_period)。")
                            else:
                                logger.warning(f"{log_prefix} 衍生特征 CLOSE_KC_POS 配置中的 kc_params 不是字典。")
                        # 相对强度 (Relative Strength)
                        # RS 列名模式: RS_{benchmark_code}_{period}_{time_level}
                        if deriv_feature_key_upper == 'RELATIVE_STRENGTH':
                            rs_config = deriv_feature_config
                            rs_periods = rs_config.get('periods', [])
                            if isinstance(rs_periods, (int, float)): rs_periods = [rs_periods]
                            if not isinstance(rs_periods, list):
                                logger.warning(f"{log_prefix} 衍生特征 RELATIVE_STRENGTH 的 periods 参数类型不正确。跳过。")
                                continue
                            # 获取基准代码列表：固定主要指数 + 股票所属同花顺板块
                            main_index_codes = bs_params.get('main_index_codes', [])
                            # 需要异步获取同花顺板块代码 (这里是 get_required_columns，不应该执行异步IO)
                            # 策略在初始化时或 prepare_data 中获取 ths_codes 更合适
                            # 暂时假设 main_index_codes 已经足够，或者 ths_codes 会在 prepare_data 中处理
                            # 实际上，IndicatorService.prepare_strategy_dataframe 会获取 ths_codes
                            # 所以这里只需要请求 main_index_codes 相关的 RS 列名
                            all_benchmark_codes_for_rs = list(set(main_index_codes)) # 暂时只考虑 main_index_codes
                            if all_benchmark_codes_for_rs and rs_periods:
                                rs_pattern_template = derivative_naming_conv.get('RELATIVE_STRENGTH', {}).get('output_column_pattern')
                                if rs_pattern_template:
                                    for benchmark_code in all_benchmark_codes_for_rs:
                                            for period in rs_periods:
                                                if isinstance(period, (int, float)) and period > 0:
                                                    formatted_benchmark_code = benchmark_code.replace(".", "_").lower()
                                                    # 格式化列名，例如 RS_000001_sh_5
                                                    base_rs_name = TrendFollowingStrategy._format_indicator_name(rs_pattern_template, benchmark_code=formatted_benchmark_code, period=period)
                                                    # 添加时间级别后缀，例如 RS_000001_sh_5_30
                                                    required.update([f"{n}_{deriv_tf_str}" for n in base_rs_name])
                                                else:
                                                    logger.warning(f"{log_prefix} 衍生特征 RELATIVE_STRENGTH 的 period 参数无效: {period}. 跳过。")
                                else:
                                    logger.warning(f"{log_prefix} 衍生特征 RELATIVE_STRENGTH 的命名模式未找到。")
                            else:
                                logger.warning(f"{log_prefix} 衍生特征 RELATIVE_STRENGTH 未配置基准代码或周期。")
                        # 滞后特征 (Lagged Features)
                        # Lagged 列名模式: {original_column_name}_lag_{lag}
                        if deriv_feature_key_upper == 'LAGGED_FEATURES':
                            lag_config = deriv_feature_config
                            columns_to_lag = lag_config.get('columns_to_lag', [])
                            lags = lag_config.get('lags', [])
                            if isinstance(columns_to_lag, str): columns_to_lag = [columns_to_lag]
                            if isinstance(lags, (int, float)): lags = [lags]
                            if not isinstance(columns_to_lag, list) or not isinstance(lags, list):
                                logger.warning(f"{log_prefix} 衍生特征 LAGGED_FEATURES 的 columns_to_lag 或 lags 参数类型不正确。跳过。")
                                continue
                            if columns_to_lag and lags:
                                lag_pattern_template = derivative_naming_conv.get('LAGGED_FEATURES', {}).get('output_column_pattern')
                                if lag_pattern_template:
                                    for original_col_base_name in columns_to_lag:
                                            # 原始列名需要带时间级别后缀
                                            original_col_with_tf = f"{original_col_base_name}_{deriv_tf_str}"
                                            for lag in lags:
                                                if isinstance(lag, (int, float)) and lag > 0:
                                                    lagged_col_name = TrendFollowingStrategy._format_indicator_name(lag_pattern_template, original_column_name=original_col_with_tf, lag=lag)
                                                    required.update(lagged_col_name)
                                                else:
                                                    logger.warning(f"{log_prefix} 衍生特征 LAGGED_FEATURES 的 lag 参数无效: {lag}. 跳过。")
                                else:
                                    logger.warning(f"{log_prefix} 衍生特征 LAGGED_FEATURES 的命名模式未找到。")
                            else:
                                logger.warning(f"{log_prefix} 衍生特征 LAGGED_FEATURES 未配置 columns_to_lag 或 lags。")
                        # 滚动统计特征 (Rolling Features)
                        # Rolling 列名模式: {original_column_name}_rolling_{stat}_{window}
                        if deriv_feature_key_upper == 'ROLLING_FEATURES':
                            roll_config = deriv_feature_config
                            columns_to_roll = roll_config.get('columns_to_roll', [])
                            windows = roll_config.get('windows', [])
                            stats = roll_config.get('stats', [])
                            if isinstance(columns_to_roll, str): columns_to_roll = [columns_to_roll]
                            if isinstance(windows, (int, float)): windows = [windows]
                            if isinstance(stats, str): stats = [stats]
                            if not isinstance(columns_to_roll, list) or not isinstance(windows, list) or not isinstance(stats, list):
                                logger.warning(f"{log_prefix} 衍生特征 ROLLING_FEATURES 的 columns_to_roll, windows 或 stats 参数类型不正确。跳过。")
                                continue
                            if columns_to_roll and windows and stats:
                                roll_pattern_template = derivative_naming_conv.get('ROLLING_FEATURES', {}).get('output_column_pattern')
                                if roll_pattern_template:
                                    for original_col_base_name in columns_to_roll:
                                            original_col_with_tf = f"{original_col_base_name}_{deriv_tf_str}"
                                            for window in windows:
                                                if isinstance(window, (int, float)) and window > 0:
                                                    for stat in stats:
                                                        if isinstance(stat, str) and stat:
                                                                rolling_col_name = TrendFollowingStrategy._format_indicator_name(roll_pattern_template, original_column_name=original_col_with_tf, stat=stat, window=window)
                                                                required.update(rolling_col_name)
                                                        else:
                                                                logger.warning(f"{log_prefix} 衍生特征 ROLLING_FEATURES 的 stat 参数无效: {stat}. 跳过。")
                                                else:
                                                    logger.warning(f"{log_prefix} 衍生特征 ROLLING_FEATURES 的 window 参数无效: {window}. 跳过。")
                                else:
                                    logger.warning(f"{log_prefix} 衍生特征 ROLLING_FEATURES 的命名模式未找到。")
                            else:
                                logger.warning(f"{log_prefix} 衍生特征 ROLLING_FEATURES 未配置 columns_to_roll, windows 或 stats。")
        # 6. 添加策略内部计算所需的其他基础指标列 (如果它们没有在评分/分析/特征工程中包含)
        # 例如，计算 EMA 排列需要多个周期的 EMA。这些周期的列表可以在参数中配置。
        ta_params_global = self.params.get('trend_analysis', {})
        if not isinstance(ta_params_global, dict): ta_params_global = {}
        ema_periods_align = ta_params_global.get('ema_periods', [])
        if not ema_periods_align or not isinstance(ema_periods_align, list):
            ema_periods_list_fe = self.fe_params.get('ema_periods', [])
            if isinstance(ema_periods_list_fe, list):
                ema_periods_align = ema_periods_list_fe
            else:
                logger.warning(f"{log_prefix} 'trend_analysis.ema_periods' 和 'feature_engineering_params.ema_periods' 参数配置无效或缺失，无法为 EMA 排列请求 EMA 列。")
                ema_periods_align = []
        if ema_periods_align and 'EMA' in indicator_naming_conv:
            ema_naming_conf = indicator_naming_conv['EMA']
            if isinstance(ema_naming_conf, dict) and 'output_columns' in ema_naming_conf and isinstance(ema_naming_conf['output_columns'], list):
                ema_patterns = [c['name_pattern'] for c in ema_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                for period in ema_periods_align:
                    if isinstance(period, (int, float)) and period > 0:
                        for tf_str in timeframes: # EMA排列信号通常在所有timeframes上计算，所以为所有timeframes请求
                            for name in TrendFollowingStrategy._format_indicator_name(ema_patterns, period=period):
                                required.add(f"{name}_{tf_str}")
        # 7. 外部特征列 (指数、板块、筹码、资金流向)
        # 这些列名由 enrich_features 生成，并带有前缀 (index_, ths_, cyq_, ff_)，不带时间级别后缀
        # 需要根据命名规范和参数（如 main_index_codes）构建这些列名
        external_features_config = self.fe_params.get('external_features', {})
        if isinstance(external_features_config, dict) and external_features_config.get('enabled', False):
            # 指数和板块数据
            index_sector_config = external_features_config.get('index_sector_data', {})
            if isinstance(index_sector_config, dict) and index_sector_config.get('enabled', False):
                main_index_codes = bs_params.get('main_index_codes', [])
                # 策略需要知道它所属的同花顺板块代码，以便请求这些板块的数据
                # IndicatorService.prepare_strategy_dataframe 会获取 ths_codes
                # 这里只需要请求 main_index_codes 相关的列名模式
                all_benchmark_codes = list(set(main_index_codes)) # 暂时只考虑 main_index_codes
                if all_benchmark_codes:
                    # 普通指数命名规范
                    index_naming_conf = external_naming_conv.get('INDEX_DAILY', {})
                    if isinstance(index_naming_conf, dict) and 'output_columns' in index_naming_conf and isinstance(index_naming_conf['output_columns'], list):
                            index_patterns = [c['name_pattern'] for c in index_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                            for benchmark_code in all_benchmark_codes:
                                if '.' in benchmark_code: # 假设包含 '.' 是普通指数代码
                                    formatted_benchmark_code = benchmark_code.replace(".", "_").lower()
                                    for pattern in index_patterns:
                                        # 格式化列名，例如 index_000001_sh_close
                                        formatted_names = TrendFollowingStrategy._format_indicator_name(pattern, index_code=formatted_benchmark_code)
                                        required.update(formatted_names)
                    else:
                            logger.warning(f"{log_prefix} 普通指数命名规范无效。")
                    # 同花顺板块命名规范
                    ths_naming_conf = external_naming_conv.get('THS_INDEX_DAILY', {})
                    if isinstance(ths_naming_conf, dict) and 'output_columns' in ths_naming_conf and isinstance(ths_naming_conf['output_columns'], list):
                            ths_patterns = [c['name_pattern'] for c in ths_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                            # 这里无法预知股票所属的所有 ths_codes，只能请求所有可能的 ths_codes 的列名
                            # 或者依赖 IndicatorService 返回所有已获取的 ths_codes 数据
                            # 考虑到 IndicatorService 已经获取并合并了，这里只需要知道命名模式
                            # 策略在 prepare_data_for_transformer 中选择特征时，会根据实际列名来选择
                            # 所以这里不需要列出所有 ths_codes 的列名，只需要知道模式
                            # 暂时跳过 ths_codes 的具体列名请求，依赖 prepare_data_for_transformer 根据实际列名选择
                            pass # 依赖 prepare_data_for_transformer 根据实际列名选择
            # 筹码分布数据
            cyq_config = external_features_config.get('cyq_data', {})
            if isinstance(cyq_config, dict) and cyq_config.get('enabled', False):
                cyq_naming_conf = external_naming_conv.get('STOCK_CYQ_PERF', {})
                if isinstance(cyq_naming_conf, dict) and 'output_columns' in cyq_naming_conf and isinstance(cyq_naming_conf['output_columns'], list):
                    cyq_patterns = [c['name_pattern'] for c in cyq_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    required.update(cyq_patterns) # 筹码列名不带参数和后缀
            # 资金流向数据
            fund_flow_config = external_features_config.get('fund_flow_data', {})
            if isinstance(fund_flow_config, dict) and fund_flow_config.get('enabled', False):
                # 股票资金流向 (FundFlowDaily, FundFlowDailyTHS, FundFlowDailyDC)
                for ff_type in ['FUND_FLOW_DAILY', 'FUND_FLOW_DAILY_THS', 'FUND_FLOW_DAILY_DC']:
                    ff_naming_conf = external_naming_conv.get(ff_type, {})
                    if isinstance(ff_naming_conf, dict) and 'output_columns' in ff_naming_conf and isinstance(ff_naming_conf['output_columns'], list):
                            ff_patterns = [c['name_pattern'] for c in ff_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                            required.update(ff_patterns) # 资金流向列名不带参数和后缀
                # 板块/行业资金流向 (FundFlowCntTHS, FundFlowIndustryTHS)
                # 同样，这里无法预知股票所属的所有 ths_codes，依赖 prepare_data_for_transformer 根据实际列名选择
                pass # 依赖 prepare_data_for_transformer 根据实际列名选择
        logger.info(f"[{log_prefix}] get_required_columns 将请求 {len(required)} 列。")
        # logger.debug(f"[{log_prefix}] 请求的列包括: {sorted(list(required))}") # 可能非常冗长
        return list(sorted(list(required))) # 返回排序后的列表

    def _calculate_rule_based_signal(self, data: pd.DataFrame, stock_code: str, indicator_configs: List[Dict]) -> Tuple[pd.Series, Dict]:
        """
        计算基于规则的信号。
        使用JSON配置获取指标列名。
        """
        if data is None or data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 输入数据为空，无法生成规则信号。")
            return pd.Series(dtype=float, name='final_rule_signal'), {}
        if not self.params:
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算：策略参数 (self.params) 为空。")
            return pd.Series(50.0, index=data.index, name='final_rule_signal'), {}

        def _get_param_val(sources: List[Dict], key: str, default: Any = None) -> Any:
            """从参数源列表中按顺序查找键的值，返回第一个找到的值或默认值。"""
            for source_dict in sources:
                if isinstance(source_dict, dict) and key in source_dict:
                    return source_dict[key]
            return default

        # 准备参数源列表，优先级从高到低 (策略特定参数通常优先级高)
        bs_params = self.params.get('base_scoring', {})
        vc_params = self.params.get('volume_confirmation', {})
        ia_params = self.params.get('indicator_analysis_params', {})
        dd_params = self.params.get('divergence_detection', {})
        tf_params = self.params.get('trend_following_params', {}) # 策略特定参数

        param_sources = [
            tf_params, # trend_following_params (策略特定参数)
            vc_params, # volume_confirmation
            ia_params, # indicator_analysis_params
            self.params.get('feature_engineering_params', {}), # feature_engineering_params
            bs_params # base_scoring
        ]
        focus_tf = self.focus_timeframe

        # 获取命名规范字典
        ohlcv_naming_conv = NAMING_CONFIG.get('ohlcv_naming_convention', {})
        indicator_naming_conv = NAMING_CONFIG.get('indicator_naming_conventions', {})
        derivative_naming_conv = NAMING_CONFIG.get('derivative_feature_naming_conventions', {})
        strategy_internal_naming_conv = NAMING_CONFIG.get('strategy_internal_columns', {}) # 新增：策略内部列命名规范

        # 增加类型检查
        if not isinstance(ohlcv_naming_conv, dict): ohlcv_naming_conv = {}
        if not isinstance(indicator_naming_conv, dict): indicator_naming_conv = {}
        if not isinstance(derivative_naming_conv, dict): derivative_naming_conv = {}
        if not isinstance(strategy_internal_naming_conv, dict): strategy_internal_naming_conv = {}

        # 检查关键OHLCV列
        ohlcv_base_names_list = ohlcv_naming_conv.get('output_columns', [])
        # 增加类型检查
        if not isinstance(ohlcv_base_names_list, list):
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算：OHLCV 命名规范 output_columns 类型不正确 (应为列表)。")
            ohlcv_base_names_list = []
        ohlcv_base_names = [c['name_pattern'] for c in ohlcv_base_names_list if isinstance(c, dict) and 'name_pattern' in c]
        # 检查 focus_tf 的 close 和 volume 列
        close_col_focus = f"close_{focus_tf}" # 使用带后缀的列名
        volume_col_focus = f"volume_{focus_tf}" # 使用带后缀的列名
        critical_ohlcv_cols = [f"{base_col}_{focus_tf}" for base_col in ohlcv_base_names if base_col in ['close', 'volume']]

        missing_critical_cols = [col for col in critical_ohlcv_cols if col not in data.columns or data[col].isnull().all()]
        if missing_critical_cols:
            logger.error(f"[{self.strategy_name}][{stock_code}] 规则信号计算缺少关键OHLCV列或数据全为 NaN: {missing_critical_cols}。")
            return pd.Series(50.0, index=data.index), {}

        # 动态调整波动率参数
        self._adjust_volatility_parameters(data)

        # 计算基础指标得分 (假定 strategy_utils 包含此函数)
        logger.debug(f"[{self.strategy_name}][{stock_code}] 计算基础指标评分...")
        indicator_scores_df = strategy_utils.calculate_all_indicator_scores(data, bs_params, indicator_configs, NAMING_CONFIG)

        if indicator_scores_df is None or indicator_scores_df.empty or not any(col.startswith('SCORE_') for col in (indicator_scores_df.columns if isinstance(indicator_scores_df, pd.DataFrame) else [])):
            logger.warning(f"[{self.strategy_name}][{stock_code}] 未计算或未找到任何指标评分列。基础评分将为中性50。")
            indicator_scores_df = pd.DataFrame(50.0, index=data.index, columns=['SCORE_DEFAULT_50'])

        # 应用时间框架权重并合并基础评分
        logger.debug(f"[{self.strategy_name}][{stock_code}] 应用时间框架权重并合并基础评分...")
        current_weights: Dict[str, float]
        timeframes_from_config = bs_params.get('timeframes', [])
        if not isinstance(timeframes_from_config, list) or not timeframes_from_config:
            logger.error(f"[{self.strategy_name}][{stock_code}] 'base_scoring.timeframes' 为空或无效，无法计算基础评分。")
            return pd.Series(50.0, index=data.index, name='final_rule_signal'), {}

        if self.timeframe_weights is not None and isinstance(self.timeframe_weights, dict):
            current_weights = self.timeframe_weights.copy()
            defined_tfs_set = set(timeframes_from_config)
            for tf_w in list(current_weights.keys()):
                if tf_w not in defined_tfs_set:
                    del current_weights[tf_w]
            for tf_d in defined_tfs_set:
                if tf_d not in current_weights:
                    current_weights[tf_d] = 0.0
        else:
            focus_weight_val = tf_params.get('focus_weight', 0.45)
            if not isinstance(focus_weight_val, (int, float)) or not (0 <= focus_weight_val <= 1):
                logger.warning(f"[{self.strategy_name}][{stock_code}] 'focus_weight' 参数无效 ({focus_weight_val})，使用默认值 0.45。")
                focus_weight_val = 0.45

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
            if self.focus_timeframe in timeframes_from_config:
                current_weights[self.focus_timeframe] = focus_weight_val
            elif timeframes_from_config:
                logger.warning(f"[{self.strategy_name}][{stock_code}] focus_timeframe '{self.focus_timeframe}' 不在配置的时间框架列表中。将权重平均分配。")
                avg_weight = 1.0 / len(timeframes_from_config) if timeframes_from_config else 0.0
                current_weights = {tf: avg_weight for tf in timeframes_from_config}

        self._normalize_weights(current_weights)

        base_score_raw = pd.Series(0.0, index=data.index)
        total_effective_weight = 0.0

        for tf_s in timeframes_from_config:
            tf_weight = current_weights.get(tf_s, 0)
            if tf_weight == 0: continue

            tf_score_cols = [col for col in indicator_scores_df.columns if col.endswith(f'_{tf_s}') and col.startswith('SCORE_')]

            if tf_score_cols:
                valid_tf_score_cols = [col for col in tf_score_cols if not indicator_scores_df[col].isnull().all()]
                if valid_tf_score_cols:
                    tf_average_score = indicator_scores_df[valid_tf_score_cols].mean(axis=1).fillna(50.0)
                    base_score_raw = base_score_raw.add(tf_average_score * tf_weight, fill_value=0.0)
                    total_effective_weight += tf_weight
                else:
                    logger.debug(f"[{self.strategy_name}][{stock_code}] 时间框架 '{tf_s}' (权重 {tf_weight:.2f}) 的所有指标评分列全为 NaN。将使用中性分50参与加权。")
                    base_score_raw = base_score_raw.add(pd.Series(50.0, index=data.index) * tf_weight, fill_value=0.0)
                    total_effective_weight += tf_weight
            else:
                logger.debug(f"[{self.strategy_name}][{stock_code}] 时间框架 '{tf_s}' (权重 {tf_weight:.2f}) 没有找到任何指标评分列。将使用中性分50参与加权。")
                base_score_raw = base_score_raw.add(pd.Series(50.0, index=data.index) * tf_weight, fill_value=0.0)
                total_effective_weight += tf_weight

        if total_effective_weight == 0 and timeframes_from_config:
             logger.warning(f"[{self.strategy_name}][{stock_code}] 所有时间框架的有效权重总和为零，基础评分将为中性50。")
             base_score_raw = pd.Series(50.0, index=data.index)
        elif not np.isclose(total_effective_weight, sum(current_weights.values())) and sum(current_weights.values()) > 0:
            pass

        base_score_raw = base_score_raw.clip(0, 100).fillna(50.0)

        # 量能调整基础评分 (假定 strategy_utils 包含此函数)
        logger.debug(f"[{self.strategy_name}][{stock_code}] 执行量能调整/分析模块...")
        vc_params_adjusted = vc_params.copy()
        if not isinstance(vc_params_adjusted, dict):
            vc_params_adjusted = {}
        vc_params_adjusted['boost_factor'] = self.volume_boost_factor
        vc_params_adjusted['penalty_factor'] = self.volume_penalty_factor
        vc_params_adjusted['volume_spike_threshold'] = self.volume_spike_threshold

        vol_ma_period_vc = _get_param_val(param_sources, 'volume_ma_period', 20)
        obv_ma_period_vc = _get_param_val(param_sources, 'obv_ma_period', 10)

        vc_tf_list_raw = vc_params.get('timeframes', [self.focus_timeframe])
        vc_tf_list = [vc_tf_list_raw] if isinstance(vc_tf_list_raw, str) else vc_tf_list_raw
        if not isinstance(vc_tf_list, list) or not vc_tf_list:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 量能确认时间框架配置无效或为空，量能调整将跳过。")
            volume_adjusted_results_df = pd.DataFrame(index=data.index)
            volume_adjusted_results_df['ADJUSTED_SCORE'] = base_score_raw
            internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
            vol_spike_pattern = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'].startswith("VOL_SPIKE_SIGNAL")), "VOL_SPIKE_SIGNAL_{timeframe}")
            for tf in ([self.focus_timeframe] if not vc_tf_list else vc_tf_list):
                vol_spike_col_name = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=tf)[0]
                volume_adjusted_results_df[vol_spike_col_name] = 0.0
        else:
            try:
                volume_adjusted_results_df = strategy_utils.adjust_score_with_volume(
                    preliminary_score=base_score_raw,
                    data=data,
                    vc_params=vc_params_adjusted,
                    vc_tf_list=vc_tf_list,
                    vol_ma_period=vol_ma_period_vc,
                    obv_ma_period=obv_ma_period_vc,
                    naming_config=NAMING_CONFIG
                )
                if 'ADJUSTED_SCORE' not in volume_adjusted_results_df.columns:
                    logger.error(f"[{self.strategy_name}][{stock_code}] 量能调整模块未能生成 'ADJUSTED_SCORE' 列。将使用原始基础评分。")
                    volume_adjusted_results_df['ADJUSTED_SCORE'] = base_score_raw
                logger.debug(f"[{self.strategy_name}][{stock_code}] 量能调整/分析完成。")
            except Exception as e:
                logger.error(f"[{self.strategy_name}][{stock_code}] 执行量能调整/分析模块出错: {e}", exc_info=True)
                volume_adjusted_results_df = pd.DataFrame(index=data.index)
                volume_adjusted_results_df['ADJUSTED_SCORE'] = base_score_raw
                internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
                vol_spike_pattern = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'].startswith("VOL_SPIKE_SIGNAL")), "VOL_SPIKE_SIGNAL_{timeframe}")
                for tf in vc_tf_list:
                    vol_spike_col_name = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=tf)[0]
                    volume_adjusted_results_df[vol_spike_col_name] = 0.0
        base_score_volume_adjusted = volume_adjusted_results_df['ADJUSTED_SCORE']

        # 执行趋势分析 (假定 _perform_trend_analysis 已实现)
        trend_analysis_df = self._perform_trend_analysis(data, base_score_volume_adjusted)
        # 执行量价背离检测 (假定 strategy_utils 包含此函数)
        logger.debug(f"[{self.strategy_name}][{stock_code}] 执行量价背离检测...")
        divergence_signals_df = pd.DataFrame(index=data.index)
        if isinstance(dd_params, dict) and dd_params.get('enabled', True):
            div_indicator_scoring_info = {}
            glob_base_scoring_params = INDICATOR_PARAMETERS.get('base_scoring', {})
            glob_ia_params = INDICATOR_PARAMETERS.get('indicator_analysis_params', {})
            glob_indicator_naming_conventions = NAMING_CONFIG.get('indicator_naming_conventions', {})
            configured_div_indicators = dd_params.get('indicators', {})
            for indi_key_from_dd, is_enabled_in_dd in configured_div_indicators.items():
                if not is_enabled_in_dd:
                    continue
                indi_key_lower = indi_key_from_dd.lower()
                current_defaults = {}
                current_prefixes = []
                if indi_key_lower == 'rsi':
                    current_defaults['period'] = glob_base_scoring_params.get('rsi_period', 14)
                    rsi_conf = glob_indicator_naming_conventions.get('RSI', {})
                    if rsi_conf.get('output_columns'):
                        pattern_base = rsi_conf['output_columns'][0].get('name_pattern', '').split('_')[0]
                        if pattern_base: current_prefixes.append(pattern_base)
                elif indi_key_lower == 'macd_hist':
                    current_defaults['period_fast'] = glob_base_scoring_params.get('macd_fast', 12)
                    current_defaults['period_slow'] = glob_base_scoring_params.get('macd_slow', 26)
                    current_defaults['signal_period'] = glob_base_scoring_params.get('macd_signal', 9)
                    macd_conf = glob_indicator_naming_conventions.get('MACD', {})
                    if macd_conf.get('output_columns'):
                        for col_spec in macd_conf['output_columns']:
                            if col_spec.get('name_pattern', '').startswith('MACDh'):
                                current_prefixes.append('MACDh')
                                break
                elif indi_key_lower == 'mfi':
                    current_defaults['period'] = glob_base_scoring_params.get('mfi_period', 14)
                    mfi_conf = glob_indicator_naming_conventions.get('MFI', {})
                    if mfi_conf.get('output_columns'):
                        pattern_base = mfi_conf['output_columns'][0].get('name_pattern', '').split('_')[0]
                        if pattern_base: current_prefixes.append(pattern_base)
                elif indi_key_lower == 'obv':
                    current_defaults = {}
                    obv_conf = glob_indicator_naming_conventions.get('OBV', {})
                    if obv_conf.get('output_columns'):
                        pattern_base = obv_conf['output_columns'][0].get('name_pattern', '').split('_')[0]
                        if pattern_base: current_prefixes.append(pattern_base)
                elif indi_key_lower in ['stoch_k', 'stoch_d']:
                    current_defaults['k_period'] = glob_ia_params.get('stoch_k', 14)
                    current_defaults['d_period'] = glob_ia_params.get('stoch_d', 3)
                    current_defaults['smooth_k_period'] = glob_ia_params.get('stoch_smooth_k', 3)
                    stoch_conf = glob_indicator_naming_conventions.get('STOCH', {})
                    if stoch_conf.get('output_columns'):
                        target_pattern_start = 'STOCHK' if indi_key_lower == 'stoch_k' else 'STOCHD'
                        for col_spec in stoch_conf['output_columns']:
                            pattern = col_spec.get('name_pattern', '')
                            if pattern.upper().startswith(target_pattern_start):
                                current_prefixes.append(pattern.split('_')[0])
                                break
                if not current_prefixes:
                    if indi_key_lower == 'macd_hist': current_prefixes.append('MACDH')
                    elif indi_key_lower == 'stoch_k': current_prefixes.append('STOCHK')
                    elif indi_key_lower == 'stoch_d': current_prefixes.append('STOCHD')
                    else: current_prefixes.append(indi_key_from_dd.upper())
                div_indicator_scoring_info[indi_key_lower] = {
                    'defaults': current_defaults,
                    'prefixes': list(set(p for p in current_prefixes if p))
                }
            try:
                divergence_signals_df = strategy_utils.detect_divergence(data=data, dd_params=dd_params, naming_config=NAMING_CONFIG, indicator_scoring_info=div_indicator_scoring_info)
                if not divergence_signals_df.empty:
                    internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
                    has_bearish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BEARISH_DIVERGENCE"), "HAS_BEARISH_DIVERGENCE")
                    has_bullish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BULLISH_DIVERGENCE"), "HAS_BULLISH_DIVERGENCE")

                    if has_bearish_div_col not in divergence_signals_df.columns:
                        divergence_signals_df[has_bearish_div_col] = False
                    if has_bullish_div_col not in divergence_signals_df.columns:
                        divergence_signals_df[has_bullish_div_col] = False

                    logger.debug(f"[{self.strategy_name}][{stock_code}] 背离检测完成，最新信号: {divergence_signals_df.iloc[-1].to_dict() if not divergence_signals_df.empty else '无'}")
                else:
                    internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
                    has_bearish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BEARISH_DIVERGENCE"), "HAS_BEARISH_DIVERGENCE")
                    has_bullish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BULLISH_DIVERGENCE"), "HAS_BULLISH_DIVERGENCE")
                    divergence_signals_df[has_bearish_div_col] = False
                    divergence_signals_df[has_bullish_div_col] = False

            except Exception as e:
                logger.error(f"[{self.strategy_name}][{stock_code}] 执行量价背离检测时出错: {e}", exc_info=True)
                internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
                has_bearish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BEARISH_DIVERGENCE"), "HAS_BEARISH_DIVERGENCE")
                has_bullish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BULLISH_DIVERGENCE"), "HAS_BULLISH_DIVERGENCE")
                divergence_signals_df[has_bearish_div_col] = False
                divergence_signals_df[has_bullish_div_col] = False

        else:
            logger.info(f"[{self.strategy_name}][{stock_code}] 量价背离检测未启用。")
            internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
            has_bearish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BEARISH_DIVERGENCE"), "HAS_BEARISH_DIVERGENCE")
            has_bullish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BULLISH_DIVERGENCE"), "HAS_BULLISH_DIVERGENCE")
            divergence_signals_df[has_bearish_div_col] = False
            divergence_signals_df[has_bullish_div_col] = False

        # 综合规则信号 (假定 strategy_utils 包含 combine_rule_signals 函数)
        logger.debug(f"[{self.strategy_name}][{stock_code}] 计算综合规则信号...")
        weights = self.rule_signal_weights

        # 提取各种信号列，如果缺失则用 0 填充
        base_score_norm = (base_score_volume_adjusted.fillna(50.0) - 50) / 50 # 归一化到 [-1, 1]
        alignment_norm = trend_analysis_df.get('alignment_signal', pd.Series(0.0, index=data.index)).fillna(0.0) / 3.0 # 归一化到 [-1, 1] (假设排列信号范围是-3到3)
        long_context_norm = trend_analysis_df.get('long_term_context', pd.Series(0.0, index=data.index)).fillna(0.0) # 假定范围是-1, 0, 1
        score_momentum_series = trend_analysis_df.get('score_momentum', pd.Series(0.0, index=data.index)).fillna(0.0)
        momentum_norm = np.sign(score_momentum_series).fillna(0.0) # 归一化到 -1, 0, 1
        ema_cross_norm = trend_analysis_df.get('ema_cross_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # 假定范围是-1, 0, 1
        boll_breakout_norm = trend_analysis_df.get('boll_breakout_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # 假定范围是-1, 0, 1
        adx_strength_norm = trend_analysis_df.get('adx_strength_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # 假定范围是-1, -0.5, 0, 0.5, 1
        stoch_signal_norm = trend_analysis_df.get('stoch_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # 新增：STOCH信号
        vwap_dev_norm = trend_analysis_df.get('vwap_deviation_signal', pd.Series(0.0, index=data.index)).fillna(0.0) # 新增：VWAP偏离信号

        # 从 volume_adjusted_results_df 获取量价异动信号
        vc_tf_list_vol_spike = vc_params.get('timeframes', [self.focus_timeframe]) # 获取量能分析时间框架列表
        if isinstance(vc_tf_list_vol_spike, str): vc_tf_list_vol_spike = [vc_tf_list_vol_spike]
        if not isinstance(vc_tf_list_vol_spike, list) or not vc_tf_list_vol_spike: vc_tf_list_vol_spike = [self.focus_timeframe]

        # 从JSON配置获取 VOL_SPIKE_SIGNAL 列名模式
        vol_spike_pattern = "VOL_SPIKE_SIGNAL_{timeframe}" # 默认模式
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        for item in internal_cols_conf:
            if isinstance(item, dict) and item.get('name_pattern', '').startswith("VOL_SPIKE_SIGNAL"):
                vol_spike_pattern = item['name_pattern']
                break

        vol_spike_signal_col = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=vc_tf_list_vol_spike[0])[0] if vc_tf_list_vol_spike else None

        if vol_spike_signal_col and vol_spike_signal_col in volume_adjusted_results_df.columns:
            volume_spike_norm = volume_adjusted_results_df.get(vol_spike_signal_col, pd.Series(0.0, index=data.index)).fillna(0.0) # 假定范围是-1, 0, 1
        else:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 量价异动信号列 '{vol_spike_signal_col}' (基于时间框架 {vc_tf_list_vol_spike[0] if vc_tf_list_vol_spike else '无'}) 不存在或量能时间框架未配置。规则组合中量价异动贡献为 0。")
            volume_spike_norm = pd.Series(0.0, index=data.index)

        total_weighted_contribution = pd.Series(0.0, index=data.index)
        total_weighted_contribution += base_score_norm * weights.get('base_score', 0)
        total_weighted_contribution += alignment_norm * weights.get('alignment', 0)
        total_weighted_contribution += long_context_norm * weights.get('long_context', 0)
        total_weighted_contribution += momentum_norm * weights.get('momentum', 0)
        total_weighted_contribution += ema_cross_norm * weights.get('ema_cross', 0)
        total_weighted_contribution += boll_breakout_norm * weights.get('boll_breakout', 0)
        total_weighted_contribution += adx_strength_norm * weights.get('adx_strength', 0)
        total_weighted_contribution += stoch_signal_norm * weights.get('stoch_signal', 0)
        total_weighted_contribution += vwap_dev_norm * weights.get('vwap_deviation', 0)
        total_weighted_contribution += volume_spike_norm * weights.get('volume_spike', 0)

        # 将加权贡献转换回 0-100 的分数范围
        base_rule_signal_before_adjust = 50.0 + total_weighted_contribution * 50.0
        base_rule_signal_before_adjust = base_rule_signal_before_adjust.clip(0, 100)

        # 应用 ADX 增强 (假定 _apply_adx_boost 已实现)
        # 修改行: 更改变量名以便后续打印
        final_rule_signal_after_adx = self._apply_adx_boost(
            base_rule_signal_before_adjust,
            adx_strength_norm,
            (base_rule_signal_before_adjust.fillna(50.0) - 50.0) / 50.0
        )

        # 应用背离惩罚 (假定 _apply_divergence_penalty 已实现)
        # 修改行: 更改变量名以便后续打印
        final_rule_signal_after_div = self._apply_divergence_penalty(final_rule_signal_after_adx, divergence_signals_df, dd_params)

        # 应用趋势确认过滤 (假定 _apply_trend_confirmation 已实现)
        # 修改行: 更改变量名以便后续打印
        final_rule_signal = self._apply_trend_confirmation(final_rule_signal_after_div)

        # 最终剪切和四舍五入
        final_rule_signal = final_rule_signal.clip(0, 100).round(2)

        # 准备返回的中间结果字典
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
        使用带有时间级别后缀的列名访问数据。
        所有指标参数从 self.params (代表 indicator_parameters.json) 读取。
        """
        analysis_df = pd.DataFrame(index=data.index) # 使用原始数据的索引，包含所有时间点
        def _get_param_val(sources: List[Dict], key: str, default: Any = None) -> Any:
            """从参数源列表中按顺序查找键的值，返回第一个找到的值或默认值。"""
            for source_dict in sources:
                if isinstance(source_dict, dict) and key in source_dict:
                    return source_dict[key]
            return default
        # 从 self.params (代表 indicator_parameters.json) 获取各主要配置块
        # 使用更明确的变量名从 self.params 获取配置，确保来源是 JSON 文件
        trend_analysis_config = self.params.get('trend_analysis', {})
        if not isinstance(trend_analysis_config, dict):
             logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 'trend_analysis' 参数类型不正确或缺失。")
             trend_analysis_config = {}
        base_scoring_config = self.params.get('base_scoring', {})
        if not isinstance(base_scoring_config, dict):
             logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 'base_scoring' 参数类型不正确或缺失。")
             base_scoring_config = {}
        indicator_analysis_config = self.params.get('indicator_analysis_params', {})
        if not isinstance(indicator_analysis_config, dict):
             logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 'indicator_analysis_params' 参数类型不正确或缺失。")
             indicator_analysis_config = {}
        # feature_engineering_params 在全局配置中 (来自 self.fe_params)
        feature_engineering_config = self.fe_params # 假设 self.fe_params 已被正确设置为 self.params['feature_engineering_params']
        if not isinstance(feature_engineering_config, dict):
             logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 'feature_engineering_params' (self.fe_params) 参数类型不正确或缺失。")
             feature_engineering_config = {}
        trend_following_config = self.params.get('trend_following_params', {})
        if not isinstance(trend_following_config, dict):
             logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 'trend_following_params' 参数类型不正确或缺失。")
             trend_following_config = {}
        # volume_confirmation 参数，用于 param_sources
        volume_confirmation_config = self.params.get('volume_confirmation', {})
        if not isinstance(volume_confirmation_config, dict):
            logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 'volume_confirmation' 参数类型不正确或缺失。")
            volume_confirmation_config = {}

        # focus_tf 来自 trend_following_params, 已在 __init__ 中设置到 self.focus_timeframe
        focus_tf = self.focus_timeframe
        # logger.debug(f"[{self.strategy_name}] _perform_trend_analysis: Using focus_timeframe: {focus_tf} from self.focus_timeframe")

        if base_score_series is None or base_score_series.empty or base_score_series.isnull().all():
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 基础分数 Series 无效或全为 NaN，无法执行趋势分析。所有分析信号将为默认值。")
            default_ema_periods = trend_analysis_config.get('ema_periods', [5, 10, 20, 60]) # 从配置读取用于默认信号
            if not isinstance(default_ema_periods, list) or not default_ema_periods: default_ema_periods = [5,10,20,60]

            default_signals = {
                f'ema_score_{p}': 50.0 for p in default_ema_periods
            }
            default_signals.update({
                'alignment_signal': 0.0, 'ema_cross_signal': 0.0, 'ema_strength': 0.0,
                'score_momentum': 0.0, 'score_momentum_acceleration': 0.0, 'score_volatility': 0.0,
                'volatility_signal': 0.0, 'long_term_context': 0.0, 'adx_strength_signal': 0.0,
                'stoch_signal': 0.0, 'vwap_deviation_signal': 0.0, 'vwap_deviation_percent': 0.0,
                'boll_breakout_signal': 0.0
            })
            return pd.DataFrame(default_signals, index=data.index)

        score_series_filled = base_score_series.fillna(50.0)
        # 准备参数源列表，与 get_required_columns 一致
        # param_sources 使用新的配置变量
        param_sources = [
            trend_following_config,
            volume_confirmation_config,
            indicator_analysis_config,
            feature_engineering_config,
            base_scoring_config
        ]
        # 1. 计算分数 EMA
        # 使用 trend_analysis_config 获取 ema_periods
        all_ema_periods = trend_analysis_config.get('ema_periods', [5, 10, 20, 60])
        if not isinstance(all_ema_periods, list) or not all_ema_periods: # 确保是列表且非空
             logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 'trend_analysis.ema_periods' 参数无效或为空，使用默认值 [5, 10, 20, 60]。")
             all_ema_periods = [5, 10, 20, 60]
        for period in all_ema_periods:
            try:
                if isinstance(period, (int, float)) and period > 0:
                    ema_result = ta.ema(score_series_filled, length=int(period))
                    analysis_df[f'ema_score_{period}'] = ema_result
                    # print(f"[{self.strategy_name}] _perform_trend_analysis: EMA Score {period} 计算完成。结果后10行:\n{ema_result.tail(10)}") # 将 .head() 修改为 .tail(10) 以显示最后10个值
                else:
                    logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: EMA Score 周期参数无效: {period}. 跳过计算。")
                    analysis_df[f'ema_score_{period}'] = np.nan # 原始逻辑: 无效周期设为 NaN
            except Exception as e:
                logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 EMA Score {period} 时出错: {e}", exc_info=True)
                analysis_df[f'ema_score_{period}'] = np.nan # 原始逻辑: 出错时设为 NaN
        # 2. 计算 EMA 排列信号
        # 使用 trend_following_config 获取 ema_alignment_periods
        ema_periods_align = trend_following_config.get('ema_alignment_periods', all_ema_periods[:4] if len(all_ema_periods) >=4 else [])
        if not isinstance(ema_periods_align, list) or len(ema_periods_align) < 4:
             logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 'trend_following_params.ema_alignment_periods' 参数无效或数量不足4个，尝试使用 'trend_analysis.ema_periods' 的前4个 ({all_ema_periods[:4]})。")
             ema_periods_align = all_ema_periods[:4] if len(all_ema_periods) >=4 else [] # 提供备用值或空列表
             if len(ema_periods_align) < 4:
                  logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 可用的 EMA Score 周期不足4个 ({len(ema_periods_align)} found)，无法计算完整的 EMA 排列信号。")
                  analysis_df['alignment_signal'] = 0.0 # 无法计算则赋默认值

        if len(ema_periods_align) == 4 : # 只有当周期足够时才计算
            ema_alignment_cols = [f'ema_score_{p}' for p in ema_periods_align[:4]]
            if all(col in analysis_df.columns for col in ema_alignment_cols):
                try:
                    s_ema, m1_ema, m2_ema, l_ema = (analysis_df[col].fillna(50.0) for col in ema_alignment_cols)
                    alignment = pd.Series(0, index=analysis_df.index, dtype=float) # 明确 dtype
                    alignment += np.sign(s_ema - m1_ema).fillna(0)
                    alignment += np.sign(m1_ema - m2_ema).fillna(0)
                    alignment += np.sign(m2_ema - l_ema).fillna(0)
                    analysis_df['alignment_signal'] = alignment
                except Exception as e:
                     logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 EMA 排列信号时出错: {e}", exc_info=True)
                     analysis_df['alignment_signal'] = 0.0
            else:
                logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 无法计算 EMA 排列信号，所需 EMA Score 列不足或缺失。所需: {ema_alignment_cols}, 实际可用: {[col for col in ema_alignment_cols if col in analysis_df.columns]}")
                analysis_df['alignment_signal'] = 0.0
        else: # 如果 ema_periods_align 长度不足4
            analysis_df['alignment_signal'] = 0.0

        # 3. 计算 EMA 交叉信号
        # 使用 trend_following_config 获取 ema_cross_short/long
        ema_cross_short = trend_following_config.get('ema_cross_short', all_ema_periods[0] if len(all_ema_periods) > 0 else 5)
        ema_cross_long = trend_following_config.get('ema_cross_long', all_ema_periods[1] if len(all_ema_periods) > 1 else 20)
        short_ema_col = f'ema_score_{ema_cross_short}'
        long_ema_col = f'ema_score_{ema_cross_long}'
        if short_ema_col in analysis_df.columns and long_ema_col in analysis_df.columns:
            try:
                short_ema = analysis_df[short_ema_col].fillna(50.0)
                long_ema = analysis_df[long_ema_col].fillna(50.0)
                golden_cross = (short_ema > long_ema) & (short_ema.shift(1).fillna(long_ema.shift(1)) <= long_ema.shift(1).fillna(short_ema.shift(1))) # fillna 策略
                death_cross = (short_ema < long_ema) & (short_ema.shift(1).fillna(long_ema.shift(1)) >= long_ema.shift(1).fillna(short_ema.shift(1)))  # fillna 策略
                analysis_df['ema_cross_signal'] = 0.0
                analysis_df.loc[golden_cross, 'ema_cross_signal'] = 1.0
                analysis_df.loc[death_cross, 'ema_cross_signal'] = -1.0
                analysis_df['ema_cross_signal'] = analysis_df['ema_cross_signal'].fillna(0.0)
            except Exception as e:
                 logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 EMA 交叉信号 ({ema_cross_short}/{ema_cross_long}) 时出错: {e}", exc_info=True)
                 analysis_df['ema_cross_signal'] = 0.0
        else:
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 缺少 EMA Score 列 ({short_ema_col}, {long_ema_col})，无法计算 EMA 交叉信号。")
            analysis_df['ema_cross_signal'] = 0.0
        # 4. 计算 EMA 强度
        # 使用 trend_following_config 和 trend_analysis_config
        ema_strength_short_p = trend_following_config.get('ema_strength_short', all_ema_periods[0] if len(all_ema_periods) > 0 else 5)
        long_term_ema_period_for_strength = trend_analysis_config.get('long_term_ema_period', all_ema_periods[-1] if all_ema_periods else 60)
        ema_strength_long_p = trend_following_config.get('ema_strength_long', long_term_ema_period_for_strength)
        short_ema_col_strength = f'ema_score_{ema_strength_short_p}'
        long_ema_col_strength = f'ema_score_{ema_strength_long_p}'
        if short_ema_col_strength in analysis_df.columns and long_ema_col_strength in analysis_df.columns:
            try:
                analysis_df['ema_strength'] = (analysis_df[short_ema_col_strength].fillna(50.0) - analysis_df[long_ema_col_strength].fillna(50.0)).fillna(0.0)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 EMA 强度 ({ema_strength_short_p}/{ema_strength_long_p}) 时出错: {e}", exc_info=True)
                analysis_df['ema_strength'] = 0.0
        else:
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 缺少 EMA Score 列 ({short_ema_col_strength}, {long_ema_col_strength})，无法计算 EMA 强度。")
            analysis_df['ema_strength'] = 0.0
        # 5. 计算得分动量及动量加速
        analysis_df['score_momentum'] = score_series_filled.diff().fillna(0.0)
        analysis_df['score_momentum_acceleration'] = analysis_df['score_momentum'].diff().fillna(0.0)
        # 6. 计算得分波动率
        # 使用 trend_analysis_config 获取 volatility_window
        volatility_window = trend_analysis_config.get('volatility_window', 10)
        if not (isinstance(volatility_window, int) and volatility_window > 0): # 验证参数有效性
             logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 'trend_analysis.volatility_window' 参数无效 ({volatility_window})，使用默认值 10。")
             volatility_window = 10
        min_periods_volatility = max(1, volatility_window // 2)
        if len(score_series_filled) < min_periods_volatility:
             logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 数据长度不足 {min_periods_volatility} (基于窗口 {volatility_window})，无法计算得分波动率。")
             analysis_df['score_volatility'] = 0.0
        else:
             try:
                 analysis_df['score_volatility'] = score_series_filled.rolling(window=volatility_window, min_periods=min_periods_volatility).std().fillna(0.0)
             except Exception as e:
                 logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算得分波动率时出错: {e}", exc_info=True)
                 analysis_df['score_volatility'] = 0.0
        # volatility_signal 使用 self.volatility_threshold_high/low (来自 trend_following_params, 已在 __init__ 中设置)
        analysis_df['volatility_signal'] = 0.0
        analysis_df.loc[analysis_df['score_volatility'] > self.volatility_threshold_high, 'volatility_signal'] = -1.0
        analysis_df.loc[analysis_df['score_volatility'] < self.volatility_threshold_low, 'volatility_signal'] = 1.0
        analysis_df['volatility_signal'] = analysis_df['volatility_signal'].fillna(0.0)
        # 7. 长期趋势背景
        # 使用 trend_analysis_config 获取 long_term_ema_period
        long_term_ema_period_context = trend_analysis_config.get('long_term_ema_period', all_ema_periods[-1] if all_ema_periods else 60)
        if not (isinstance(long_term_ema_period_context, (int, float)) and long_term_ema_period_context > 0): # 验证参数有效性
             logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 'trend_analysis.long_term_ema_period' 参数无效 ({long_term_ema_period_context})，使用默认值 60。")
             long_term_ema_period_context = 60
        long_term_ema_col_context = f'ema_score_{long_term_ema_period_context}'
        if long_term_ema_col_context in analysis_df.columns:
            try:
                long_term_ema_filled = analysis_df[long_term_ema_col_context].fillna(50.0)
                analysis_df['long_term_context'] = 0.0
                analysis_df.loc[score_series_filled > long_term_ema_filled, 'long_term_context'] = 1.0
                analysis_df.loc[score_series_filled < long_term_ema_filled, 'long_term_context'] = -1.0
                analysis_df['long_term_context'] = analysis_df['long_term_context'].fillna(0.0)
            except Exception as e:
                 logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算长期趋势背景时出错: {e}", exc_info=True)
                 analysis_df['long_term_context'] = 0.0
        else:
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 缺少长期 EMA Score 列 ({long_term_ema_col_context})，无法计算长期趋势背景。")
            analysis_df['long_term_context'] = 0.0
        # 8. ADX 趋势强度判断
        # dmi_period_signal 从 param_sources (包含 base_scoring_config) 中查找 'dmi_period'
        dmi_period_signal = _get_param_val(param_sources, 'dmi_period', 14)
        if not (isinstance(dmi_period_signal, (int, float)) and dmi_period_signal > 0): # 验证参数有效性
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: DMI 周期参数无效 ({dmi_period_signal})，使用默认值 14。")
            dmi_period_signal = 14
        pdi_col = f'PDI_{dmi_period_signal}_{focus_tf}'
        ndi_col = f'NDI_{dmi_period_signal}_{focus_tf}'
        adx_col = f'ADX_{dmi_period_signal}_{focus_tf}'
        missing_dmi_cols = [col for col in [adx_col, pdi_col, ndi_col] if col not in data.columns]
        if missing_dmi_cols:
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 缺少 ADX/PDI/NDI 列 ({', '.join(missing_dmi_cols)}) for focus_tf='{focus_tf}' (period={dmi_period_signal})，无法计算 ADX 强度信号。")
            analysis_df['adx_strength_signal'] = 0.0
        else:
            try:
                adx = data[adx_col].fillna(0.0)
                pdi = data[pdi_col].fillna(0.0) # PDI 通常与 +DI 对应
                mdi = data[ndi_col].fillna(0.0) # NDI 通常与 -DI 对应
                # ADX 阈值 self.adx_strong_threshold / self.adx_moderate_threshold (来自 trend_following_params, 已在 __init__ 中设置)
                analysis_df['adx_strength_signal'] = 0.0
                strong_trend = adx >= self.adx_strong_threshold
                moderate_trend = (adx >= self.adx_moderate_threshold) & (adx < self.adx_strong_threshold)
                bullish_dmi = pdi > mdi
                bearish_dmi = mdi > pdi
                analysis_df.loc[strong_trend & bullish_dmi, 'adx_strength_signal'] = 1.0
                analysis_df.loc[strong_trend & bearish_dmi, 'adx_strength_signal'] = -1.0
                analysis_df.loc[moderate_trend & bullish_dmi, 'adx_strength_signal'] = 0.5
                analysis_df.loc[moderate_trend & bearish_dmi, 'adx_strength_signal'] = -0.5
                analysis_df['adx_strength_signal'] = analysis_df['adx_strength_signal'].fillna(0.0)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 ADX 强度信号出错: {e}", exc_info=True)
                analysis_df['adx_strength_signal'] = 0.0
        # 9. STOCH 超买超卖判断
        # STOCH 参数从 param_sources (包含 indicator_analysis_config) 查找
        stoch_k_p_signal = _get_param_val(param_sources, 'stoch_k', 14)
        stoch_d_p_signal = _get_param_val(param_sources, 'stoch_d', 3)
        stoch_smooth_k_p_signal = _get_param_val(param_sources, 'stoch_smooth_k', 3)
        if not (isinstance(stoch_k_p_signal, (int, float)) and stoch_k_p_signal > 0 and
                isinstance(stoch_d_p_signal, (int, float)) and stoch_d_p_signal > 0 and
                isinstance(stoch_smooth_k_p_signal, (int, float)) and stoch_smooth_k_p_signal > 0): # 验证参数有效性
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: STOCH 信号参数无效: k={stoch_k_p_signal}, d={stoch_d_p_signal}, smooth_k={stoch_smooth_k_p_signal}. 使用默认值 14,3,3。")
            stoch_k_p_signal, stoch_d_p_signal, stoch_smooth_k_p_signal = 14, 3, 3
        k_col = f'STOCHk_{stoch_k_p_signal}_{stoch_d_p_signal}_{stoch_smooth_k_p_signal}_{focus_tf}'
        d_col = f'STOCHd_{stoch_k_p_signal}_{stoch_d_p_signal}_{stoch_smooth_k_p_signal}_{focus_tf}'
        # STOCH 阈值 self.stoch_oversold_threshold / self.stoch_overbought_threshold (来自 trend_following_params, 已在 __init__ 中设置)
        missing_stoch_cols = [col for col in [k_col, d_col] if col not in data.columns]
        if missing_stoch_cols:
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 缺少 STOCH K/D 列 ({', '.join(missing_stoch_cols)}) for focus_tf='{focus_tf}' (params={stoch_k_p_signal},{stoch_d_p_signal},{stoch_smooth_k_p_signal})，无法计算 STOCH 信号。")
            analysis_df['stoch_signal'] = 0.0
        else:
            try:
                k_val = data[k_col].fillna(50.0)
                d_val = data[d_col].fillna(50.0)
                is_oversold = (k_val < self.stoch_oversold_threshold) & (d_val < self.stoch_oversold_threshold)
                is_overbought = (k_val > self.stoch_overbought_threshold) & (d_val > self.stoch_overbought_threshold)
                # 使用 fillna 避免比较 NaN 和数字时产生的问题
                turning_up = (k_val > d_val) & (k_val.shift(1).fillna(d_val.shift(1)) <= d_val.shift(1).fillna(k_val.shift(1))) # fillna 策略
                turning_down = (k_val < d_val) & (k_val.shift(1).fillna(d_val.shift(1)) >= d_val.shift(1).fillna(k_val.shift(1))) # fillna 策略
                analysis_df['stoch_signal'] = 0.0
                analysis_df.loc[is_oversold & turning_up, 'stoch_signal'] = 1.0
                analysis_df.loc[is_overbought & turning_down, 'stoch_signal'] = -1.0
                analysis_df.loc[is_oversold & ~(turning_up), 'stoch_signal'] = 0.5
                analysis_df.loc[is_overbought & ~(turning_down), 'stoch_signal'] = -0.5
                analysis_df['stoch_signal'] = analysis_df['stoch_signal'].fillna(0.0)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 STOCH 信号出错: {e}", exc_info=True)
                analysis_df['stoch_signal'] = 0.0
        # 10. VWAP 偏离判断
        # vwap_anchor_signal 从 param_sources 查找 'vwap_anchor' (JSON 中此键不存在于相关部分，故为 None)
        vwap_anchor_signal = _get_param_val(param_sources, 'vwap_anchor', None)
        vwap_col_name_base = 'VWAP' if vwap_anchor_signal is None else f'VWAP_{vwap_anchor_signal}'
        vwap_col = f"{vwap_col_name_base}_{focus_tf}"
        close_col = f'close_{focus_tf}'
        # VWAP 偏离阈值 self.vwap_deviation_threshold (来自 trend_following_params, 已在 __init__ 中设置)
        missing_vwap_cols = [col for col in [vwap_col, close_col] if col not in data.columns]
        if missing_vwap_cols:
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 缺少 VWAP 或收盘价列 ({', '.join(missing_vwap_cols)}) for focus_tf='{focus_tf}'，无法计算 VWAP 偏离信号。")
            analysis_df['vwap_deviation_signal'] = 0.0
            analysis_df['vwap_deviation_percent'] = 0.0
        else:
            try:
                vwap = data[vwap_col]
                close_price = data[close_col]
                vwap_safe = vwap.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
                if vwap_safe.isnull().all() or vwap_safe.eq(0).all(): # 增加对全0的检查
                    logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: VWAP ({vwap_col}) 数据无效 (全为 NaN 或 0)，无法计算 VWAP 偏离。")
                    analysis_df['vwap_deviation_signal'] = 0.0
                    analysis_df['vwap_deviation_percent'] = 0.0
                else:
                    # 确保vwap_safe不为0，避免除零错误
                    deviation = ((close_price - vwap_safe) / vwap_safe.replace(0, np.nan)).fillna(0.0) # 再次确保除数不为0
                    analysis_df['vwap_deviation_signal'] = 0.0
                    deviation_filled = deviation # deviation 已经被fillna(0.0)
                    analysis_df.loc[deviation_filled > self.vwap_deviation_threshold, 'vwap_deviation_signal'] = 1.0
                    analysis_df.loc[deviation_filled < -self.vwap_deviation_threshold, 'vwap_deviation_signal'] = -1.0
                    analysis_df['vwap_deviation_signal'] = analysis_df['vwap_deviation_signal'].fillna(0.0)
                    analysis_df['vwap_deviation_percent'] = (deviation_filled * 100).fillna(0.0)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 VWAP 偏离信号出错: {e}", exc_info=True)
                analysis_df['vwap_deviation_signal'] = 0.0
                analysis_df['vwap_deviation_percent'] = 0.0
        # 11. BOLL 突破判断
        # boll_breakout_params_tf 从 trend_following_config 获取
        boll_breakout_params_tf = trend_following_config.get('boll_breakout_params', {})
        boll_period_signal, boll_std_dev_signal = None, None # 初始化
        if isinstance(boll_breakout_params_tf, dict) and 'period' in boll_breakout_params_tf and 'std_dev' in boll_breakout_params_tf:
             boll_period_signal = boll_breakout_params_tf['period']
             boll_std_dev_signal = boll_breakout_params_tf['std_dev']
        else:
             # 如果策略特定参数没有配置，回退使用 base_scoring_config 中的 BOLL 参数
             boll_period_signal = base_scoring_config.get("boll_period", 20) # 默认值与注释中讨论的默认值(20,2.0)保持一致或根据实际需要
             boll_std_dev_signal = base_scoring_config.get("boll_std_dev", 2.0)
             logger.debug(f"[{self.strategy_name}] _perform_trend_analysis: 未找到 'trend_following_params.boll_breakout_params' 或配置无效，回退使用 'base_scoring' BOLL 参数 (period={boll_period_signal}, std_dev={boll_std_dev_signal})。")
        # 确保参数是有效数字 (此段逻辑原先被注释，根据需要决定是否启用)
        if not (isinstance(boll_period_signal, (int, float)) and boll_period_signal > 0 and
                isinstance(boll_std_dev_signal, (int, float)) and boll_std_dev_signal > 0):
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: BOLL 突破信号参数无效或未正确读取: period={boll_period_signal}, std_dev={boll_std_dev_signal}. 使用默认值 20, 2.0。")
            boll_period_signal, boll_std_dev_signal = 20, 2.0 # 确保有默认值

        std_str_signal = f"{float(boll_std_dev_signal):.1f}" # 确保 boll_std_dev_signal 转为 float
        bbu_col = f'BBU_{int(boll_period_signal)}_{std_str_signal}_{focus_tf}' # 确保 boll_period_signal 转为 int
        bbl_col = f'BBL_{int(boll_period_signal)}_{std_str_signal}_{focus_tf}' # 确保 boll_period_signal 转为 int
        # close_col 已在 VWAP 部分获取
        missing_boll_cols = [col for col in [bbu_col, bbl_col, close_col] if col not in data.columns]
        if missing_boll_cols:
            # 日志信息已在原始代码中，这里保持不变。此问题根源在于 data DataFrame 缺少对应列。
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 缺少 BOLL 上轨/下轨或收盘价列 ({', '.join(missing_boll_cols)}) for focus_tf='{focus_tf}' (period={boll_period_signal}, std_dev={boll_std_dev_signal})，无法计算 BOLL 突破信号。")
            analysis_df['boll_breakout_signal'] = 0.0
        else:
            try:
                # 当上/下轨本身是NaN时，用收盘价填充可能不是最佳策略，但遵循原逻辑。更好的方式是上游处理好指标的NaN。
                bbu = data[bbu_col].fillna(data[close_col])
                bbl = data[bbl_col].fillna(data[close_col])
                close_price = data[close_col] # 已定义
                analysis_df['boll_breakout_signal'] = 0.0
                # 使用 fillna 替换可能因 shift 产生的 NaN，使其可与另一侧比较
                # shifted_close 用 bbu/bbl 的 shifted value 来 fillna，反之亦然，或者用一个通用的
                shifted_close_price = data[close_col].shift(1)
                shifted_bbu = bbu.shift(1)
                shifted_bbl = bbl.shift(1)

                # 填充NaN值以进行稳健比较
                # 如果前一期值为NaN，则认为未满足突破前置条件
                bullish_breakout = (close_price > bbu) & (shifted_close_price.fillna(shifted_bbu) <= shifted_bbu.fillna(shifted_close_price))
                bearish_breakout = (close_price < bbl) & (shifted_close_price.fillna(shifted_bbl) >= shifted_bbl.fillna(shifted_close_price))
                analysis_df.loc[bullish_breakout, 'boll_breakout_signal'] = 1.0
                analysis_df.loc[bearish_breakout, 'boll_breakout_signal'] = -1.0
                analysis_df['boll_breakout_signal'] = analysis_df['boll_breakout_signal'].fillna(0.0)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 BOLL 突破信号 (period={boll_period_signal}, std_dev={boll_std_dev_signal}) 出错: {e}", exc_info=True)
                analysis_df['boll_breakout_signal'] = 0.0
        # 12. 计算综合趋势强度 (可选)
        logger.debug(f"[{self.strategy_name}] 趋势分析完成，最新分析信号 (部分): "
                     f"Alignment: {analysis_df['alignment_signal'].iloc[-1] if not analysis_df.empty and 'alignment_signal' in analysis_df.columns and not analysis_df['alignment_signal'].empty else 'N/A'}, "
                     f"ADX Strength: {analysis_df['adx_strength_signal'].iloc[-1] if not analysis_df.empty and 'adx_strength_signal' in analysis_df.columns and not analysis_df['adx_strength_signal'].empty else 'N/A'}")

        # “底部放量起涨”信号
        # 0. 底部判定：近60根K线收盘价处于最低20%（即底部区域）
        bottom_window = indicator_analysis_config.get('bottom_window', 55)  # 可配置，默认60
        bottom_percent = indicator_analysis_config.get('bottom_percent', 0.2)  # 可配置，默认0.2

        if close_col in analysis_df.columns:
            analysis_df['min_close_60'] = analysis_df[close_col].rolling(window=bottom_window, min_periods=1).min()
            analysis_df['max_close_60'] = analysis_df[close_col].rolling(window=bottom_window, min_periods=1).max()
            analysis_df['bottom_zone'] = (
                (analysis_df[close_col] - analysis_df['min_close_60']) /
                (analysis_df['max_close_60'] - analysis_df['min_close_60'] + 1e-6) < bottom_percent
            ).astype(float)
        else:
            analysis_df['bottom_zone'] = 0.0

        # 1. 获取参数
        volume_spike_factor = indicator_analysis_config.get('volume_spike_factor', 2.0)  # 放量倍数，默认2.0
        # volume_ma_period = indicator_analysis_config.get('volume_ma_period', 20)         # 均量周期，默认20

        # 2. 字段名
        vol_col = f'volume_{focus_tf}'
        vol_ma_col = f'VOL_MA_{focus_tf}'
        close_col = f'close_{focus_tf}'
        open_col = f'open_{focus_tf}'
        high_col = f'high_{focus_tf}'

        # 3. 计算近20日最高价
        if high_col in analysis_df.columns:
            analysis_df['high_20'] = analysis_df[high_col].rolling(window=20, min_periods=1).max()
        else:
            analysis_df['high_20'] = None

        # 4. 放量判断
        if vol_col in analysis_df.columns and vol_ma_col in analysis_df.columns:
            analysis_df['is_volume_surge'] = (analysis_df[vol_col] > volume_spike_factor * analysis_df[vol_ma_col]).astype(float)
        else:
            analysis_df['is_volume_surge'] = 0.0

        # 5. 起涨判断（两种方式：突破近20日高点 或 大阳线涨幅>3%）
        if close_col in analysis_df.columns and open_col in analysis_df.columns:
            # 价格突破近20日高点
            analysis_df['is_price_breakout'] = (
                (analysis_df['high_20'].notnull()) & (analysis_df[close_col] > analysis_df['high_20'])
            ).astype(float)
            # 大阳线且涨幅>3%
            analysis_df['is_strong_bull'] = (
                (analysis_df[close_col] > analysis_df[open_col]) &
                ((analysis_df[close_col] - analysis_df[open_col]) / analysis_df[open_col] > 0.03)
            ).astype(float)
        else:
            analysis_df['is_price_breakout'] = 0.0
            analysis_df['is_strong_bull'] = 0.0

        # 6. 综合“放量起涨”信号
        analysis_df['volume_breakout_signal'] = (
            (analysis_df['is_volume_surge'] > 0) &
            ((analysis_df['is_price_breakout'] > 0) | (analysis_df['is_strong_bull'] > 0))
        ).astype(float)
        # 7. 综合“底部放量起涨”信号
        analysis_df['bottom_volume_breakout_signal'] = (
            (analysis_df['bottom_zone'] > 0) &
            (analysis_df['volume_breakout_signal'] > 0)
        ).astype(float)
        return analysis_df

    def _adjust_volatility_parameters(self, data: pd.DataFrame):
        """
        根据股票波动率动态调整参数，如波动率阈值。
        使用JSON配置获取close列名。
        """
        focus_tf = self.focus_timeframe
        # 使用JSON配置获取close列名
        ohlcv_configs = NAMING_CONFIG.get('ohlcv_naming_convention', {}).get('output_columns', [])
        close_base_name = next((c['name_pattern'] for c in ohlcv_configs if isinstance(c, dict) and c.get('name_pattern') == 'close'), 'close')
        close_col = f'{close_base_name}_{focus_tf}' # 使用带后缀的列名

        if close_col not in data.columns or data[close_col].isnull().all():
            logger.warning(f"[{self.strategy_name}] 动态调整波动率：缺少收盘价列 {close_col} 或数据全为空。")
            return

        ta_params_global = self.params.get('trend_analysis', {})
        if not isinstance(ta_params_global, dict): ta_params_global = {}
        volatility_window = ta_params_global.get('volatility_window', 10)
        if not isinstance(volatility_window, int) or volatility_window <= 0:
             logger.warning(f"[{self.strategy_name}] 动态调整波动率：'trend_analysis.volatility_window' 参数无效 ({volatility_window})，使用默认值 10。")
             volatility_window = 10

        min_periods_volatility = max(1, volatility_window // 2)
        if len(data) < min_periods_volatility:
            logger.warning(f"[{self.strategy_name}] 动态调整波动率：数据长度不足 {min_periods_volatility}，无法计算价格波动率。")
            return

        try:
            price_volatility = data[close_col].rolling(window=volatility_window, min_periods=min_periods_volatility).std()

            if price_volatility.isnull().all() or price_volatility.empty:
                logger.warning(f"[{self.strategy_name}] 动态调整波动率：价格波动率数据计算后全为空或空 Series。")
                return

            latest_volatility = price_volatility.iloc[-1]
            # 确保 latest_volatility 不是无穷大或非常大的值
            if not np.isfinite(latest_volatility) or latest_volatility <= 0:
                 logger.warning(f"[{self.strategy_name}] 动态调整波动率：最新波动率 ({latest_volatility}) 无效或非正数。")
                 return

            # volatility_benchmark 来自 trend_following_params
            base_volatility_benchmark = self.tf_params.get('volatility_benchmark', 5.0)
            if not isinstance(base_volatility_benchmark, (int, float)) or base_volatility_benchmark <= 0:
                 logger.warning(f"[{self.strategy_name}] 动态调整波动率：'volatility_benchmark' 参数无效或非正数 ({base_volatility_benchmark})。")
                 # 如果基准无效，不进行调整，使用原始阈值
                 self.volatility_adjust_factor = 1.0
                 self.volatility_threshold_high = self.tf_params.get('volatility_threshold_high', 10.0)
                 self.volatility_threshold_low = self.tf_params.get('volatility_threshold_low', 5.0)
                 logger.debug(f"[{self.strategy_name}] 动态调整波动率阈值：基准无效，使用原始阈值 high={self.volatility_threshold_high:.2f}, low={self.volatility_threshold_low:.2f}, factor={self.volatility_adjust_factor:.2f}")
                 return

            self.volatility_adjust_factor = max(0.5, min(2.0, latest_volatility / base_volatility_benchmark)) # 将调整因子限制在 0.5 到 2.0 之间

            original_high = self.tf_params.get('volatility_threshold_high', 10.0)
            original_low = self.tf_params.get('volatility_threshold_low', 5.0)

            self.volatility_threshold_high = original_high * self.volatility_adjust_factor
            self.volatility_threshold_low = original_low * self.volatility_adjust_factor

            # 确保调整后的阈值在一个合理范围内，例如不低于原始值的一半，不高于原始值的两倍
            # 并且确保 low <= high
            self.volatility_threshold_high = np.clip(self.volatility_threshold_high, original_high * 0.5, original_high * 2.0)
            self.volatility_threshold_low = np.clip(self.volatility_threshold_low, original_low * 0.5, original_low * 2.0)
            if self.volatility_threshold_low >= self.volatility_threshold_high:
                 self.volatility_threshold_low = self.volatility_threshold_high * 0.9 # 确保 low 略小于 high


            logger.debug(f"[{self.strategy_name}] 动态调整波动率阈值: high={self.volatility_threshold_high:.2f}, low={self.volatility_threshold_low:.2f}, factor={self.volatility_adjust_factor:.2f} (based on latest_vol={latest_volatility:.2f}, benchmark={base_volatility_benchmark:.2f})")

        except Exception as e:
             logger.error(f"[{self.strategy_name}] 动态调整波动率参数出错: {e}", exc_info=True)
             # 出现错误时，不进行调整，使用原始阈值
             self.volatility_adjust_factor = 1.0
             self.volatility_threshold_high = self.tf_params.get('volatility_threshold_high', 10.0)
             self.volatility_threshold_low = self.tf_params.get('volatility_threshold_low', 5.0)
             logger.debug(f"[{self.strategy_name}] 动态调整波动率阈值：计算出错，使用原始阈值 high={self.volatility_threshold_high:.2f}, low={self.volatility_threshold_low:.2f}, factor={self.volatility_adjust_factor:.2f}")

    def _apply_adx_boost(self, final_signal: pd.Series, adx_strength_norm: pd.Series, base_signal_direction_norm: pd.Series) -> pd.Series:
        """
        模块化调整逻辑：使用 ADX 强度增强信号。
        """
        if final_signal.empty or adx_strength_norm.empty or base_signal_direction_norm.empty:
            logger.warning(f"[{self.strategy_name}] ADX 增强调整：输入 Series 为空，跳过调整。")
            return final_signal # 返回原始 Series

        # 确保索引对齐，不对齐则尝试 reindex，无法对齐则跳过调整
        if not (final_signal.index.equals(adx_strength_norm.index) and final_signal.index.equals(base_signal_direction_norm.index)):
             logger.warning(f"[{self.strategy_name}] ADX 增强调整：输入 Series 索引不一致，尝试重新对齐。")
             try:
                 adx_strength_norm_aligned = adx_strength_norm.reindex(final_signal.index).fillna(0.0)
                 base_signal_direction_norm_aligned = base_signal_direction_norm.reindex(final_signal.index).fillna(0.0)
             except Exception as e:
                  logger.error(f"[{self.strategy_name}] ADX 增强调整：数据重新对齐失败: {e}. 跳过调整。", exc_info=True)
                  return final_signal # 返回原始 Series
        else:
            adx_strength_norm_aligned = adx_strength_norm.fillna(0.0)
            base_signal_direction_norm_aligned = base_signal_direction_norm.fillna(0.0)


        final_signal_filled = final_signal.fillna(50.0)
        adx_adjustment_factor = self.tf_params.get('adx_adjustment_factor', 10.0)
        
        # 确保 adjustment_factor 是有效数字
        if not isinstance(adx_adjustment_factor, (int, float)):
             logger.warning(f"[{self.strategy_name}] ADX 增强调整：'adx_adjustment_factor' 参数无效 ({adx_adjustment_factor})，使用默认值 10.0。")
             adx_adjustment_factor = 10.0


        adjustment = pd.Series(0.0, index=final_signal.index)

        # 只在 ADX 方向与基础信号方向一致时进行增强
        effective_mask = (np.sign(adx_strength_norm_aligned) == np.sign(base_signal_direction_norm_aligned)) & \
                         (base_signal_direction_norm_aligned != 0) & \
                         (adx_strength_norm_aligned.abs() > 0) # ADX强度必须大于0才有增强效果

        adjustment.loc[effective_mask] = np.sign(base_signal_direction_norm_aligned.loc[effective_mask]) * \
                                         np.abs(adx_strength_norm_aligned.loc[effective_mask]) * \
                                         adx_adjustment_factor

        # 调整幅度限制，防止过度增强/减弱，例如限制在 +/- 15 分以内
        max_adjustment = self.tf_params.get('adx_max_adjustment', 15.0)
        if not isinstance(max_adjustment, (int, float)) or max_adjustment < 0:
            logger.warning(f"[{self.strategy_name}] ADX 增强调整：'adx_max_adjustment' 参数无效 ({max_adjustment})，使用默认值 15.0。")
            max_adjustment = 15.0

        adjustment = adjustment.clip(-max_adjustment, max_adjustment)

        if not adjustment.empty:
             logger.debug(f"[{self.strategy_name}] ADX 增强调整 (最新值): {adjustment.iloc[-1] if not adjustment.empty else 'N/A'}")
        return (final_signal_filled + adjustment).clip(0, 100)

    def _apply_divergence_penalty(self, final_signal: pd.Series, divergence_signals_df: pd.DataFrame, dd_params: Dict) -> pd.Series:
        """
        模块化调整逻辑：应用背离惩罚。
        使用JSON配置获取背离信号列名。
        """
        final_signal_filled = final_signal.fillna(50.0)
        if divergence_signals_df.empty:
            logger.debug(f"[{self.strategy_name}] 背离信号 DataFrame 为空，跳过背离惩罚。")
            return final_signal_filled # 返回原始 Series

        # 确保 divergence_signals_df 是一个 DataFrame
        if not isinstance(divergence_signals_df, pd.DataFrame):
            logger.error(f"[{self.strategy_name}] 背离惩罚：传入的背离信号不是 DataFrame。跳过惩罚。")
            return final_signal_filled

        penalty_factor = dd_params.get('divergence_penalty_factor', 0.45)
        if not isinstance(penalty_factor, (int, float)) or not (0 <= penalty_factor <= 1):
             logger.warning(f"[{self.strategy_name}] 背离惩罚：'divergence_penalty_factor' 参数无效 ({penalty_factor})，使用默认值 0.45。")
             penalty_factor = 0.45

        # 从JSON获取背离信号列名 (这些是策略内部列)
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        has_bearish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c.get('name_pattern') == "HAS_BEARISH_DIVERGENCE"), "HAS_BEARISH_DIVERGENCE")
        has_bullish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c.get('name_pattern') == "HAS_BULLISH_DIVERGENCE"), "HAS_BULLISH_DIVERGENCE")

        if has_bearish_div_col not in divergence_signals_df.columns or \
           has_bullish_div_col not in divergence_signals_df.columns:
            logger.warning(f"[{self.strategy_name}] 背离信号 DataFrame 缺少聚合信号列 ('{has_bearish_div_col}', '{has_bullish_div_col}')，跳过背离惩罚。")
            return final_signal_filled

        # 确保信号和背离数据索引对齐
        if not final_signal_filled.index.equals(divergence_signals_df.index):
            logger.warning(f"[{self.strategy_name}] 背离惩罚：信号和背离数据索引不一致，尝试重新对齐。")
            try:
                # 使用 reindex 并保持布尔类型，用 False 填充缺失值
                divergence_signals_aligned = divergence_signals_df.reindex(final_signal_filled.index).fillna(False)
                has_bearish_div = divergence_signals_aligned[has_bearish_div_col].astype(bool)
                has_bullish_div = divergence_signals_aligned[has_bullish_div_col].astype(bool)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] 背离数据重新对齐失败: {e}。跳过背离惩罚。", exc_info=True)
                return final_signal_filled # 返回原始 Series
        else:
            has_bearish_div = divergence_signals_df[has_bearish_div_col].astype(bool)
            has_bullish_div = divergence_signals_df[has_bullish_div_col].astype(bool)


        is_bullish_signal = final_signal_filled > 50
        is_bearish_signal = final_signal_filled < 50
        adjusted_signal = final_signal_filled.copy()

        # 看涨信号遇到看跌背离时惩罚
        mask_bullish_penalty = is_bullish_signal & has_bearish_div
        # 将信号拉向中性 50，惩罚越大，拉得越多
        adjusted_signal.loc[mask_bullish_penalty] = 50 + (adjusted_signal.loc[mask_bullish_penalty] - 50) * (1 - penalty_factor)

        # 看跌信号遇到看涨背离时惩罚
        mask_bearish_penalty = is_bearish_signal & has_bullish_div
         # 将信号拉向中性 50，惩罚越大，拉得越多
        adjusted_signal.loc[mask_bearish_penalty] = 50 + (adjusted_signal.loc[mask_bearish_penalty] - 50) * (1 - penalty_factor)

        if not adjusted_signal.empty:
             logger.debug(f"[{self.strategy_name}] 背离惩罚调整后信号 (最新值): {adjusted_signal.iloc[-1] if not adjusted_signal.empty else 'N/A'}")
        return adjusted_signal.clip(0, 100)

    def _apply_trend_confirmation(self, final_signal: pd.Series) -> pd.Series:
        """
        增强假信号过滤：要求信号突破阈值后持续若干周期才视为有效。
        """
        if final_signal is None or final_signal.empty:
            logger.warning(f"[{self.strategy_name}] 趋势确认过滤：输入 Series 为空，跳过过滤。")
            return pd.Series(dtype=float) # 返回空 Series

        # trend_confirmation_threshold_upper/lower 和 confirmation_periods 来自 trend_following_params
        trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55)
        trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45)
        confirmation_periods = self.trend_confirmation_periods # 已在 __init__ 中从 tf_params 获取并有默认值

        # 确保阈值和周期是有效数字
        if not isinstance(trend_threshold_upper, (int, float)):
             logger.warning(f"[{self.strategy_name}] 趋势确认过滤：'trend_confirmation_threshold_upper' 参数无效 ({trend_threshold_upper})，使用默认值 55。")
             trend_threshold_upper = 55.0
        if not isinstance(trend_threshold_lower, (int, float)):
             logger.warning(f"[{self.strategy_name}] 趋势确认过滤：'trend_confirmation_threshold_lower' 参数无效 ({trend_threshold_lower})，使用默认值 45。")
             trend_threshold_lower = 45.0
        if not isinstance(confirmation_periods, int) or confirmation_periods <= 0:
             logger.warning(f"[{self.strategy_name}] 趋势确认过滤：'trend_confirmation_periods' 参数无效 ({confirmation_periods})，使用默认值 3。")
             confirmation_periods = 3

        # 确保下限阈值不高于上限阈值
        if trend_threshold_lower >= trend_threshold_upper:
             logger.warning(f"[{self.strategy_name}] 趋势确认过滤：lower ({trend_threshold_lower}) >= upper ({trend_threshold_upper}) 阈值，调整 lower 阈值。")
             trend_threshold_lower = trend_threshold_upper - 5 # 确保有一个间隔

        final_signal_filled = final_signal.fillna(50.0)
        # 初始化 filtered_signal 为 NaN，以便后续进行前向填充
        filtered_signal = pd.Series(index=final_signal.index, dtype=float) # 初始化为NaN，用于趋势保持
        # print(f"DEBUG: 初始 filtered_signal (NaN): {filtered_signal.head()}") # 调试信息

        if len(final_signal_filled) < confirmation_periods:
             logger.warning(f"[{self.strategy_name}] 趋势确认过滤：数据长度 ({len(final_signal_filled)}) 不足确认周期 ({confirmation_periods})。跳过过滤。")
             return final_signal_filled # 数据不足时不进行过滤

        try:
            # 判断信号是否持续高于上限或低于下限足够周期
            # min_periods=confirmation_periods 确保只有足够的数据点才开始计算
            above_upper_streak = (final_signal_filled >= trend_threshold_upper).rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() == confirmation_periods
            below_lower_streak = (final_signal_filled <= trend_threshold_lower).rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() == confirmation_periods

            # 将确认后的信号赋值给 filtered_signal
            # 只有在连续满足条件时，才将原始信号值传递过来
            filtered_signal.loc[above_upper_streak] = final_signal_filled.loc[above_upper_streak]
            filtered_signal.loc[below_lower_streak] = final_signal_filled.loc[below_lower_streak]
            # print(f"DEBUG: 赋值后的 filtered_signal (仅确认点有值): {filtered_signal.head()}") # 调试信息

            # 使用 ffill() 保持上一个确认的趋势值
            filtered_signal = filtered_signal.ffill() # 前向填充，保持趋势状态
            # print(f"DEBUG: ffill 后的 filtered_signal (趋势保持): {filtered_signal.head()}") # 调试信息
            # 填充 NaN，特别是开头 confirmation_periods-1 个点以及未被任何趋势确认覆盖的区域
            filtered_signal = filtered_signal.fillna(50.0) # 填充剩余的NaN（如序列开头），默认中性50
            # print(f"DEBUG: fillna(50.0) 后的 filtered_signal (最终填充): {filtered_signal.head()}") # 调试信息

        except Exception as e:
            logger.error(f"[{self.strategy_name}] 趋势确认过滤出错: {e}", exc_info=True)
            # 出现错误时，不进行过滤，返回原始信号
            return final_signal_filled.clip(0, 100) # 至少确保范围正确

        if not filtered_signal.empty:
             logger.debug(f"[{self.strategy_name}] 趋势确认过滤后信号 (最新值): {filtered_signal.iloc[-1] if not filtered_signal.empty else 'N/A'}")
        return filtered_signal.clip(0, 100) # 确保过滤后的信号也在 0-100 范围内
    
    # 这个方法是实际的信号生成入口，返回 DataFrame
    def generate_signals(self, data: pd.DataFrame, stock_code: Optional[str] = None, indicator_configs: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        生成趋势跟踪信号，整合规则信号和Transformer模型预测。
        返回包含所有中间计算和最终信号的 DataFrame。
        """
        if data is None or data.empty:
            logger.warning(f"[{self.strategy_name}] generate_signals: 输入数据为空，无法生成信号。")
            # 返回一个包含默认信号列的空DataFrame，避免下游错误
            empty_df = pd.DataFrame(columns=['final_rule_signal', 'transformer_signal', 'combined_signal', 'signal'])
            return empty_df
        if stock_code is None:
            logger.error(f"[{self.strategy_name}] generate_signals: 必须提供 stock_code。")
            empty_df = pd.DataFrame(index=data.index, columns=['final_rule_signal', 'transformer_signal', 'combined_signal', 'signal'])
            empty_df.fillna(50.0, inplace=True) # 信号默认中性
            empty_df['signal'] = 0 # 最终信号默认 0
            return empty_df
        if indicator_configs is None:
            logger.warning(f"[{self.strategy_name}][{stock_code}] generate_signals: 未提供 indicator_configs。基础评分计算可能不准确。")
            # 尝试从 self.params 中获取基础指标配置，作为备用
            bs_params = self.params.get('base_scoring', {})
            if not isinstance(bs_params, dict): bs_params = {}
            # 这里的 indicator_configs 应该包含 IndicatorService 实际计算了哪些指标的信息
            # 如果没有提供，我们无法知道具体计算了哪些参数组合，这可能导致问题
            # 暂时使用一个空的 indicator_configs 列表，依赖 calculate_all_indicator_scores 内部的容错或参数默认值
            indicator_configs_to_use = []
        else:
             indicator_configs_to_use = indicator_configs
        logger.info(f"[{self.strategy_name}] 开始为股票 {stock_code} 生成信号 (Focus: {self.focus_timeframe})...")
        # --- 阶段 1: 计算规则信号及中间数据 ---
        logger.debug(f"[{self.strategy_name}][{stock_code}] 计算规则信号及中间数据...")
        final_rule_signal, intermediate_results_dict = self._calculate_rule_based_signal(
            data=data, stock_code=stock_code, indicator_configs=indicator_configs_to_use
        )
        logger.debug(f"[{self.strategy_name}][{stock_code}] 规则信号计算完成。")
        # 将原始数据复制，并合并规则信号和中间结果
        processed_data = data.copy()
        # 添加 final_rule_signal 列 (这是一个策略内部列)
        processed_data['final_rule_signal'] = final_rule_signal # final_rule_signal 是一个 Series
        # 合并 intermediate_results_dict 中的 DataFrame/Series
        # 注意：这里的键名 (如 'base_score_raw', 'trend_analysis_df' 等) 可能会与最终的列名不同
        # 需要将 dict 中的 DataFrame/Series 的列合并到 processed_data 中
        for key, df_or_series in intermediate_results_dict.items():
            if isinstance(df_or_series, pd.DataFrame) and not df_or_series.empty:
                # 合并 DataFrame 的所有列
                for col_join in df_or_series.columns:
                    # 避免覆盖原始数据或已有的中间列，除非是特定的内部列需要更新
                    # 例如，ADJUSTED_SCORE 是临时列，应该被最终的 base_score_volume_adjusted 替代
                    # 假设 strategy_utils 返回的 DataFrame 列名是最终想要的列名（除了 ADJUSTED_SCORE）
                    if col_join != 'ADJUSTED_SCORE' and col_join not in processed_data.columns:
                         processed_data[col_join] = df_or_series[col_join]
                    elif col_join == 'ADJUSTED_SCORE':
                        # 特殊处理 ADJUSTED_SCORE，将其值赋值给 'base_score_volume_adjusted' 列 (如果还没有的话)
                        if 'base_score_volume_adjusted' not in processed_data.columns:
                            processed_data['base_score_volume_adjusted'] = df_or_series[col_join]
                        else:
                            logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果列 'ADJUSTED_SCORE' 已存在于 processed_data['base_score_volume_adjusted']，跳过合并。")
                    else:
                        logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果列 '{col_join}' 已存在于 processed_data，跳过合并来自 '{key}' 的同名列。")
            elif isinstance(df_or_series, pd.Series) and not df_or_series.empty:
                # 合并 Series
                # 如果 Series 有 name 属性，使用 name 作为列名
                series_col_name = df_or_series.name
                if series_col_name and series_col_name not in processed_data.columns:
                    processed_data[series_col_name] = df_or_series
                # 如果 Series 没有 name，或者 name 已存在，可以考虑使用 dict 的 key 作为列名（需要确保不冲突）
                elif not series_col_name and key not in processed_data.columns:
                    processed_data[key] = df_or_series
                else:
                    logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果Series '{series_col_name or key}' 已存在或无名，跳过合并。")
        # --- 阶段 2: Transformer 模型预测增强 ---
        logger.debug(f"[{self.strategy_name}][{stock_code}] 准备 Transformer 模型预测...")
        # 添加 transformer_signal 列 (内部列)，默认填充 50.0
        processed_data['transformer_signal'] = pd.Series(50.0, index=processed_data.index)
        # print(f"generate_signals.processed_data: {stock_code} - {processed_data['final_rule_signal']}")
        self.set_model_paths(stock_code)
        # 调用加载 Transformer 模型和转换器的方法
        # load_prepared_data 会加载数据、Scalers 和可选的特征工程转换器
        # 虽然这里不需要加载训练数据本身，但调用它可以确保 Scalers 和转换器被加载到 self 属性中
        # 更好的做法是有一个专门加载模型和转换器的方法，而不是复用 load_prepared_data
        # 为了简化，我们假设 load_prepared_data 已经将所需的 self.transformer_model, self.feature_scaler, self.target_scaler, self.selected_feature_names_for_transformer, self.pca_model, self.scaler_for_pca, self.feature_selector_model 加载好了
        # 如果 load_prepared_data 返回了数据，这里可以忽略它们，只关注 self 属性
        _, _, _, _, _, _, feature_scaler_loaded, target_scaler_loaded = self.load_prepared_data(stock_code)
        # 调用 load_transformer_model 来加载 Transformer 模型本身
        
        self.load_transformer_model(stock_code) # 调用此方法加载 Transformer 模型

        # ----------- 自动trace并保存traced模型 -----------
        # 假设模型路径和traced模型路径如下
        model_dir = os.path.dirname(self.model_path)
        traced_model_path = os.path.join(model_dir, f"transformer_{stock_code}_traced.pt")

        # 优先加载traced模型
        if os.path.exists(traced_model_path):
            try:
                self.transformer_model = torch.jit.load(traced_model_path, map_location='cpu')
                self.transformer_model.eval()
                self.jit_traced = True
                print(f"[{self.strategy_name}][{stock_code}] 已加载traced模型: {traced_model_path}")
            except Exception as e:
                print(f"[{self.strategy_name}][{stock_code}] 加载traced模型失败: {e}，将尝试trace原始模型。")
                self.jit_traced = False
        else:
            self.jit_traced = False

        # 如果未trace过，自动trace并保存
        if self.transformer_model and not getattr(self, 'jit_traced', False):
            try:
                self.transformer_model.eval()
                # 构造示例输入，shape需与实际推理一致
                example_input = torch.zeros(1, self.transformer_window_size, len(self.selected_feature_names_for_transformer))
                example_input = example_input.to('cpu')
                self.transformer_model.to('cpu')
                scripted_model = torch.jit.script(self.transformer_model)
                scripted_model.save(traced_model_path)
                self.transformer_model = scripted_model
                self.jit_traced = True
                print(f"[{self.strategy_name}][{stock_code}] 已自动script并保存scripted模型: {traced_model_path}")
            except Exception as e:
                print(f"[{self.strategy_name}][{stock_code}] torch.jit.script失败: {e}")
        # ----------- 后续推理用 self.transformer_model 即可 -----------
        
        # 检查模型和必需的 Scaler/特征列表是否已加载
        if self.transformer_model and self.feature_scaler and self.target_scaler and self.selected_feature_names_for_transformer:
            try:
                logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型和转换器已加载，开始进行预测...")
                # --- 应用特征工程管道到最新数据窗口 ---
                # 提取用于预测的最新数据窗口 (最后 window_size 行)
                latest_data_window = processed_data.tail(self.transformer_window_size).copy() # 修改行：提取最新窗口数据
                # 应用特征工程管道
                processed_data_for_prediction = self._apply_feature_engineering_pipeline(latest_data_window) # 修改行：调用新的辅助方法
                if processed_data_for_prediction is None or processed_data_for_prediction.empty:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 应用特征工程管道后数据无效或为空，无法进行 Transformer 预测。")
                else:
                    # 再次调用 predict_with_transformer_model，这次传入的是经过管道处理的 DataFrame
                    # 注意：predict_with_transformer_model 内部会取这个 DataFrame 的最后 window_size 行
                    # 但我们的 processed_data_for_prediction 已经是最后 window_size 行经过处理的结果
                    # 所以这里传入 processed_data_for_prediction 即可
                    predicted_signal_value = predict_with_transformer_model( # 修改行：接收单个预测值
                        model=self.transformer_model,
                        data=processed_data_for_prediction, # 传入经过特征工程管道处理后的数据 (一个窗口)
                        feature_scaler=self.feature_scaler,
                        target_scaler=self.target_scaler,
                        selected_feature_names=self.selected_feature_names_for_transformer,
                        window_size=self.transformer_window_size,
                        device=self.device,
                    )
                    # 将预测的单个信号值赋给 processed_data 的最后一行
                    if not processed_data.empty and 'transformer_signal' in processed_data.columns:
                        # 预测值对应于输入窗口的最后一个时间步
                        latest_index = processed_data.index[-1]
                        processed_data.loc[latest_index, 'transformer_signal'] = predicted_signal_value
                        logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型预测完成，最新预测信号 ({latest_index}): {predicted_signal_value:.2f}")
                    else:
                        logger.warning(f"[{self.strategy_name}][{stock_code}] processed_data 为空或缺少 'transformer_signal' 列，无法记录 Transformer 预测结果。")
            except Exception as e:
                logger.error(f"[{self.strategy_name}][{stock_code}] Transformer 模型预测出错: {e}", exc_info=True)
                # 预测出错时，transformer_signal 列保持默认值 50.0
        # else:
            # logger.warning(f"[{self.strategy_name}][{stock_code}] Transformer 模型/Scaler/特征列表未加载，跳过 Transformer 预测。Transformer_signal 将保持默认值 50.0。")
        # --- 阶段 3: 组合规则信号和 Transformer 信号 ---
        logger.debug(f"[{self.strategy_name}][{stock_code}] 组合规则信号和 Transformer 信号...")
        try:
            # 获取信号组合权重
            # 从 tf_params 中获取 signal_combination_weights
            signal_combination_weights = self.tf_params.get('signal_combination_weights', {})
            if not isinstance(signal_combination_weights, dict) or not signal_combination_weights:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 'signal_combination_weights' 参数无效或为空，使用默认权重 {{'rule_weight': 0.6, 'transformer_weight': 0.4}}。")
                signal_combination_weights = {'rule_weight': 0.6, 'transformer_weight': 0.4}
            # 确保权重归一化
            self._normalize_weights(signal_combination_weights)
            rule_weight = signal_combination_weights.get('rule_weight', 0)
            transformer_weight = signal_combination_weights.get('transformer_weight', 0)
            # 确保所需的信号列存在，如果不存在则用 50 填充 (对于 0-100 分数)
            rule_signal_col = 'final_rule_signal'
            transformer_signal_col = 'transformer_signal'
            if rule_signal_col not in processed_data.columns:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 规则信号列 ('{rule_signal_col}') 不存在，组合信号时将其视为 50。")
                processed_data[rule_signal_col] = 50.0
            # Transformer 信号已经在前面添加并默认填充 50.0
            # 计算组合信号
            # 使用 fillna(50.0) 确保计算过程中没有 NaN 影响，并且对齐索引
            processed_data['combined_signal'] = (
                processed_data[rule_signal_col].fillna(50.0) * rule_weight +
                processed_data[transformer_signal_col].fillna(50.0) * transformer_weight
            )
            # 确保组合信号在 0-100 范围内并四舍五入
            processed_data['combined_signal'] = processed_data['combined_signal'].clip(0, 100).round(2)
            latest_combined_signal_val = processed_data['combined_signal'].iloc[-1] if not processed_data.empty and 'combined_signal' in processed_data.columns else np.nan
            logger.info(f"[{self.strategy_name}][{stock_code}] 组合信号计算完成 (权重: 规则={rule_weight:.2f}, Transformer={transformer_weight:.2f})，最新值: {latest_combined_signal_val:.2f}.")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 组合规则信号和 Transformer 信号出错: {e}", exc_info=True)
            # 出现错误时，添加默认的组合信号列
            processed_data['combined_signal'] = 50.0 # 默认中性
        # --- 阶段 4: 生成最终交易信号 (例如，[-1, 0, 1]) ---
        # 这个最终信号通常是基于 combined_signal 的离散值
        logger.debug(f"[{self.strategy_name}][{stock_code}] 生成最终交易信号...")
        try:
            # 可以基于 combined_signal 设定阈值来生成最终信号
            buy_threshold = self.tf_params.get('final_signal_buy_threshold', 55.0) # 默认值 55
            sell_threshold = self.tf_params.get('final_signal_sell_threshold', 45.0) # 默认值 45
             # 确保阈值是有效数字且 sell_threshold <= buy_threshold
            if not isinstance(buy_threshold, (int, float)): buy_threshold = 55.0
            if not isinstance(sell_threshold, (int, float)): sell_threshold = 45.0
            if sell_threshold > buy_threshold:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 最终信号阈值配置异常: sell_threshold ({sell_threshold}) > buy_threshold ({buy_threshold}). 调整 sell_threshold。")
                sell_threshold = buy_threshold - 10 # 确保有一个间隔
            processed_data['signal'] = 0 # 最终交易信号列 (内部列)，默认信号为 0 (观望)
            # 确保 combined_signal 列存在且不是全 NaN
            combined_signal_col = 'combined_signal'
            if combined_signal_col in processed_data.columns and not processed_data[combined_signal_col].isnull().all():
                # 使用填充后的组合信号进行阈值判断
                combined_signal_filled = processed_data[combined_signal_col].fillna(50.0)
                processed_data.loc[combined_signal_filled >= buy_threshold, 'signal'] = 1 # 买入信号
                processed_data.loc[combined_signal_filled <= sell_threshold, 'signal'] = -1 # 卖出信号
                # 考虑 A 股 T+1 交易限制，可能需要额外的逻辑来处理卖出信号
                # 例如，只有在持有仓位时，-1 信号才有效
                # 这部分逻辑通常在回测或实时交易执行模块中处理，策略只负责生成意向信号。
                # 如果策略需要直接输出可执行的交易指令 (如 'BUY', 'SELL', 'HOLD'), 则在这里转换
                logger.info(f"[{self.strategy_name}][{stock_code}] 最终交易信号生成完成。")
            else:
                logger.error(f"[{self.strategy_name}][{stock_code}] 组合信号列 ('{combined_signal_col}') 不存在或全为 NaN，无法生成最终交易信号！信号列将全为 0。")
                # processed_data['signal'] 已经默认为 0
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 生成最终交易信号出错: {e}", exc_info=True)
            # 出现错误时，也添加一个全为 0 的信号列
            processed_data['signal'] = 0
        # --- 阶段 5: 存储中间数据并返回结果 ---
        self.intermediate_data = processed_data.copy()
        if not self.intermediate_data.empty:
            latest_signal_val = self.intermediate_data['signal'].iloc[-1] if 'signal' in self.intermediate_data.columns else 0
            logger.info(f"[{self.strategy_name}][{stock_code}] 信号生成流程完成。最新最终信号: {latest_signal_val}.")
        else:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 信号生成流程未产生有效数据 (intermediate_data 为空)。")
        # 返回包含所有中间计算和最终信号的 DataFrame
        return self.intermediate_data

    # 负责在预测时对最新的数据窗口应用已拟合的转换器
    def _apply_feature_engineering_pipeline(self, raw_data_window: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        对原始数据窗口应用特征工程管道，返回处理后的特征DataFrame。
        包括NaN填充、根据训练时特征名筛选列。
        """
        if raw_data_window is None or raw_data_window.empty:
            logger.warning(f"[{self.strategy_name}] 输入数据为空，无法应用特征工程管道。")
            return None
        processed_df = raw_data_window.copy()
        # 1. 处理NaN，先前向填充，再后向填充，最后用0填充
        if processed_df.isnull().any().any():
            processed_df = processed_df.ffill().bfill().fillna(0)
            if processed_df.isnull().any().any():
                logger.error(f"[{self.strategy_name}] NaN填充后仍存在缺失值，无法继续处理。")
                return None
        print(f"[{self.strategy_name}] NaN填充后数据维度: {processed_df.shape}") # ADDED LINE: 调试信息
        # 2. 根据训练时保存的特征名列表筛选列，确保顺序和数量一致
        # 确保 self.selected_feature_names_for_transformer 已加载
        if not hasattr(self, 'selected_feature_names_for_transformer') or not self.selected_feature_names_for_transformer:
            logger.error(f"[{self.strategy_name}] 未加载训练时特征名列表，无法进行特征筛选。请确保已调用 _load_selected_feature_names。") # MODIFIED LINE: 提示用户加载特征名
            return None
        missing_features = set(self.selected_feature_names_for_transformer) - set(processed_df.columns)
        if missing_features:
            logger.error(f"[{self.strategy_name}] 缺少训练时特征列: {missing_features}")
            return None
        # 严格按照训练时顺序筛选列
        processed_df = processed_df[self.selected_feature_names_for_transformer]
        print(f"[{self.strategy_name}] 根据JSON特征列表筛选后数据维度: {processed_df.shape}") # ADDED LINE: 调试信息
        # 3. 转为numpy数组，类型为float32，方便后续模型处理
        current_features_np = processed_df.values.astype(np.float32)
        # 4. 确保输出是二维数组
        if current_features_np.ndim == 1:
            current_features_np = current_features_np.reshape(1, -1)
        # 5. 构建最终DataFrame，列名用训练时特征名，索引与输入数据一致
        processed_df_final = pd.DataFrame(current_features_np, columns=self.selected_feature_names_for_transformer, index=raw_data_window.index)
        print(f"[{self.strategy_name}] 最终处理后的特征DataFrame维度: {processed_df_final.shape}") # ADDED LINE: 调试信息
        return processed_df_final

    # --- 为 Transformer 训练准备数据子集 ---
    def _prepare_transformer_training_data_subset(self, data: pd.DataFrame, stock_code: Optional[str] = None, indicator_configs: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        从完整的 OHLCV+指标数据中提取和计算用于 Transformer 训练的特征和目标列。
        此方法旨在减少内存使用，只返回训练所需的最小数据集。
        逻辑基于 generate_signals 方法中数据准备的前半部分。
        """
        logger.info(f"[{self.strategy_name}][{stock_code}] 开始提取 Transformer 训练数据子集...")

        if data is None or data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] _prepare_transformer_training_data_subset: 输入数据为空。")
            return pd.DataFrame()

        # 1. 计算规则信号的中间结果
        # 这是必要的，因为 Transformer 的特征和目标可能依赖于这些结果。
        # 假设 _calculate_rule_based_signal 返回 final_rule_signal (Series) 和 intermediate_results_dict (Dict[str, Union[pd.DataFrame, pd.Series]])
        logger.info(f"[{self.strategy_name}][{stock_code}] 计算规则信号中间数据以确定特征和目标...")
        try:
            final_rule_signal, intermediate_results_dict = self._calculate_rule_based_signal(
                data=data, stock_code=stock_code, indicator_configs=indicator_configs
            )
            logger.info(f"[{self.strategy_name}][{stock_code}] 规则信号中间数据计算完成。")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 计算规则信号中间数据时出错: {e}", exc_info=True)
            return pd.DataFrame() # 计算失败则返回空 DataFrame

        # 2. 构建一个包含所有潜在特征和目标列的临时 DataFrame
        # 这个临时 DataFrame 包含了原始数据、final_rule_signal 和所有中间结果。
        # 这是为了方便后续根据列名选取特征和目标，模拟 generate_signals 构建 processed_data 的过程（但不包含预测和组合信号）。
        temp_data_container = data.copy() # 复制原始数据作为容器
        temp_data_container['final_rule_signal'] = final_rule_signal # 添加 final_rule_signal

        # 合并 intermediate_results_dict 中的 DataFrame/Series 到临时容器
        for key, df_or_series in intermediate_results_dict.items():
            if isinstance(df_or_series, pd.DataFrame) and not df_or_series.empty:
                # 合并 DataFrame 的所有列
                for col_join in df_or_series.columns:
                    # 避免覆盖原始数据或已有的中间列，除非是特定的内部列需要更新
                    # 假设 strategy_utils 返回的 DataFrame 列名是最终想要的列名（除了 ADJUSTED_SCORE）
                    if col_join != 'ADJUSTED_SCORE' and col_join not in temp_data_container.columns:
                        temp_data_container[col_join] = df_or_series[col_join]
                    elif col_join == 'ADJUSTED_SCORE':
                        # 特殊处理 ADJUSTED_SCORE，将其值赋值给 'base_score_volume_adjusted' 列 (如果还没有的话)
                        if 'base_score_volume_adjusted' not in temp_data_container.columns:
                            temp_data_container['base_score_volume_adjusted'] = df_or_series[col_join]
                        else:
                            logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果列 'ADJUSTED_SCORE' 已存在于 temp_data_container['base_score_volume_adjusted']，跳过合并。")
                    else:
                        logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果列 '{col_join}' 已存在于 temp_data_container，跳过合并来自 '{key}' 的同名列。")

            elif isinstance(df_or_series, pd.Series) and not df_or_series.empty:
                # 合并 Series
                # 如果 Series 有 name 属性，使用 name 作为列名
                series_col_name = df_or_series.name
                if series_col_name and series_col_name not in temp_data_container.columns:
                    temp_data_container[series_col_name] = df_or_series
                # 如果 Series 没有 name，或者 name 已存在，可以考虑使用 dict 的 key 作为列名（需要确保不冲突）
                elif not series_col_name and key not in temp_data_container.columns:
                    temp_data_container[key] = df_or_series
                else:
                    logger.debug(f"[{self.strategy_name}][{stock_code}] 中间结果Series '{series_col_name or key}' 已存在或无名，跳过合并。")

        logger.info(f"[{self.strategy_name}][{stock_code}] 临时数据容器构建完成，形状: {temp_data_container.shape}")
        print(f"[{self.strategy_name}][{stock_code}] temp_data_container 内存使用 (MB): {temp_data_container.memory_usage(deep=True).sum() / 1024**2:.2f}") # 打印内存使用

        # 3. 确定 Transformer 的目标列
        target_column = self.transformer_target_column # 从策略属性获取目标列名

        if target_column not in temp_data_container.columns:
            # 如果目标列不存在，可能是因为它需要在这里计算（例如，未来收益率）
            # 或者配置错误。根据原 generate_signals 逻辑，目标列应该在规则计算后就存在。
            # 如果目标列是未来值，它可能需要在规则计算后，但在选取特征/目标前计算。
            # 假设目标列是基于 temp_data_container 中的列计算的，例如未来收益率。
            # **重要：** 这里的计算逻辑需要根据您的实际策略定义 Transformer 目标的方式来实现。
            # 示例：计算未来 N 天的收盘价变化率作为目标 (如果 target_column 是 'future_return')
            # 如果您的目标列是 'future_return' 并且它不在 temp_data_container 中，您需要在这里计算它。
            # 例如：
            # if target_column == 'future_return' and 'close' in temp_data_container.columns:
            #     future_period = self.tf_params.get('target_future_period', 5)
            #     temp_data_container[target_column] = temp_data_container['close'].pct_change(periods=future_period).shift(-future_period)
            #     logger.info(f"[{self.strategy_name}][{stock_code}] 计算目标列 '{target_column}' 完成 (未来 {future_period} 天收益率)。")
            # else:
            #     logger.error(f"[{self.strategy_name}][{stock_code}] Transformer 目标列 '{target_column}' 不存在于临时数据中，且无法计算。可用列: {temp_data_container.columns.tolist()}")
            #     del temp_data_container # 释放内存
            #     return pd.DataFrame() # 无法确定目标列，返回空 DataFrame

            # 严格按照原 generate_signals 逻辑，它假设目标列在合并所有中间结果后就存在。
            # 因此，如果此时目标列仍不存在，说明规则计算或配置有问题。
            logger.error(f"[{self.strategy_name}][{stock_code}] Transformer 目标列 '{target_column}' 不存在于临时数据容器中。请检查规则计算或目标列配置。可用列: {temp_data_container.columns.tolist()}")
            del temp_data_container # 释放内存
            return pd.DataFrame() # 无法确定目标列，返回空 DataFrame

        logger.info(f"[{self.strategy_name}][{stock_code}] 确定 Transformer 目标列: '{target_column}'.")

        # 4. 确定 Transformer 的特征列
        # 根据原 Celery 任务逻辑，特征列是 generate_signals 输出 DataFrame 中除了目标列之外的所有列。
        # 在这里，这个 DataFrame 对应于我们的 temp_data_container。
        transformer_feature_columns = [col for col in temp_data_container.columns if col != target_column]

        if not transformer_feature_columns:
            logger.error(f"[{self.strategy_name}][{stock_code}] 未找到用于 Transformer 训练的特征列 (目标列 '{target_column}' 是唯一列)。")
            del temp_data_container # 释放内存
            return pd.DataFrame()

        logger.info(f"[{self.strategy_name}][{stock_code}] 确定 {len(transformer_feature_columns)} 个 Transformer 特征列。")
        # logger.debug(f"[{self.strategy_name}][{stock_code}] 特征列 (部分): {transformer_feature_columns[:10]}...") # 调试时打印

        # 5. 构建包含特征和目标列的精简 DataFrame
        columns_for_transformer_prep = transformer_feature_columns + [target_column]

        # 创建精简后的 DataFrame，只选取需要的列
        data_subset = temp_data_container[columns_for_transformer_prep].copy() # 创建一个只包含必要列的副本

        # 显式删除不再需要的临时 DataFrame，释放内存
        del temp_data_container # 显式删除临时 DataFrame
        print(f"[{self.strategy_name}][{stock_code}] 已删除 temp_data_container。") # 打印删除信息

        # 6. 移除包含 NaN 的行，特别是目标列中的 NaN
        initial_rows = len(data_subset)
        # prepare_data_for_transformer 会处理 NaN，但移除目标列为 NaN 的行是必要的，因为这些样本无法用于训练
        data_subset.dropna(subset=[target_column], inplace=True) # 移除目标列为 NaN 的行
        rows_after_dropna = len(data_subset)
        if rows_after_dropna < initial_rows:
             logger.warning(f"[{self.strategy_name}][{stock_code}] 移除了 {initial_rows - rows_after_dropna} 行目标列为 NaN 的数据。剩余 {rows_after_dropna} 行。")

        if data_subset.empty:
             logger.error(f"[{self.strategy_name}][{stock_code}] 移除目标列 NaN 后，数据子集为空。无法进行训练。")
             return pd.DataFrame()

        logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 训练数据子集提取完成。最终形状: {data_subset.shape}")
        # print(f"[{self.strategy_name}][{stock_code}] data_subset 数据类型:\n{data_subset.dtypes}") # 打印数据类型
        print(f"[{self.strategy_name}][{stock_code}] data_subset 内存使用 (MB): {data_subset.memory_usage(deep=True).sum() / 1024**2:.2f}") # 打印内存使用

        return data_subset

    # 将 load_lstm_model 更名为 load_transformer_model
    def load_transformer_model(self, stock_code: str):
        """
        为特定股票加载 Transformer 模型权重和 scaler。
        """
        # 检查模型和 scaler 的路径是否已设置
        if not all([self.model_path, self.feature_scaler_path, self.target_scaler_path, self.selected_features_path]):
            logger.error(f"[{self.strategy_name}][{stock_code}] Transformer 模型或Scaler路径未正确设置，无法加载。")
            self._reset_model_components() # 重置状态
            return

        # 检查模型和 scaler 文件是否存在
        required_files_exist = all([
            os.path.exists(self.model_path),
            os.path.exists(self.feature_scaler_path),
            os.path.exists(self.target_scaler_path),
            os.path.exists(self.selected_features_path)
        ])

        if not required_files_exist:
            # logger.warning(f"[{self.strategy_name}][{stock_code}] 缺失必需的 Transformer 模型/Scaler/特征文件，无法加载。")
            self._reset_model_components() # 重置状态
            return

        try:
            # 加载选中特征列表
            with open(self.selected_features_path, 'r', encoding='utf-8') as f:
                self.selected_feature_names_for_transformer = json.load(f)
            logger.debug(f"[{self.strategy_name}][{stock_code}] 选中特征名列表 ({len(self.selected_feature_names_for_transformer)}个) 从 {self.selected_features_path} 加载。")

            current_num_features_from_json = len(self.selected_feature_names_for_transformer)
            if current_num_features_from_json == 0:
                 logger.error(f"[{self.strategy_name}][{stock_code}] 加载的选中特征列表为空，无法构建模型。")
                 self._reset_model_components() # 重置状态
                 return

            # --- 动态加载并推断已保存模型的参数 ---
            logger.debug(f"[{self.strategy_name}][{stock_code}] 尝试从 {self.model_path} 加载模型状态字典以推断参数。")
            # 1. 加载模型的 state_dict
            model_state_dict = torch.load(self.model_path, map_location=self.device)

            # 2. 定义辅助函数，从 state_dict 推断模型参数
            def _infer_model_params_from_state_dict(state_dict: dict) -> dict:
                inferred_params = {}
                # 推断 num_features (输入特征维度)
                if 'embedding.weight' in state_dict:
                    inferred_params['num_features'] = state_dict['embedding.weight'].shape[1]
                else:
                    logger.error(f"[{self.strategy_name}][{stock_code}] 无法从 state_dict 推断 'num_features' (缺少 'embedding.weight')。")
                    return {}

                # 推断 d_model (模型内部维度)
                if 'embedding.weight' in state_dict:
                    inferred_params['d_model'] = state_dict['embedding.weight'].shape[0]
                elif 'pos_encoder.pe' in state_dict: # 备用推断方式
                    inferred_params['d_model'] = state_dict['pos_encoder.pe'].shape[1]
                else:
                    logger.error(f"[{self.strategy_name}][{stock_code}] 无法从 state_dict 推断 'd_model'。")
                    return {}

                # 推断 nlayers (Transformer Encoder 层数)
                max_layer_idx = -1
                layer_pattern = re.compile(r'transformer_encoder\.layers\.(\d+)\..*')
                for key in state_dict.keys():
                    match = layer_pattern.match(key)
                    if match:
                        layer_idx = int(match.group(1))
                        if layer_idx > max_layer_idx:
                            max_layer_idx = layer_idx
                inferred_params['nlayers'] = max_layer_idx + 1 if max_layer_idx != -1 else 0
                if inferred_params['nlayers'] == 0:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 未在 state_dict 中找到 Transformer Encoder 层。")

                # 推断 dim_feedforward (前馈网络维度)
                # 检查是否存在第一层的 linear1.weight
                if f'transformer_encoder.layers.0.linear1.weight' in state_dict:
                    inferred_params['dim_feedforward'] = state_dict[f'transformer_encoder.layers.0.linear1.weight'].shape[0]
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 无法从 state_dict 推断 'dim_feedforward' (缺少 'transformer_encoder.layers.0.linear1.weight')。")
                    inferred_params['dim_feedforward'] = None # 如果无法推断，则设为 None，让 build_transformer_model 使用默认值或配置值

                # 推断 nhead (注意力头数)
                # nhead 无法直接从 state_dict 的键中推断，因为它不是一个权重或偏置的维度。
                # 它通常是 d_model / head_dim。
                # 我们可以尝试从 self_attn.in_proj_weight 的形状来间接推断 d_model，但 nhead 仍需假设。
                # 最佳实践是，如果 config 中有 nhead，就用 config 的；否则，根据 d_model 给出常见值。
                # 假设 in_proj_weight 的形状是 (3 * d_model, d_model)
                # 并且 d_model = nhead * head_dim
                # 我们可以尝试从 self_attn.out_proj.weight 的形状来推断 head_dim
                # out_proj.weight 的形状是 (d_model, d_model)
                # 这是一个复杂的问题，通常 nhead 是一个超参数，在模型训练时确定。
                # 鉴于之前的日志，d_model=64 时 nhead=8 是一个合理的推断。
                # 这里我们优先使用当前配置中的 nhead，如果配置中没有，则根据推断的 d_model 给出常见值。
                inferred_params['nhead'] = self.transformer_model_config.get('nhead')
                if inferred_params['nhead'] is None:
                    if inferred_params['d_model'] == 64:
                        inferred_params['nhead'] = 8
                    elif inferred_params['d_model'] == 128:
                        inferred_params['nhead'] = 8 # 也可以是 4
                    else:
                        inferred_params['nhead'] = 4 # 一个通用默认值
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 'nhead' 未在配置中找到，根据推断的 d_model={inferred_params['d_model']} 猜测为 {inferred_params['nhead']}。")

                return inferred_params

            # 3. 调用辅助函数推断已保存模型的参数
            saved_model_params = _infer_model_params_from_state_dict(model_state_dict)
            if not saved_model_params:
                logger.error(f"[{self.strategy_name}][{stock_code}] 无法从模型状态字典推断参数，加载失败。")
                self._reset_model_components()
                return

            # 4. 获取推断出的参数
            saved_model_num_features = saved_model_params.get('num_features')
            saved_model_d_model = saved_model_params.get('d_model')
            saved_model_nlayers = saved_model_params.get('nlayers')
            saved_model_dim_feedforward = saved_model_params.get('dim_feedforward')
            saved_model_nhead = saved_model_params.get('nhead')

            config_mismatch_detected = False

            # 5. 比较当前配置与推断出的参数，并进行调整
            # 调整 num_features
            if current_num_features_from_json != saved_model_num_features:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 特征数量不匹配: JSON配置 {current_num_features_from_json}，已保存模型 {saved_model_num_features}。将使用已保存模型的特征数量。")
                num_features_for_build = saved_model_num_features
                config_mismatch_detected = True
            else:
                num_features_for_build = current_num_features_from_json
            
            # 创建一个临时配置字典，用于构建模型，优先使用推断出的参数
            temp_model_config = self.transformer_model_config.copy()

            # 调整 d_model
            current_d_model = self.transformer_model_config.get('d_model')
            if current_d_model != saved_model_d_model:
                logger.warning(f"[{self.strategy_name}][{stock_code}] d_model 不匹配: 配置 {current_d_model}，已保存模型 {saved_model_d_model}。将使用已保存模型的 d_model。")
                temp_model_config['d_model'] = saved_model_d_model
                config_mismatch_detected = True
            
            # 调整 nlayers
            current_nlayers = self.transformer_model_config.get('nlayers')
            if current_nlayers != saved_model_nlayers:
                logger.warning(f"[{self.strategy_name}][{stock_code}] nlayers 不匹配: 配置 {current_nlayers}，已保存模型 {saved_model_nlayers}。将使用已保存模型的 nlayers。")
                temp_model_config['nlayers'] = saved_model_nlayers
                config_mismatch_detected = True
            
            # 调整 dim_feedforward (如果推断出且与当前配置不同)
            current_dim_feedforward = self.transformer_model_config.get('dim_feedforward')
            if saved_model_dim_feedforward is not None and current_dim_feedforward != saved_model_dim_feedforward:
                logger.warning(f"[{self.strategy_name}][{stock_code}] dim_feedforward 不匹配: 配置 {current_dim_feedforward}，已保存模型 {saved_model_dim_feedforward}。将使用已保存模型的 dim_feedforward。")
                temp_model_config['dim_feedforward'] = saved_model_dim_feedforward
                config_mismatch_detected = True
            
            # 调整 nhead (如果推断出且与当前配置不同，或当前配置中没有)
            current_nhead = self.transformer_model_config.get('nhead')
            if saved_model_nhead is not None and current_nhead != saved_model_nhead:
                logger.warning(f"[{self.strategy_name}][{stock_code}] nhead 不匹配: 配置 {current_nhead}，已保存模型 {saved_model_nhead}。将使用已保存模型的 nhead。")
                temp_model_config['nhead'] = saved_model_nhead
                config_mismatch_detected = True
            elif saved_model_nhead is not None and current_nhead is None:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 配置中缺少 nhead，将使用推断的 nhead: {saved_model_nhead}。")
                temp_model_config['nhead'] = saved_model_nhead
                config_mismatch_detected = True

            if config_mismatch_detected:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 注意: 模型配置已临时调整以匹配已保存的 .pth 文件。强烈建议更新策略参数文件和/或重新训练模型以保持一致性。")
            
            # 6. 使用调整后的参数构建模型
            self.transformer_model = build_transformer_model(
                num_features=num_features_for_build, # 使用修正后的 num_features
                model_config=temp_model_config, # 使用修正后的 model_config
                summary=False, # 加载模型时不打印摘要
                window_size=self.transformer_window_size
            )
            self.transformer_model.to(self.device)

            # 7. 加载模型权重 (现在模型结构与 state_dict 匹配，加载应该成功)
            self.transformer_model.load_state_dict(model_state_dict) # 使用已加载的 state_dict
            logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型权重从 {self.model_path} 加载成功。")

            # 加载 Scaler
            self.feature_scaler = joblib.load(self.feature_scaler_path)
            logger.info(f"[{self.strategy_name}][{stock_code}] 特征 Scaler 从 {self.feature_scaler_path} 加载成功。")

            self.target_scaler = joblib.load(self.target_scaler_path)
            logger.info(f"[{self.strategy_name}][{stock_code}] 目标 Scaler 从 {self.target_scaler_path} 加载成功。")

            # 确保所有组件都已成功加载
            if not all([self.transformer_model, self.feature_scaler, self.target_scaler, self.selected_feature_names_for_transformer]):
                 logger.warning(f"[{self.strategy_name}][{stock_code}] 加载 Transformer 模型或相关组件失败，部分对象为 None/空。")
                 self._reset_model_components() # 重置状态
            else:
                 # 设置模型为评估模式
                 self.transformer_model.eval()
                 logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型及相关组件加载完成并设置为评估模式。")

        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 加载 Transformer 模型或Scaler或特征列表出错: {e}", exc_info=True)
            self._reset_model_components() # 出现错误时重置状态
                        
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
                            selected_feature_names: List[str],
                            pca_model: Any = None, # 新增参数: PCA 模型，使用 Any 类型提示，并设置默认值为 None
                            scaler_for_pca: Any = None, # 新增参数: PCA 前使用的 Scaler，使用 Any 类型提示，并设置默认值为 None
                            feature_selector_model: Any = None): # 新增参数: 特征选择模型，使用 Any 类型提示，并设置默认值为 None
        """
        保存准备好的 Transformer 训练数据、Scaler、PCA 模型、PCA Scaler 和特征选择模型。
        """
        self.set_model_paths(stock_code)
        # 检查所有必要的路径是否已设置，包括新增的模型/Scaler路径
        # 增加对新增模型/Scaler路径的检查
        if not all([self.all_prepared_data_npz_path, self.feature_scaler_path, self.target_scaler_path,
                    self.selected_features_path, self.pca_model_path, self.scaler_for_pca_path,
                    self.feature_selector_model_path]):
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存准备数据：部分或全部路径未设置。")
            # 不是 raise 异常，而是返回 False 表示保存失败
            return False

        try:
            # 保存训练、验证、测试数据集
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

            # 保存特征和目标 Scaler
            joblib.dump(feature_scaler, self.feature_scaler_path)
            joblib.dump(target_scaler, self.target_scaler_path)
            # 更新日志信息以包含更多保存的对象
            logger.debug(f"[{self.strategy_name}][{stock_code}] Feature Scaler 和 Target Scaler 已保存。")

            # 保存 PCA 模型 (如果存在)
            # 新增行: 检查 pca_model 是否为 None，如果不为 None 则保存
            if pca_model is not None:
                joblib.dump(pca_model, self.pca_model_path)
                # 新增行: 记录 PCA 模型保存日志
                logger.debug(f"[{self.strategy_name}][{stock_code}] PCA 模型已保存到 {self.pca_model_path}。")

            # 保存 PCA 前使用的 Scaler (如果存在)
            # 新增行: 检查 scaler_for_pca 是否为 None，如果不为 None 则保存
            if scaler_for_pca is not None:
                 joblib.dump(scaler_for_pca, self.scaler_for_pca_path)
                 # 新增行: 记录 PCA Scaler 保存日志
                 logger.debug(f"[{self.strategy_name}][{stock_code}] PCA Scaler 已保存到 {self.scaler_for_pca_path}。")

            # 保存特征选择模型 (如果存在)
            # 新增行: 检查 feature_selector_model 是否为 None，如果不为 None 则保存
            if feature_selector_model is not None:
                 joblib.dump(feature_selector_model, self.feature_selector_model_path)
                 # 新增行: 记录特征选择模型保存日志
                 logger.debug(f"[{self.strategy_name}][{stock_code}] 特征选择模型已保存到 {self.feature_selector_model_path}。")

            # 保存选中特征名列表
            with open(self.selected_features_path, 'w', encoding='utf-8') as f:
                 json.dump(selected_feature_names, f, ensure_ascii=False, indent=4)
            logger.debug(f"[{self.strategy_name}][{stock_code}] 选中特征名列表已保存到 {self.selected_features_path}。")

            return True # 返回 True 表示保存成功
        except Exception as e:
            # 更新错误日志信息，包含新增的模型/Scaler
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存准备好的数据、Scaler、模型或特征列表时出错: {e}", exc_info=True)
            # 返回 False 表示保存失败
            return False

    def load_prepared_data(self, stock_code: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]], Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]]]:
        """
        从文件加载特定股票准备好的 Transformer 训练数据和 Scaler。
        同时尝试加载 PCA 模型、PCA 前缩放器和特征选择器模型（如果存在）。
        返回 NumPy 数组和 Scaler 对象。加载失败时返回空 NumPy 数组和 None。
        """
        self.set_model_paths(stock_code)
        empty_array = np.array([], dtype=np.float32) # 修改：指定 dtype 以匹配 prepare_data_for_transformer 的输出

        # 检查必需的文件路径是否已设置
        if not all([self.all_prepared_data_npz_path, self.feature_scaler_path, self.target_scaler_path, self.selected_features_path]):
            logger.warning(f"[{self.strategy_name}][{stock_code}] 加载准备数据：部分或全部必需路径未设置。")
            self.selected_feature_names_for_transformer = [] # 重置列表
            # 返回空 NumPy 数组和 None
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None

        # 检查必需的文件是否存在
        required_files_exist = all([
            os.path.exists(self.all_prepared_data_npz_path),
            os.path.exists(self.feature_scaler_path),
            os.path.exists(self.target_scaler_path),
            os.path.exists(self.selected_features_path)
        ])

        if not required_files_exist:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 缺失必需的准备数据/Scaler/特征文件，无法加载。")
            self.selected_feature_names_for_transformer = [] # 重置列表
             # 返回空 NumPy 数组和 None
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None

        # 初始化可选的转换器属性为 None
        self.pca_model = None
        self.scaler_for_pca = None
        self.feature_selector_model = None

        try:
            # --- 加载必需的数据和 Scaler ---
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

            # --- 尝试加载可选的特征工程转换器 ---
            # 加载 PCA 模型
            # 检查路径是否存在且有效
            if hasattr(self, 'pca_model_path') and self.pca_model_path and os.path.exists(self.pca_model_path): # 修改行：检查路径是否存在
                try:
                    self.pca_model = joblib.load(self.pca_model_path)
                    logger.info(f"[{self.strategy_name}][{stock_code}] PCA 模型从 {self.pca_model_path} 加载成功。") # 修改行：日志信息
                except Exception as e_pca_load:
                    logger.error(f"[{self.strategy_name}][{stock_code}] 加载 PCA 模型时出错: {e_pca_load}", exc_info=True) # 修改行：日志信息
                    self.pca_model = None # 加载失败则设为 None
            else:
                logger.debug(f"[{self.strategy_name}][{stock_code}] PCA 模型文件不存在或路径未设置，跳过加载。") # 修改行：日志信息级别和内容

            # 加载 PCA 前缩放器
            if hasattr(self, 'scaler_for_pca_path') and self.scaler_for_pca_path and os.path.exists(self.scaler_for_pca_path): # 修改行：检查路径是否存在
                try:
                    self.scaler_for_pca = joblib.load(self.scaler_for_pca_path)
                    logger.info(f"[{self.strategy_name}][{stock_code}] PCA 前缩放器从 {self.scaler_for_pca_path} 加载成功。") # 修改行：日志信息
                except Exception as e_scaler_pca_load:
                    logger.error(f"[{self.strategy_name}][{stock_code}] 加载 PCA 前缩放器时出错: {e_scaler_pca_load}", exc_info=True) # 修改行：日志信息
                    self.scaler_for_pca = None # 加载失败则设为 None
            else:
                logger.debug(f"[{self.strategy_name}][{stock_code}] PCA 前缩放器文件不存在或路径未设置，跳过加载。") # 修改行：日志信息级别和内容

            # 加载特征选择器模型
            if hasattr(self, 'feature_selector_model_path') and self.feature_selector_model_path and os.path.exists(self.feature_selector_model_path): # 修改行：检查路径是否存在
                try:
                    self.feature_selector_model = joblib.load(self.feature_selector_model_path)
                    logger.info(f"[{self.strategy_name}][{stock_code}] 特征选择器模型从 {self.feature_selector_model_path} 加载成功。") # 修改行：日志信息
                except Exception as e_fs_load:
                    logger.error(f"[{self.strategy_name}][{stock_code}] 加载特征选择器模型时出错: {e_fs_load}", exc_info=True) # 修改行：日志信息
                    self.feature_selector_model = None # 加载失败则设为 None
            else:
                logger.debug(f"[{self.strategy_name}][{stock_code}] 特征选择器模型文件不存在或路径未设置，跳过加载。") # 修改行：日志信息级别和内容


            # 检查加载的数据和 scaler 是否匹配 (维度检查)
            # 这个检查应该基于最终的特征维度，无论是否使用了可选的特征工程
            # 选定特征列表的长度应该与加载的训练集特征的列数匹配
            if features_train.shape[1] != len(self.selected_feature_names_for_transformer):
                 logger.error(f"[{self.strategy_name}][{stock_code}] 加载的数据特征维度 ({features_train.shape[1]}) 与选中特征列表长度 ({len(self.selected_feature_names_for_transformer)}) 不匹配！")
                 self.selected_feature_names_for_transformer = [] # 重置列表
                 # 加载失败，重置所有加载的属性
                 self.pca_model = None
                 self.scaler_for_pca = None
                 self.feature_selector_model = None
                 return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None # 返回 None 表示不匹配

            logger.debug(f"[{self.strategy_name}][{stock_code}] 准备好的数据、Scaler 和可选转换器已成功加载。")
            # 返回必需的数据和 Scaler
            return features_train, targets_train, features_val, targets_val, features_test, targets_test, feature_scaler, target_scaler
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 加载准备好的数据、Scaler或特征列表时发生错误: {e}", exc_info=True)
            self.selected_feature_names_for_transformer = [] # 重置列表
            # 加载失败，重置所有加载的属性
            self.pca_model = None
            self.scaler_for_pca = None
            self.feature_selector_model = None
            # 返回空 NumPy 数组和 None
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None

    def prepare_data_and_train(self, data: pd.DataFrame, stock_code: str, retrain: bool = False, indicator_configs: Optional[List[Dict]] = None):
        """
        准备数据并训练 Transformer 模型。

        Args:
            data (pd.DataFrame): 包含多时间框架 OHLCV 和指标数据的 DataFrame。
            stock_code (str): 股票代码。
            retrain (bool): 是否强制重新训练模型。如果为 False 且模型已存在，则跳过训练。
        """
        log_prefix = f"[{self.strategy_name}][{stock_code}]"
        self.set_model_paths(stock_code)

        # 检查模型是否已存在且不需要重新训练
        model_exists = os.path.exists(self.model_path) if self.model_path else False
        feature_scaler_exists = os.path.exists(self.feature_scaler_path) if self.feature_scaler_path else False
        target_scaler_exists = os.path.exists(self.target_scaler_path) if self.target_scaler_path else False
        selected_features_exists = os.path.exists(self.selected_features_path) if self.selected_features_path else False

        if model_exists and feature_scaler_exists and target_scaler_exists and selected_features_exists and not retrain:
            logger.info(f"{log_prefix} 模型和 Scalers 已存在，跳过数据准备和训练。")
            # 尝试加载它们，以便在 run 方法中使用 (这里实际上是在训练任务中调用，加载后用于评估等)
            try:
                 # 直接调用 load_prepared_data 加载 scaler 和 selected features
                 # 数据本身 (NPZ) 不需要在这里加载到内存，train_transformer_model_from_prepared_data 会按需加载
                 _, _, _, _, _, _, loaded_feature_scaler, loaded_target_scaler = self.load_prepared_data(stock_code)
                 if loaded_feature_scaler and loaded_target_scaler and self.selected_feature_names_for_transformer:
                     self.feature_scaler = loaded_feature_scaler
                     self.target_scaler = loaded_target_scaler
                     logger.info(f"{log_prefix} 已加载现有 Scalers 和特征列表。")
                     # 如果数据已准备好，直接进入训练阶段（即使不重新训练，也需要加载数据进行评估等）
                     self.train_transformer_model_from_prepared_data(stock_code) # 调用训练方法，它会从文件加载数据
                     return # 加载成功并尝试训练/评估后返回

                 else:
                      logger.warning(f"[{log_prefix}] 加载现有 Scalers/特征列表失败，即使文件存在。可能需要重新准备数据和训练。")
                      retrain = True # 强制重新训练
            except Exception as e:
                 logger.error(f"[{log_prefix}] 加载现有 Scalers/特征列表时发生错误: {e}", exc_info=True)
                 retrain = True # 强制重新训练


        if retrain or not (model_exists and feature_scaler_exists and target_scaler_exists and selected_features_exists):
            logger.info(f"[{log_prefix}] 开始准备 Transformer 模型训练数据...")

            # 确保输入数据 data 包含计算 target_column 所需的列以及所有潜在的特征列
            # Transformer 的 target_column 是 'final_rule_signal'，这需要通过规则计算得到
            # 所以在准备数据前，需要先运行规则信号计算逻辑

            if data is None or data.empty:
                 logger.error(f"[{log_prefix}] 输入数据为空，无法准备 Transformer 数据。")
                 return

            # 假设 indicator_configs 已经由外部提供或在此之前加载
            # 如果没有提供 indicator_configs，_calculate_rule_based_signal 内部会尝试使用默认配置或跳过计算
            # 为了确保 target_column 'final_rule_signal' 存在，我们需要在这里运行规则计算的一部分。

            logger.info(f"[{log_prefix}] 在准备数据前计算规则信号以生成目标列 '{self.transformer_target_column}'...")
            try:
                 # 调用规则信号计算，生成 final_rule_signal
                 # 需要传递 indicator_configs，这里假设 prepare_data_and_train 的调用者会提供
                 # 如果没有提供，可能会导致 _calculate_rule_based_signal 出错或生成默认信号
                 # 更好的做法是，由 IndicatorService 提供所有需要的指标和衍生特征列，然后在策略内部计算目标列
                 # 假设 prepare_data_and_train 接收到的 data 已经包含了足够计算 final_rule_signal 的所有基础指标和衍生特征

                 # 模拟获取 indicator_configs，如果外部未提供，尝试使用默认配置
                 if indicator_configs is None:
                     logger.warning(f"[{log_prefix}] 准备数据时未提供 indicator_configs，规则信号计算可能不完整。")
                     # 尝试从参数文件中加载 indicator_configs，但这通常是由 IndicatorService 动态生成的
                     # 这里的备用逻辑可能不健壮，依赖于 IndicatorService 的实际工作流程
                     # 暂时跳过从文件加载，接受 None 并依赖 strategy_utils 的内部处理
                     configs_for_rule_calc = None # 传递 None

                 else:
                      configs_for_rule_calc = indicator_configs # 使用传入的 configs

                 final_rule_signal_for_train, intermediate_results_dict_for_train = self._calculate_rule_based_signal(
                     data=data,
                     stock_code=stock_code,
                     indicator_configs=configs_for_rule_calc # 传递 configs
                 )

                 # 将计算出的 final_rule_signal 添加到 data 中作为目标列
                 data_with_target = data.copy()
                 data_with_target['final_rule_signal'] = final_rule_signal_for_train # 添加目标列

                 # 将中间结果中的 DataFrame/Series 也合并到 data_with_target 中，作为潜在特征
                 for key, df_or_series in intermediate_results_dict_for_train.items():
                    if isinstance(df_or_series, pd.DataFrame) and not df_or_series.empty:
                         for col_join in df_or_series.columns:
                             if col_join != 'ADJUSTED_SCORE' and col_join not in data_with_target.columns:
                                 data_with_target[col_join] = df_or_series[col_join]
                             elif col_join == 'ADJUSTED_SCORE':
                                 if 'base_score_volume_adjusted' not in data_with_target.columns:
                                      data_with_target['base_score_volume_adjusted'] = df_or_series[col_join]

                    elif isinstance(df_or_series, pd.Series) and not df_or_series.empty:
                         series_col_name = df_or_series.name
                         if series_col_name and series_col_name not in data_with_target.columns:
                              data_with_target[series_col_name] = df_or_series
                         elif not series_col_name and key not in data_with_target.columns:
                              data_with_target[key] = df_or_series

            except Exception as e:
                logger.error(f"[{log_prefix}] 规则信号计算出错，无法生成目标列 '{self.transformer_target_column}': {e}", exc_info=True)
                logger.error(f"[{log_prefix}] 数据准备和模型训练已中断。")
                return # 规则计算失败，无法生成目标列，中断训练流程

            # 调用 prepare_data_for_transformer 进行数据准备
            # prepare_data_for_transformer 会处理特征选择、标准化、窗口化、分割
            # 并返回 NumPy 数组以及训练过程中生成的 scalers 和 selected_feature_names

            try:
                features_scaled_train_np, targets_scaled_train_np, \
                features_scaled_val_np, targets_scaled_val_np, \
                features_scaled_test_np, targets_scaled_test_np, \
                feature_scaler, target_scaler, selected_feature_names = prepare_data_for_transformer(
                    data_with_target, # 使用包含目标列和中间特征的 DataFrame
                    window_size=self.transformer_window_size,
                    target_column=self.transformer_target_column, # 指定目标列
                    data_prep_config=self.transformer_data_prep_config,
                    # 保存路径
                    save_npz_path=self.all_prepared_data_npz_path,
                    save_feature_scaler_path=self.feature_scaler_path,
                    save_target_scaler_path=self.target_scaler_path,
                    save_selected_features_path=self.selected_features_path,
                    # 参数传递
                    train_split=self.transformer_data_prep_config.get('train_split', 0.7),
                    val_split=self.transformer_data_prep_config.get('val_split', 0.15),
                    apply_variance_threshold=self.transformer_data_prep_config.get('apply_variance_threshold', False),
                    variance_threshold_value=self.transformer_data_prep_config.get('variance_threshold_value', 0.01),
                    use_pca=self.transformer_data_prep_config.get('use_pca', False),
                    pca_n_components=self.transformer_data_prep_config.get('pca_n_components', 0.99),
                    pca_solver=self.transformer_data_prep_config.get('pca_solver', 'auto'),
                    use_feature_selection=self.transformer_data_prep_config.get('use_feature_selection', True),
                    feature_selector_model_type=self.transformer_data_prep_config.get('feature_selector_model_type', 'rf'),
                    fs_model_n_estimators=self.transformer_data_prep_config.get('fs_model_n_estimators', 100),
                    fs_model_max_depth=self.transformer_data_prep_config.get('fs_model_max_depth'), # None is allowed
                    fs_max_features=self.transformer_data_prep_config.get('fs_max_features', None),
                    fs_selection_threshold=self.transformer_data_prep_config.get('fs_selection_threshold', 'median'),
                    target_scaler_type=self.transformer_data_prep_config.get('target_scaler_type', 'minmax')
                )

                self.feature_scaler = feature_scaler
                self.target_scaler = target_scaler
                self.selected_feature_names_for_transformer = selected_feature_names

                if features_scaled_train_np is None or features_scaled_train_np.shape[0] == 0 or \
                   targets_scaled_train_np is None or targets_scaled_train_np.shape[0] == 0 or \
                   feature_scaler is None or target_scaler is None or not self.selected_feature_names_for_transformer:
                     logger.error(f"[{log_prefix}] Transformer 数据准备失败或训练集数据无效/不足，无法继续训练。")
                     # 重置模型相关状态
                     self._reset_model_components()
                     return # 准备失败则停止

                logger.info(f"[{log_prefix}] Transformer 数据准备完成，保存到 '{self.all_prepared_data_npz_path}' 等文件。")

                # 调用训练方法
                logger.info(f"[{log_prefix}] 开始训练 Transformer 模型...")
                # train_transformer_model_from_prepared_data 会从文件加载 NPZ 数据并训练
                self.train_transformer_model_from_prepared_data(stock_code)

            except Exception as e:
                logger.error(f"[{log_prefix}] 准备 Transformer 数据或训练出错: {e}", exc_info=True)
                # 重置模型相关状态
                self._reset_model_components()

    def _calculate_trend_duration(self, data_with_signals: pd.DataFrame) -> Dict[str, Any]:
        """
        计算趋势的持续时间和强度，基于 'final_rule_signal' 列。
        使用JSON配置获取内部列名。
        深化：增加更详细的参数校验和调试信息，以及数据点数量检查。
        优化：在保证严格性的前提下，提升代码效率和清晰度。
        """
        # print(f"[{self.strategy_name}] 开始计算趋势持续时间。")
        trend_duration_info = {
            'bullish_duration': 0, 'bearish_duration': 0,
            'bullish_duration_text': '0分钟', 'bearish_duration_text': '0分钟',
            'current_trend': '中性', 'trend_strength': '不明', 'duration_status': '短'
        }

        # 获取 final_rule_signal 的列名，确保配置有效
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        if not isinstance(internal_cols_conf, list):
            logger.error(f"[{self.strategy_name}] NAMING_CONFIG 中 'strategy_internal_columns.output_columns' 配置无效，应为列表。")
            internal_cols_conf = []
        final_rule_signal_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c.get('name_pattern') == "final_rule_signal"), "final_rule_signal")

        # 严格检查输入数据框和信号列的有效性
        # 增加对输入DataFrame类型的检查
        if not isinstance(data_with_signals, pd.DataFrame) or data_with_signals.empty:
            logger.warning(f"[{self.strategy_name}] 输入数据框为空或不是有效的DataFrame，无法计算趋势持续时间。")
            print(f"[{self.strategy_name}] 输入数据无效，返回默认趋势信息。")
            return trend_duration_info
        
        if final_rule_signal_col not in data_with_signals.columns:
            logger.warning(f"[{self.strategy_name}] 规则信号列 '{final_rule_signal_col}' 不存在于数据中，无法计算趋势持续时间。")
            print(f"[{self.strategy_name}] 规则信号列 '{final_rule_signal_col}' 不存在，返回默认趋势信息。")
            return trend_duration_info

        # 提取信号序列并去除空值，转换为NumPy数组以提高迭代效率
        # 直接获取NumPy数组，避免Pandas Series的额外开销
        final_signal_values = data_with_signals[final_rule_signal_col].dropna().values
        
        if final_signal_values.size < 2: # 至少需要两个点才能判断趋势持续
            logger.warning(f"[{self.strategy_name}] 规则信号列 '{final_rule_signal_col}' 去除空值后数据点少于2个 ({final_signal_values.size}个)，无法有效计算趋势持续时间。")
            print(f"[{self.strategy_name}] 信号数据点过少，返回默认趋势信息。")
            return trend_duration_info

        # --- 获取并校验趋势判断阈值 ---
        # 使用局部变量存储阈值，避免直接修改 self.tf_params
        # 确保所有阈值都是数字类型，并提供默认值
        # 优化阈值获取和校验逻辑，使用更简洁的赋值方式
        thresholds_map = {
            'trend_confirmation_threshold_upper': 55,
            'trend_confirmation_threshold_lower': 45,
            'strong_bullish_threshold': 75,
            'strong_bearish_threshold': 25,
            'moderate_bullish_threshold': 60,
            'moderate_bearish_threshold': 40,
            'trend_duration_threshold_strong': 10,
            'trend_duration_threshold_moderate': 5,
            'trading_day_minutes': 240,
        }
        # 批量获取并校验阈值
        validated_params = {}
        for key, default_val in thresholds_map.items():
            value = self.tf_params.get(key, default_val)
            if not isinstance(value, (int, float)):
                logger.warning(f"[{self.strategy_name}] 参数 '{key}' 无效 ({value})，应为数字。将使用默认值 {default_val}。")
                value = default_val
            # 针对信号阈值，额外检查范围
            if key in ['trend_confirmation_threshold_upper', 'trend_confirmation_threshold_lower',
                       'strong_bullish_threshold', 'strong_bearish_threshold',
                       'moderate_bullish_threshold', 'moderate_bearish_threshold']:
                if not (0 <= value <= 100):
                    logger.warning(f"[{self.strategy_name}] 信号阈值 '{key}' ({value}) 超出合理范围 (0-100)。")
            # 针对持续时间阈值和交易日分钟数，额外检查是否大于0
            if key in ['trend_duration_threshold_strong', 'trend_duration_threshold_moderate', 'trading_day_minutes']:
                if value <= 0:
                    logger.warning(f"[{self.strategy_name}] 参数 '{key}' ({value}) 必须大于0。将使用默认值 {default_val}。")
                    value = default_val
            validated_params[key] = value

        # 将校验后的参数赋值给局部变量
        trend_threshold_upper = validated_params['trend_confirmation_threshold_upper']
        trend_threshold_lower = validated_params['trend_confirmation_threshold_lower']
        strong_bullish_threshold = validated_params['strong_bullish_threshold']
        strong_bearish_threshold = validated_params['strong_bearish_threshold']
        moderate_bullish_threshold = validated_params['moderate_bullish_threshold']
        moderate_bearish_threshold = validated_params['moderate_bearish_threshold']
        trend_duration_threshold_strong = validated_params['trend_duration_threshold_strong']
        trend_duration_threshold_moderate = validated_params['trend_duration_threshold_moderate']
        trading_day_minutes = validated_params['trading_day_minutes']

        # 确保强趋势持续时间阈值不小于中等趋势阈值
        if trend_duration_threshold_strong < trend_duration_threshold_moderate:
             logger.warning(f"[{self.strategy_name}] 趋势持续时间状态：强趋势阈值 ({trend_duration_threshold_strong}) 小于中等趋势阈值 ({trend_duration_threshold_moderate})。已调整强趋势阈值以确保逻辑正确。")
             trend_duration_threshold_strong = trend_duration_threshold_moderate + 1

        current_bullish_streak = 0
        current_bearish_streak = 0
        # 优化：直接迭代NumPy数组，避免Pandas Series的额外开销
        # 从最新数据向前计算连续周期数
        for signal_val in final_signal_values[::-1]: # 使用 [::-1] 创建反向视图，高效
            if signal_val >= trend_threshold_upper:
                current_bullish_streak += 1
                current_bearish_streak = 0
            elif signal_val <= trend_threshold_lower:
                current_bearish_streak += 1
                current_bullish_streak = 0
            else: # 信号回到中性区域，趋势中断
                # print(f"[{self.strategy_name}] 信号 {signal_val:.2f} 进入中性区域，趋势中断。")
                break

        trend_duration_info['bullish_duration'] = current_bullish_streak
        trend_duration_info['bearish_duration'] = current_bearish_streak

        # 将周期数转换为时间文本
        try:
            timeframe_minutes = int(self.focus_timeframe)
            bullish_total_minutes = current_bullish_streak * timeframe_minutes
            bearish_total_minutes = current_bearish_streak * timeframe_minutes

            def format_duration_with_trading_day(total_minutes: int, trading_day_minutes: int) -> str:
                """将总分钟数格式化为交易日、小时、分钟的字符串。"""
                if total_minutes == 0: return "0分钟"
                full_trading_days = total_minutes // trading_day_minutes
                remaining_minutes_after_days = total_minutes % trading_day_minutes
                parts = []
                if full_trading_days > 0: parts.append(f"{full_trading_days}交易日")
                if remaining_minutes_after_days > 0:
                    rem_hours, rem_minutes = divmod(remaining_minutes_after_days, 60)
                    if rem_hours > 0: parts.append(f"{rem_hours}小时")
                    if rem_minutes > 0: parts.append(f"{rem_minutes}分钟")
                if not parts and total_minutes > 0: parts.append(f"{total_minutes}分钟")
                return "".join(parts)

            def format_duration_simple(minutes: int) -> str:
                """将总分钟数格式化为天、小时、分钟的字符串。"""
                if minutes == 0: return "0分钟"
                if minutes < 60: return f"{minutes}分钟"
                hours, rem_minutes = divmod(minutes, 60)
                if hours < 24: return f"{hours}小时{rem_minutes}分钟" if rem_minutes else f"{hours}小时"
                else:
                    days, rem_hours = divmod(hours, 24)
                    parts = []
                    if days > 0: parts.append(f"{days}天")
                    if rem_hours > 0: parts.append(f"{rem_hours}小时")
                    if rem_minutes > 0 or (minutes > 0 and not parts): parts.append(f"{rem_minutes}分钟")
                    return "".join(parts)

            # 再次检查 trading_day_minutes，以防万一在函数内部被修改或传入无效值
            if trading_day_minutes <= 0:
                 logger.warning(f"[{self.strategy_name}] 趋势持续时间计算：'trading_day_minutes' 参数无效 ({trading_day_minutes})，使用简单天/小时/分钟格式。")
                 trend_duration_info['bullish_duration_text'] = format_duration_simple(bullish_total_minutes)
                 trend_duration_info['bearish_duration_text'] = format_duration_simple(bearish_total_minutes)
            else:
                trend_duration_info['bullish_duration_text'] = format_duration_with_trading_day(bullish_total_minutes, trading_day_minutes)
                # 修正 bearish_duration_text 的参数传递，确保使用 trading_day_minutes
                trend_duration_info['bearish_duration_text'] = format_duration_with_trading_day(bearish_total_minutes, trading_day_minutes)
                
            print(f"[{self.strategy_name}] 格式化后的看涨持续时间: {trend_duration_info['bullish_duration_text']}, 看跌持续时间: {trend_duration_info['bearish_duration_text']}。")

        except ValueError:
            logger.warning(f"[{self.strategy_name}] 趋势持续时间计算：无法将 focus_timeframe '{self.focus_timeframe}' 转换为分钟数 (非数字)。持续时间将以周期数显示。")
            trend_duration_info['bullish_duration_text'] = f"{current_bullish_streak}个周期"
            trend_duration_info['bearish_duration_text'] = f"{current_bearish_streak}个周期"
            print(f"[{self.strategy_name}] 格式化时间失败，使用周期数显示。")
        except Exception as e:
             logger.error(f"[{self.strategy_name}] 趋势持续时间计算：格式化持续时间时发生未知错误: {e}", exc_info=True)
             trend_duration_info['bullish_duration_text'] = f"{current_bullish_streak}个周期 (格式化错误)"
             trend_duration_info['bearish_duration_text'] = f"{current_bearish_streak}个周期 (格式化错误)"
             print(f"[{self.strategy_name}] 格式化时间发生未知错误。")

        # 判断当前趋势方向和强度
        # 直接从NumPy数组获取最后一个值，更高效
        latest_rule_signal_val = final_signal_values[-1]
        
        print(f"[{self.strategy_name}] 最新规则信号值: {latest_rule_signal_val:.2f}。")

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
        else:
             trend_duration_info.update({'current_trend': '中性', 'trend_strength': '不明'})
        print(f"[{self.strategy_name}] 判断当前趋势：方向='{trend_duration_info['current_trend']}', 强度='{trend_duration_info['trend_strength']}'。")

        # 判断趋势持续状态（短、中、长）
        current_duration_periods = max(current_bullish_streak, current_bearish_streak)
        if current_duration_periods >= trend_duration_threshold_strong:
             trend_duration_info['duration_status'] = '长'
        elif current_duration_periods >= trend_duration_threshold_moderate:
             trend_duration_info['duration_status'] = '中'
        else:
             trend_duration_info['duration_status'] = '短'

        return trend_duration_info

    def analyze_signals(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        分析趋势策略信号，生成解读和建议。
        使用JSON配置获取内部列名。
        优化：增加严格的输入检查，参数预加载，统一信号值获取，提升效率和清晰度。
        """
        # print(f"[{self.strategy_name}][{stock_code}] 开始分析信号。")
        stock_basic_dao = StockBasicInfoDao()
        stock = asyncio.run(stock_basic_dao.get_stock_by_code(stock_code))

        # 严格的输入检查
        if not isinstance(stock_code, str) or not stock_code.strip():
            logger.error(f"[{self.strategy_name}] 无效的股票代码: '{stock_code}'。")
            print(f"[{self.strategy_name}] 无效股票代码，返回 None。")
            return None
        if not isinstance(self.intermediate_data, pd.DataFrame) or self.intermediate_data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 中间数据为空或不是有效的DataFrame，无法进行信号分析。")
            print(f"[{self.strategy_name}][{stock_code}] 中间数据为空，返回 None。")
            return None

        analysis_results_dict = {}
        latest_data_row = self.intermediate_data.iloc[-1]
        # print(f"analyze_signals.latest_data_row: {latest_data_row}")
        # print(f"[{self.strategy_name}][{stock}] 获取最新数据行。")
        # 新增：用于记录每次add_signal_impact的详细影响
        signal_impact_records = []

        # 参数预加载，减少重复字典查找
        # 从 self.tf_params 中一次性获取所有需要的阈值和参数
        moderate_bullish_thresh = self.tf_params.get('moderate_bullish_threshold', 60)
        strong_bullish_thresh = self.tf_params.get('strong_bullish_threshold', 75)
        moderate_bearish_thresh =   self.tf_params.get('moderate_bearish_threshold', 40)
        strong_bearish_thresh = self.tf_params.get('strong_bearish_threshold', 25)
        trend_conf_upper_thresh = self.tf_params.get('trend_confirmation_threshold_upper', 55)
        trend_conf_lower_thresh = self.tf_params.get('trend_confirmation_threshold_lower', 45)
        stoch_oversold_thresh = self.tf_params.get('stoch_oversold_threshold', 20)
        stoch_overbought_thresh = self.tf_params.get('stoch_overbought_threshold', 80)
        vwap_deviation_thresh = self.tf_params.get('vwap_deviation_threshold', 0.01)
        volatility_thresh_high = self.tf_params.get('volatility_threshold_high', 0.03)
        volatility_thresh_low = self.tf_params.get('volatility_threshold_low', 0.01)
        adx_strong_thresh = self.tf_params.get('adx_strong_threshold', 30) # 用于ADX强度判断
        adx_moderate_thresh = self.tf_params.get('adx_moderate_threshold', 20) # 用于ADX强度判断
        # 新增多时间段分数变化相关参数的加载
        score_momentum_short_period = self.tf_params.get('score_momentum_short_period', 3)
        score_momentum_long_period = self.tf_params.get('score_momentum_long_period', 5)
        score_momentum_threshold_abs = self.tf_params.get('score_momentum_threshold_abs', 5)
        score_momentum_long_term_multiplier = self.tf_params.get('score_momentum_long_term_multiplier', 1.5)

        # print(f"[{self.strategy_name}][{stock_code}] 策略参数已加载。")

        # 获取策略内部列名，使用 JSON 配置
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        if not isinstance(internal_cols_conf, list):
            logger.error(f"[{self.strategy_name}][{stock_code}] NAMING_CONFIG 中 'strategy_internal_columns.output_columns' 配置无效，应为列表。")
            internal_cols_conf = []

        # 动态映射所有 NAMING_CONFIG 中的列名到实际数据中的列名
        actual_col_names = {}
        for item in internal_cols_conf:
            if isinstance(item, dict) and 'name_pattern' in item:
                pattern = item['name_pattern']
                actual_name = None
                if '{timeframe}' in pattern:
                    formatted_names = self._format_indicator_name(pattern, timeframe=self.focus_timeframe)
                    if formatted_names: actual_name = formatted_names[0]
                elif '{period}' in pattern:
                    # 尝试在数据中查找匹配该模式的列，取第一个找到的
                    found_period_col = None
                    prefix = pattern.replace('_{period}', '_')
                    for col in latest_data_row.index:
                        if col.startswith(prefix) and col[len(prefix):].isdigit():
                            found_period_col = col
                            break
                    actual_name = found_period_col
                else:
                    actual_name = pattern
                if actual_name and actual_name in latest_data_row:
                    actual_col_names[pattern] = actual_name
                else:
                    logger.debug(f"[{self.strategy_name}][{stock}] 内部列模式 '{pattern}' (或其格式化名称 '{actual_name}') 未在数据中找到。")
        # print(f"[{self.strategy_name}][{stock}] 动态获取内部列名完成。")

        # 从 actual_col_names 字典中安全获取列名，如果不存在则使用默认字符串
        def get_col_name(pattern: str, default_str: str) -> str:
            return actual_col_names.get(pattern, default_str)

        # 统一获取所有信号值，使用 .get() 避免 KeyError
        # 获取所有需要分析的信号值对应的实际列名
        combined_signal_col = get_col_name("combined_signal", "combined_signal")
        final_rule_signal_col = get_col_name("final_rule_signal", "final_rule_signal")
        transformer_signal_col = get_col_name("transformer_signal", "transformer_signal")
        base_score_raw_col = get_col_name("base_score_raw", "base_score_raw")
        base_score_volume_adjusted_col = get_col_name("ADJUSTED_SCORE", "ADJUSTED_SCORE")
        alignment_signal_col = get_col_name("alignment_signal", "alignment_signal")
        long_term_context_col = get_col_name("long_term_context", "long_term_context")
        adx_strength_signal_col = get_col_name("adx_strength_signal", "adx_strength_signal") # adx_strength_signal 列名
        stoch_signal_col = get_col_name("stoch_signal", "stoch_signal")
        has_bearish_div_col = get_col_name("HAS_BEARISH_DIVERGENCE", "HAS_BEARISH_DIVERGENCE")
        has_bullish_div_col = get_col_name("HAS_BULLISH_DIVERGENCE", "HAS_BULLISH_DIVERGENCE")
        vol_spike_signal_col = get_col_name("VOL_SPIKE_SIGNAL_{timeframe}", f"VOL_SPIKE_SIGNAL_{self.focus_timeframe}") # 确保默认值也带时间后缀
        volatility_col = get_col_name("score_volatility", "score_volatility")
        ema_score_col = next((col for col in latest_data_row.index if col.startswith('ema_score_') and col[len('ema_score_'):].isdigit()), "ema_score_default")
        ema_cross_signal_col = get_col_name("ema_cross_signal", "ema_cross_signal")
        ema_strength_col = get_col_name("ema_strength", "ema_strength")
        score_momentum_col = get_col_name("score_momentum", "score_momentum")
        score_momentum_acceleration_col = get_col_name("score_momentum_acceleration", "score_momentum_acceleration")
        volatility_signal_col = get_col_name("volatility_signal", "volatility_signal")
        vwap_deviation_signal_col = get_col_name("vwap_deviation_signal", "vwap_deviation_signal")
        vwap_deviation_percent_col = get_col_name("vwap_deviation_percent", "vwap_deviation_percent")
        boll_breakout_signal_col = get_col_name("boll_breakout_signal", "boll_breakout_signal")
        # 底部放量
        volume_breakout_signal_col = get_col_name("volume_breakout_signal", "volume_breakout_signal")
        bottom_volume_breakout_signal_col = get_col_name("bottom_volume_breakout_signal", "bottom_volume_breakout_signal")

        # 获取所有信号的最新值
        final_score_val = latest_data_row.get(combined_signal_col, 50.0)
        final_rule_score_val = latest_data_row.get(final_rule_signal_col, 50.0)
        transformer_score_val = latest_data_row.get(transformer_signal_col, 50.0)
        base_score_raw_val = latest_data_row.get(base_score_raw_col, np.nan)
        base_score_volume_adjusted_val = latest_data_row.get(base_score_volume_adjusted_col, np.nan)
        alignment_val = latest_data_row.get(alignment_signal_col, np.nan)
        long_term_context_val = latest_data_row.get(long_term_context_col, None)
        adx_strength_signal_val = latest_data_row.get(adx_strength_signal_col, np.nan) # adx_strength_signal_val 的赋值
        stoch_signal_val = latest_data_row.get(stoch_signal_col, np.nan)
        has_bearish_div_val = latest_data_row.get(has_bearish_div_col, False)
        has_bullish_div_val = latest_data_row.get(has_bullish_div_col, False)
        vol_spike_signal_val = latest_data_row.get(vol_spike_signal_col, np.nan)
        volatility_val = latest_data_row.get(volatility_col, np.nan)
        ema_score_val = latest_data_row.get(ema_score_col, np.nan)
        ema_cross_signal_val = latest_data_row.get(ema_cross_signal_col, np.nan)
        ema_strength_val = latest_data_row.get(ema_strength_col, np.nan)
        score_momentum_val = latest_data_row.get(score_momentum_col, np.nan)
        score_momentum_acceleration_val = latest_data_row.get(score_momentum_acceleration_col, np.nan)
        volatility_signal_val = latest_data_row.get(volatility_signal_col, np.nan)
        vwap_deviation_signal_val = latest_data_row.get(vwap_deviation_signal_col, np.nan)
        vwap_deviation_percent_val = latest_data_row.get(vwap_deviation_percent_col, np.nan)
        boll_breakout_signal_val = latest_data_row.get(boll_breakout_signal_col, np.nan)
        # 底部放量
        volume_breakout_signal_val = latest_data_row.get(volume_breakout_signal_col, np.nan)
        bottom_volume_breakout_signal_val = latest_data_row.get(bottom_volume_breakout_signal_col, np.nan)

        print(f"[{self.strategy_name}][{stock}] 最新信号值：组合={final_score_val:.2f}, 规则={final_rule_score_val:.2f}, Transformer={transformer_score_val:.2f}。")

        # 计算趋势持续时间
        trend_duration_info_dict = self._calculate_trend_duration(self.intermediate_data)
        analysis_results_dict.update(trend_duration_info_dict)
        # print(f"[{self.strategy_name}][{stock}] 趋势持续时间信息已计算。")

        signal_judgment_dict = {}
        operation_advice_str = "中性观望"
        risk_warning_str = ""
        t_plus_1_note_str = "（受 T+1 限制，建议次日操作）"

        duration_status_rule_str = trend_duration_info_dict.get('duration_status', '短')
        current_trend_direction = trend_duration_info_dict.get('current_trend', '中性')
        current_trend_strength = trend_duration_info_dict.get('trend_strength', '不明')
        is_in_bottom = trend_duration_info_dict.get('is_in_bottom', False)
        is_bottom_breakout = trend_duration_info_dict.get('is_bottom_breakout', False)

        print(f"[{self.strategy_name}][{stock}] 趋势持续状态: '{duration_status_rule_str}', 方向: '{current_trend_direction}', 强度: '{current_trend_strength}'。")

        # ================== 多变量深化判断与智能建议 ==================
        confidence_score = 0 # 初始信心分数，范围 -100 到 100

        # 辅助函数：用于统一添加信号判断结果、更新信心分数和风险提示
        def add_signal_impact(key: str, status: str, confidence_change: int, risk_msg: str = ""):
            nonlocal confidence_score, risk_warning_str
            signal_judgment_dict[key] = status
            confidence_score += confidence_change
            if risk_msg:
                risk_warning_str += risk_msg + " "
            signal_impact_records.append({
                "key": key,
                "status": status,
                "confidence_change": confidence_change,
                "risk_msg": risk_msg,
                "confidence_score_after": confidence_score
            })
            # print(f"[{self.strategy_name}][{stock}] 信号影响 - {key}: {status}, 信心变化: {confidence_change}, 当前信心: {confidence_score}。")

        # 1. 整体信号强度评估
        if final_score_val >= strong_bullish_thresh:
            add_signal_impact('overall_signal_strength', '非常强劲看涨', 30)
        elif final_score_val >= moderate_bullish_thresh:
            add_signal_impact('overall_signal_strength', '强劲看涨', 20)
        elif final_score_val >= trend_conf_upper_thresh:
            add_signal_impact('overall_signal_strength', '温和看涨', 10)
        elif final_score_val <= strong_bearish_thresh:
            add_signal_impact('overall_signal_strength', '非常强劲看跌', -30)
        elif final_score_val <= moderate_bearish_thresh:
            add_signal_impact('overall_signal_strength', '强劲看跌', -20)
        elif final_score_val <= trend_conf_lower_thresh:
            add_signal_impact('overall_signal_strength', '温和看跌', -10)
        else:
            add_signal_impact('overall_signal_strength', '中性', 0)

        # 2. 趋势持续时间对信心的影响
        if duration_status_rule_str == '长':
            if (current_trend_direction.startswith('看涨') and final_score_val > 50) or \
               (current_trend_direction.startswith('看跌') and final_score_val < 50):
                add_signal_impact('trend_duration_impact', '趋势持续且与信号一致', 10)
            else:
                add_signal_impact('trend_duration_impact', '趋势持续但与信号有分歧', -5, "长期趋势与当前信号方向不完全一致，注意风险。")
        elif duration_status_rule_str == '短':
            add_signal_impact('trend_duration_impact', '趋势刚启动或不稳定', -5, "趋势持续时间较短，信号可能不稳定。")
        else: # '中'
            add_signal_impact('trend_duration_impact', '趋势发展中', 0)

        # 3. 趋势分与量能分配合/背离
        print(f"[{self.strategy_name}][{stock}] 原始分: {base_score_raw_val:.2f}, 量能调整分: {base_score_volume_adjusted_val:.2f}。")
        if not np.isnan(base_score_raw_val) and not np.isnan(base_score_volume_adjusted_val):
            score_diff = base_score_volume_adjusted_val - base_score_raw_val
            if abs(score_diff) > 10:
                add_signal_impact('trend_volume_consistency', "分歧", -10, "基础趋势分与量能调整分分歧较大，趋势信号稳定性存疑。")
            else:
                add_signal_impact('trend_volume_consistency', "一致", 5)
            if score_diff > 8:
                add_signal_impact('volume_effect', "量能显著强化趋势", 15 if final_score_val > 50 else 0)
            elif score_diff < -8:
                add_signal_impact('volume_effect', "量能显著削弱趋势", -15 if final_score_val < 50 else 0)
            else:
                add_signal_impact('volume_effect', "量能影响一般", 0)

        # 4. 长期趋势与当前信号配合/背离
        # print(f"[{self.strategy_name}][{stock}] 长期趋势背景: '{long_term_context_val}'。")
        if long_term_context_val is not None:
            if long_term_context_val == "多头" and final_score_val < 50:
                add_signal_impact('long_term_vs_current', "多头背景下信号偏弱", -10, "长期趋势为多头，但当前信号偏弱，警惕短期回调风险。")
            elif long_term_context_val == "空头" and final_score_val > 50:
                add_signal_impact('long_term_vs_current', "空头背景下信号偏强", -10, "长期趋势为空头，但当前信号偏强，警惕反弹结束或诱多。")
            elif long_term_context_val == "多头" and final_score_val > 50:
                add_signal_impact('long_term_vs_current', "长期多头与当前信号一致", 10)
            elif long_term_context_val == "空头" and final_score_val < 50:
                add_signal_impact('long_term_vs_current', "长期空头与当前信号一致", 10)
            else:
                add_signal_impact('long_term_vs_current', f"长期趋势：{long_term_context_val}", 0)

        # 5. 动量指标 (STOCH) 与趋势信号共振/背离
        # print(f"[{self.strategy_name}][{stock}] STOCH 信号: {stoch_signal_val:.2f}, 超买阈值={stoch_overbought_thresh}, 超卖阈值={stoch_oversold_thresh}。")
        if not np.isnan(stoch_signal_val):
            if stoch_signal_val >= stoch_overbought_thresh and final_score_val >= moderate_bullish_thresh:
                add_signal_impact('stoch_trend_relation', "趋势与超买共振", -15, "趋势强劲但随机指标超买，短线追高风险较大，注意回调。")
            elif stoch_signal_val <= stoch_oversold_thresh and final_score_val <= moderate_bearish_thresh:
                add_signal_impact('stoch_trend_relation', "趋势与超卖共振", 15, "趋势疲软但随机指标超卖，短线反弹概率增加，可关注。")
            elif stoch_signal_val >= stoch_overbought_thresh and final_score_val < 50:
                add_signal_impact('stoch_trend_relation', "超买与趋势分背离", -20, "随机指标超买但趋势分偏弱，警惕假突破或诱多。")
            elif stoch_signal_val <= stoch_oversold_thresh and final_score_val > 50:
                add_signal_impact('stoch_trend_relation', "超卖与趋势分背离", -20, "随机指标超卖但趋势分偏强，警惕假反弹或洗盘。")
            else:
                add_signal_impact('stoch_trend_relation', "随机指标与趋势分无明显共振", 0)

        # 6. 量能异动与趋势信号配合/背离
        # print(f"[{self.strategy_name}][{stock}] 量能异动信号: {vol_spike_signal_val:.2f}。")
        if not np.isnan(vol_spike_signal_val):
            if vol_spike_signal_val > 0 and final_score_val >= moderate_bullish_thresh:
                add_signal_impact('vol_spike_trend', "量能异动强化看涨趋势", 10, "趋势强劲且伴随放量上涨，可能预示加速，但需注意追高风险。")
            elif vol_spike_signal_val < 0 and final_score_val <= moderate_bearish_thresh:
                add_signal_impact('vol_spike_trend', "量能异动强化看跌趋势", 10, "趋势疲软且伴随放量下跌，可能预示加速，但需注意抄底风险。")
            elif vol_spike_signal_val > 0 and final_score_val < 50:
                add_signal_impact('vol_spike_trend', "量能异动与弱趋势", -5, "趋势偏弱但出现放量，警惕反抽或短期诱多。")
            elif vol_spike_signal_val < 0 and final_score_val > 50:
                add_signal_impact('vol_spike_trend', "量能异动与强趋势", -5, "趋势偏强但出现放量下跌，警惕短期洗盘或回调。")
            elif vol_spike_signal_val != 0:
                add_signal_impact('vol_spike_trend', "量能异动", 0, "市场出现量能异动，短线波动可能加剧。")
            else:
                add_signal_impact('vol_spike_trend', "无量能异动", 0)
        else:
            add_signal_impact('vol_spike_trend', "量能异动数据缺失", 0)

        # 7. EMA 排列状态判断
        if not np.isnan(alignment_val):
             if alignment_val == 3: add_signal_impact('alignment_status', "完全多头排列", 15)
             elif alignment_val == -3: add_signal_impact('alignment_status', "完全空头排列", -15)
             elif alignment_val > 0: add_signal_impact('alignment_status', "多头排列形成中", 5)
             elif alignment_val < 0: add_signal_impact('alignment_status', "空头排列形成中", -5)
             else: add_signal_impact('alignment_status', "排列不明朗", -5, "EMA排列不明朗，市场可能处于震荡。")
        else:
             add_signal_impact('alignment_status', "数据缺失", 0)

        # 8. EMA 交叉信号判断
        if not np.isnan(ema_cross_signal_val):
            if ema_cross_signal_val == 1:
                if final_score_val > 50: add_signal_impact('ema_cross_status', "EMA金叉，看涨确认", 10)
                else: add_signal_impact('ema_cross_status', "EMA金叉，但主信号偏弱", 0, "EMA金叉但主信号未确认，可能为假突破。")
            elif ema_cross_signal_val == -1:
                if final_score_val < 50: add_signal_impact('ema_cross_status', "EMA死叉，看跌确认", -10)
                else: add_signal_impact('ema_cross_status', "EMA死叉，但主信号偏强", 0, "EMA死叉但主信号未确认，可能为假跌破。")
            else: add_signal_impact('ema_cross_status', "无EMA交叉", 0)
        else:
            add_signal_impact('ema_cross_status', "数据缺失", 0)

        # 9. EMA 强度判断
        if not np.isnan(ema_strength_val):
            if ema_strength_val > 0.02:
                add_signal_impact('ema_strength_status', "EMA多头强度强", 5)
            elif ema_strength_val < -0.02:
                add_signal_impact('ema_strength_status', "EMA空头强度强", -5)
            else:
                add_signal_impact('ema_strength_status', "EMA强度中性", 0)
        else:
            add_signal_impact('ema_strength_status', "数据缺失", 0)

        # 10. 信号动量和加速判断
        if not np.isnan(score_momentum_val):
            if score_momentum_val > 5: add_signal_impact('score_momentum_status', "信号分动量强劲向上", 5)
            elif score_momentum_val < -5: add_signal_impact('score_momentum_status', "信号分动量强劲向下", -5)
            else: add_signal_impact('score_momentum_status', "信号分动量中性", 0)
        else: add_signal_impact('score_momentum_status', "数据缺失", 0)
        if not np.isnan(score_momentum_acceleration_val):
            if score_momentum_acceleration_val > 2: add_signal_impact('score_momentum_acceleration_status', "信号分加速向上", 5)
            elif score_momentum_acceleration_val < -2: add_signal_impact('score_momentum_acceleration_status', "信号分加速向下", -5)
            else: add_signal_impact('score_momentum_acceleration_status', "信号分加速中性", 0)
        else: add_signal_impact('score_momentum_acceleration_status', "数据缺失", 0)

        # 11. 多时间段的分数变化
        # print(f"[{self.strategy_name}][{stock}] 分析信号分在多时间段的变化。")
        score_change_short = np.nan
        score_change_long = np.nan
        score_change_consistency_status = "数据不足"
        if combined_signal_col in self.intermediate_data.columns and len(self.intermediate_data) > 1:
            combined_signals = self.intermediate_data[combined_signal_col]
            if len(combined_signals) > score_momentum_short_period:
                score_val_short_ago = combined_signals.iloc[-(score_momentum_short_period + 1)]
                score_change_short = final_score_val - score_val_short_ago
                print(f"[{self.strategy_name}][{stock}] 短期 ({score_momentum_short_period}期) 信号分变化: {score_change_short:.2f}。")
                if score_change_short >= score_momentum_threshold_abs:
                    add_signal_impact('score_change_short_term', f"短期信号分强劲上涨 ({score_momentum_short_period}期)", 10)
                elif score_change_short <= -score_momentum_threshold_abs:
                    add_signal_impact('score_change_short_term', f"短期信号分强劲下跌 ({score_momentum_short_period}期)", -10)
                else:
                    add_signal_impact('score_change_short_term', f"短期信号分变化不大 ({score_momentum_short_period}期)", 0)
            else:
                add_signal_impact('score_change_short_term', "数据不足，无法计算短期信号分变化", 0)
            if len(combined_signals) > score_momentum_long_period:
                score_val_long_ago = combined_signals.iloc[-(score_momentum_long_period + 1)]
                score_change_long = final_score_val - score_val_long_ago
                print(f"[{self.strategy_name}][{stock}] 长期 ({score_momentum_long_period}期) 信号分变化: {score_change_long:.2f}。")
                long_term_threshold = score_momentum_threshold_abs * score_momentum_long_term_multiplier
                if score_change_long >= long_term_threshold:
                    add_signal_impact('score_change_long_term', f"长期信号分强劲上涨 ({score_momentum_long_period}期)", 15)
                elif score_change_long <= -long_term_threshold:
                    add_signal_impact('score_change_long_term', f"长期信号分强劲下跌 ({score_momentum_long_period}期)", -15)
                else:
                    add_signal_impact('score_change_long_term', f"长期信号分变化不大 ({score_momentum_long_period}期)", 0)
            else:
                add_signal_impact('score_change_long_term', "数据不足，无法计算长期信号分变化", 0)
            if not np.isnan(score_change_short) and not np.isnan(score_change_long):
                if (score_change_short > 0 and score_change_long > 0) or \
                   (score_change_short < 0 and score_change_long < 0):
                    add_signal_impact('score_change_consistency', "短期与长期信号分变化方向一致", 5)
                    score_change_consistency_status = "一致"
                else:
                    add_signal_impact('score_change_consistency', "短期与长期信号分变化方向不一致", -5, "信号分短期与长期变化方向不一致，趋势可能不稳定。")
                    score_change_consistency_status = "不一致"
            else:
                add_signal_impact('score_change_consistency', "数据不足，无法判断信号分变化一致性", 0)
        else:
            add_signal_impact('score_change_short_term', "数据不足", 0)
            add_signal_impact('score_change_long_term', "数据不足", 0)
            add_signal_impact('score_change_consistency', "数据不足", 0)

        # 12. VWAP 偏离信号判断
        if not np.isnan(vwap_deviation_signal_val) and not np.isnan(vwap_deviation_percent_val):
            if vwap_deviation_signal_val == 1:
                if vwap_deviation_percent_val > vwap_deviation_thresh and final_score_val > 60:
                    add_signal_impact('vwap_deviation_status', "价格偏离VWAP过高，短期超买", -5, "价格短期偏离VWAP过高，注意回调风险。")
                else: add_signal_impact('vwap_deviation_status', "价格在VWAP之上", 5)
            elif vwap_deviation_signal_val == -1:
                if vwap_deviation_percent_val < -vwap_deviation_thresh and final_score_val < 40:
                    add_signal_impact('vwap_deviation_status', "价格偏离VWAP过低，短期超卖", 5, "价格短期偏离VWAP过低，注意反弹机会。")
                else: add_signal_impact('vwap_deviation_status', "价格在VWAP之下", -5)
            else: add_signal_impact('vwap_deviation_status', "价格接近VWAP", 0)
        else:
            add_signal_impact('vwap_deviation_status', "数据缺失", 0)

        # 13. Bollinger Band 突破信号判断
        if not np.isnan(boll_breakout_signal_val):
            if boll_breakout_signal_val == 1:
                if final_score_val > 60: add_signal_impact('boll_breakout_status', "向上突破布林带上轨，趋势可能加速", 10)
                else: add_signal_impact('boll_breakout_status', "向上突破布林带上轨，但主信号偏弱", -5, "布林带向上突破但主信号未确认，警惕假突破。")
            elif boll_breakout_signal_val == -1:
                if final_score_val < 40: add_signal_impact('boll_breakout_status', "向下突破布林带下轨，趋势可能加速下跌", -10)
                else: add_signal_impact('boll_breakout_status', "向下突破布林带下轨，但主信号偏强", 5, "布林带向下突破但主信号未确认，警惕假跌破。")
            else: add_signal_impact('boll_breakout_status', "无布林带突破", 0)
        else:
            add_signal_impact('boll_breakout_status', "数据缺失", 0)

        # 14. 波动率信号判断
        if not np.isnan(volatility_signal_val):
            if volatility_signal_val == 1: add_signal_impact('volatility_signal_status', "高波动信号", -5, "波动率信号提示高波动，增加交易风险。")
            elif volatility_signal_val == -1: add_signal_impact('volatility_signal_status', "低波动信号", 5)
            else: add_signal_impact('volatility_signal_status', "中性波动信号", 0)
        else:
            add_signal_impact('volatility_signal_status', "数据缺失", 0)

        # 15. ADX 强度判断 (基于原始ADX列)
        param_sources = [self.tf_params, self.params.get('volume_confirmation', {}), self.params.get('indicator_analysis_params', {}), self.params.get('base_scoring', {})]
        dmi_period_bs = self._get_param_val(param_sources, 'dmi_period', 14)
        adx_col = f'ADX_{dmi_period_bs}_{self.focus_timeframe}'
        adx_val = latest_data_row.get(adx_col, np.nan)
        # print(f"[{self.strategy_name}][{stock}] 原始 ADX 值 (来自列 {adx_col}): {adx_val}。")
        if not np.isnan(adx_val):
             if adx_val >= adx_strong_thresh: add_signal_impact('adx_status', "趋势非常强劲 (原始ADX)", 10)
             elif adx_val >= adx_moderate_thresh: add_signal_impact('adx_status', "趋势强劲 (原始ADX)", 5)
             else: add_signal_impact('adx_status', "趋势较弱 (原始ADX)", -5, "原始ADX显示趋势强度不足，市场可能处于盘整状态。")
             if adx_val < adx_moderate_thresh and abs(final_score_val - 50) > 10:
                 risk_warning_str += f"原始ADX ({adx_val:.2f}) 显示趋势强度不足 ({adx_moderate_thresh})，当前信号偏离中性，注意假信号或震荡风险。 "
        else:
             add_signal_impact('adx_status', "原始ADX数据缺失", 0)

        # 16. 背离状态判断
        if has_bearish_div_val and final_score_val > 50:
            add_signal_impact('divergence_status', "检测到顶背离", -20, "检测到顶背离，强烈预示趋势衰竭或反转！")
        elif has_bullish_div_val and final_score_val < 50:
            add_signal_impact('divergence_status', "检测到底背离", 20, "检测到底背离，强烈预示趋势衰竭或反转！")
        else:
            add_signal_impact('divergence_status', "无明显背离", 0)

        # 17. Transformer 信号与规则信号的一致性
        if not np.isnan(transformer_score_val) and not np.isnan(final_rule_score_val):
            transformer_rule_diff = abs(transformer_score_val - final_rule_score_val)
            if transformer_rule_diff > 20:
                add_signal_impact('transformer_rule_consistency', "分歧较大", -15, "Transformer信号与规则信号存在较大分歧，模型预测风险较高。")
            elif transformer_rule_diff > 10:
                add_signal_impact('transformer_rule_consistency', "有分歧", -5, "Transformer信号与规则信号存在一定分歧，需谨慎。")
            else:
                add_signal_impact('transformer_rule_consistency', "一致", 5)

        # 18. EMA Score
        if not np.isnan(ema_score_val):
            if final_score_val > 60 and ema_score_val > 60:
                add_signal_impact('ema_score_status', f"EMA得分 ({ema_score_col}) 强劲看涨，与主信号一致", 5)
            elif final_score_val < 40 and ema_score_val < 40:
                add_signal_impact('ema_score_status', f"EMA得分 ({ema_score_col}) 强劲看跌，与主信号一致", -5)
            elif abs(final_score_val - ema_score_val) > 10:
                add_signal_impact('ema_score_status', f"EMA得分 ({ema_score_col}) 与主信号存在分歧", -5, "EMA得分与主信号存在分歧，趋势可能不稳定。")
            else:
                add_signal_impact('ema_score_status', f"EMA得分 ({ema_score_col}) 中性或一致", 0)
        else:
            add_signal_impact('ema_score_status', "数据缺失", 0)

        # 19. 波动率 (score_volatility)
        print(f"[{self.strategy_name}][{stock}] 波动率值: {volatility_val:.4f}。")
        if not np.isnan(volatility_val):
            if volatility_val >= volatility_thresh_high:
                add_signal_impact('volatility_status', "高波动", -10, f"当前波动率较高({volatility_val:.2%})，市场不确定性增加。")
            elif volatility_val <= volatility_thresh_low:
                add_signal_impact('volatility_status', "低波动", 5, f"当前波动率较低({volatility_val:.2%})，市场可能处于盘整或趋势启动前夕。")
            else:
                add_signal_impact('volatility_status', "中等波动", 0)
        else:
            add_signal_impact('volatility_status', "波动率数据缺失", 0)

        # 20. 放量起涨与底部放量起涨信号判断
        if not np.isnan(volume_breakout_signal_val) and volume_breakout_signal_val > 0:
            add_signal_impact('volume_breakout_signal', "放量起涨信号", 15, "出现放量起涨信号，主力资金有进场迹象，关注短线拉升机会。")
            volume_breakout_signal_str = "【放量起涨信号】：出现放量起涨，主力资金有进场迹象，关注短线拉升机会。\n"
        else:
            volume_breakout_signal_str = "【放量起涨信号】：无明显放量起涨信号。\n"
        if not np.isnan(bottom_volume_breakout_signal_val) and bottom_volume_breakout_signal_val > 0:
            add_signal_impact('bottom_volume_breakout_signal', "底部放量起涨", 25, "底部区域出现放量起涨信号，极有可能是阶段性反转起点，建议重点关注。")
            bottom_volume_breakout_signal_str = "【底部放量起涨】：底部区域出现放量起涨，极有可能是阶段性反转起点，建议重点关注。\n"
        else:
            bottom_volume_breakout_signal_str = "【底部放量起涨】：无底部放量起涨信号。\n"

        # 21. ADX 强度信号判断 (来自 adx_strength_signal 列, 结合方向)
        print(f"[{self.strategy_name}][{stock}] ADX 强度信号值 (来自列 {adx_strength_signal_col}, 结合方向): {adx_strength_signal_val:.2f}。")
        if not np.isnan(adx_strength_signal_val):
            trend_direction_from_adx_signal = "中性"
            # 使用一个小的epsilon来比较浮点数，避免等于0的精确比较问题
            epsilon = 1e-9
            if adx_strength_signal_val > epsilon:
                trend_direction_from_adx_signal = "看涨"
            elif adx_strength_signal_val < -epsilon:
                trend_direction_from_adx_signal = "看跌"

            abs_adx_strength = abs(adx_strength_signal_val)
            strength_desc = "不明"
            confidence_adj = 0
            risk_msg_adx_processed = ""

            # 主信号方向判断简化
            main_signal_is_strong_bullish = final_score_val >= strong_bullish_thresh
            main_signal_is_moderate_bullish = final_score_val >= moderate_bullish_thresh and not main_signal_is_strong_bullish
            main_signal_is_mildly_bullish = final_score_val >= trend_conf_upper_thresh and not (main_signal_is_strong_bullish or main_signal_is_moderate_bullish)

            main_signal_is_strong_bearish = final_score_val <= strong_bearish_thresh
            main_signal_is_moderate_bearish = final_score_val <= moderate_bearish_thresh and not main_signal_is_strong_bearish
            main_signal_is_mildly_bearish = final_score_val <= trend_conf_lower_thresh and not (main_signal_is_strong_bearish or main_signal_is_moderate_bearish)

            main_signal_overall_bullish = final_score_val > trend_conf_upper_thresh # 主信号总体看涨
            main_signal_overall_bearish = final_score_val < trend_conf_lower_thresh # 主信号总体看跌

            if abs_adx_strength >= adx_strong_thresh:
                strength_desc = "非常强劲"
                if trend_direction_from_adx_signal == "看涨":
                    confidence_adj = 10
                    if main_signal_overall_bearish: # 主信号总体看跌
                        confidence_adj = -7 # 显著冲突
                        risk_msg_adx_processed = f"处理后ADX信号{strength_desc}看涨，但与主信号(score:{final_score_val:.2f})看跌方向冲突。"
                elif trend_direction_from_adx_signal == "看跌":
                    confidence_adj = -10
                    if main_signal_overall_bullish: # 主信号总体看涨
                        confidence_adj = 7 # 显著冲突 (绝对值减小，但仍为负向影响，因为ADX信号是看跌的，所以应该是负的) -> 改为 -7
                        risk_msg_adx_processed = f"处理后ADX信号{strength_desc}看跌，但与主信号(score:{final_score_val:.2f})看涨方向冲突。"
                        if confidence_adj > 0 : confidence_adj = -7 # 确保冲突时，ADX看跌信号不会导致正向信心
            elif abs_adx_strength >= adx_moderate_thresh:
                strength_desc = "强劲"
                if trend_direction_from_adx_signal == "看涨":
                    confidence_adj = 7
                    if main_signal_is_strong_bearish or main_signal_is_moderate_bearish: # 主信号明确看跌
                        confidence_adj = -5
                        risk_msg_adx_processed = f"处理后ADX信号{strength_desc}看涨，但与主信号(score:{final_score_val:.2f})看跌方向冲突。"
                    elif main_signal_is_mildly_bearish: # 主信号温和看跌
                        confidence_adj = 0
                        risk_msg_adx_processed = f"处理后ADX信号{strength_desc}看涨，但与主信号(score:{final_score_val:.2f})温和看跌不一致。"
                elif trend_direction_from_adx_signal == "看跌":
                    confidence_adj = -7
                    if main_signal_is_strong_bullish or main_signal_is_moderate_bullish: # 主信号明确看涨
                        confidence_adj = 5 # 同样，ADX看跌信号与主信号看涨冲突，信心应该是负向或减弱 -> 改为 -5
                        risk_msg_adx_processed = f"处理后ADX信号{strength_desc}看跌，但与主信号(score:{final_score_val:.2f})看涨方向冲突。"
                        if confidence_adj > 0 : confidence_adj = -5
                    elif main_signal_is_mildly_bullish: # 主信号温和看涨
                        confidence_adj = 0
                        risk_msg_adx_processed = f"处理后ADX信号{strength_desc}看跌，但与主信号(score:{final_score_val:.2f})温和看涨不一致。"
            elif abs_adx_strength > epsilon: # 温和趋势 (大于epsilon，小于moderate_thresh)
                strength_desc = "温和"
                if trend_direction_from_adx_signal == "看涨":
                    confidence_adj = 3
                    if main_signal_is_strong_bearish or main_signal_is_moderate_bearish: # 主信号明确看跌
                         confidence_adj = -3
                         risk_msg_adx_processed = f"处理后ADX信号{strength_desc}看涨，但与主信号(score:{final_score_val:.2f})看跌方向冲突。"
                elif trend_direction_from_adx_signal == "看跌":
                    confidence_adj = -3
                    if main_signal_is_strong_bullish or main_signal_is_moderate_bullish: # 主信号明确看涨
                         confidence_adj = 3 # ADX看跌与主信号看涨冲突 -> 改为 -3
                         risk_msg_adx_processed = f"处理后ADX信号{strength_desc}看跌，但与主信号(score:{final_score_val:.2f})看涨方向冲突。"
                         if confidence_adj > 0: confidence_adj = -3
            else: # abs_adx_strength is very small (无明显趋势)
                strength_desc = "无明显趋势"
                confidence_adj = 0
                if main_signal_overall_bullish or main_signal_overall_bearish: # 主信号有明显趋势
                    confidence_adj = -3
                    risk_msg_adx_processed = f"处理后ADX信号显示{strength_desc}，但主信号(score:{final_score_val:.2f})有明显趋势倾向。"

            status_msg = f"处理后ADX信号：{trend_direction_from_adx_signal}趋势{strength_desc}"
            add_signal_impact('adx_processed_strength_status', status_msg, confidence_adj, risk_msg_adx_processed)
        else:
            add_signal_impact('adx_processed_strength_status', "处理后ADX强度信号数据缺失", 0)

        # 根据综合信心分数生成更高效和有效的操作建议
        normalized_confidence = max(-1.0, min(1.0, confidence_score / 100.0))
        print(f"[{self.strategy_name}][{stock}] 最终综合信心分数: {confidence_score}, 归一化信心: {normalized_confidence:.2f}。")

        if normalized_confidence >= 0.7:
            operation_advice_str = f"【强烈买入/积极加仓】信号极度强劲，多重指标共振看涨，趋势稳固。建议积极介入。{t_plus_1_note_str}"
        elif normalized_confidence >= 0.3:
            operation_advice_str = f"【买入/持有】信号强劲，趋势明确。建议持有或逢低加仓。{t_plus_1_note_str}"
        elif normalized_confidence > 0.05:
            operation_advice_str = f"【谨慎买入/观望】信号偏多，但有不确定性。建议轻仓试探或等待更明确信号。{t_plus_1_note_str}"
        elif normalized_confidence <= -0.7:
            operation_advice_str = f"【强烈卖出/坚决空仓】信号极度疲软，多重指标共振看跌，趋势明确。建议坚决离场。{t_plus_1_note_str}"
        elif normalized_confidence <= -0.3:
            operation_advice_str = f"【卖出/减仓】信号疲软，趋势向下。建议减仓或逢高减仓。{t_plus_1_note_str}"
        elif normalized_confidence < -0.05:
            operation_advice_str = f"【谨慎卖出/观望】信号偏空，但有不确定性。建议轻仓试空或等待更明确信号。{t_plus_1_note_str}"
        else:
            operation_advice_str = f"【中性观望】市场信号不明朗。建议耐心等待。{t_plus_1_note_str}"

        if not risk_warning_str:
            risk_warning_str = "无明显风险提示。"

        # 汇总所有关键变量到 signal_judgment_dict
        signal_judgment_dict['final_score'] = final_score_val
        signal_judgment_dict['base_score_raw'] = base_score_raw_val
        signal_judgment_dict['base_score_volume_adjusted'] = base_score_volume_adjusted_val
        signal_judgment_dict['long_term_context'] = long_term_context_val
        signal_judgment_dict['stoch_signal'] = stoch_signal_val
        signal_judgment_dict['vol_spike_signal'] = vol_spike_signal_val
        signal_judgment_dict['volatility_val'] = volatility_val
        signal_judgment_dict['ema_score_val'] = ema_score_val
        signal_judgment_dict['ema_cross_signal_val'] = ema_cross_signal_val
        signal_judgment_dict['ema_strength_val'] = ema_strength_val
        signal_judgment_dict['score_momentum_val'] = score_momentum_val
        signal_judgment_dict['score_momentum_acceleration_val'] = score_momentum_acceleration_val
        signal_judgment_dict['volatility_signal_val'] = volatility_signal_val
        signal_judgment_dict['vwap_deviation_signal_val'] = vwap_deviation_signal_val
        signal_judgment_dict['vwap_deviation_percent_val'] = vwap_deviation_percent_val
        signal_judgment_dict['boll_breakout_signal_val'] = boll_breakout_signal_val
        signal_judgment_dict['has_bearish_divergence'] = has_bearish_div_val
        signal_judgment_dict['has_bullish_divergence'] = has_bullish_div_val
        signal_judgment_dict['adx_val_raw'] = adx_val
        signal_judgment_dict['adx_strength_signal_val'] = adx_strength_signal_val # 确保存储
        signal_judgment_dict['confidence_score'] = confidence_score
        signal_judgment_dict['normalized_confidence'] = normalized_confidence
        signal_judgment_dict['signal_impact_records'] = signal_impact_records
        signal_judgment_dict['score_change_short'] = score_change_short
        signal_judgment_dict['score_change_long'] = score_change_long
        signal_judgment_dict['score_change_consistency_status'] = score_change_consistency_status
        # 放量起涨与底部放量起涨信号判断
        signal_judgment_dict['volume_breakout_signal'] = volume_breakout_signal_val
        signal_judgment_dict['bottom_volume_breakout_signal'] = bottom_volume_breakout_signal_val

        # now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        chinese_interpretation_str = (
            f"【趋势跟踪策略分析 - {stock} - {str(latest_data_row.name)}】\n"
            f"焦点时间框架: {self.focus_timeframe}\n"
            f"最新组合信号分: {final_score_val:.2f} (规则: {final_rule_score_val:.2f}, Transformer: {transformer_score_val:.2f})\n"
            f"当前策略信号强度: {signal_judgment_dict.get('overall_signal_strength', '中性')}\n"
            f"基于规则趋势判断: {current_trend_direction} (强度: {current_trend_strength})\n"
            f"趋势持续时间: {trend_duration_info_dict.get('bullish_duration_text' if current_trend_direction.startswith('看涨') else 'bearish_duration_text','未知')} (状态: {duration_status_rule_str})\n"
            f"{volume_breakout_signal_str}"
            f"{bottom_volume_breakout_signal_str}"
            f"----------------------------------------\n"
            f"【多维度信号交叉验证】\n"
            f"量能与趋势一致性: {signal_judgment_dict.get('trend_volume_consistency', '未知')}, 量能效果: {signal_judgment_dict.get('volume_effect', '未知')}\n"
            f"长期趋势与当前信号: {signal_judgment_dict.get('long_term_vs_current', '未知')}\n"
            f"随机指标(STOCH)与趋势关系: {signal_judgment_dict.get('stoch_trend_relation', '未知')}\n"
            f"量能异动与趋势: {signal_judgment_dict.get('vol_spike_trend', '未知')}\n"
            f"EMA排列状态: {signal_judgment_dict.get('alignment_status', '未知')}\n"
            f"EMA交叉信号: {signal_judgment_dict.get('ema_cross_status', '未知')}\n"
            f"EMA强度: {signal_judgment_dict.get('ema_strength_status', '未知')}\n"
            f"信号分动量: {signal_judgment_dict.get('score_momentum_status', '未知')}, 加速: {signal_judgment_dict.get('score_momentum_acceleration_status', '未知')}\n"
            f"信号分变化: 短期({score_momentum_short_period}期) {signal_judgment_dict.get('score_change_short_term', '未知')}, 长期({score_momentum_long_period}期) {signal_judgment_dict.get('score_change_long_term', '未知')}, 一致性: {signal_judgment_dict.get('score_change_consistency_status', '未知')}\n"
            f"VWAP偏离状态: {signal_judgment_dict.get('vwap_deviation_status', '未知')}\n"
            f"布林带突破: {signal_judgment_dict.get('boll_breakout_status', '未知')}\n"
            f"原始ADX强度: {signal_judgment_dict.get('adx_status', '未知')} (值: {adx_val:.2f})\n"
            f"处理后ADX强度信号: {signal_judgment_dict.get('adx_processed_strength_status', '未知')} (原始值: {adx_strength_signal_val:.2f})\n" # 更新解读
            f"背离状态: {signal_judgment_dict.get('divergence_status', '未知')}\n"
            f"模型信号一致性: {signal_judgment_dict.get('transformer_rule_consistency', '未知')}\n"
            f"EMA得分: {signal_judgment_dict.get('ema_score_status', '未知')}\n"
            f"波动率状态: {signal_judgment_dict.get('volatility_status', '未知')}\n"
            f"波动率信号: {signal_judgment_dict.get('volatility_signal_status', '未知')}\n"
            f"----------------------------------------\n"
            f"【综合评估与操作建议】\n"
            f"综合信心分数: {confidence_score} (归一化: {normalized_confidence:.2f})\n"
            f"操作建议: {operation_advice_str}\n"
            f"风险提示: {risk_warning_str}\n"
            # f"信号影响明细: {signal_impact_records}\n"
        )

        from collections import defaultdict
        signal_contribution_summary = defaultdict(int)
        for record in signal_impact_records:
            signal_contribution_summary[record['key']] += record['confidence_change']
        signal_judgment_dict['signal_contribution_summary'] = dict(signal_contribution_summary)

        SIGNAL_WEIGHTS = {
            'overall_signal_strength': 1.5,
            'trend_duration_impact': 1.2,
            'trend_volume_consistency': 1.0,
            'volume_effect': 1.0,
            'long_term_vs_current': 1.3,
            'stoch_trend_relation': 1.1,
            'vol_spike_trend': 1.0,
            'alignment_status': 1.0,
            'ema_cross_status': 1.0,
            'ema_strength_status': 1.0,
            'score_momentum_status': 1.0,
            'score_momentum_acceleration_status': 1.0,
            'score_change_short_term': 1.0,
            'score_change_long_term': 1.2,
            'score_change_consistency': 1.0,
            'vwap_deviation_status': 1.0,
            'boll_breakout_status': 1.0,
            'adx_status': 1.2,
            'adx_processed_strength_status': 1.2, # 调整权重，使其与原始ADX判断权重相似或略高，因其包含方向
            'divergence_status': 1.5,
            'transformer_rule_consistency': 1.2,
            'ema_score_status': 1.0,
            'volatility_status': 1.0,
            'volatility_signal_status': 1.0,
        }
        weighted_confidence_score = 0
        for record in signal_impact_records:
            weight = SIGNAL_WEIGHTS.get(record['key'], 1.0)
            weighted_confidence_score += record['confidence_change'] * weight
        signal_judgment_dict['weighted_confidence_score'] = weighted_confidence_score

        analysis_results_dict['signal_judgment'] = signal_judgment_dict
        analysis_results_dict['operation_advice'] = operation_advice_str
        analysis_results_dict['risk_warning'] = risk_warning_str
        analysis_results_dict['chinese_interpretation'] = chinese_interpretation_str
        self.analysis_results = analysis_results_dict
        # logger.debug(f"[{self.strategy_name}][{stock}] 信号分析完成。")
        logger.info(chinese_interpretation_str)
        # print(f"[{self.strategy_name}][{stock}] 信号分析完成。")

        return analysis_results_dict

    def get_analysis_results(self, stock_code: str, timestamp: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        读取股票趋势跟踪策略的分析结果。
        如果提供了时间戳，则读取该时间戳的分析结果；否则，读取最新的分析结果。
        参数:
            stock_code (str): 股票代码，例如 '000001.SZ'。
            timestamp (Optional[datetime]): 可选的分析时间戳。如果为 None，则返回最新的结果。
        返回:
            Optional[Dict[str, Any]]: 包含分析结果的字典，如果未找到则返回 None。
        """
        print(f"[{self.strategy_name}][{stock_code}] 开始读取分析结果。") # 调试信息：开始读取操作
        try:
            # 确保 StockAnalysisResultTrendFollowing 和 StockInfo 模型以及 StockBasicInfoDao 可用
            stock_basic_dao = StockBasicInfoDao()
            # 异步获取 StockInfo 对象
            stock_obj = asyncio.run(stock_basic_dao.get_stock_by_code(stock_code))
            if not stock_obj:
                print(f"[{self.strategy_name}][{stock_code}] 读取分析结果失败：股票代码 {stock_code} 不存在于 StockInfo 模型中。") # 调试信息：股票不存在
                return None
            # 构建查询集，过滤指定股票
            query_set = StockAnalysisResultTrendFollowing.objects.filter(stock=stock_obj)
            if timestamp:
                # 如果提供了时间戳，精确查找该时间点的分析结果
                analysis_record = query_set.filter(timestamp=timestamp).first()
                if not analysis_record:
                    print(f"[{self.strategy_name}][{stock_code}] 在时间点 {timestamp.strftime('%Y-%m-%d %H:%M')} 未找到分析结果。") # 调试信息：指定时间未找到
                    return None
            else:
                # 如果未提供时间戳，查找该股票最新的分析结果
                analysis_record = query_set.order_by('-timestamp').first()
                if not analysis_record:
                    print(f"[{self.strategy_name}][{stock_code}] 未找到最新的分析结果。") # 调试信息：未找到最新结果
                    return None
            print(f"[{self.strategy_name}][{stock_code}] 成功读取到时间点 {analysis_record.timestamp.strftime('%Y-%m-%d %H:%M')} 的分析结果。") # 调试信息：读取成功
            # 将模型实例的字段转换为字典
            results_dict = {
                'stock_code': analysis_record.stock.stock_code,
                'timestamp': analysis_record.timestamp.isoformat(), # 将 datetime 对象转换为 ISO 格式字符串
                'score': analysis_record.score,
                'rule_signal': analysis_record.rule_signal,
                'lstm_signal': analysis_record.lstm_signal,
                'base_score_raw': analysis_record.base_score_raw,
                'base_score_volume_adjusted': analysis_record.base_score_volume_adjusted,
                'alignment_signal': analysis_record.alignment_signal,
                'long_term_context': analysis_record.long_term_context,
                'adx_strength_signal': analysis_record.adx_strength_signal,
                'stoch_signal': analysis_record.stoch_signal,
                'div_has_bearish_divergence': analysis_record.div_has_bearish_divergence,
                'div_has_bullish_divergence': analysis_record.div_has_bullish_divergence,
                'volume_spike_signal': analysis_record.volume_spike_signal,
                'close_price': analysis_record.close_price,
                'current_trend': analysis_record.current_trend,
                'trend_strength': analysis_record.trend_strength,
                'trend_duration_bullish': analysis_record.trend_duration_bullish,
                'trend_duration_bearish': analysis_record.trend_duration_bearish,
                'trend_duration_text_bullish': analysis_record.trend_duration_text_bullish,
                'trend_duration_text_bearish': analysis_record.trend_duration_text_bearish,
                'trend_duration_status': analysis_record.trend_duration_status,
                'operation_advice': analysis_record.operation_advice,
                'risk_warning': analysis_record.risk_warning,
                'chinese_interpretation': analysis_record.chinese_interpretation,
                'weighted_confidence_score': analysis_record.weighted_confidence_score,
                'confidence_score': analysis_record.confidence_score,
                'normalized_confidence': analysis_record.normalized_confidence,
            }
            # Django的JSONField会自动反序列化为Python对象，直接访问属性即可
            results_dict['signal_impact_records'] = analysis_record.signal_impact_records_json
            results_dict['signal_contribution_summary'] = analysis_record.signal_contribution_summary_json
            results_dict['raw_analysis_data'] = analysis_record.raw_analysis_data
            return results_dict
        except StockInfo.DoesNotExist:
            print(f"[{self.strategy_name}][{stock_code}] 读取分析结果失败：股票代码 {stock_code} 不存在于 StockInfo 模型中。") # 调试信息：股票不存在异常
            return None
        except Exception as e:
            print(f"[{self.strategy_name}][{stock_code}] 读取 StockAnalysisResultTrendFollowing 记录出错: {e}") # 调试信息：其他读取异常
            return None

    # 添加 timestamp 参数，用于记录分析发生的时间点，data 参数可选，方便获取最新价格
    def save_analysis_results(self, stock_code: str, timestamp: pd.Timestamp, data: Optional[pd.DataFrame]=None):
        """
        保存趋势跟踪策略的分析结果到数据库。
        使用JSON配置获取内部列名和OHLCV列名。
        """
        # 引入 json 模块，用于序列化复杂数据结构
        # 移除 asyncio 导入，因为不再使用异步缓存操作
        # import asyncio # 原始代码中的导入，已移除
        # stock_basic_dao = StockBasicInfoDao()
        strategy_dao = StrategiesDAO()
        # stock_obj = asyncio.run(stock_basic_dao.get_stock_by_code(stock_code))

        if self.analysis_results is None:
            print(f"[{self.strategy_name}][{stock_code}] 无分析结果可保存。请先运行 analyze_signals。")
            return

        # 尝试从 self.intermediate_data 获取最新的数据行，如果 self.intermediate_data 为 None 或空，则尝试从传入的 data 获取
        latest_intermediate_row = pd.Series(dtype=object)
        if self.intermediate_data is not None and not self.intermediate_data.empty:
            try:
                # 使用 asof 查找小于等于 timestamp 的最新一行
                latest_intermediate_row = self.intermediate_data.loc[self.intermediate_data.index.asof(timestamp)]
                if latest_intermediate_row.isnull().all():
                    print(f"[{self.strategy_name}][{stock_code}] 无法通过 asof 在 intermediate_data 中找到时间戳 {timestamp} 对应的行，使用最后一行。")
                    latest_intermediate_row = self.intermediate_data.iloc[-1]
            except Exception as e:
                print(f"[{self.strategy_name}][{stock_code}] 在 intermediate_data 中查找时间戳 {timestamp} 对应的行出错: {e}，使用最后一行。")
                try:
                    latest_intermediate_row = self.intermediate_data.iloc[-1]
                except IndexError:
                    print(f"[{self.strategy_name}][{stock_code}] intermediate_data 为空，无法获取最新数据行。")
                    latest_intermediate_row = pd.Series(dtype=object)
        elif data is not None and not data.empty:
            print(f"[{self.strategy_name}][{stock_code}] intermediate_data 为空，尝试从原始输入 data 获取最新数据行。")
            try:
                latest_data_row_from_input = data.loc[data.index.asof(timestamp)]
                if latest_data_row_from_input.isnull().all():
                    latest_data_row_from_input = data.iloc[-1]
                latest_intermediate_row = latest_data_row_from_input
            except Exception as e:
                print(f"[{self.strategy_name}][{stock_code}] 从原始输入 data 获取最新数据行出错: {e}。")
                latest_intermediate_row = pd.Series(dtype=object)

        if latest_intermediate_row.empty:
            print(f"[{self.strategy_name}][{stock_code}] 无法获取最新的数据行用于保存分析结果。")
            # 继续保存，但部分字段可能为 None

        try:
            # 辅助函数，将 NaN, Inf 或 None 转换为 None，以便保存到数据库字段
            def convert_nan_to_none(value):
                if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
                    return None
                return value if pd.notna(value) else None

            # 获取策略内部列名
            internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
            if not isinstance(internal_cols_conf, list): internal_cols_conf = []
            def get_internal_col_name(pattern, default_name):
                for item in internal_cols_conf:
                    if isinstance(item, dict) and item.get('name_pattern') == pattern:
                        return item['name_pattern']
                return default_name

            combined_signal_col = get_internal_col_name("combined_signal", "combined_signal")
            final_rule_signal_col = get_internal_col_name("final_rule_signal", "final_rule_signal")
            transformer_signal_col = get_internal_col_name("transformer_signal", "transformer_signal")
            base_score_raw_col = get_internal_col_name("base_score_raw", "base_score_raw")
            base_score_volume_adjusted_col = get_internal_col_name("ADJUSTED_SCORE", "ADJUSTED_SCORE")
            alignment_signal_col = get_internal_col_name("alignment_signal", "alignment_signal")
            long_term_context_col = get_internal_col_name("long_term_context", "long_term_context")
            adx_strength_signal_col = get_internal_col_name("adx_strength_signal", "adx_strength_signal")
            stoch_signal_col = get_internal_col_name("stoch_signal", "stoch_signal")
            has_bearish_div_col = get_internal_col_name("HAS_BEARISH_DIVERGENCE", "HAS_BEARISH_DIVERGENCE")
            has_bullish_div_col = get_internal_col_name("HAS_BULLISH_DIVERGENCE", "HAS_BULLISH_DIVERGENCE")

            vol_spike_pattern = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c.get('name_pattern', '').startswith("VOL_SPIKE_SIGNAL")), "VOL_SPIKE_SIGNAL_{timeframe}")
            vol_spike_signal_col_tf = self._format_indicator_name(vol_spike_pattern, timeframe=self.focus_timeframe)
            vol_spike_signal_col = vol_spike_signal_col_tf[0] if vol_spike_signal_col_tf else None

            ohlcv_configs = NAMING_CONFIG.get('ohlcv_naming_convention', {}).get('output_columns', [])
            close_base_name = next((c['name_pattern'] for c in ohlcv_configs if isinstance(c, dict) and c.get('name_pattern') == 'close'), 'close')
            close_price_col_name = f'{close_base_name}_{self.focus_timeframe}'

            signal_judgment = self.analysis_results.get('signal_judgment', {})

            # 构建 defaults 字典，用于 update_or_create
            # 注意：stock 和 timestamp 是查找条件，不放在 defaults 里
            defaults_kwargs = {
                'score': convert_nan_to_none(latest_intermediate_row.get(combined_signal_col)),
                'rule_signal': convert_nan_to_none(latest_intermediate_row.get(final_rule_signal_col)),
                'lstm_signal': convert_nan_to_none(latest_intermediate_row.get(transformer_signal_col)),
                'base_score_raw': convert_nan_to_none(latest_intermediate_row.get(base_score_raw_col)),
                'base_score_volume_adjusted': convert_nan_to_none(latest_intermediate_row.get(base_score_volume_adjusted_col)),
                'alignment_signal': convert_nan_to_none(latest_intermediate_row.get(alignment_signal_col)),
                'long_term_context': convert_nan_to_none(latest_intermediate_row.get(long_term_context_col)),
                'adx_strength_signal': convert_nan_to_none(latest_intermediate_row.get(adx_strength_signal_col)),
                'stoch_signal': convert_nan_to_none(latest_intermediate_row.get(stoch_signal_col)),
                'div_has_bearish_divergence': bool(latest_intermediate_row.get(has_bearish_div_col, False)),
                'div_has_bullish_divergence': bool(latest_intermediate_row.get(has_bullish_div_col, False)),
                'volume_spike_signal': convert_nan_to_none(latest_intermediate_row.get(vol_spike_signal_col)),
                'close_price': convert_nan_to_none(latest_intermediate_row.get(close_price_col_name)),
                'current_trend': self.analysis_results.get('current_trend'),
                'trend_strength': self.analysis_results.get('trend_strength'),
                'trend_duration_bullish': self.analysis_results.get('bullish_duration'),
                'trend_duration_bearish': self.analysis_results.get('bearish_duration'),
                'trend_duration_text_bullish': self.analysis_results.get('bullish_duration_text'),
                'trend_duration_text_bearish': self.analysis_results.get('bearish_duration_text'),
                'trend_duration_status': self.analysis_results.get('duration_status'),
                'operation_advice': self.analysis_results.get('operation_advice'),
                'risk_warning': self.analysis_results.get('risk_warning'),
                'volume_breakout_signal': self.analysis_results.get('volume_breakout_signal'),
                'bottom_volume_breakout_signal': self.analysis_results.get('bottom_volume_breakout_signal'),
                'chinese_interpretation': self.analysis_results.get('chinese_interpretation'),
                'signal_impact_records_json': json.dumps(signal_judgment.get('signal_impact_records', []), ensure_ascii=False, default=lambda x: str(x)),
                'signal_contribution_summary_json': json.dumps(signal_judgment.get('signal_contribution_summary', {}), ensure_ascii=False, default=lambda x: str(x)),
                'weighted_confidence_score': convert_nan_to_none(signal_judgment.get('weighted_confidence_score')),
                'confidence_score': convert_nan_to_none(signal_judgment.get('confidence_score')),
                'normalized_confidence': convert_nan_to_none(signal_judgment.get('normalized_confidence')),
                # 'raw_analysis_data': json.dumps(self.analysis_results, ensure_ascii=False, default=lambda x: str(x))
            }
            asyncio.run(strategy_dao.save_strategy_results(stock_code=stock_code,timestamp=timestamp,defaults_kwargs=defaults_kwargs))

        except StockInfo.DoesNotExist:
            print(f"[{self.strategy_name}][{stock_code}] 保存分析结果失败：股票代码 {stock_code} 不存在于 StockInfo 模型中。")
        except Exception as e:
            print(f"[{self.strategy_name}][{stock_code}] 保存 StockAnalysisResultTrendFollowing 记录出错: {e}")

    def _get_intermediate_file_path(self, stock_code: str):
        """
        生成中间数据文件的基础路径 (不含扩展名)，使用股票专属目录结构。
        """
        if not self.base_data_dir:
             raise ValueError("base_data_dir 未设置，无法生成中间文件路径。")

        # 使用 set_model_paths 的逻辑来确定股票根目录
        stock_root_dir = os.path.join(self.base_data_dir, stock_code)
        # 在股票根目录下创建中间数据子目录
        intermediate_dir = os.path.join(stock_root_dir, "intermediate_processing")
        os.makedirs(intermediate_dir, exist_ok=True)

        # 使用 stock_code 命名，确保唯一性
        return os.path.join(intermediate_dir, f"{stock_code}_intermediate")

    # --- 保存中间数据 ---
    def save_intermediate_data(self, stock_code: str, data_df: pd.DataFrame, target_column: str, tf_params: dict, window_size: int, required_columns: list):
        """
        保存阶段1和阶段2生成的中间数据到文件。
        保存 DataFrame 到 Parquet，元数据到 Pickle。
        """
        base_path = self._get_intermediate_file_path(stock_code)
        dataframe_path = f"{base_path}.parquet"
        metadata_path = f"{base_path}.pkl"

        try:
            # 保存 DataFrame 到 Parquet
            data_df.to_parquet(dataframe_path, index=True) # 保存索引 (日期)
            logger.info(f"DataFrame 已保存到 {dataframe_path}")

            # 保存元数据到 Pickle
            metadata = {
                'transformer_target_column': target_column,
                'tf_params': tf_params,
                'transformer_window_size': window_size,
                'required_columns_for_transformer': required_columns,
                # 可以根据需要添加其他元数据
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"元数据已保存到 {metadata_path}")

            return dataframe_path, metadata_path # 返回保存的文件路径
        except Exception as e:
            logger.error(f"保存 {stock_code} 的中间数据失败: {e}", exc_info=True)
            raise # 重新抛出异常

    # --- 加载中间数据 ---
    def load_intermediate_data(self, dataframe_path: str, metadata_path: str):
        """
        从文件加载阶段1和阶段2保存的中间数据。
        """
        try:
            # 加载 DataFrame
            data_df = pd.read_parquet(dataframe_path)
            # 确保索引是 datetime 类型
            data_df.index = pd.to_datetime(data_df.index)
            logger.info(f"DataFrame 已从 {dataframe_path} 加载，{len(data_df)}行。")

            # 加载元数据
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"元数据已从 {metadata_path} 加载。")

            target_column = metadata.get('transformer_target_column')
            tf_params = metadata.get('tf_params')
            window_size = metadata.get('transformer_window_size')
            required_columns = metadata.get('required_columns_for_transformer')

            # 检查关键数据是否存在
            if not all([data_df is not None, target_column, tf_params, window_size is not None, required_columns]):
                 raise ValueError("加载的中间数据不完整。")

            return data_df, target_column, tf_params, window_size, required_columns
        except FileNotFoundError:
            logger.error(f"中间数据文件未找到: {dataframe_path} 或 {metadata_path}")
            raise
        except Exception as e:
            logger.error(f"加载中间数据失败: {e}", exc_info=True)
            raise # 重新抛出异常

    # --- 处理加载后的中间数据并调用 prepare_data_for_transformer ---
    def process_loaded_intermediate_data(self, data_for_prep: pd.DataFrame, transformer_target_column: str, tf_params: dict, required_columns_for_transformer: list):
        """
        接收加载后的中间数据，提取参数，并调用 prepare_data_for_transformer。
        """
        # print("DEBUG: process_loaded_intermediate_data called") # 调试信息
        logger.info("开始处理加载后的中间数据并准备 Transformer 数据...")
        # 从加载的 tf_params 中提取数据准备配置
        data_prep_config = tf_params.get('transformer_data_prep_config', {})
        # 从 data_prep_config 中提取参数，并为 prepare_data_for_transformer 提供默认值
        scaler_type = data_prep_config.get('scaler_type', 'minmax')
        train_split = data_prep_config.get('train_split', 0.7)
        val_split = data_prep_config.get('val_split', 0.15)
        apply_variance_threshold = data_prep_config.get('apply_variance_threshold', False)
        variance_threshold_value = data_prep_config.get('variance_threshold_value', 0.01)
        use_pca = data_prep_config.get('use_pca', False)
        pca_n_components = data_prep_config.get('pca_n_components', 0.99)
        pca_solver = data_prep_config.get('pca_solver', 'auto')
        use_feature_selection = data_prep_config.get('use_feature_selection', True)
        feature_selector_model_type = data_prep_config.get('feature_selector_model_type', 'rf')
        fs_model_n_estimators = data_prep_config.get('fs_model_n_estimators', 100)
        fs_model_max_depth = data_prep_config.get('fs_model_max_depth', None)
        fs_max_features = data_prep_config.get('fs_max_features', 120)
        fs_selection_threshold = data_prep_config.get('fs_selection_threshold', 'median')
        target_scaler_type = data_prep_config.get('target_scaler_type', 'minmax')

        # 调用 prepare_data_for_transformer 函数
        # print("DEBUG: Calling prepare_data_for_transformer from process_loaded_intermediate_data") # 调试信息
        return prepare_data_for_transformer(
            data=data_for_prep,
            required_columns=required_columns_for_transformer,
            target_column=transformer_target_column,
            scaler_type=scaler_type,
            train_split=train_split,
            val_split=val_split,
            apply_variance_threshold=apply_variance_threshold,
            variance_threshold_value=variance_threshold_value,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            pca_solver=pca_solver,
            use_feature_selection=use_feature_selection,
            feature_selector_model_type=feature_selector_model_type,
            fs_model_n_estimators=fs_model_n_estimators,
            fs_model_max_depth=fs_model_max_depth,
            fs_max_features=fs_max_features,
            fs_selection_threshold=fs_selection_threshold,
            target_scaler_type=target_scaler_type
        )











