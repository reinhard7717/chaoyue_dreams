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
from services.indicator_services import IndicatorService # 确保导入 IndicatorService
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas_ta as ta
from dao_manager.tushare_daos.industry_dao import IndustryDao
from .utils import strategy_utils
from .utils.deep_learning_utils import (
    build_transformer_model,
    evaluate_transformer_model,
    train_transformer_model,
    predict_with_transformer_model,
    TimeSeriesDataset,
    prepare_data_for_transformer # prepare_data_for_transformer 仍然从这里导入
)

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
        self.industry_dao = IndustryDao()
        self.indicator_service = IndicatorService() # 初始化 IndicatorService
        # 检查settings中是否配置了INDICATOR_PARAMETERS_CONFIG_PATH
        if params_file is None:
            if not hasattr(settings, 'INDICATOR_PARAMETERS_CONFIG_PATH') or not settings.INDICATOR_PARAMETERS_CONFIG_PATH:
                 logger.error("CRITICAL: Django settings.INDICATOR_PARAMETERS_CONFIG_PATH 未配置!")
                 # 如果路径未配置，使用一个无效值，后续加载会失败
                 params_file = ""
            else:
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
        # 在加载参数之前初始化 fe_params 为空字典
        self.fe_params: Dict[str, Any] = {} # 初始化 fe_params 为空字典
        self.base_data_dir = base_data_dir
        # --- 阶段 0: 初始化局部变量 ---
        loaded_params: Dict[str, Any] = {} # 用于存储从文件加载的参数
        resolved_params_file_path = params_file # 初始化解析后的路径为传入的路径
        file_load_success = False # 标记参数文件是否成功加载并解析
        # 将解析后的参数文件路径存储为实例属性
        self.params_file_path = resolved_params_file_path # 存储解析后的参数文件路径
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
                        # 将 CWD 路径解析成功的日志级别降低，并强调建议使用相对 BASE_DIR 的路径
                        logger.info(f"{temp_log_prefix} 参数文件在当前工作目录 '{os.getcwd()}' 找到: '{resolved_params_file_path}'. 建议使用相对于项目根目录 (BASE_DIR) 的路径以提高健壮性。")
                    else:
                        # 对于相对路径在 BASE_DIR 和 CWD 都找不到的情况，提升日志级别到 ERROR 或 CRITICAL
                        logger.error(f"{temp_log_prefix} CRITICAL: 相对参数文件 '{params_file}' 在 BASE_DIR 和 CWD (解析为 '{path_based_on_cwd}') 中均未找到。无法加载参数。")
                        resolved_params_file_path = None # 标记为无效路径
            else:
                logger.warning(f"{temp_log_prefix} Django settings.BASE_DIR 未定义。尝试基于 CWD 解析相对路径 '{params_file}'...")
                path_based_on_cwd = os.path.abspath(params_file)
                if os.path.exists(path_based_on_cwd) and os.path.isfile(path_based_on_cwd):
                    resolved_params_file_path = path_based_on_cwd
                    # 将 CWD 路径解析成功的日志级别降低，并强调建议定义 BASE_DIR
                    logger.info(f"{temp_log_prefix} Django settings.BASE_DIR 未定义。相对参数文件在当前工作目录 '{os.getcwd()}' 找到: '{resolved_params_file_path}'. 强烈建议定义 settings.BASE_DIR。")
                else:
                    # 对于 BASE_DIR 未定义且相对路径在 CWD 找不到的情况，提升日志级别到 ERROR 或 CRITICAL
                    logger.error(f"{temp_log_prefix} CRITICAL: Django settings.BASE_DIR 未定义，且相对参数文件在 CWD (解析为 '{path_based_on_cwd}') 也未找到。无法加载参数。")
                    resolved_params_file_path = None # 标记为无效路径
        else:
            logger.debug(f"{temp_log_prefix} 参数文件路径 '{params_file}' 是绝对路径，直接使用。")
            resolved_params_file_path = params_file

        # --- 阶段 2: 从解析后的路径加载参数文件 ---
        # 在尝试加载前，先检查 resolved_params_file_path 是否有效
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
                        # 文件存在但内容为空或无效 JSON，提升日志级别到 ERROR 或 CRITICAL
                        logger.error(f"{temp_log_prefix} CRITICAL: 参数文件 '{resolved_params_file_path}' 内容为空或不是有效的JSON对象 (解析后类型: {type(loaded_params)}).")
                        loaded_params = {} # 确保 loaded_params 是一个空字典
                except FileNotFoundError:
                    # 尽管之前检查存在，但打开时未找到，这非常异常，提升日志级别
                    logger.error(f"{temp_log_prefix} CRITICAL: 文件 '{resolved_params_file_path}' 在尝试打开时未找到 (despite previous check).")
                except PermissionError:
                    # 权限错误，提升日志级别
                    logger.error(f"{temp_log_prefix} CRITICAL: 没有权限读取参数文件 '{resolved_params_file_path}'。")
                except json.JSONDecodeError as e_json:
                    # JSON 解析错误，提升日志级别
                    logger.error(f"{temp_log_prefix} CRITICAL: 解析参数文件 '{resolved_params_file_path}' 时发生JSON解码错误: {e_json}")
                except Exception as e_load:
                    # 其他未知加载错误，提升日志级别
                    logger.error(f"{temp_log_prefix} CRITICAL: 加载参数文件 '{resolved_params_file_path}' 时发生未知错误: {e_load}", exc_info=True)
            else:
                # 最终确认文件不存在或不是文件，提升日志级别
                logger.error(f"{temp_log_prefix} CRITICAL: 最终确认参数文件 '{resolved_params_file_path}' (原始输入: '{params_file}') 不存在或不是文件。无法加载参数。")
        else:
            # 如果 resolved_params_file_path 在解析阶段就无效，直接记录无法加载
             logger.error(f"{temp_log_prefix} CRITICAL: 参数文件路径解析失败。无法加载参数文件 '{params_file}'.")


        if not file_load_success:
            # 如果文件加载失败，明确记录策略将使用默认参数运行
            logger.warning(f"{temp_log_prefix} 参数文件加载失败或内容无效，策略将使用默认参数运行。")
            loaded_params = {} # 确保 loaded_params 是一个空字典

        # --- 阶段 3: 设置 self.params (之前由 BaseStrategy 完成) ---
        self.params: Dict[str, Any] = loaded_params # 直接将加载的参数赋值给 self.params
        # 在设置 self.params 后，安全获取 fe_params
        self.fe_params = self.params.get('feature_engineering_params', {}) # 在加载参数后获取 fe_params
        # 增加对 self.params 是否为空的日志判断
        logger.debug(f"{temp_log_prefix} self.params 已设置。是否为空: {not bool(self.params)}. 顶层键 (部分): {list(self.params.keys())[:5] if self.params else 'None'}")

        # --- 阶段 4: 设置 TrendFollowingStrategy 实例的最终 strategy_name ---
        # 优化 strategy_name 的设置逻辑，确保总能得到一个字符串名称
        strategy_name_from_params = self.params.get('trend_following_strategy_name')
        if isinstance(strategy_name_from_params, str) and strategy_name_from_params:
            self.strategy_name = strategy_name_from_params
            logger.info(f"[{self.strategy_name}-init-阶段4] 实例策略名从参数文件成功设置为: '{self.strategy_name}'")
        else:
            self.strategy_name = TrendFollowingStrategy.strategy_name_class_default
            if strategy_name_from_params is None:
                 logger.warning(f"[{self.strategy_name}-init-阶段4] 参数中未找到 'trend_following_strategy_name'键，实例策略名使用类默认值: '{self.strategy_name}'")
            elif not isinstance(strategy_name_from_params, str) or not strategy_name_from_params:
                 logger.warning(f"[{self.strategy_name}-init-阶段4] 参数中的 'trend_following_strategy_name' 无效 (非字符串或为空): '{strategy_name_from_params}'，实例策略名使用类默认值: '{self.strategy_name}'")

        log_prefix = f"[{self.strategy_name}]"

        # --- 阶段 5: 初始化 TrendFollowingStrategy 特有的其他属性 ---
        self.base_data_dir = base_data_dir
        logger.debug(f"{log_prefix} base_data_dir 设置为: '{self.base_data_dir}'")

        # 安全获取 trend_following_params，即使 self.params 为空
        self.tf_params: Dict[str, Any] = self.params.get('trend_following_params', {})
        if not self.params:
            logger.error(f"{log_prefix} CRITICAL INIT (最终属性设置前): 策略参数 (self.params) 仍为空！后续属性将完全依赖代码默认值。")
        elif not self.tf_params:
            logger.error(f"{log_prefix} CRITICAL INIT (最终属性设置前): 'trend_following_params' 块在已加载的参数中缺失或为空！后续特定参数将依赖代码默认值。")
        # 使用 .get() 方法安全获取参数，提供默认值
        self.focus_timeframe: str = str(self.tf_params.get('focus_timeframe', self.default_focus_timeframe))
        self.timeframe_weights: Optional[Dict[str, float]] = self.tf_params.get('timeframe_weights', None)
        # 确保 trend_indicators 是列表，即使参数中不是
        self.trend_indicators: List[str] = self.tf_params.get('trend_indicators', ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi'])
        if not isinstance(self.trend_indicators, list):
            logger.warning(f"{log_prefix} 参数 'trend_indicators' 不是一个列表，使用默认值。")
            self.trend_indicators = ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi']
        # 确保 rule_signal_weights 是字典，即使参数中不是
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
        # 安全获取 transformer_model_config 和 transformer_training_config
        self.transformer_model_config: Dict[str, Any] = self.tf_params.get('transformer_model_config', {})
        self.transformer_training_config: Dict[str, Any] = self.tf_params.get('transformer_training_config', {})

        # 确保训练配置中的 batch_size 覆盖策略参数中的值（如果存在）
        if 'batch_size' in self.transformer_training_config:
             self.transformer_batch_size = self.transformer_training_config['batch_size']
        # 安全获取 transformer_data_prep_config
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
            # 在验证过程中发生错误，提升日志级别
            logger.error(f"{log_prefix} CRITICAL: 在执行 _validate_params 时发生错误: {e_validate}", exc_info=True)

        # --- 阶段 7: 初始化完成最终日志 ---
        logger.info(f"策略 '{self.strategy_name}' 初始化流程完成。")
        logger.info(f"{log_prefix} 最终确定的主要关注时间框架: {self.focus_timeframe}.")
        logger.info(f"{log_prefix} 参数最终来源: '{resolved_params_file_path if file_load_success else '无或加载失败'}'.")
        logger.info(f"{log_prefix} self.params 最终是否为空: {not bool(self.params)} (True表示空).")
        logger.info(f"{log_prefix} self.tf_params 最终是否为空: {not bool(self.tf_params)} (True表示空).")
        logger.info(f"{log_prefix} 最终使用的 transformer_window_size: {self.transformer_window_size}")
        logger.info(f"{log_prefix} 最终使用的 transformer_target_column: '{self.transformer_target_column}'")
        # 安全访问 rule_signal_weights 中的键
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
                return None
            data_df, indicator_configs_from_service = prepared_data_tuple # 解包返回的元组
            if data_df is None or data_df.empty:
                logger.error(f"{log_prefix} IndicatorService.prepare_strategy_dataframe 返回空 DataFrame。数据准备失败。")
                return None
            # 将 IndicatorService 返回的 DataFrame 存储到实例变量，可能用于调试或后续步骤
            self.intermediate_data = data_df # 存储 IndicatorService 返回的 DataFrame
            logger.info(f"{log_prefix} 数据准备完成。DataFrame Shape: {data_df.shape}, 列数: {len(data_df.columns)}")
            # logger.debug(f"{log_prefix} 准备好的数据列 (部分): {data_df.columns.tolist()[:30]}...") # 调试输出
            # IndicatorService 已经处理了缺失值填充，这里不再需要额外的填充步骤
            return data_df, indicator_configs_from_service # 返回 DataFrame 和 indicator_configs
        except Exception as e:
            logger.error(f"{log_prefix} 调用 IndicatorService.prepare_strategy_dataframe 时出错: {e}", exc_info=True)
            return None

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
            # 检查验证集数据量是否足够创建 Dataset 和 DataLoader
            if features_scaled_val_np is not None and features_scaled_val_np.shape[0] >= self.transformer_window_size and targets_scaled_val_np is not None and targets_scaled_val_np.shape[0] >= self.transformer_window_size:
                val_dataset = TimeSeriesDataset(features_scaled_val_np, targets_scaled_val_np, self.transformer_window_size)
                if len(val_dataset) > 0:
                    # 验证集 DataLoader 不需要 shuffle
                    val_loader = DataLoader(val_dataset, batch_size=self.transformer_batch_size, shuffle=False)
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 验证集 Dataset 为空 (数据量不足 {self.transformer_window_size} 或其他原因)。验证阶段将跳过。")
            else:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 验证集数据量不足 {self.transformer_window_size}。验证阶段将跳过。")


            test_loader = None
            # 检查测试集数据量是否足够创建 Dataset 和 DataLoader
            if features_scaled_test_np is not None and features_scaled_test_np.shape[0] >= self.transformer_window_size and targets_scaled_test_np is not None and targets_scaled_test_np.shape[0] >= self.transformer_window_size:
                test_dataset = TimeSeriesDataset(features_scaled_test_np, targets_scaled_test_np, self.transformer_window_size)
                if len(test_dataset) > 0:
                    # 测试集 DataLoader 不需要 shuffle
                    test_loader = DataLoader(test_dataset, batch_size=self.transformer_batch_size, shuffle=False)
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] 测试集 Dataset 为空 (数据量不足 {self.transformer_window_size} 或其他原因)。测试评估将跳过。")
            else:
                logger.warning(f"[{self.strategy_name}][{stock_code}] 测试集数据量不足 {self.transformer_window_size}。测试评估将跳过。")


            # 检查训练集数据量是否足够创建 Dataset 和 DataLoader
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
                window_size=self.transformer_window_size # 传递 window_size 给 build_transformer_model
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
        # 提取参数的辅助函数 (保留原有的，因为它从多个源查找参数)
        def _get_param_val(sources: List[Dict], key: str, default: Any = None) -> Any:
            """从参数源列表中按顺序查找键的值，返回第一个找到的值或默认值。"""
            for source_dict in sources:
                if isinstance(source_dict, dict) and key in source_dict:
                    return source_dict[key]
            return default
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
        macd_fast = _get_param_val(param_sources, 'period_fast', 12) # 优先从 tf_params, ia_params, fe_params, bs_params 查找 period_fast
        macd_slow = _get_param_val(param_sources, 'period_slow', 26) # 优先从 tf_params, ia_params, fe_params, bs_params 查找 period_slow
        macd_sig = _get_param_val(param_sources, 'signal_period', 9) # 优先从 tf_params, ia_params, fe_params, bs_params 查找 signal_period
        # RSI参数
        rsi_period = _get_param_val(param_sources, 'period', 14) # 优先从 tf_params, ia_params, fe_params, bs_params 查找 period
        # KDJ参数 (注意JSON和策略中参数名的对应)
        kdj_k_period = _get_param_val(param_sources, 'kdj_period_k', _get_param_val(param_sources, 'period', 9)) # 优先 kdj_period_k，其次 period
        kdj_d_period = _get_param_val(param_sources, 'kdj_period_d', _get_param_val(param_sources, 'signal_period', 3)) # 优先 kdj_period_d，其次 signal_period
        kdj_j_smooth_k = _get_param_val(param_sources, 'kdj_period_j', _get_param_val(param_sources, 'smooth_k_period', 3)) # 优先 kdj_period_j，其次 smooth_k_period
        # BOLL参数 (基础评分用的BOLL)
        boll_period = _get_param_val(param_sources, 'boll_period', _get_param_val(param_sources, 'period', 20)) # 优先 boll_period，其次 period
        boll_std_dev = _get_param_val(param_sources, 'boll_std_dev', _get_param_val(param_sources, 'std_dev', 2.0)) # 优先 boll_std_dev，其次 std_dev
        # CCI参数
        cci_period = _get_param_val(param_sources, 'period', 14)
        # MFI参数
        mfi_period = _get_param_val(param_sources, 'period', 14)
        # ROC参数
        roc_period = _get_param_val(param_sources, 'period', 12)
        # DMI参数
        dmi_period = _get_param_val(param_sources, 'period', 14)
        # SAR参数
        sar_step = _get_param_val(param_sources, 'af_step', 0.02)
        sar_max_af = _get_param_val(param_sources, 'max_af', 0.2)
        # EMA/SMA 基础参数 (如果 score_indicators 中包含它们，通常是某个默认周期)
        ema_base_period = _get_param_val(param_sources, 'ema_base_period', _get_param_val(param_sources, 'period', 20)) # 优先 ema_base_period，其次 period
        sma_base_period = _get_param_val(param_sources, 'sma_base_period', _get_param_val(param_sources, 'period', 20)) # 优先 sma_base_period，其次 period
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
                for col_conf in indi_naming_conv['output_columns']:
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
            amt_ma_period_vc = _get_param_val(param_sources, 'amount_ma_period', 15)
            cmf_period_vc = _get_param_val(param_sources, 'cmf_period', 20)
            obv_ma_period_vc = _get_param_val(param_sources, 'obv_ma_period', 10)
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
            stoch_k_ia = _get_param_val(param_sources, 'stoch_k', _get_param_val(param_sources, 'k_period', 14))
            stoch_d_ia = _get_param_val(param_sources, 'stoch_d', _get_param_val(param_sources, 'd_period', 3))
            stoch_smooth_k_ia = _get_param_val(param_sources, 'stoch_smooth_k', _get_param_val(param_sources, 'smooth_k_period', 3))
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
            vol_ma_period_ia = _get_param_val(param_sources, 'volume_ma_period', _get_param_val(param_sources, 'period', 20))
            if ia_params_global.get('calculate_vol_ma', False) and 'VOL_MA' in indicator_naming_conv:
                vol_ma_naming_conf = indicator_naming_conv['VOL_MA']
                if isinstance(vol_ma_naming_conf, dict) and 'output_columns' in vol_ma_naming_conf and isinstance(vol_ma_naming_conf['output_columns'], list):
                    vol_ma_name_patterns = [c['name_pattern'] for c in vol_ma_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(vol_ma_period_ia, (int, float)) and vol_ma_period_ia > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(vol_ma_name_patterns, period=vol_ma_period_ia):
                            required.add(f"{name}_{analyze_tf_str}")
            # VWAP
            vwap_anchor_ia = _get_param_val(param_sources, 'vwap_anchor', None)
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
                ichimoku_tenkan_ia = _get_param_val(param_sources, 'ichimoku_tenkan', _get_param_val(param_sources, 'tenkan_period', 9))
                ichimoku_kijun_ia = _get_param_val(param_sources, 'ichimoku_kijun', _get_param_val(param_sources, 'kijun_period', 26))
                ichimoku_senkou_ia = _get_param_val(param_sources, 'ichimoku_senkou', _get_param_val(param_sources, 'senkou_period', 52))
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
            atr_period_fe = _get_param_val(param_sources, 'atr_period', _get_param_val(param_sources, 'period', 14))
            if fe_params_global.get('calculate_atr', False) and 'ATR' in indicator_naming_conv:
                atr_naming_conf = indicator_naming_conv['ATR']
                if isinstance(atr_naming_conf, dict) and 'output_columns' in atr_naming_conf and isinstance(atr_naming_conf['output_columns'], list):
                    atr_patterns = [c['name_pattern'] for c in atr_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(atr_period_fe, (int, float)) and atr_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(atr_patterns, period=atr_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # HV (Historical Volatility)
            hv_period_fe = _get_param_val(param_sources, 'hv_period', _get_param_val(param_sources, 'period', 20))
            if fe_params_global.get('calculate_hv', False) and 'HV' in indicator_naming_conv:
                hv_naming_conf = indicator_naming_conv['HV']
                if isinstance(hv_naming_conf, dict) and 'output_columns' in hv_naming_conf and isinstance(hv_naming_conf['output_columns'], list):
                    hv_patterns = [c['name_pattern'] for c in hv_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(hv_period_fe, (int, float)) and hv_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(hv_patterns, period=hv_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # KC (Keltner Channels)
            kc_ema_period_fe = _get_param_val(param_sources, 'kc_ema_period', _get_param_val(param_sources, 'ema_period', 20))
            kc_atr_period_fe = _get_param_val(param_sources, 'kc_atr_period', _get_param_val(param_sources, 'atr_period', 10))
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
            mom_period_fe = _get_param_val(param_sources, 'mom_period', _get_param_val(param_sources, 'period', 10))
            if fe_params_global.get('calculate_mom', False) and 'MOM' in indicator_naming_conv:
                mom_naming_conf = indicator_naming_conv['MOM']
                if isinstance(mom_naming_conf, dict) and 'output_columns' in mom_naming_conf and isinstance(mom_naming_conf['output_columns'], list):
                    mom_patterns = [c['name_pattern'] for c in mom_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(mom_period_fe, (int, float)) and mom_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(mom_patterns, period=mom_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # WILLR
            willr_period_fe = _get_param_val(param_sources, 'willr_period', _get_param_val(param_sources, 'period', 14))
            if fe_params_global.get('calculate_willr', False) and 'WILLR' in indicator_naming_conv:
                willr_naming_conf = indicator_naming_conv['WILLR']
                if isinstance(willr_naming_conf, dict) and 'output_columns' in willr_naming_conf and isinstance(willr_naming_conf['output_columns'], list):
                    willr_patterns = [c['name_pattern'] for c in willr_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(willr_period_fe, (int, float)) and willr_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(willr_patterns, period=willr_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # VROC
            vroc_period_fe = _get_param_val(param_sources, 'vroc_period', _get_param_val(param_sources, 'period', 10))
            if fe_params_global.get('calculate_vroc', False) and 'VROC' in indicator_naming_conv:
                vroc_naming_conf = indicator_naming_conv['VROC']
                if isinstance(vroc_naming_conf, dict) and 'output_columns' in vroc_naming_conf and isinstance(vroc_naming_conf['output_columns'], list):
                    vroc_patterns = [c['name_pattern'] for c in vroc_naming_conf['output_columns'] if isinstance(c, dict) and 'name_pattern' in c]
                    if isinstance(vroc_period_fe, (int, float)) and vroc_period_fe > 0:
                        for name in TrendFollowingStrategy._format_indicator_name(vroc_patterns, period=vroc_period_fe):
                            required.add(f"{name}_{analyze_tf_str}")
            # AROC
            aroc_period_fe = _get_param_val(param_sources, 'aroc_period', _get_param_val(param_sources, 'period', 10))
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
        # IndicatorService 已经计算了基础指标，但策略内部可能需要根据这些指标计算一个综合评分
        # strategy_utils.calculate_all_indicator_scores 应该接收 IndicatorService 返回的 DataFrame
        # 并根据 bs_params 和 NAMING_CONFIG 找到对应的指标列（带后缀），然后计算 SCORE 列
        logger.info(f"[{self.strategy_name}][{stock_code}] 计算基础指标评分...")
        # 传递命名规范和 IndicatorService 返回的 indicator_configs (如果需要的话)
        # strategy_utils.calculate_all_indicator_scores 需要知道如何找到带后缀的指标列
        # 它可以根据 bs_params 中的 score_indicators 列表和 NAMING_CONFIG 来构建带后缀的列名
        indicator_scores_df = strategy_utils.calculate_all_indicator_scores(data, bs_params, indicator_configs, NAMING_CONFIG)

        if indicator_scores_df is None or indicator_scores_df.empty or not any(col.startswith('SCORE_') for col in (indicator_scores_df.columns if isinstance(indicator_scores_df, pd.DataFrame) else [])):
            logger.warning(f"[{self.strategy_name}][{stock_code}] 未计算或未找到任何指标评分列。基础评分将为中性50。")
            # 如果没有评分列，创建一个全为50的 DataFrame 作为基础
            indicator_scores_df = pd.DataFrame(50.0, index=data.index, columns=['SCORE_DEFAULT_50'])

        # 应用时间框架权重并合并基础评分
        logger.info(f"[{self.strategy_name}][{stock_code}] 应用时间框架权重并合并基础评分...")
        # current_weights 的计算逻辑与 __init__ 中验证时基本一致
        current_weights: Dict[str, float]
        timeframes_from_config = bs_params.get('timeframes', [])
        if not isinstance(timeframes_from_config, list) or not timeframes_from_config:
            logger.error(f"[{self.strategy_name}][{stock_code}] 'base_scoring.timeframes' 为空或无效，无法计算基础评分。")
             # 返回一个全为50的Series和一个空的Dict
            return pd.Series(50.0, index=data.index, name='final_rule_signal'), {}

        # 确定时间框架权重 current_weights
        if self.timeframe_weights is not None and isinstance(self.timeframe_weights, dict):
            current_weights = self.timeframe_weights.copy()
            defined_tfs_set = set(timeframes_from_config)
            # 移除不在配置列表中的权重
            for tf_w in list(current_weights.keys()):
                if tf_w not in defined_tfs_set:
                    del current_weights[tf_w]
            # 为配置列表中但没有权重的添加0权重
            for tf_d in defined_tfs_set:
                if tf_d not in current_weights:
                    current_weights[tf_d] = 0.0
        else:
            # 使用 focus_weight 逻辑计算权重
            focus_weight_val = tf_params.get('focus_weight', 0.45)
            # 确保 focus_weight_val 是有效数字且在 [0, 1] 范围内
            if not isinstance(focus_weight_val, (int, float)) or not (0 <= focus_weight_val <= 1):
                 logger.warning(f"[{self.strategy_name}][{stock_code}] 'focus_weight' 参数无效 ({focus_weight_val})，使用默认值 0.45。")
                 focus_weight_val = 0.45

            num_other_tfs = len(timeframes_from_config) - 1
            if num_other_tfs > 0:
                base_weight_val = (1.0 - focus_weight_val) / num_other_tfs
            elif len(timeframes_from_config) == 1:
                base_weight_val = 0.0
                focus_weight_val = 1.0
            else: # timeframes_from_config 为空的情况已在前面处理，这里理论上不会发生
                base_weight_val = 0.0
                focus_weight_val = 0.0

            current_weights = {tf: base_weight_val for tf in timeframes_from_config if tf != self.focus_timeframe}
            if self.focus_timeframe in timeframes_from_config: # 确保 focus_timeframe 在列表中才添加权重
                current_weights[self.focus_timeframe] = focus_weight_val
            elif timeframes_from_config: # 如果 timeframes_from_config 不为空但 focus_tf 不在，将权重平均分配
                 logger.warning(f"[{self.strategy_name}][{stock_code}] focus_timeframe '{self.focus_timeframe}' 不在配置的时间框架列表中。将权重平均分配。")
                 avg_weight = 1.0 / len(timeframes_from_config) if timeframes_from_config else 0.0
                 current_weights = {tf: avg_weight for tf in timeframes_from_config}

        self._normalize_weights(current_weights)

        # 初始化 base_score_raw，用于累加加权得分
        # 在计算权重后，但在加权求和循环之前初始化 base_score_raw
        base_score_raw = pd.Series(0.0, index=data.index)
        total_effective_weight = 0.0 # 用于确认加权总和

        # 根据确定的权重 current_weights 计算加权平均基础得分
        # 将加权求和循环移动到确定 current_weights 之后
        for tf_s in timeframes_from_config:
            tf_weight = current_weights.get(tf_s, 0)
            if tf_weight == 0: continue

            # SCORE_ 列名由 strategy_utils.calculate_all_indicator_scores 内部逻辑决定
            tf_score_cols = [col for col in indicator_scores_df.columns if col.endswith(f'_{tf_s}') and col.startswith('SCORE_')]

            if tf_score_cols:
                # 过滤掉所有值都是 NaN 的列，避免拉低平均分
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

        # 如果所有有效权重总和为零，或者 timeframes_from_config 为空，则基础评分全为 50
        # 这个检查现在在加权求和之后进行
        if total_effective_weight == 0 and timeframes_from_config: # 如果timeframes不为空但有效权重总和为零
             logger.warning(f"[{self.strategy_name}][{stock_code}] 所有时间框架的有效权重总和为零，基础评分将为中性50。")
             base_score_raw = pd.Series(50.0, index=data.index)
        # elif not timeframes_from_config: # timeframes为空的情况已在前面处理并返回
        #      pass # base_score_raw 已经是全0或全50，取决于初始化
        elif not np.isclose(total_effective_weight, sum(current_weights.values())) and sum(current_weights.values()) > 0:
            # logger.warning(f"[{self.strategy_name}][{stock_code}] 有效权重总和 {total_effective_weight:.4f} 与预期权重总和 {sum(current_weights.values()):.4f} 不符。可能存在计算问题。")
            pass # 这是一个调试信息，不是必须的警告

        # 对计算出的原始基础评分进行剪切和填充 NaN
        # 这一行现在在加权求和计算之后执行
        base_score_raw = base_score_raw.clip(0, 100).fillna(50.0)

        # 量能调整基础评分 (假定 strategy_utils 包含此函数)
        logger.info(f"[{self.strategy_name}][{stock_code}] 执行量能调整/分析模块...")
        vc_params_adjusted = vc_params.copy()
        # 确保 vc_params_adjusted 是字典
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
             # 如果量能调整跳过，adjusted_score 就等于 raw_score
             volume_adjusted_results_df = pd.DataFrame(index=data.index)
             volume_adjusted_results_df['ADJUSTED_SCORE'] = base_score_raw
             # 添加量价异动默认列 (全0)
             internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
             vol_spike_pattern = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'].startswith("VOL_SPIKE_SIGNAL")), "VOL_SPIKE_SIGNAL_{timeframe}")
             # 为每个量能时间框架添加默认列
             for tf in ([self.focus_timeframe] if not vc_tf_list else vc_tf_list):
                 vol_spike_col_name = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=tf)[0]
                 volume_adjusted_results_df[vol_spike_col_name] = 0.0

        else:
             try:
                 # strategy_utils.adjust_score_with_volume 需要知道如何找到带后缀的 OHLCV, OBV, AMT_MA, VOL_MA 列
                 volume_adjusted_results_df = strategy_utils.adjust_score_with_volume(
                    preliminary_score=base_score_raw,
                    data=data, # 传递包含所有数据的 DataFrame
                    vc_params=vc_params_adjusted,
                    vc_tf_list=vc_tf_list, # 传递时间框架列表
                    vol_ma_period=vol_ma_period_vc, # 传递 VOL_MA 周期
                    obv_ma_period=obv_ma_period_vc, # 传递 OBV_MA 周期
                    naming_config=NAMING_CONFIG # 传递命名规范
                 )
                 # 检查 ADJUSTED_SCORE 列是否存在
                 if 'ADJUSTED_SCORE' not in volume_adjusted_results_df.columns:
                      logger.error(f"[{self.strategy_name}][{stock_code}] 量能调整模块未能生成 'ADJUSTED_SCORE' 列。将使用原始基础评分。")
                      volume_adjusted_results_df['ADJUSTED_SCORE'] = base_score_raw
                 logger.info(f"[{self.strategy_name}][{stock_code}] 量能调整/分析完成。")
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
        logger.info(f"[{self.strategy_name}][{stock_code}] 执行量价背离检测...")
        divergence_signals_df = pd.DataFrame(index=data.index)
        if isinstance(dd_params, dict) and dd_params.get('enabled', True):
            try:
                # strategy_utils.detect_divergence 需要知道如何找到带后缀的指标列
                divergence_signals_df = strategy_utils.detect_divergence(data=data, dd_params=dd_params, naming_config=NAMING_CONFIG)
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

        else: # 背离检测未启用
            logger.info(f"[{self.strategy_name}][{stock_code}] 量价背离检测未启用。")
            internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
            has_bearish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BEARISH_DIVERGENCE"), "HAS_BEARISH_DIVERGENCE")
            has_bullish_div_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c['name_pattern'] == "HAS_BULLISH_DIVERGENCE"), "HAS_BULLISH_DIVERGENCE")
            divergence_signals_df[has_bearish_div_col] = False
            divergence_signals_df[has_bullish_div_col] = False


        # 综合规则信号 (假定 strategy_utils 包含 combine_rule_signals 函数)
        logger.info(f"[{self.strategy_name}][{stock_code}] 计算综合规则信号...")
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

        # 假设我们使用第一个量能时间框架的量价异动信号进行规则组合，或者对多个时间框架的信号进行某种聚合（这里简化为使用第一个）
        vol_spike_signal_col = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=vc_tf_list_vol_spike[0])[0] if vc_tf_list_vol_spike else None

        # 检查量价异动列是否存在，如果不存在或没有指定时间框架，则使用全0 Series
        if vol_spike_signal_col and vol_spike_signal_col in volume_adjusted_results_df.columns:
             volume_spike_norm = volume_adjusted_results_df.get(vol_spike_signal_col, pd.Series(0.0, index=data.index)).fillna(0.0) # 假定范围是-1, 0, 1
        else:
             logger.warning(f"[{self.strategy_name}][{stock_code}] 量价异动信号列 '{vol_spike_signal_col}' (基于时间框架 {vc_tf_list_vol_spike[0] if vc_tf_list_vol_spike else '无'}) 不存在或量能时间框架未配置。规则组合中量价异动贡献为 0。")
             volume_spike_norm = pd.Series(0.0, index=data.index)


        total_weighted_contribution = pd.Series(0.0, index=data.index)
        # 使用 .get() 方法安全获取权重，避免 KeyError
        total_weighted_contribution += base_score_norm * weights.get('base_score', 0)
        total_weighted_contribution += alignment_norm * weights.get('alignment', 0)
        total_weighted_contribution += long_context_norm * weights.get('long_context', 0)
        total_weighted_contribution += momentum_norm * weights.get('momentum', 0)
        total_weighted_contribution += ema_cross_norm * weights.get('ema_cross', 0)
        total_weighted_contribution += boll_breakout_norm * weights.get('boll_breakout', 0)
        total_weighted_contribution += adx_strength_norm * weights.get('adx_strength', 0)
        total_weighted_contribution += stoch_signal_norm * weights.get('stoch_signal', 0) # 新增：STOCH信号权重
        total_weighted_contribution += vwap_dev_norm * weights.get('vwap_deviation', 0) # 新增：VWAP偏离信号权重
        total_weighted_contribution += volume_spike_norm * weights.get('volume_spike', 0)

        # 将加权贡献转换回 0-100 的分数范围
        base_rule_signal_before_adjust = 50.0 + total_weighted_contribution * 50.0
        base_rule_signal_before_adjust = base_rule_signal_before_adjust.clip(0, 100)

        # 应用 ADX 增强 (假定 _apply_adx_boost 已实现)
        final_rule_signal = self._apply_adx_boost(
            base_rule_signal_before_adjust,
            adx_strength_norm,
            (base_rule_signal_before_adjust.fillna(50.0) - 50.0) / 50.0 # 传递归一化后的基础信号方向
        )

        # 应用背离惩罚 (假定 _apply_divergence_penalty 已实现)
        # _apply_divergence_penalty 需要接收背离信号 DataFrame 和 dd_params
        final_rule_signal = self._apply_divergence_penalty(final_rule_signal, divergence_signals_df, dd_params) # 传递 divergence_signals_df 和 dd_params

        # 应用趋势确认过滤 (假定 _apply_trend_confirmation 已实现)
        # _apply_trend_confirmation 需要知道趋势持续时间阈值 (self.trend_duration_threshold_strong/moderate)
        # 它还需要访问 ADX 强度信号 (adx_strength_norm) 和基础信号方向 ((final_rule_signal - 50)/50)
        # 可以在 _apply_trend_confirmation 内部获取这些，或者作为参数传递
        # 为了简化，假设 _apply_trend_confirmation 内部可以访问这些信息
        final_rule_signal = self._apply_trend_confirmation(final_rule_signal) # 保持原签名，假设内部获取所需数据

        # 最终剪切和四舍五入
        final_rule_signal = final_rule_signal.clip(0, 100).round(2)

        # 准备返回的中间结果字典
        intermediate_results = {
            'base_score_raw': base_score_raw,
            'base_score_volume_adjusted': base_score_volume_adjusted,
            'indicator_scores_df': indicator_scores_df, # 包含 SCORE_ 列
            'volume_analysis_df': volume_adjusted_results_df, # 包含 ADJUSTED_SCORE 和 VOL_SPIKE_SIGNAL_* 列
            'trend_analysis_df': trend_analysis_df, # 包含 alignment_signal, adx_strength_signal 等列
            'divergence_signals_df': divergence_signals_df # 包含 HAS_BEARISH_DIVERGENCE, HAS_BULLISH_DIVERGENCE 列
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
        # 修改: 使用更明确的变量名从 self.params 获取配置，确保来源是 JSON 文件
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
            default_ema_periods = trend_analysis_config.get('ema_periods', [5, 10, 20, 60]) # 修改: 从配置读取用于默认信号
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
        # 修改: param_sources 使用新的配置变量
        param_sources = [
            trend_following_config,
            volume_confirmation_config,
            indicator_analysis_config,
            feature_engineering_config,
            base_scoring_config
        ]
        # 1. 计算分数 EMA
        # 修改: 使用 trend_analysis_config 获取 ema_periods
        all_ema_periods = trend_analysis_config.get('ema_periods', [5, 10, 20, 60])
        if not isinstance(all_ema_periods, list) or not all_ema_periods: # 确保是列表且非空
             logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 'trend_analysis.ema_periods' 参数无效或为空，使用默认值 [5, 10, 20, 60]。")
             all_ema_periods = [5, 10, 20, 60]
        for period in all_ema_periods:
            try:
                if isinstance(period, (int, float)) and period > 0:
                    ema_result = ta.ema(score_series_filled, length=int(period))
                    analysis_df[f'ema_score_{period}'] = ema_result
                    print(f"[{self.strategy_name}] _perform_trend_analysis: EMA Score {period} 计算完成。结果后10行:\n{ema_result.tail(10)}") # 将 .head() 修改为 .tail(10) 以显示最后10个值
                else:
                    logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: EMA Score 周期参数无效: {period}. 跳过计算。")
                    analysis_df[f'ema_score_{period}'] = np.nan # 原始逻辑: 无效周期设为 NaN
            except Exception as e:
                logger.error(f"[{self.strategy_name}] _perform_trend_analysis: 计算 EMA Score {period} 时出错: {e}", exc_info=True)
                analysis_df[f'ema_score_{period}'] = np.nan # 原始逻辑: 出错时设为 NaN
        # 2. 计算 EMA 排列信号
        # 修改: 使用 trend_following_config 获取 ema_alignment_periods
        ema_periods_align = trend_following_config.get('ema_alignment_periods', all_ema_periods[:4] if len(all_ema_periods) >=4 else [])
        if not isinstance(ema_periods_align, list) or len(ema_periods_align) < 4:
             logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 'trend_following_params.ema_alignment_periods' 参数无效或数量不足4个，尝试使用 'trend_analysis.ema_periods' 的前4个 ({all_ema_periods[:4]})。")
             ema_periods_align = all_ema_periods[:4] if len(all_ema_periods) >=4 else [] # 修改: 提供备用值或空列表
             if len(ema_periods_align) < 4:
                  logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: 可用的 EMA Score 周期不足4个 ({len(ema_periods_align)} found)，无法计算完整的 EMA 排列信号。")
                  analysis_df['alignment_signal'] = 0.0 # 修改: 无法计算则赋默认值

        if len(ema_periods_align) == 4 : # 修改: 只有当周期足够时才计算
            ema_alignment_cols = [f'ema_score_{p}' for p in ema_periods_align[:4]]
            if all(col in analysis_df.columns for col in ema_alignment_cols):
                try:
                    s_ema, m1_ema, m2_ema, l_ema = (analysis_df[col].fillna(50.0) for col in ema_alignment_cols)
                    alignment = pd.Series(0, index=analysis_df.index, dtype=float) # 修改: 明确 dtype
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
        # 修改: 使用 trend_following_config 获取 ema_cross_short/long
        ema_cross_short = trend_following_config.get('ema_cross_short', all_ema_periods[0] if len(all_ema_periods) > 0 else 5)
        ema_cross_long = trend_following_config.get('ema_cross_long', all_ema_periods[1] if len(all_ema_periods) > 1 else 20)
        short_ema_col = f'ema_score_{ema_cross_short}'
        long_ema_col = f'ema_score_{ema_cross_long}'
        if short_ema_col in analysis_df.columns and long_ema_col in analysis_df.columns:
            try:
                short_ema = analysis_df[short_ema_col].fillna(50.0)
                long_ema = analysis_df[long_ema_col].fillna(50.0)
                golden_cross = (short_ema > long_ema) & (short_ema.shift(1).fillna(long_ema.shift(1)) <= long_ema.shift(1).fillna(short_ema.shift(1))) # 修改: fillna 策略
                death_cross = (short_ema < long_ema) & (short_ema.shift(1).fillna(long_ema.shift(1)) >= long_ema.shift(1).fillna(short_ema.shift(1)))  # 修改: fillna 策略
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
        # 修改: 使用 trend_following_config 和 trend_analysis_config
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
        # 修改: 使用 trend_analysis_config 获取 volatility_window
        volatility_window = trend_analysis_config.get('volatility_window', 10)
        if not (isinstance(volatility_window, int) and volatility_window > 0): # 修改: 验证参数有效性
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
        # 修改: 使用 trend_analysis_config 获取 long_term_ema_period
        long_term_ema_period_context = trend_analysis_config.get('long_term_ema_period', all_ema_periods[-1] if all_ema_periods else 60)
        if not (isinstance(long_term_ema_period_context, (int, float)) and long_term_ema_period_context > 0): # 修改: 验证参数有效性
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
        if not (isinstance(dmi_period_signal, (int, float)) and dmi_period_signal > 0): # 修改: 验证参数有效性
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
                isinstance(stoch_smooth_k_p_signal, (int, float)) and stoch_smooth_k_p_signal > 0): # 修改: 验证参数有效性
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
                turning_up = (k_val > d_val) & (k_val.shift(1).fillna(d_val.shift(1)) <= d_val.shift(1).fillna(k_val.shift(1))) # 修改: fillna 策略
                turning_down = (k_val < d_val) & (k_val.shift(1).fillna(d_val.shift(1)) >= d_val.shift(1).fillna(k_val.shift(1))) # 修改: fillna 策略
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
                if vwap_safe.isnull().all() or vwap_safe.eq(0).all(): # 修改: 增加对全0的检查
                    logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: VWAP ({vwap_col}) 数据无效 (全为 NaN 或 0)，无法计算 VWAP 偏离。")
                    analysis_df['vwap_deviation_signal'] = 0.0
                    analysis_df['vwap_deviation_percent'] = 0.0
                else:
                    # 确保vwap_safe不为0，避免除零错误
                    deviation = ((close_price - vwap_safe) / vwap_safe.replace(0, np.nan)).fillna(0.0) # 修改: 再次确保除数不为0
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
        # 修改: boll_breakout_params_tf 从 trend_following_config 获取
        boll_breakout_params_tf = trend_following_config.get('boll_breakout_params', {})
        boll_period_signal, boll_std_dev_signal = None, None # 修改: 初始化
        if isinstance(boll_breakout_params_tf, dict) and 'period' in boll_breakout_params_tf and 'std_dev' in boll_breakout_params_tf:
             boll_period_signal = boll_breakout_params_tf['period']
             boll_std_dev_signal = boll_breakout_params_tf['std_dev']
        else:
             # 修改: 如果策略特定参数没有配置，回退使用 base_scoring_config 中的 BOLL 参数
             boll_period_signal = base_scoring_config.get("boll_period", 20) # 修改: 默认值与注释中讨论的默认值(20,2.0)保持一致或根据实际需要
             boll_std_dev_signal = base_scoring_config.get("boll_std_dev", 2.0)
             logger.debug(f"[{self.strategy_name}] _perform_trend_analysis: 未找到 'trend_following_params.boll_breakout_params' 或配置无效，回退使用 'base_scoring' BOLL 参数 (period={boll_period_signal}, std_dev={boll_std_dev_signal})。")
        # 确保参数是有效数字 (此段逻辑原先被注释，根据需要决定是否启用)
        if not (isinstance(boll_period_signal, (int, float)) and boll_period_signal > 0 and
                isinstance(boll_std_dev_signal, (int, float)) and boll_std_dev_signal > 0):
            logger.warning(f"[{self.strategy_name}] _perform_trend_analysis: BOLL 突破信号参数无效或未正确读取: period={boll_period_signal}, std_dev={boll_std_dev_signal}. 使用默认值 20, 2.0。")
            boll_period_signal, boll_std_dev_signal = 20, 2.0 # 确保有默认值

        std_str_signal = f"{float(boll_std_dev_signal):.1f}" # 修改: 确保 boll_std_dev_signal 转为 float
        bbu_col = f'BBU_{int(boll_period_signal)}_{std_str_signal}_{focus_tf}' # 修改: 确保 boll_period_signal 转为 int
        bbl_col = f'BBL_{int(boll_period_signal)}_{std_str_signal}_{focus_tf}' # 修改: 确保 boll_period_signal 转为 int
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
        filtered_signal = pd.Series(50.0, index=final_signal.index) # 默认过滤后信号是中性 50

        if len(final_signal_filled) < confirmation_periods:
             logger.warning(f"[{self.strategy_name}] 趋势确认过滤：数据长度 ({len(final_signal_filled)}) 不足确认周期 ({confirmation_periods})。跳过过滤。")
             return final_signal_filled # 数据不足时不进行过滤

        try:
            # 判断信号是否持续高于上限或低于下限足够周期
            above_upper_streak = (final_signal_filled >= trend_threshold_upper).rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() == confirmation_periods
            below_lower_streak = (final_signal_filled <= trend_threshold_lower).rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() == confirmation_periods

            # 将确认后的信号赋值给 filtered_signal
            # 只有在连续满足条件时，才将原始信号值传递过来
            filtered_signal.loc[above_upper_streak] = final_signal_filled.loc[above_upper_streak]
            filtered_signal.loc[below_lower_streak] = final_signal_filled.loc[below_lower_streak]

            # 填充 NaN，特别是开头 confirmation_periods-1 个点
            filtered_signal = filtered_signal.fillna(50.0) # 默认填充 50

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
        logger.info(f"[{self.strategy_name}][{stock_code}] 计算规则信号及中间数据...")
        final_rule_signal, intermediate_results_dict = self._calculate_rule_based_signal(
            data=data, stock_code=stock_code, indicator_configs=indicator_configs_to_use
        )
        logger.info(f"[{self.strategy_name}][{stock_code}] 规则信号计算完成。")

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
        logger.info(f"[{self.strategy_name}][{stock_code}] 准备 Transformer 模型预测...")
        # 添加 transformer_signal 列 (内部列)，默认填充 50.0
        processed_data['transformer_signal'] = pd.Series(50.0, index=processed_data.index)

        self.set_model_paths(stock_code)
        # 调用加载 Transformer 模型的方法
        self.load_transformer_model(stock_code)

        if self.transformer_model and self.feature_scaler and self.target_scaler and self.selected_feature_names_for_transformer:
            try:
                logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型已加载，开始进行预测...")

                # 预测函数 predict_with_transformer_model 应该处理数据的选取、标准化、窗口化和预测
                # 确保 processed_data 包含所有 selected_feature_names_for_transformer 中的列
                # 预测函数会返回与原始数据索引对齐的预测结果 Series
                predicted_signal_series = predict_with_transformer_model(
                    model=self.transformer_model,
                    data=processed_data, # 传入包含所有可能特征的 DataFrame
                    feature_scaler=self.feature_scaler,
                    target_scaler=self.target_scaler,
                    selected_feature_names=self.selected_feature_names_for_transformer,
                    window_size=self.transformer_window_size,
                    device=self.device,
                    # 可能需要传递 data_prep_config 中的一些参数给 predict_with_transformer_model
                    data_prep_config=self.transformer_data_prep_config
                )

                if predicted_signal_series is not None and not predicted_signal_series.empty:
                    # 将预测结果合并回 processed_data 的 transformer_signal 列
                    # predicted_signal_series 的索引应该与原始 processed_data 对齐 (考虑 window_size 的偏移)
                    # 预测结果对应的是窗口的最后一个时间点
                    # predict_with_transformer_model 应该已经处理好索引对齐问题，直接赋值即可
                    processed_data.loc[predicted_signal_series.index, 'transformer_signal'] = predicted_signal_series

                    latest_pred_idx = predicted_signal_series.index[-1] if not predicted_signal_series.empty else 'N/A'
                    latest_pred_val = predicted_signal_series.iloc[-1] if not predicted_signal_series.empty else np.nan
                    logger.info(f"[{self.strategy_name}][{stock_code}] Transformer 模型预测完成，最新预测信号 ({latest_pred_idx}): {latest_pred_val:.2f}")
                else:
                    logger.warning(f"[{self.strategy_name}][{stock_code}] Transformer 模型预测返回空 Series 或 None。")

            except Exception as e:
                logger.error(f"[{self.strategy_name}][{stock_code}] Transformer 模型预测出错: {e}", exc_info=True)
                # 预测出错时，transformer_signal 列保持默认值 50.0
        else:
            logger.warning(f"[{self.strategy_name}][{stock_code}] Transformer 模型/Scaler/特征列表未加载，跳过 Transformer 预测。Transformer_signal 将保持默认值 50.0。")


        # --- 阶段 3: 组合规则信号和 Transformer 信号 ---
        logger.info(f"[{self.strategy_name}][{stock_code}] 组合规则信号和 Transformer 信号...")
        try:
            # 获取信号组合权重
            # 从 tf_params 中获取 signal_combination_weights，并确保是字典
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
        logger.info(f"[{self.strategy_name}][{stock_code}] 生成最终交易信号...")
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
            logger.warning(f"[{self.strategy_name}][{stock_code}] 缺失必需的 Transformer 模型/Scaler/特征文件，无法加载。")
            self._reset_model_components() # 重置状态
            return

        try:
            # 加载选中特征列表
            with open(self.selected_features_path, 'r', encoding='utf-8') as f:
                 self.selected_feature_names_for_transformer = json.load(f)
            logger.debug(f"[{self.strategy_name}][{stock_code}] 选中特征名列表 ({len(self.selected_feature_names_for_transformer)}个) 从 {self.selected_features_path} 加载。")

            num_features = len(self.selected_feature_names_for_transformer)
            if num_features == 0:
                 logger.error(f"[{self.strategy_name}][{stock_code}] 加载的选中特征列表为空，无法构建模型。")
                 self._reset_model_components() # 重置状态
                 return

            # 构建模型
            self.transformer_model = build_transformer_model(
                num_features=num_features,
                model_config=self.transformer_model_config,
                summary=False, # 加载模型时不打印摘要
                window_size=self.transformer_window_size
            )
            self.transformer_model.to(self.device)

            # 加载模型权重
            self.transformer_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
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
                        selected_feature_names: List[str]):
        """
        保存准备好的 Transformer 训练数据和 Scaler。
        """
        self.set_model_paths(stock_code)
        if not all([self.all_prepared_data_npz_path, self.feature_scaler_path, self.target_scaler_path, self.selected_features_path]):
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存准备数据：部分或全部路径未设置。")
            # 不是 raise 异常，而是返回 False 表示保存失败
            return False

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

            return True # 返回 True 表示保存成功
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存准备好的数据、Scaler或特征列表时出错: {e}", exc_info=True)
            # 返回 False 表示保存失败
            return False

    def load_prepared_data(self, stock_code: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Union[MinMaxScaler, StandardScaler]], Optional[Union[MinMaxScaler, StandardScaler]]]:
        """
        从文件加载特定股票准备好的 Transformer 训练数据和 Scaler。
        返回 NumPy 数组和 Scaler 对象。加载失败时返回空 NumPy 数组和 None。
        """
        self.set_model_paths(stock_code)
        empty_array = np.array([])

        if not all([self.all_prepared_data_npz_path, self.feature_scaler_path, self.target_scaler_path, self.selected_features_path]):
            logger.warning(f"[{self.strategy_name}][{stock_code}] 加载准备数据：部分或全部路径未设置。")
            self.selected_feature_names_for_transformer = [] # 重置列表
            # 返回空 NumPy 数组和 None
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None

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

            # 检查加载的数据和 scaler 是否匹配 (维度检查)
            if feature_scaler is not None and features_train.shape[1] != len(self.selected_feature_names_for_transformer):
                 logger.error(f"[{self.strategy_name}][{stock_code}] 加载的数据特征维度 ({features_train.shape[1]}) 与选中特征列表长度 ({len(self.selected_feature_names_for_transformer)}) 不匹配！")
                 self.selected_feature_names_for_transformer = [] # 重置列表
                 return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None, None # 返回 None 表示不匹配

            logger.info(f"[{self.strategy_name}][{stock_code}] 准备好的数据和 Scaler 已成功加载。")
            return features_train, targets_train, features_val, targets_val, features_test, targets_test, feature_scaler, target_scaler
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 加载准备好的数据、Scaler或特征列表时出错: {e}", exc_info=True)
            self.selected_feature_names_for_transformer = [] # 重置列表
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
                    fs_max_features=self.transformer_data_prep_config.get('fs_max_features', 50),
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
        """
        trend_duration_info = {
            'bullish_duration': 0, 'bearish_duration': 0,
            'bullish_duration_text': '0分钟', 'bearish_duration_text': '0分钟',
            'current_trend': '中性', 'trend_strength': '不明', 'duration_status': '短'
        }
        # final_rule_signal 是内部列，从 JSON 配置获取列名
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        final_rule_signal_col = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c.get('name_pattern') == "final_rule_signal"), "final_rule_signal")

        if final_rule_signal_col not in data_with_signals.columns or data_with_signals[final_rule_signal_col].isnull().all():
            logger.warning(f"[{self.strategy_name}] 规则信号列 '{final_rule_signal_col}' 不存在或全为空，无法计算趋势持续时间。")
            return trend_duration_info

        final_signal_series = data_with_signals[final_rule_signal_col].dropna()
        if final_signal_series.empty: return trend_duration_info

        # 趋势判断阈值来自 trend_following_params
        trend_threshold_upper = self.tf_params.get('trend_confirmation_threshold_upper', 55)
        trend_threshold_lower = self.tf_params.get('trend_confirmation_threshold_lower', 45)
        strong_bullish_threshold = self.tf_params.get('strong_bullish_threshold', 75)
        strong_bearish_threshold = self.tf_params.get('strong_bearish_threshold', 25)
        moderate_bullish_threshold = self.tf_params.get('moderate_bullish_threshold', 60)
        moderate_bearish_threshold = self.tf_params.get('moderate_bearish_threshold', 40)

        # 确保阈值是有效数字
        if not isinstance(trend_threshold_upper, (int, float)): trend_threshold_upper = 55.0
        if not isinstance(trend_threshold_lower, (int, float)): trend_threshold_lower = 45.0
        if not isinstance(strong_bullish_threshold, (int, float)): strong_bullish_threshold = 75.0
        if not isinstance(strong_bearish_threshold, (int, float)): strong_bearish_threshold = 25.0
        if not isinstance(moderate_bullish_threshold, (int, float)): moderate_bullish_threshold = 60.0
        if not isinstance(moderate_bearish_threshold, (int, float)): moderate_bearish_threshold = 40.0


        current_bullish_streak = 0
        current_bearish_streak = 0
        # 从最新数据向前计算连续周期数
        for signal_val in final_signal_series.iloc[::-1]:
            if signal_val >= trend_threshold_upper:
                current_bullish_streak += 1
                current_bearish_streak = 0
            elif signal_val <= trend_threshold_lower:
                current_bearish_streak += 1
                current_bullish_streak = 0
            else:
                break # 信号回到中性区域，趋势中断

        trend_duration_info['bullish_duration'] = current_bullish_streak
        trend_duration_info['bearish_duration'] = current_bearish_streak

        # 将周期数转换为时间文本
        try:
            # 假设 focus_timeframe 是数字分钟数 (例如 '5', '15', '30', '60')
            # 如果是非数字，会捕获 ValueError，使用周期数显示
            timeframe_minutes = int(self.focus_timeframe)
            bullish_total_minutes = current_bullish_streak * timeframe_minutes
            bearish_total_minutes = current_bearish_streak * timeframe_minutes

            # 获取交易日总分钟数参数，用于按“交易日”格式化持续时间
            # 默认为 A 股交易时间（上午 9:30-11:30, 下午 1:00-3:00），共 4 小时 = 240 分钟
            trading_day_minutes = self.tf_params.get('trading_day_minutes', 240)
            if not isinstance(trading_day_minutes, (int, float)) or trading_day_minutes <= 0:
                 logger.warning(f"[{self.strategy_name}] 趋势持续时间计算：'trading_day_minutes' 参数无效 ({trading_day_minutes})，使用简单天/小时/分钟格式。")
                 # 定义简单格式化函数
                 def format_duration_simple(minutes):
                     if minutes == 0: return "0分钟"
                     if minutes < 60: return f"{minutes}分钟"
                     hours, rem_minutes = divmod(minutes, 60)
                     if hours < 24:
                         return f"{hours}小时{rem_minutes}分钟" if rem_minutes else f"{hours}小时"
                     else:
                         days, rem_hours = divmod(hours, 24)
                         # 确保分钟部分也包含
                         return f"{days}天{rem_hours}小时{rem_minutes}分钟" if rem_hours or rem_minutes else f"{days}天"

                 # 使用简单格式化函数
                 trend_duration_info['bullish_duration_text'] = format_duration_simple(bullish_total_minutes)
                 trend_duration_info['bearish_duration_text'] = format_duration_simple(bearish_total_minutes)

            else:
                # 定义按交易日格式化函数
                def format_duration_with_trading_day(total_minutes, trading_day_minutes):
                    if total_minutes == 0: return "0分钟"

                    full_trading_days = total_minutes // trading_day_minutes
                    remaining_minutes_after_days = total_minutes % trading_day_minutes

                    parts = []
                    if full_trading_days > 0:
                        parts.append(f"{full_trading_days}交易日")

                    # 格式化剩余分钟为小时和分钟
                    if remaining_minutes_after_days > 0:
                         rem_hours, rem_minutes = divmod(remaining_minutes_after_days, 60)
                         if rem_hours > 0:
                             parts.append(f"{rem_hours}小时")
                         if rem_minutes > 0 or (rem_hours == 0 and full_trading_days == 0): # 如果没有小时部分，或者总时间不足一天但有分钟，显示分钟
                             parts.append(f"{rem_minutes}分钟")

                    # 如果总分钟数非常小，不足一个完整的“小时”但大于0分钟，需要确保显示分钟数
                    if not parts and total_minutes > 0:
                         parts.append(f"{total_minutes}分钟")

                    return "".join(parts)

                # 使用按交易日格式化函数
                trend_duration_info['bullish_duration_text'] = format_duration_with_trading_day(bullish_total_minutes, trading_day_minutes)
                trend_duration_info['bearish_duration_text'] = format_duration_with_trading_day(bearish_total_minutes, trading_day_minutes)


        except ValueError:
            logger.warning(f"[{self.strategy_name}] 趋势持续时间计算：无法将 focus_timeframe '{self.focus_timeframe}' 转换为分钟数 (非数字)。持续时间将以周期数显示。")
            trend_duration_info['bullish_duration_text'] = f"{current_bullish_streak}个周期"
            trend_duration_info['bearish_duration_text'] = f"{current_bearish_streak}个周期"
        except Exception as e:
             logger.error(f"[{self.strategy_name}] 趋势持续时间计算：格式化持续时间时出错: {e}", exc_info=True)
             trend_duration_info['bullish_duration_text'] = f"{current_bullish_streak}个周期 (格式化错误)"
             trend_duration_info['bearish_duration_text'] = f"{current_bearish_streak}个周期 (格式化错误)"


        # 判断当前趋势方向和强度
        latest_rule_signal_val = final_signal_series.iloc[-1]
        # 趋势强度阈值来自 trend_following_params
        strong_bullish_threshold = self.tf_params.get('strong_bullish_threshold', 75)
        strong_bearish_threshold = self.tf_params.get('strong_bearish_threshold', 25)
        moderate_bullish_threshold = self.tf_params.get('moderate_bullish_threshold', 60)
        moderate_bearish_threshold = self.tf_params.get('moderate_bearish_threshold', 40)

        # 确保阈值是有效数字
        if not isinstance(strong_bullish_threshold, (int, float)): strong_bullish_threshold = 75.0
        if not isinstance(strong_bearish_threshold, (int, float)): strong_bearish_threshold = 25.0
        if not isinstance(moderate_bullish_threshold, (int, float)): moderate_bullish_threshold = 60.0
        if not isinstance(moderate_bearish_threshold, (int, float)): moderate_bearish_threshold = 40.0


        if latest_rule_signal_val >= strong_bullish_threshold:
            trend_duration_info.update({'current_trend': '看涨↑', 'trend_strength': '非常强烈'})
        elif latest_rule_signal_val >= moderate_bullish_threshold:
            trend_duration_info.update({'current_trend': '看涨↑', 'trend_strength': '强'})
        elif latest_rule_signal_val >= trend_threshold_upper: # 使用趋势确认的上限阈值作为温和看涨的下限
            trend_duration_info.update({'current_trend': '看涨↑', 'trend_strength': '温和'})
        elif latest_rule_signal_val <= strong_bearish_threshold:
            trend_duration_info.update({'current_trend': '看跌↓', 'trend_strength': '非常强烈'})
        elif latest_rule_signal_val <= moderate_bearish_threshold:
            trend_duration_info.update({'current_trend': '看跌↓', 'trend_strength': '强'})
        elif latest_rule_signal_val <= trend_threshold_lower: # 使用趋势确认的下限阈值作为温和看跌的上限
            trend_duration_info.update({'current_trend': '看跌↓', 'trend_strength': '温和'})
        else: # 信号在中性区域
             trend_duration_info.update({'current_trend': '中性', 'trend_strength': '不明'})


        # 判断趋势持续状态（短、中、长）
        current_duration_periods = max(current_bullish_streak, current_bearish_streak)
        # 趋势持续时间阈值来自 trend_following_params
        trend_duration_threshold_strong = self.tf_params.get('trend_duration_threshold_strong', 5)
        trend_duration_threshold_moderate = self.tf_params.get('trend_duration_threshold_moderate', 10)

        # 确保阈值是有效数字
        if not isinstance(trend_duration_threshold_strong, (int, float)) or trend_duration_threshold_strong <= 0:
             logger.warning(f"[{self.strategy_name}] 趋势持续时间状态：'trend_duration_threshold_strong' 参数无效 ({trend_duration_threshold_strong})，使用默认值 5。")
             trend_duration_threshold_strong = 5
        if not isinstance(trend_duration_threshold_moderate, (int, float)) or trend_duration_threshold_moderate <= 0:
             logger.warning(f"[{self.strategy_name}] 趋势持续时间状态：'trend_duration_threshold_moderate' 参数无效 ({trend_duration_threshold_moderate})，使用默认值 10。")
             trend_duration_threshold_moderate = 10

        # 确保 strong 阈值不低于 moderate 阈值
        if trend_duration_threshold_strong < trend_duration_threshold_moderate:
             logger.warning(f"[{self.strategy_name}] 趋势持续时间状态：strong ({trend_duration_threshold_strong}) < moderate ({trend_duration_threshold_moderate}) 阈值，调整 strong 阈值。")
             trend_duration_threshold_strong = trend_duration_threshold_moderate + 1 # 确保至少相差 1

        if current_duration_periods >= trend_duration_threshold_strong:
             trend_duration_info['duration_status'] = '长'
        elif current_duration_periods >= trend_duration_threshold_moderate:
             trend_duration_info['duration_status'] = '中'
        else:
             trend_duration_info['duration_status'] = '短'

        logger.debug(f"[{self.strategy_name}] 趋势持续时间计算完成。持续信息: {trend_duration_info}")

        return trend_duration_info

    def analyze_signals(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        分析趋势策略信号，生成解读和建议。
        使用JSON配置获取内部列名。
        """
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 中间数据为空，无法进行信号分析。")
            return None

        analysis_results_dict = {}
        latest_data_row = self.intermediate_data.iloc[-1]

        # 获取策略内部列名，使用 JSON 配置
        internal_cols_conf = NAMING_CONFIG.get('strategy_internal_columns', {}).get('output_columns', [])
        # 确保 internal_cols_conf 是列表
        if not isinstance(internal_cols_conf, list): internal_cols_conf = []

        def get_internal_col_name(pattern, default_name):
            """从配置中查找内部列名，如果找不到则返回默认名。"""
            for item in internal_cols_conf:
                if isinstance(item, dict) and item.get('name_pattern') == pattern:
                    return item['name_pattern']
            return default_name # 如果没找到，返回默认名字

        # 获取内部列名
        combined_signal_col = get_internal_col_name("combined_signal", "combined_signal")
        final_rule_signal_col = get_internal_col_name("final_rule_signal", "final_rule_signal")
        transformer_signal_col = get_internal_col_name("transformer_signal", "transformer_signal")
        base_score_raw_col = get_internal_col_name("base_score_raw", "base_score_raw")
        base_score_volume_adjusted_col = get_internal_col_name("ADJUSTED_SCORE", "base_score_volume_adjusted") # ADJUSTED_SCORE 在 adjust_score_with_volume 中生成，通常映射到 base_score_volume_adjusted
        alignment_signal_col = get_internal_col_name("alignment_signal", "alignment_signal")
        long_term_context_col = get_internal_col_name("long_term_context", "long_term_context")
        adx_strength_signal_col = get_internal_col_name("adx_strength_signal", "adx_strength_signal")
        stoch_signal_col = get_internal_col_name("stoch_signal", "stoch_signal")
        has_bearish_div_col = get_internal_col_name("HAS_BEARISH_DIVERGENCE", "HAS_BEARISH_DIVERGENCE")
        has_bullish_div_col = get_internal_col_name("HAS_BULLISH_DIVERGENCE", "HAS_BULLISH_DIVERGENCE")
        # VOL_SPIKE_SIGNAL_{timeframe} 列名模式，需要根据 focus_timeframe 获取具体列名
        vol_spike_pattern = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c.get('name_pattern', '').startswith("VOL_SPIKE_SIGNAL")), "VOL_SPIKE_SIGNAL_{timeframe}")
        vol_spike_signal_col_tf = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=self.focus_timeframe)
        vol_spike_signal_col = vol_spike_signal_col_tf[0] if vol_spike_signal_col_tf else None # 使用第一个格式化结果


        # combined_signal 是内部列
        if combined_signal_col in self.intermediate_data.columns:
            combined_signal_series = self.intermediate_data[combined_signal_col].dropna()
            if not combined_signal_series.empty:
                analysis_results_dict['combined_signal_mean'] = combined_signal_series.mean()
        else:
             logger.warning(f"[{self.strategy_name}][{stock_code}] 分析：缺少组合信号列 '{combined_signal_col}'。")
             analysis_results_dict['combined_signal_mean'] = np.nan


        # 计算趋势持续时间
        trend_duration_info_dict = self._calculate_trend_duration(self.intermediate_data)
        analysis_results_dict.update(trend_duration_info_dict)

        signal_judgment_dict = {}
        operation_advice_str = "中性观望"
        risk_warning_str = ""
        t_plus_1_note_str = "（受 T+1 限制，建议次日操作）"

        # 从 latest_data_row 中安全获取信号值，如果列不存在则默认为 50.0
        final_score_val = latest_data_row.get(combined_signal_col, 50.0)
        final_rule_score_val = latest_data_row.get(final_rule_signal_col, 50.0)
        transformer_score_val = latest_data_row.get(transformer_signal_col, 50.0)

        # 趋势判断阈值来自 trend_following_params
        moderate_bullish_thresh = self.tf_params.get('moderate_bullish_threshold', 60)
        strong_bullish_thresh = self.tf_params.get('strong_bullish_threshold', 75)
        moderate_bearish_thresh = self.tf_params.get('moderate_bearish_threshold', 40)
        strong_bearish_thresh = self.tf_params.get('strong_bearish_threshold', 25)

        # 确保阈值是有效数字
        if not isinstance(moderate_bullish_thresh, (int, float)): moderate_bullish_thresh = 60.0
        if not isinstance(strong_bullish_thresh, (int, float)): strong_bullish_thresh = 75.0
        if not isinstance(moderate_bearish_thresh, (int, float)): moderate_bearish_thresh = 40.0
        if not isinstance(strong_bearish_thresh, (int, float)): strong_bearish_thresh = 25.0

        duration_status_rule_str = trend_duration_info_dict.get('duration_status', '短') # 默认为短

        # 基于 combined_signal 进行操作建议判断
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

        # EMA 排列状态判断 (内部列)
        alignment_val = latest_data_row.get(alignment_signal_col, np.nan)
        if not np.isnan(alignment_val):
             if alignment_val == 3: signal_judgment_dict['alignment_status'] = "完全多头排列"
             elif alignment_val == -3: signal_judgment_dict['alignment_status'] = "完全空头排列"
             elif alignment_val > 0: signal_judgment_dict['alignment_status'] = "多头排列形成中"
             elif alignment_val < 0: signal_judgment_dict['alignment_status'] = "空头排列形成中"
             else: signal_judgment_dict['alignment_status'] = "排列不明朗"
        else:
             signal_judgment_dict['alignment_status'] = "数据缺失"
             logger.warning(f"[{self.strategy_name}][{stock_code}] 分析：缺少 EMA 排列信号列 '{alignment_signal_col}'。")


        # ADX 强度判断 (内部列)
        adx_strength_signal_val = latest_data_row.get(adx_strength_signal_col, np.nan) # 获取归一化后的 ADX 强度信号
        if not np.isnan(adx_strength_signal_val):
            # 使用原始 ADX 阈值进行判断，而不是归一化后的信号
            # DMI 周期来自全局 base_scoring.dmi_period 或 indicator_analysis_params.dmi_period
            param_sources = [self.tf_params, self.params.get('volume_confirmation', {}), self.params.get('indicator_analysis_params', {}), self.fe_params, self.params.get('base_scoring', {})]
            dmi_period_bs = next((_get_param_val(param_sources, 'dmi_period', 14) for _ in [None]), 14) # 使用辅助函数获取 DMI 周期
            adx_col = f'ADX_{dmi_period_bs}_{self.focus_timeframe}' # 使用带后缀的原始 ADX 列名
            adx_val = latest_data_row.get(adx_col, np.nan) # 获取原始 ADX 值

            if not np.isnan(adx_val):
                 adx_strong_threshold = self.tf_params.get('adx_strong_threshold', 30)
                 adx_moderate_threshold = self.tf_params.get('adx_moderate_threshold', 20)

                 if adx_val >= adx_strong_threshold: signal_judgment_dict['adx_status'] = "趋势非常强劲"
                 elif adx_val >= adx_moderate_threshold: signal_judgment_dict['adx_status'] = "趋势强劲"
                 else: signal_judgment_dict['adx_status'] = "趋势较弱"

                 # 如果 ADX 强度不足（例如小于 moderate_threshold），且 combined_signal 偏离中性区域较远，提示风险
                 if adx_val < adx_moderate_threshold and abs(final_score_val - 50) > 10: # 阈值 10 可配置
                     risk_warning_str += f"ADX ({adx_val:.2f}) 显示趋势强度不足 ({adx_moderate_threshold})，当前信号偏离中性，注意假信号或震荡风险。 "
            else:
                 signal_judgment_dict['adx_status'] = "原始ADX数据缺失"
                 logger.warning(f"[{self.strategy_name}][{stock_code}] 分析：缺少原始 ADX 列 '{adx_col}'，无法判断 ADX 趋势强度状态。")
        else:
            signal_judgment_dict['adx_status'] = "ADX信号数据缺失"
            logger.warning(f"[{self.strategy_name}][{stock_code}] 分析：缺少 ADX 强度信号列 '{adx_strength_signal_col}'。")


        # 背离状态判断 (内部列)
        has_bearish_div_val = latest_data_row.get(has_bearish_div_col, False)
        has_bullish_div_val = latest_data_row.get(has_bullish_div_col, False)

        if has_bearish_div_val and final_score_val > 50:
            signal_judgment_dict['divergence_status'] = "检测到顶背离"
            risk_warning_str += "检测到顶背离，可能预示趋势衰竭或反转！ "
        elif has_bullish_div_val and final_score_val < 50:
            signal_judgment_dict['divergence_status'] = "检测到底背离"
            risk_warning_str += "检测到底背离，可能预示趋势衰竭或反转！ "
        else:
            signal_judgment_dict['divergence_status'] = "无明显背离"

        # TODO: 添加其他分析项，如量能信号状态、波动率状态等
        # 例如：检查 latest_data_row.get('VOL_SPIKE_SIGNAL_{focus_tf}')
        # 检查 latest_data_row.get('volatility_signal')


        now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        chinese_interpretation_str = (
            f"【趋势跟踪策略分析 - {stock_code} - {now_str}】\n"
            f"焦点时间框架: {self.focus_timeframe}\n"
            f"最新组合信号分: {final_score_val:.2f} (规则: {final_rule_score_val:.2f}, Transformer: {transformer_score_val:.2f})\n"
            f"当前策略信号: {signal_judgment_dict.get('overall_signal', '中性')}\n"
            f"基于规则趋势判断: {trend_duration_info_dict.get('current_trend', '中性')} (强度: {trend_duration_info_dict.get('trend_strength', '不明')})\n"
            f"趋势持续时间: {trend_duration_info_dict.get('bullish_duration_text' if trend_duration_info_dict.get('current_trend','').startswith('看涨') else 'bearish_duration_text','未知')} (状态: {trend_duration_info_dict.get('duration_status', '短')})\n"
            f"EMA排列状态: {signal_judgment_dict.get('alignment_status', '未知')}\n"
            f"ADX强度信号: {latest_data_row.get(adx_strength_signal_col, np.nan):.2f} ({signal_judgment_dict.get('adx_status', '未知')})\n"
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

    # 添加 timestamp 参数，用于记录分析发生的时间点，data 参数可选，方便获取最新价格
    def save_analysis_results(self, stock_code: str, timestamp: pd.Timestamp, data: Optional[pd.DataFrame]=None):
        """
        保存趋势跟踪策略的分析结果到数据库。
        使用JSON配置获取内部列名和OHLCV列名。
        """
        from stock_models.stock_analytics import StockScoreAnalysis
        from stock_models.stock_basic import StockInfo

        if self.analysis_results is None:
            logger.warning(f"[{self.strategy_name}][{stock_code}] 无分析结果可保存。请先运行 analyze_signals。")
            return

        # 尝试从 self.intermediate_data 获取最新的数据行，如果 self.intermediate_data 为 None 或空，则尝试从传入的 data 获取
        latest_intermediate_row = pd.Series(dtype=object) # 初始化一个空 Series
        if self.intermediate_data is not None and not self.intermediate_data.empty:
             # 查找与 timestamp 最接近的索引行，或者直接使用最后一行
             try:
                  # 使用 asof 查找小于等于 timestamp 的最新一行
                  latest_intermediate_row = self.intermediate_data.loc[self.intermediate_data.index.asof(timestamp)]
                  if latest_intermediate_row.isnull().all(): # asof 可能返回全 NaN 行
                       logger.warning(f"[{self.strategy_name}][{stock_code}] 无法通过 asof 在 intermediate_data 中找到时间戳 {timestamp} 对应的行，使用最后一行。")
                       latest_intermediate_row = self.intermediate_data.iloc[-1]
             except Exception as e:
                  logger.warning(f"[{self.strategy_name}][{stock_code}] 在 intermediate_data 中查找时间戳 {timestamp} 对应的行出错: {e}，使用最后一行。")
                  try:
                       latest_intermediate_row = self.intermediate_data.iloc[-1]
                  except IndexError:
                       logger.error(f"[{self.strategy_name}][{stock_code}] intermediate_data 为空，无法获取最新数据行。")
                       latest_intermediate_row = pd.Series(dtype=object) # 确保是空 Series

        elif data is not None and not data.empty:
             # 如果 intermediate_data 为空，尝试从传入的原始数据 data 获取最后一行
             logger.warning(f"[{self.strategy_name}][{stock_code}] intermediate_data 为空，尝试从原始输入 data 获取最新数据行。")
             try:
                  # 同样尝试 asof 查找
                  latest_data_row_from_input = data.loc[data.index.asof(timestamp)]
                  if latest_data_row_from_input.isnull().all():
                       latest_data_row_from_input = data.iloc[-1]
                  latest_intermediate_row = latest_data_row_from_input # 使用从输入数据获取的行

             except Exception as e:
                  logger.warning(f"[{self.strategy_name}][{stock_code}] 从原始输入 data 获取最新数据行出错: {e}。")
                  latest_intermediate_row = pd.Series(dtype=object) # 确保是空 Series

        if latest_intermediate_row.empty:
             logger.error(f"[{self.strategy_name}][{stock_code}] 无法获取最新的数据行用于保存分析结果。")
             # 继续保存，但部分字段可能为 None

        try:
            stock_obj = StockInfo.objects.get(stock_code=stock_code)

            # 辅助函数，将 NaN, Inf 或 None 转换为 None，以便保存到数据库字段
            def convert_nan_to_none(value):
                if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
                    return None
                # 检查 pandas NotNa，用于处理 pd.NA 或 pd.NaT
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
            base_score_volume_adjusted_col = get_internal_col_name("ADJUSTED_SCORE", "base_score_volume_adjusted") # ADJUSTED_SCORE 在 adjust_score_with_volume 中生成，通常映射到 base_score_volume_adjusted
            alignment_signal_col = get_internal_col_name("alignment_signal", "alignment_signal")
            long_term_context_col = get_internal_col_name("long_term_context", "long_term_context")
            adx_strength_signal_col = get_internal_col_name("adx_strength_signal", "adx_strength_signal")
            stoch_signal_col = get_internal_col_name("stoch_signal", "stoch_signal")
            has_bearish_div_col = get_internal_col_name("HAS_BEARISH_DIVERGENCE", "HAS_BEARISH_DIVERGENCE")
            has_bullish_div_col = get_internal_col_name("HAS_BULLISH_DIVERGENCE", "HAS_BULLISH_DIVERGENCE")
            # VOL_SPIKE_SIGNAL_{timeframe} 列名模式，需要根据 focus_timeframe 获取具体列名
            vol_spike_pattern = next((c['name_pattern'] for c in internal_cols_conf if isinstance(c, dict) and c.get('name_pattern', '').startswith("VOL_SPIKE_SIGNAL")), "VOL_SPIKE_SIGNAL_{timeframe}")
            vol_spike_signal_col_tf = TrendFollowingStrategy._format_indicator_name(vol_spike_pattern, timeframe=self.focus_timeframe)
            vol_spike_signal_col = vol_spike_signal_col_tf[0] if vol_spike_signal_col_tf else None # 使用第一个格式化结果

            # 获取 close 列名，使用 JSON 配置
            ohlcv_configs = NAMING_CONFIG.get('ohlcv_naming_convention', {}).get('output_columns', [])
            close_base_name = next((c['name_pattern'] for c in ohlcv_configs if isinstance(c, dict) and c.get('name_pattern') == 'close'), 'close')
            close_price_col_name = f'{close_base_name}_{self.focus_timeframe}' # 使用带后缀的列名

            defaults_payload = {
                'score': convert_nan_to_none(latest_intermediate_row.get(combined_signal_col)),
                'rule_signal': convert_nan_to_none(latest_intermediate_row.get(final_rule_signal_col)),
                'lstm_signal': convert_nan_to_none(latest_intermediate_row.get(transformer_signal_col)),
                'base_score_raw': convert_nan_to_none(latest_intermediate_row.get(base_score_raw_col)),
                'base_score_volume_adjusted': convert_nan_to_none(latest_intermediate_row.get(base_score_volume_adjusted_col)),
                'alignment_signal': convert_nan_to_none(latest_intermediate_row.get(alignment_signal_col)),
                'long_term_context': convert_nan_to_none(latest_intermediate_row.get(long_term_context_col)),
                'adx_strength_signal': convert_nan_to_none(latest_intermediate_row.get(adx_strength_signal_col)),
                'stoch_signal': convert_nan_to_none(latest_intermediate_row.get(stoch_signal_col)),
                'div_has_bearish_divergence': bool(latest_intermediate_row.get(has_bearish_div_col, False)), # 确保是布尔值
                'div_has_bullish_divergence': bool(latest_intermediate_row.get(has_bullish_div_col, False)), # 确保是布尔值
                # 确保 volume_spike_signal_col 存在于 latest_intermediate_row 中
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
                'chinese_interpretation': self.analysis_results.get('chinese_interpretation'),

                # TODO: 可以根据需要在 analysis_results_dict 中添加更多字段，并在此处保存
                # 例如：'score_momentum', 'score_volatility', 'vwap_deviation_percent', 'boll_breakout_signal' 等
                # 这些字段需要在 _perform_trend_analysis 和 analyze_signals 中计算并添加到 analysis_results_dict

                'raw_analysis_data': json.dumps(self.analysis_results, ensure_ascii=False, default=lambda x: str(x)) # 保存完整的分析结果字典 (将不可序列化的对象转为字符串)
            }

            # 使用 update_or_create 方法避免重复创建
            obj, created = StockScoreAnalysis.objects.update_or_create(
                stock=stock_obj,
                strategy_name=self.strategy_name,
                timestamp=timestamp, # 使用传入的时间戳作为唯一标识
                timeframe=self.focus_timeframe, # 保存策略的焦点时间框架
                defaults=defaults_payload
            )

            if created:
                logger.info(f"[{self.strategy_name}][{stock_code}] 在时间点 {timestamp.strftime('%Y-%m-%d %H:%M')} 创建新的 StockScoreAnalysis 记录。")
            else:
                logger.info(f"[{self.strategy_name}][{stock_code}] 更新时间点 {timestamp.strftime('%Y-%m-%d %H:%M')} 的 StockScoreAnalysis 记录。")

        except StockInfo.DoesNotExist:
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存分析结果失败：股票代码 {stock_code} 不存在于 StockInfo 模型中。")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{stock_code}] 保存 StockScoreAnalysis 记录出错: {e}", exc_info=True)

