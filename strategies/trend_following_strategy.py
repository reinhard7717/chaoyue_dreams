# 此策略侧重于识别和跟随趋势，主要使用 EMA 排列、DMI、SAR 等指标，并以 30 分钟级别为主要权重。
# trend_following_strategy.py
import pandas as pd
import numpy as np
import json
import os
import logging
import joblib
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple
# 导入深度学习工具函数
from .utils.deep_learning_utils import prepare_data_for_lstm, build_lstm_model, train_lstm_model

# 假设 BaseStrategy 和常量在 .base 或 core.constants
from .base import BaseStrategy
from . import strategy_utils # 导入公共工具

logger = logging.getLogger("strategy_trend_following")

# --- 动态导入 pandas_ta ---
try:
    import pandas_ta as ta
    # 检查 pandas_ta 是否成功导入且包含常用的指标函数
    if not hasattr(ta, 'ema') or not hasattr(ta, 'macd') or not hasattr(ta, 'rsi'):
        logger.warning("pandas_ta 导入成功但部分核心功能可能不可用，趋势分析将受限。")
        # 如果核心功能缺失，可以考虑禁用相关策略部分或使用备用计算方法
        # 这里我们假设如果 ta 存在，至少 EMA 是可用的
        if not hasattr(ta, 'ema'):
             ta = None # 如果 EMA 不可用，则禁用 ta
except ImportError:
    ta = None
    logger.warning("pandas_ta 库未安装或导入失败，趋势分析将受限。请运行 'pip install pandas_ta'")


class TrendFollowingStrategy(BaseStrategy):
    """
    趋势跟踪策略：
    - 基于多时间框架指标评分，并根据参数侧重特定时间框架 (`focus_timeframe`)。
    - 主要关注趋势指标 (DMI, SAR, MACD, EMA排列, OBV趋势) 和趋势强度 (ADX)。
    - 结合量能确认、波动率、STOCH、VWAP、BOLL等辅助判断趋势的持续性、增强或衰竭。
    - 增加量价背离检测作为潜在反转的警示信号。
    - 适应A股 T+1 交易制度，增强假信号过滤，动态调整参数。
    - **新增：集成基于规则生成的信号作为目标变量，训练 LSTM 模型进行预测，辅助判断趋势延续或反转概率。**
    """
    strategy_name = "TrendFollowingStrategy"
    focus_timeframe = '30' # 默认主要关注的时间框架

    def __init__(self, params_file: str = "strategies/indicator_parameters.json", base_model_dir="models"):
        """初始化策略，加载参数"""
        self.params_file = params_file
        self.params = self._load_params()
        self.strategy_name = self.params.get('trend_following_strategy_name', self.strategy_name)
        self.base_model_dir = base_model_dir # 存储基础模型目录

        # --- 加载趋势跟踪特定参数 ---
        self.tf_params = self.params.get('trend_following_params', {})
        self.focus_timeframe = self.tf_params.get('focus_timeframe', self.focus_timeframe)
        self.timeframe_weights = self.tf_params.get('timeframe_weights', None)  # 使用特定的时间框架权重
        self.trend_indicators = self.tf_params.get('trend_indicators', ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi'])
        self.signal_weights = self.tf_params.get('signal_weights', {
            'base_score': 0.5, 'alignment': 0.25, 'long_context': 0.1, 'momentum': 0.15
        })
        # 优先使用 TF 参数中的量能因子，如果不存在则回退到 volume_confirmation 中的参数
        self.volume_boost_factor = self.tf_params.get('volume_boost_factor', self.params.get('volume_confirmation', {}).get('boost_factor', 1.2))
        self.volume_penalty_factor = self.tf_params.get('volume_penalty_factor', self.params.get('volume_confirmation', {}).get('penalty_factor', 0.8))
        self.volume_spike_threshold = self.tf_params.get('volume_spike_threshold', 2.0)
        self.volatility_threshold_high = self.tf_params.get('volatility_threshold_high', 10)
        self.volatility_threshold_low = self.tf_params.get('volatility_threshold_low', 5)
        self.ema_cross_weight = self.tf_params.get('ema_cross_weight', 0.15)
        self.boll_breakout_weight = self.tf_params.get('boll_breakout_weight', 0.1)
        self.adx_strong_threshold = self.tf_params.get('adx_strong_threshold', 30)
        self.adx_moderate_threshold = self.tf_params.get('adx_moderate_threshold', 20)
        self.trend_duration_threshold_strong = self.tf_params.get('trend_duration_threshold_strong', 3)
        self.trend_duration_threshold_moderate = self.tf_params.get('trend_duration_threshold_moderate', 5)
        self.stoch_oversold_threshold = self.tf_params.get('stoch_oversold_threshold', 20)
        self.stoch_overbought_threshold = self.tf_params.get('stoch_overbought_threshold', 80)
        self.vwap_deviation_threshold = self.tf_params.get('vwap_deviation_threshold', 0.01)
        # --- 趋势确认周期参数 ---
        self.trend_confirmation_periods = self.tf_params.get('trend_confirmation_periods', 3)  # 默认值为3个周期
        # --- 动态参数调整相关变量 ---
        self.volatility_adjust_factor = 1.0  # 波动率调整因子，初始值为1.0
        # --- 结束加载参数 ---
        self.intermediate_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[pd.DataFrame] = None

        if ta is None:
             logger.error(f"[{self.strategy_name}] pandas_ta 未加载或 EMA 不可用，无法计算趋势指标。")
             # 根据策略核心依赖性，如果 ta 不可用，可能需要抛出异常或禁用策略
             # raise ImportError("pandas_ta with EMA is required for TrendFollowingStrategy.")
             # 暂时允许继续，但部分功能将受限

        # --- 动态加载深度学习模型相关参数 ---
        self.lstm_model = None
        self.feature_scaler = None # 用于特征的 scaler
        self.target_scaler = None # 用于目标变量的 scaler
        # 从参数文件加载窗口大小，如果不存在则使用默认值 60
        self.window_size = self.tf_params.get('lstm_window_size', 60)
        # 股票特定的模型和 scaler 路径将在 set_model_paths 中设置
        self.model_path = None
        self.scaler_path = None
        # LSTM 模型和训练配置 (从参数文件加载或使用默认值)
        self.model_config = self.tf_params.get('lstm_model_config', {}) # 使用参数文件中的配置，空字典表示使用 build_lstm_model 的默认值
        self.training_config = self.tf_params.get('lstm_training_config', {}) # 使用参数文件中的配置，空字典表示使用 train_lstm_model 的默认值
        # 确保训练配置中的学习率覆盖模型配置中的学习率 (如果两者都指定了)
        if 'learning_rate' in self.training_config:
             # 确保 model_config 中有 optimizer 相关的键，否则可能导致 build_lstm_model 错误
             if 'optimizer' not in self.model_config:
                  self.model_config['optimizer'] = 'adam' # 默认一个优化器
             # 将训练配置中的学习率传递给模型配置
             self.model_config['learning_rate'] = self.training_config['learning_rate']

        # 从参数加载 LSTM 数据准备相关配置
        self.lstm_data_params = {
            'window_size': self.window_size,
            'scaler_type': self.tf_params.get('lstm_scaler_type', 'minmax'),
            'train_split': self.tf_params.get('lstm_train_split', 0.7),
            'val_split': self.tf_params.get('lstm_val_split', 0.15),
            'use_feature_selection': self.tf_params.get('lstm_use_feature_selection', True),
            'feature_selector_model': self.tf_params.get('lstm_feature_selector_model', 'rf'),
            'max_features_fs': self.tf_params.get('lstm_max_features_fs', 50),
            'feature_selection_threshold': self.tf_params.get('lstm_feature_selection_threshold', 'median'),
            'use_pca': self.tf_params.get('lstm_use_pca', False),
            'n_components': self.tf_params.get('lstm_pca_n_components', 0.99),
            'target_scaler_type': self.tf_params.get('lstm_target_scaler_type', 'minmax'),
            'apply_variance_threshold': self.tf_params.get('lstm_apply_variance_threshold', False), # 新增方差过滤参数
            'variance_threshold_value': self.tf_params.get('lstm_variance_threshold_value', 0.01) # 新增方差过滤阈值参数
        }
        super().__init__(self.params)

    def set_model_paths(self, stock_code: str):
        """
        为特定股票设置模型和 scaler 的保存路径。
        """
        # 每只股票单独一个目录
        stock_model_dir = os.path.join(self.base_model_dir, stock_code)
        if not os.path.exists(stock_model_dir):
            os.makedirs(stock_model_dir)
            logger.info(f"创建股票模型目录: {stock_model_dir}")

        self.model_path = os.path.join(stock_model_dir, "trend_following_lstm.keras")
        self.scaler_path = os.path.join(stock_model_dir, "trend_following_lstm_scaler.save")
        logger.info(f"设置股票 {stock_code} 的模型路径: {self.model_path}, Scaler路径: {self.scaler_path}")

    # 为特定股票训练 LSTM 模型并保存
    def train_lstm_model_for_stock(self, data: pd.DataFrame, stock_code: str, required_columns: List[str]):
        """
        为特定股票训练 LSTM 模型并保存模型和 scaler。
        训练目标是规则生成的 'final_signal' 列。
        """
        self.set_model_paths(stock_code) # 设置股票特定的路径

        logger.info(f"开始为股票 {stock_code} 训练 LSTM 模型...")

        # 1. 数据准备
        try:
            # 使用 prepare_data_for_lstm 准备数据，目标变量是规则生成的 final_signal
            # 将 self.lstm_data_params 中的配置作为 kwargs 传递
            X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = prepare_data_for_lstm(
                data=data,
                required_columns=required_columns, # 传入所有可能的原始特征列
                target_column='final_signal', # 目标列是规则生成的 final_signal
                **self.lstm_data_params # 解包传递 LSTM 数据准备参数
            )
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

            # 检查数据是否有效
            if X_train.shape[0] == 0 or X_train.shape[2] == 0:
                 logger.error(f"股票 {stock_code} 数据准备失败，训练集为空或特征维度为零。")
                 self.lstm_model = None
                 self.feature_scaler = None
                 self.target_scaler = None
                 return # 停止训练

            num_features = X_train.shape[2] # 获取实际使用的特征数量 (经过选择/PCA后)

        except Exception as e:
            logger.error(f"股票 {stock_code} 数据准备出错: {e}", exc_info=True)
            self.lstm_model = None
            self.feature_scaler = None
            self.target_scaler = None
            return # 停止训练

        # 2. 构建模型
        try:
            model = build_lstm_model(
                window_size=self.window_size, # 使用参数中的窗口大小
                num_features=num_features, # 使用数据准备后确定的特征数量
                model_config=self.model_config, # 传递模型配置
                # 从参数加载模型类型，如果不存在则使用默认值
                model_type=self.tf_params.get('lstm_model_type', 'lstm'),
                summary=True
            )
            self.lstm_model = model # 暂存模型对象
        except Exception as e:
            logger.error(f"股票 {stock_code} 构建 LSTM 模型出错: {e}", exc_info=True)
            self.lstm_model = None
            return # 停止训练

        # 3. 训练模型
        try:
            # 使用 stock-specific model_path 作为 checkpoint_path
            train_lstm_model(
                X_train, y_train, X_val, y_val, X_test, y_test,
                model=self.lstm_model,
                target_scaler=self.target_scaler, # 传入目标变量 scaler
                training_config=self.training_config, # 传递训练配置
                checkpoint_path=self.model_path, # 保存到股票特定的路径
                # 从参数加载是否绘图，如果不存在则使用默认值 False
                plot_training_history=self.tf_params.get('lstm_plot_history', False)
            )
            logger.info(f"股票 {stock_code} LSTM 模型训练完成，最佳模型已保存到 {self.model_path}")

            # 4. 保存 Scaler
            try:
                # 使用 joblib 保存 feature_scaler 和 target_scaler
                joblib.dump({'feature_scaler': self.feature_scaler, 'target_scaler': self.target_scaler}, self.scaler_path)
                logger.info(f"股票 {stock_code} Scaler 已保存到 {self.scaler_path}")
            except Exception as e:
                logger.error(f"股票 {stock_code} 保存 Scaler 出错: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"股票 {stock_code} 训练 LSTM 模型出错: {e}", exc_info=True)
            self.lstm_model = None # 训练失败，模型对象置空

    def _load_params(self) -> Dict[str, Any]:
        """从 JSON 文件加载参数"""
        if not os.path.exists(self.params_file):
            raise FileNotFoundError(f"参数文件未找到: {self.params_file}")
        try:
            with open(self.params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"成功从 {self.params_file} 加载策略参数。")
            return params
        except Exception as e:
            logger.error(f"加载或解析参数文件 {self.params_file} 时出错: {e}")
            raise

    def _validate_params(self):
        """验证策略特定参数"""
        super()._validate_params()
        if 'trend_following_params' not in self.params:
            logger.warning("参数中缺少 'trend_following_params' 部分，将使用部分默认值。")

        bs_params = self.params.get('base_scoring', {})
        ta_params = self.params.get('trend_analysis', {})
        vc_params = self.params.get('volume_confirmation', {})
        dd_params = self.params.get('divergence_detection', {})
        ia_params = self.params.get('indicator_analysis_params', {}) # 获取指标分析参数
        tf_params = self.params.get('trend_following_params', {}) # 获取趋势跟踪参数

        if 'timeframes' not in bs_params or not isinstance(bs_params['timeframes'], list):
             raise ValueError("'base_scoring.timeframes' 必须是一个列表")
        if self.focus_timeframe not in bs_params['timeframes']:
             raise ValueError(f"'focus_timeframe' ({self.focus_timeframe}) 必须在 'base_scoring.timeframes' 中")

        # 验证 timeframe_weights (如果提供)
        if self.timeframe_weights:
            if not isinstance(self.timeframe_weights, dict):
                raise ValueError("'trend_following_params.timeframe_weights' 必须是一个字典")
            # 检查权重字典的键是否与 timeframes 列表匹配
            if set(self.timeframe_weights.keys()) != set(bs_params['timeframes']):
                logger.warning(f"'trend_following_params.timeframe_weights' 的键与 'base_scoring.timeframes' 不完全匹配，请检查: {list(self.timeframe_weights.keys())} vs {bs_params['timeframes']}")
                # 对于缺失的时间框架，可以添加默认权重（例如0）
                for tf in bs_params['timeframes']:
                    if tf not in self.timeframe_weights:
                        self.timeframe_weights[tf] = 0.0
                        logger.warning(f"为时间框架 '{tf}' 添加默认权重 0.0。")

            # 归一化权重，确保总和为 1
            total_weight = sum(self.timeframe_weights.values())
            if total_weight > 0 and not np.isclose(total_weight, 1.0):
                 logger.warning(f"'trend_following_params.timeframe_weights' 的权重之和不为 1 ({total_weight:.4f})，将重新归一化。")
                 self.timeframe_weights = {k: v / total_weight for k, v in self.timeframe_weights.items()}
            elif total_weight <= 0: # 如果总权重为0或负数，则重置为均等权重
                  logger.error("timeframe_weights 总和为0或负数，重置为均等权重。")
                  num_tf = len(bs_params['timeframes'])
                  self.timeframe_weights = {tf: 1.0 / num_tf for tf in bs_params['timeframes']} if num_tf > 0 else {}

        if 'trend_indicators' not in tf_params or not isinstance(tf_params['trend_indicators'], list):
            logger.warning("'trend_following_params.trend_indicators' 未定义或格式错误，将使用默认趋势指标 ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi']")
            self.trend_indicators = ['dmi', 'sar', 'macd', 'ema_alignment', 'obv', 'rsi'] # 设置默认值
        if not ta_params.get('ema_periods') or not isinstance(ta_params['ema_periods'], list):
             raise ValueError("'trend_analysis.ema_periods' 必须是一个列表")
        if ta_params.get('long_term_ema_period') not in ta_params['ema_periods']:
             raise ValueError("'trend_analysis.long_term_ema_period' 必须在 'ema_periods' 列表中")
        if 'signal_weights' not in tf_params or not isinstance(tf_params['signal_weights'], dict):
            logger.warning("'trend_following_params.signal_weights' 未定义或格式错误，将使用默认权重。")
            # 使用 self.signal_weights 中定义的默认值

        # 检查辅助指标所需的参数
        if 'stoch' in self.trend_indicators:
            if 'stoch_k' not in ia_params or 'stoch_d' not in ia_params:
                logger.warning("趋势指标包含 'stoch'，但 'indicator_analysis_params' 中缺少 'stoch_k' 或 'stoch_d'。")
        # VWAP 通常不需要额外参数，但需要确保数据中包含 VWAP 列
        # BOLL 参数已在 base_scoring 中定义，这里检查是否在 trend_indicators 中使用
        if 'boll' in self.trend_indicators and ('boll_period' not in bs_params or 'boll_std_dev' not in bs_params):
            logger.warning("趋势指标包含 'boll'，但 'base_scoring' 中缺少 'boll_period' 或 'boll_std_dev'。")

        # 验证 LSTM 相关参数 (可选，prepare_data_for_lstm 中会进行更详细的检查)
        lstm_params = self.lstm_data_params
        if lstm_params['train_split'] + lstm_params['val_split'] > 1.0:
             logger.warning(f"LSTM 数据分割比例 train_split ({lstm_params['train_split']}) + val_split ({lstm_params['val_split']}) > 1.0，请检查参数。")
        if lstm_params['use_feature_selection'] and lstm_params['feature_selector_model'].lower() == 'xgb' and xgb is None:
             logger.warning("参数指定使用 XGBoost 进行特征选择，但 XGBoost 库未安装。请安装 XGBoost 或修改参数。")
        if lstm_params['use_pca'] and lstm_params['use_feature_selection']:
             logger.warning("参数同时启用 PCA 和基于模型的特征选择，将优先使用 PCA。")

        logger.info(f"[{self.strategy_name}] 参数验证通过，主要关注时间框架: {self.focus_timeframe}")
        logger.info(f"[{self.strategy_name}] 使用的时间框架权重: {self.timeframe_weights}")
        logger.info(f"[{self.strategy_name}] 关注的趋势指标: {self.trend_indicators}")
        logger.info(f"[{self.strategy_name}] 信号组合权重: {self.signal_weights}")
        logger.info(f"[{self.strategy_name}] LSTM 数据准备参数: {self.lstm_data_params}")

    def get_required_columns(self) -> List[str]:
        """返回趋势跟踪策略所需的列"""
        required = set()
        bs_params = self.params['base_scoring']
        vc_params = self.params.get('volume_confirmation', {}) # 使用 .get() 防止 key 不存在
        ta_params = self.params.get('trend_analysis', {})
        dd_params = self.params.get('divergence_detection', {})
        ia_params = self.params.get('indicator_analysis_params', {}) # 指标分析参数
        timeframes = bs_params['timeframes']
        all_score_indicators = bs_params.get('score_indicators', []) # 基础分计算所需指标
        trend_indicators = self.trend_indicators # 策略关注的趋势指标

        # 基础分计算需要的列 (所有时间框架)
        for tf in timeframes:
            required.update([f'open_{tf}', f'high_{tf}', f'low_{tf}', f'close_{tf}', f'volume_{tf}', f'amount_{tf}']) # K线基础数据
            # 确保所有 base_scoring.score_indicators 定义的指标列都被包含
            # MACD
            if 'macd' in all_score_indicators:
                macd_fast = bs_params.get("macd_fast", 12) # 使用默认值防止参数缺失
                macd_slow = bs_params.get("macd_slow", 26)
                macd_signal = bs_params.get("macd_signal", 9)
                required.update([f'MACD_{macd_fast}_{macd_slow}_{macd_signal}_{tf}',
                                 f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}_{tf}',
                                 f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}_{tf}'])
            # RSI
            if 'rsi' in all_score_indicators:
                rsi_period = bs_params.get("rsi_period", 14)
                required.add(f'RSI_{rsi_period}_{tf}')
            # KDJ
            if 'kdj' in all_score_indicators:
                k_period = bs_params.get("kdj_period_k", 9) # 修正默认值以匹配常见设置
                d_period = bs_params.get("kdj_period_d", 3)
                # J周期参数通常不影响列名，列名通常只包含 K 和 D 的周期
                required.add(f'K_{k_period}_{d_period}_{tf}')
                required.add(f'D_{k_period}_{d_period}_{tf}')
                required.add(f'J_{k_period}_{d_period}_{tf}') # 假设 J 的列名也包含 K 和 D 周期
            # BOLL
            if 'boll' in all_score_indicators:
                boll_period = bs_params.get("boll_period", 20)
                # 假设列名格式为 BB_UPPER_period_tf, BB_MIDDLE_period_tf, BB_LOWER_period_tf
                required.update([f'BB_UPPER_{boll_period}_{tf}', f'BB_MIDDLE_{boll_period}_{tf}', f'BB_LOWER_{boll_period}_{tf}'])
            # CCI
            if 'cci' in all_score_indicators:
                cci_period = bs_params.get("cci_period", 14)
                required.add(f'CCI_{cci_period}_{tf}')
            # MFI
            if 'mfi' in all_score_indicators:
                mfi_period = bs_params.get("mfi_period", 14)
                required.add(f'MFI_{mfi_period}_{tf}')
            # ROC
            if 'roc' in all_score_indicators:
                roc_period = bs_params.get("roc_period", 12)
                required.add(f'ROC_{roc_period}_{tf}')
            # DMI
            if 'dmi' in all_score_indicators:
                dmi_period = bs_params.get("dmi_period", 14)
                required.update([f'+DI_{dmi_period}_{tf}', f'-DI_{dmi_period}_{tf}', f'ADX_{dmi_period}_{tf}'])
            # SAR
            if 'sar' in all_score_indicators:
                 # SAR 的列名通常不包含周期参数，直接是 SAR_tf
                 required.add(f'SAR_{tf}')

        # 趋势分析和信号生成需要的额外列 (根据 trend_indicators)
        # 这些指标可能在 base_scoring 中已经包含，但如果 trend_indicators 中有而 base_scoring 没有，则需要在这里确保添加
        for tf in timeframes: # 可能需要所有时间框架的数据
            # OBV (通常不包含周期参数，但可能需要其均线)
            if 'obv' in trend_indicators:
                required.add(f'OBV_{tf}')
                # 可能需要 OBV 的均线
                obv_ma_period = vc_params.get("obv_ma_period") # 从量能确认参数中获取 OBV 均线周期
                if obv_ma_period:
                    required.add(f'OBV_MA_{obv_ma_period}_{tf}')
            # STOCH
            if 'stoch' in trend_indicators:
                stoch_k = ia_params.get('stoch_k', 14)
                stoch_d = ia_params.get('stoch_d', 3)
                # 根据参数生成列名, 假设为 STOCH_K_k_tf, STOCH_D_k_d_tf
                required.add(f'STOCH_K_{stoch_k}_{tf}')
                required.add(f'STOCH_D_{stoch_k}_{stoch_d}_{tf}')
            # VWAP
            if 'vwap' in trend_indicators:
                 required.add(f'VWAP_{tf}') # 假设 VWAP 列存在且列名格式为 VWAP_tf
            # DMI, SAR, MACD, RSI, BOLL 如果只在 score_indicators 中定义，则上面已包含
            # 如果它们在 trend_indicators 中但不在 score_indicators 中，也需要在这里添加
            # DMI (如果 trend_indicators 包含但 score_indicators 不包含)
            if 'dmi' in trend_indicators and 'dmi' not in all_score_indicators:
                 dmi_period = bs_params.get("dmi_period", 14)
                 required.update([f'+DI_{dmi_period}_{tf}', f'-DI_{dmi_period}_{tf}', f'ADX_{dmi_period}_{tf}'])
            # SAR (如果 trend_indicators 包含但 score_indicators 不包含)
            if 'sar' in trend_indicators and 'sar' not in all_score_indicators:
                 required.add(f'SAR_{tf}')
            # MACD (如果 trend_indicators 包含但 score_indicators 不包含)
            if 'macd' in trend_indicators and 'macd' not in all_score_indicators:
                macd_fast = bs_params.get("macd_fast", 12)
                macd_slow = bs_params.get("macd_slow", 26)
                macd_signal = bs_params.get("macd_signal", 9)
                required.update([f'MACD_{macd_fast}_{macd_slow}_{macd_signal}_{tf}',
                                 f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}_{tf}',
                                 f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}_{tf}'])
            # RSI (如果 trend_indicators 包含但 score_indicators 不包含)
            if 'rsi' in trend_indicators and 'rsi' not in all_score_indicators:
                 rsi_period = bs_params.get("rsi_period", 14)
                 required.add(f'RSI_{rsi_period}_{tf}')
            # BOLL (如果 trend_indicators 包含但 score_indicators 不包含)
            if 'boll' in trend_indicators and 'boll' not in all_score_indicators:
                 boll_period = bs_params.get("boll_period", 20)
                 required.update([f'BB_UPPER_{boll_period}_{tf}', f'BB_MIDDLE_{boll_period}_{tf}', f'BB_LOWER_{boll_period}_{tf}'])


        # 量能确认指标 (如果启用)
        if vc_params.get('enabled', False):
            vol_tf = vc_params.get('tf', self.focus_timeframe) # 默认使用焦点时间框架
            # 确保相关周期的价格和量能数据存在
            required.update([f'close_{vol_tf}', f'high_{vol_tf}', f'low_{vol_tf}', f'volume_{vol_tf}', f'amount_{vol_tf}'])
            amount_ma_period = vc_params.get("amount_ma_period")
            if amount_ma_period:
                 required.add(f'AMT_MA_{amount_ma_period}_{vol_tf}')
            cmf_period = vc_params.get("cmf_period")
            if cmf_period:
                 required.add(f'CMF_{cmf_period}_{vol_tf}')
            # OBV 相关列已在上面根据 trend_indicators 添加，如果 OBV 不在 trend_indicators 中但量能确认需要，则在这里添加
            if 'obv' not in trend_indicators:
                 required.add(f'OBV_{vol_tf}')
                 obv_ma_period = vc_params.get("obv_ma_period")
                 if obv_ma_period:
                      required.add(f'OBV_MA_{obv_ma_period}_{vol_tf}')


        # 背离检测所需的列 (如果启用)
        if dd_params.get('enabled', True): # 假设默认启用或参数指定
            div_tf = dd_params.get('tf', self.focus_timeframe)
            # 确保相关周期的价格数据存在
            required.update([f'low_{div_tf}', f'high_{div_tf}', f'close_{div_tf}'])
            div_indicators = dd_params.get('indicators', {})
            # MACD Hist (需要 MACD 参数)
            if div_indicators.get('macd_hist', True):
                macd_fast = bs_params.get("macd_fast", 12)
                macd_slow = bs_params.get("macd_slow", 26)
                macd_signal = bs_params.get("macd_signal", 9)
                required.add(f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}_{div_tf}')
            # RSI (需要 RSI 参数)
            if div_indicators.get('rsi', True):
                 rsi_period = bs_params.get("rsi_period", 14)
                 required.add(f'RSI_{rsi_period}_{div_tf}')
            # MFI (需要 MFI 参数)
            if div_indicators.get('mfi', True):
                 mfi_period = bs_params.get("mfi_period", 14)
                 required.add(f'MFI_{mfi_period}_{div_tf}')
            # OBV
            if div_indicators.get('obv', False):
                 required.add(f'OBV_{div_tf}')
            # CMF 可能也用于背离判断 (需要 CMF 参数)
            cmf_period = vc_params.get("cmf_period")
            if cmf_period:
                 required.add(f'CMF_{cmf_period}_{div_tf}')

        # K线形态检测所需的列 (如果启用)
        kp_params = self.params.get('kline_pattern_detection', {})
        if kp_params.get('enabled', True):
             kp_tf = kp_params.get('tf', self.focus_timeframe)
             # K线形态检测通常需要开高低收数据
             required.update([f'open_{kp_tf}', f'high_{kp_tf}', f'low_{kp_tf}', f'close_{kp_tf}'])
             # 可能还需要量能数据辅助判断
             required.update([f'volume_{kp_tf}', f'amount_{kp_tf}'])
             # 还需要各种形态的识别结果列，这些列名通常由形态检测函数生成，例如 'CDLENGULFING_tf', 'CDLHAMMER_tf' 等
             # 这里无法列出所有可能的形态列，但需要确保形态检测函数生成这些列并添加到数据中
             # 假设形态检测函数会生成以 'CDL' 开头的列
             # required.update([f'CDLENGULFING_{kp_tf}', f'CDLHAMMER_{kp_tf}', ...]) # 示例，实际应根据形态检测函数确定

        # 确保焦点时间框架的基础数据存在 (已在基础分计算中包含)
        # required.update([f'open_{self.focus_timeframe}', f'high_{self.focus_timeframe}', f'low_{self.focus_timeframe}', f'close_{self.focus_timeframe}', f'volume_{self.focus_timeframe}', f'amount_{self.focus_timeframe}'])

        # 确保规则生成的 final_signal 列存在，它是 LSTM 的目标变量
        required.add('final_signal')

        # 移除重复项并排序（可选）
        return sorted(list(required))
    
    def _calculate_trend_focused_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算多时间框架加权的基础评分，使用 trend_following_params.timeframe_weights。
        分数基于 base_scoring.score_indicators 定义的指标。
        统一 NaN 处理，返回中性分 50。
        """
        scores = pd.DataFrame(index=data.index)
        bs_params = self.params['base_scoring']
        timeframes = bs_params['timeframes']
        all_score_indicators = bs_params.get('score_indicators', []) # 使用基础配置中的指标计算分数
        # --- 获取时间框架权重 ---
        # 优先使用策略特定的 timeframe_weights
        if self.timeframe_weights:
            weights = self.timeframe_weights.copy() # 使用副本，避免修改原始参数
            # 确保权重覆盖所有时间框架，对于缺失的权重，可以分配0
            for tf in timeframes:
                if tf not in weights:
                    weights[tf] = 0.0
                    logger.warning(f"timeframe_weights 中缺少时间框架: {tf}，其权重将设为 0。")
            # 重新归一化权重，确保总和为 1 (已在 _validate_params 中处理，这里再次确保)
            total_weight = sum(weights.values())
            if total_weight > 0 and not np.isclose(total_weight, 1.0):
                 weights = {k: v / total_weight for k, v in weights.items()}
                 logger.warning(f"重新归一化 timeframe_weights，总和为 {sum(weights.values()):.4f}")
            elif total_weight <= 0:
                 logger.error("timeframe_weights 总和为0或负数，无法计算加权分数。")
                 # 返回一个全为 50 的 DataFrame
                 scores['total_weighted_score'] = 50.0
                 scores['base_score_raw'] = 50.0
                 return scores
        else: # 如果未提供特定权重，则使用旧的基于 focus_weight 的方法或均等权重
            focus_weight = self.tf_params.get('focus_weight', 0.45) # 使用参数文件中的 focus_weight
            num_other_tfs = len(timeframes) - 1
            base_weight = (1.0 - focus_weight) / num_other_tfs if num_other_tfs > 0 else 0
            weights = {tf: base_weight for tf in timeframes if tf != self.focus_timeframe}
            weights[self.focus_timeframe] = focus_weight
            logger.info(f"未提供 timeframe_weights，使用 focus_weight ({focus_weight}) 计算权重: {weights}")


        scores['total_weighted_score'] = 0.0
        for tf in timeframes:
            weight = weights.get(tf, 0)
            if weight == 0: # 如果权重为0，跳过该时间框架的计算
                 continue
            tf_score_sum = pd.Series(0.0, index=data.index)
            indicator_count_in_tf = 0
            close_price_col = f'close_{tf}'
            close_price = data.get(close_price_col) # 使用 .get() 防止列不存在

            # --- 使用 strategy_utils 计算各指标分数，统一 NaN 处理为 50 ---
            # MACD
            if 'macd' in all_score_indicators:
                macd_fast = bs_params.get("macd_fast", 12)
                macd_slow = bs_params.get("macd_slow", 26)
                macd_signal = bs_params.get("macd_signal", 9)
                macd_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}_{tf}'
                macdh_col = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}_{tf}'
                macds_col = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}_{tf}'
                if all(c in data.columns for c in [macd_col, macdh_col, macds_col]): # 检查列是否存在
                    score = strategy_utils.calculate_macd_score(data[macd_col], data[macds_col], data[macdh_col])
                    tf_score_sum += score.fillna(50.0)
                    scores[f'macd_score_{tf}'] = score # 记录原始分数 (可能含 NaN)
                    indicator_count_in_tf += 1
                else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 MACD 列，跳过计算。")

            # RSI
            if 'rsi' in all_score_indicators:
                rsi_period = bs_params.get("rsi_period", 14)
                rsi_col = f'RSI_{rsi_period}_{tf}'
                if rsi_col in data.columns:
                     score = strategy_utils.calculate_rsi_score(data[rsi_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'rsi_score_{tf}'] = score
                     indicator_count_in_tf += 1
                else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 RSI 列，跳过计算。")

            # KDJ
            if 'kdj' in all_score_indicators:
                k_period = bs_params.get("kdj_period_k", 9)
                d_period = bs_params.get("kdj_period_d", 3)
                # J周期参数通常不影响列名，列名通常只包含 K 和 D 的周期
                k_col = f'K_{k_period}_{d_period}_{tf}'
                d_col = f'D_{k_period}_{d_period}_{tf}'
                j_col = f'J_{k_period}_{d_period}_{tf}' # 假设 J 的列名也包含 K 和 D 周期

                if all(c in data.columns for c in [k_col, d_col, j_col]):
                    # 确保使用的 KDJ 计算函数能处理可能为 NaN 的情况
                    # 简单填充NaN，或者在计算分数函数内部处理
                    # 这里假设 strategy_utils.calculate_kdj_score 内部能处理 NaN 或我们传入的数据已填充
                    # 为了安全，可以在传入前进行填充，但更好的做法是在计算函数内部处理或返回 NaN
                    # 假设 strategy_utils 函数返回 NaN，这里用 50 填充
                    score = strategy_utils.calculate_kdj_score(data[k_col], data[d_col], data[j_col], bs_params)
                    tf_score_sum += score.fillna(50.0) # 用中性分填充计算结果中的NaN
                    scores[f'kdj_score_{tf}'] = score
                    indicator_count_in_tf += 1
                else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 KDJ 列: {k_col}, {d_col}, 或 {j_col}，跳过计算。")

            # BOLL
            if 'boll' in all_score_indicators:
                 boll_period = bs_params.get("boll_period", 20)
                 upper_col, mid_col, lower_col = f'BB_UPPER_{boll_period}_{tf}', f'BB_MIDDLE_{boll_period}_{tf}', f'BB_LOWER_{boll_period}_{tf}'
                 if all(c in data.columns for c in [upper_col, mid_col, lower_col]) and close_price is not None and not close_price.isnull().all():
                     score = strategy_utils.calculate_boll_score(close_price, data[upper_col], data[mid_col], data[lower_col])
                     tf_score_sum += score.fillna(50.0)
                     scores[f'boll_score_{tf}'] = score
                     indicator_count_in_tf += 1
                 else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 BOLL 列或收盘价，跳过计算。")

            # CCI
            if 'cci' in all_score_indicators:
                 cci_period = bs_params.get("cci_period", 14)
                 cci_col = f'CCI_{cci_period}_{tf}'
                 if cci_col in data.columns:
                     score = strategy_utils.calculate_cci_score(data[cci_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'cci_score_{tf}'] = score
                     indicator_count_in_tf += 1
                 else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 CCI 列，跳过计算。")

            # MFI
            if 'mfi' in all_score_indicators:
                 mfi_period = bs_params.get("mfi_period", 14)
                 mfi_col = f'MFI_{mfi_period}_{tf}'
                 if mfi_col in data.columns:
                     score = strategy_utils.calculate_mfi_score(data[mfi_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'mfi_score_{tf}'] = score
                     indicator_count_in_tf += 1
                 else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 MFI 列，跳过计算。")

            # ROC
            if 'roc' in all_score_indicators:
                 roc_period = bs_params.get("roc_period", 12)
                 roc_col = f'ROC_{roc_period}_{tf}'
                 if roc_col in data.columns:
                     score = strategy_utils.calculate_roc_score(data[roc_col])
                     tf_score_sum += score.fillna(50.0)
                     scores[f'roc_score_{tf}'] = score
                     indicator_count_in_tf += 1
                 else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 ROC 列，跳过计算。")

            # DMI
            if 'dmi' in all_score_indicators:
                 dmi_period = bs_params.get("dmi_period", 14)
                 pdi_col, mdi_col, adx_col = f'+DI_{dmi_period}_{tf}', f'-DI_{dmi_period}_{tf}', f'ADX_{dmi_period}_{tf}'
                 if all(c in data.columns for c in [pdi_col, mdi_col, adx_col]):
                     score = strategy_utils.calculate_dmi_score(data[pdi_col], data[mdi_col], data[adx_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'dmi_score_{tf}'] = score
                     indicator_count_in_tf += 1
                 else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 DMI 列，跳过计算。")

            # SAR
            if 'sar' in all_score_indicators:
                 sar_col = f'SAR_{tf}'
                 if sar_col in data.columns and close_price is not None and not close_price.isnull().all():
                     score = strategy_utils.calculate_sar_score(close_price, data[sar_col])
                     tf_score_sum += score.fillna(50.0)
                     scores[f'sar_score_{tf}'] = score
                     indicator_count_in_tf += 1
                 else:
                    logger.warning(f"[{self.strategy_name}] 时间框架 {tf} 缺少 SAR 列或收盘价，跳过计算。")


            # --- 计算加权平均分 ---
            if indicator_count_in_tf > 0:
                # 确保 tf_score_sum 在除法前不是全 NaN
                if not tf_score_sum.isnull().all():
                    avg_tf_score = tf_score_sum / indicator_count_in_tf
                    scores[f'avg_score_{tf}'] = avg_tf_score # 记录每个周期的平均分
                    scores['total_weighted_score'] += avg_tf_score.fillna(50.0) * weight # 使用 timeframe_weights 加权，用 50 填充周期平均分中的 NaN
                else:
                    logger.warning(f"时间框架 '{tf}' 所有指标计算结果均为 NaN，将使用中性分 50.0 进行加权。")
                    scores[f'avg_score_{tf}'] = 50.0
                    scores['total_weighted_score'] += 50.0 * weight # 无指标贡献中性分
            else:
                logger.warning(f"时间框架 '{tf}' 没有可用的指标来计算分数，将使用中性分 50.0 进行加权。")
                scores[f'avg_score_{tf}'] = 50.0
                scores['total_weighted_score'] += 50.0 * weight # 无指标贡献中性分

        # 最终的总加权分数也需要处理可能的 NaN (如果所有时间框架都没有有效分数)
        scores['total_weighted_score'] = scores['total_weighted_score'].fillna(50.0)
        scores['base_score_raw'] = scores['total_weighted_score'].clip(0, 100) # 确保分数在 0-100 范围内
        return scores
        
    def _perform_trend_analysis(self, data: pd.DataFrame, base_score_series: pd.Series) -> pd.DataFrame:
        """
        增强趋势分析，加入 ADX, STOCH, VWAP, BOLL 等辅助判断。
        :param data: 包含所有所需指标的原始 DataFrame。
        :param base_score_series: 用于计算趋势的基础分数 Series (通常是量能调整后的)。
        :return: 包含趋势分析结果的 DataFrame。
        """
        analysis_df = pd.DataFrame(index=data.index) # 使用原始数据的索引
        ta_params = self.params.get('trend_analysis', {})
        bs_params = self.params.get('base_scoring', {})
        ia_params = self.params.get('indicator_analysis_params', {})  # 指标分析参数
        focus_tf = self.focus_timeframe  # 使用焦点时间框架的数据进行部分分析

        if base_score_series.isnull().all():
            logger.warning("基础分数全为 NaN，无法执行趋势分析。")
            # 返回包含所有预期列但全为 NaN 或默认值的 DataFrame
            # 预填充所有可能的列名，以便后续合并不会出错
            expected_cols = [
                'ema_score_5', 'ema_score_13', 'ema_score_21', 'ema_score_55', 'ema_score_233', # 示例 EMA 列
                'alignment_signal', 'ema_cross_signal', 'ema_strength',
                'score_momentum', 'score_momentum_acceleration', 'score_volatility', 'volatility_signal',
                'long_term_context', 'adx_strength_signal', 'stoch_signal',
                'vwap_deviation_signal', 'vwap_deviation_percent', 'boll_breakout_signal', 'boll_percent_b'
            ]
            for col in expected_cols:
                 analysis_df[col] = np.nan
            analysis_df['adx_strength_signal'] = 0.0 # ADX 默认无趋势
            analysis_df['stoch_signal'] = 0.0 # STOCH 默认无信号
            analysis_df['vwap_deviation_signal'] = 0 # VWAP 默认无信号
            analysis_df['boll_breakout_signal'] = 0 # BOLL 默认无信号
            analysis_df['vwap_deviation_percent'] = 0.0 # VWAP 偏离默认 0
            analysis_df['boll_percent_b'] = 50.0 # BOLL %b 默认 50
            return analysis_df

        score_series = base_score_series.copy() # 使用副本，避免修改原始 Series

        # --- 1. 计算分数 EMA ---
        all_ema_periods = ta_params.get('ema_periods', [5, 13, 21, 55, 233])
        if ta is not None:
            for period in all_ema_periods:
                try:
                    # 确保输入 Series 没有 None，虽然 pandas_ta 应该能处理 NaN
                    analysis_df[f'ema_score_{period}'] = ta.ema(score_series.dropna(), length=period)
                    # 将计算结果与原始索引对齐，并用 NaN 填充计算前的 NaN
                    analysis_df[f'ema_score_{period}'] = analysis_df[f'ema_score_{period}'].reindex(data.index)
                except Exception as e:
                    logger.error(f"计算 EMA Score {period} 时出错: {e}", exc_info=True)
                    analysis_df[f'ema_score_{period}'] = np.nan
        else:
             logger.warning("pandas_ta 未加载，跳过 EMA Score 计算。")
             for period in all_ema_periods:
                  analysis_df[f'ema_score_{period}'] = np.nan

        # 彻底消除所有 None，防止后续比较报错，用 NaN 替换 None
        for col in analysis_df.columns:
            if col.startswith('ema_score_'):
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce') # 确保是数值类型
                # print(f"EMA 计算结果: {col} - {analysis_df[col]} - {analysis_df[col].dtype}")

        # --- 2. 计算 EMA 排列信号 ---
        # 假设使用前4个周期进行排列判断，确保周期数足够
        ema_periods_for_align = all_ema_periods[:4]
        ema_cols_align = [f'ema_score_{p}' for p in ema_periods_for_align]
        if len(ema_cols_align) == 4 and all(col in analysis_df.columns for col in ema_cols_align):
            short_ema_col, mid1_ema_col, mid2_ema_col, long_ema_col = ema_cols_align
            # 使用 .loc 进行条件赋值，处理 NaN
            analysis_df['alignment_signal'] = 0 # 默认中性
            # 上升排列: 短 > 中1 > 中2 > 长
            analysis_condition_up = (analysis_df[short_ema_col] > analysis_df[mid1_ema_col]) & \
                                    (analysis_df[mid1_ema_col] > analysis_df[mid2_ema_col]) & \
                                    (analysis_df[mid2_ema_col] > analysis_df[long_ema_col])
            analysis_df.loc[analysis_condition_up, 'alignment_signal'] = 1 # 看涨信号

            # 下降排列: 短 < 中1 < 中2 < 长
            analysis_condition_down = (analysis_df[short_ema_col] < analysis_df[mid1_ema_col]) & \
                                      (analysis_df[mid1_ema_col] < analysis_df[mid2_ema_col]) & \
                                      (analysis_df[mid2_ema_col] < analysis_df[long_ema_col])
            analysis_df.loc[analysis_condition_down, 'alignment_signal'] = -1 # 看跌信号

            # 处理任何参与计算的 EMA 列为 NaN 的情况，将 alignment_signal 设为 NaN
            analysis_df.loc[analysis_df[ema_cols_align].isnull().any(axis=1), 'alignment_signal'] = np.nan

        else:
            logger.warning(f"无法计算 EMA 排列信号，所需的 EMA 列不足或缺失: {ema_cols_align} (可用列: {list(analysis_df.columns)})")
            analysis_df['alignment_signal'] = np.nan # 无法计算则为 NaN

        # --- 3. 计算 EMA 交叉信号 (短期 vs 中期) ---
        if len(all_ema_periods) >= 2:
            short_ema_col = f'ema_score_{all_ema_periods[0]}'
            mid_ema_col = f'ema_score_{all_ema_periods[1]}'
            if short_ema_col in analysis_df.columns and mid_ema_col in analysis_df.columns:
                short_ema = analysis_df[short_ema_col]
                mid_ema = analysis_df[mid_ema_col]
                # 确保 shift 操作不会引入额外的 NaN，或者在判断时处理 NaN
                short_ema_shift = short_ema.shift(1)
                mid_ema_shift = mid_ema.shift(1)

                # 金叉条件: 短期 EMA 上穿 中期 EMA
                golden_cross = (short_ema > mid_ema) & (short_ema_shift <= mid_ema_shift)
                # 死叉条件: 短期 EMA 下穿 中期 EMA
                death_cross = (short_ema < mid_ema) & (short_ema_shift >= mid_ema_shift)

                analysis_df['ema_cross_signal'] = 0 # 默认无交叉
                analysis_df.loc[golden_cross.fillna(False), 'ema_cross_signal'] = 1 # 金叉信号
                analysis_df.loc[death_cross.fillna(False), 'ema_cross_signal'] = -1 # 死叉信号

                # 如果当前或前一周期任一 EMA 为 NaN，则交叉信号为 NaN
                analysis_df.loc[short_ema.isna() | mid_ema.isna() | short_ema_shift.isna() | mid_ema_shift.isna(), 'ema_cross_signal'] = np.nan
            else:
                logger.warning(f"缺少短期或中期 EMA 列 ({short_ema_col}, {mid_ema_col})，无法计算 EMA 交叉信号。")
                analysis_df['ema_cross_signal'] = np.nan # 无法计算则为 NaN
        else:
            logger.warning("EMA 周期数不足，无法计算 EMA 交叉信号。")
            analysis_df['ema_cross_signal'] = np.nan # 无法计算则为 NaN


        # --- 4. 计算 EMA 强度 (短期 vs 长期) ---
        # 确保长期 EMA 周期存在且在计算出的 EMA 列中
        long_term_ema_period_context = ta_params.get('long_term_ema_period')
        if long_term_ema_period_context and long_term_ema_period_context in all_ema_periods:
            short_ema_col = f'ema_score_{all_ema_periods[0]}'
            long_ema_col = f'ema_score_{long_term_ema_period_context}'
            if short_ema_col in analysis_df.columns and long_ema_col in analysis_df.columns:
                analysis_df['ema_strength'] = analysis_df[short_ema_col] - analysis_df[long_ema_col]
            else:
                logger.warning(f"缺少短期或长期 EMA 列 ({short_ema_col}, {long_ema_col})，无法计算 EMA 强度。")
                analysis_df['ema_strength'] = np.nan
        else:
            logger.warning("长期 EMA 周期未指定或不在计算出的 EMA 周期列表中，无法计算 EMA 强度。")
            analysis_df['ema_strength'] = np.nan


        # --- 5. 计算得分动量及动量加速 ---
        # 确保 score_series 不是全 NaN
        if not score_series.isnull().all():
            analysis_df['score_momentum'] = score_series.diff()
            analysis_df['score_momentum_acceleration'] = analysis_df['score_momentum'].diff()
        else:
            analysis_df['score_momentum'] = np.nan
            analysis_df['score_momentum_acceleration'] = np.nan


        # --- 6. 计算得分波动率 ---
        volatility_window = ta_params.get('volatility_window', 10)
        # 确保 score_series 有足够的数据计算滚动标准差
        if len(score_series.dropna()) >= volatility_window:
            analysis_df['score_volatility'] = score_series.rolling(window=volatility_window).std()
            # 使用参数文件中的波动率阈值
            analysis_df['volatility_signal'] = 0 # 默认中性
            # 高波动率，趋势不稳定 (-1)
            analysis_df.loc[analysis_df['score_volatility'] > self.volatility_threshold_high, 'volatility_signal'] = -1
            # 低波动率，趋势稳定 (1)
            analysis_df.loc[analysis_df['score_volatility'] < self.volatility_threshold_low, 'volatility_signal'] = 1
            # NaN 视为中性 (0)
            analysis_df.loc[analysis_df['score_volatility'].isna(), 'volatility_signal'] = 0
        else:
            logger.warning(f"得分数据不足 ({len(score_series.dropna())} < {volatility_window})，无法计算得分波动率。")
            analysis_df['score_volatility'] = np.nan
            analysis_df['volatility_signal'] = 0 # 数据不足时信号为中性


        # 7. 长期趋势背景 (基于分数与长期 EMA)
        # 确保长期 EMA 周期存在且在计算出的 EMA 列中
        long_term_ema_period_context = ta_params.get('long_term_ema_period')
        if long_term_ema_period_context and long_term_ema_period_context in all_ema_periods:
            long_term_ema_col_context = f'ema_score_{long_term_ema_period_context}'
            if long_term_ema_col_context in analysis_df.columns:
                analysis_df['long_term_context'] = 0 # 默认中性
                # 分数在长均线上方 - 看涨背景 (1)
                analysis_df.loc[score_series > analysis_df[long_term_ema_col_context], 'long_term_context'] = 1
                # 分数在长均线下方 - 看跌背景 (-1)
                analysis_df.loc[score_series < analysis_df[long_term_ema_col_context], 'long_term_context'] = -1
                # 如果分数或长均线为 NaN，则背景为 NaN
                analysis_df.loc[score_series.isna() | analysis_df[long_term_ema_col_context].isna(), 'long_term_context'] = np.nan
            else:
                logger.warning(f"缺少长期 EMA 列 ({long_term_ema_col_context})，无法计算长期趋势背景。")
                analysis_df['long_term_context'] = np.nan
        else:
            logger.warning("长期 EMA 周期未指定或不在计算出的 EMA 周期列表中，无法计算长期趋势背景。")
            analysis_df['long_term_context'] = np.nan


        # --- 8. ADX 趋势强度判断 (使用 focus_timeframe) ---
        dmi_period = bs_params.get("dmi_period", 14)
        adx_col = f'ADX_{dmi_period}_{focus_tf}'
        pdi_col = f'+DI_{dmi_period}_{focus_tf}'
        mdi_col = f'-DI_{dmi_period}_{focus_tf}'
        # 检查所需的 ADX/PDI/MDI 列是否存在于原始数据中
        if all(c in data.columns for c in [adx_col, pdi_col, mdi_col]):
            adx = data[adx_col]
            pdi = data[pdi_col]
            mdi = data[mdi_col]
            # ADX 信号：1 (强上升), 0.5 (中等上升), 0 (无趋势), -0.5 (中等下降), -1 (强下降)
            analysis_df['adx_strength_signal'] = 0.0 # 默认无趋势
            # 强趋势 (ADX >= strong_threshold)
            strong_trend_condition = adx >= self.adx_strong_threshold
            # 强上升趋势 (ADX >= strong_threshold 且 +DI > -DI)
            analysis_df.loc[strong_trend_condition.fillna(False) & (pdi > mdi).fillna(False), 'adx_strength_signal'] = 1.0
            # 强下降趋势 (ADX >= strong_threshold 且 +DI < -DI)
            analysis_df.loc[strong_trend_condition.fillna(False) & (pdi < mdi).fillna(False), 'adx_strength_signal'] = -1.0

            # 中等趋势 (ADX >= moderate_threshold 且 ADX < strong_threshold)
            moderate_trend_condition = (adx >= self.adx_moderate_threshold) & (adx < self.adx_strong_threshold)
            # 中等上升趋势
            analysis_df.loc[moderate_trend_condition.fillna(False) & (pdi > mdi).fillna(False), 'adx_strength_signal'] = 0.5
            # 中等下降趋势
            analysis_df.loc[moderate_trend_condition.fillna(False) & (pdi < mdi).fillna(False), 'adx_strength_signal'] = -0.5

            # 处理可能的 NaN 值，如果 ADX, PDI, MDI 任一为 NaN，则信号为 NaN
            analysis_df.loc[adx.isna() | pdi.isna() | mdi.isna(), 'adx_strength_signal'] = np.nan
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 ADX/PDI/MDI 列 ({adx_col}, {pdi_col}, {mdi_col})，无法计算 ADX 强度信号。")
            analysis_df['adx_strength_signal'] = np.nan # 默认无法计算则为 NaN


        # --- 9. STOCH 超买超卖判断 (使用 focus_timeframe) ---
        if 'stoch' in self.trend_indicators:
            stoch_k = ia_params.get('stoch_k', 14)
            stoch_d = ia_params.get('stoch_d', 3)
            # 根据参数生成列名, 假设为 STOCH_K_k_tf, STOCH_D_k_d_tf
            k_col = f'STOCH_K_{stoch_k}_{focus_tf}'
            d_col = f'STOCH_D_{stoch_k}_{stoch_d}_{focus_tf}' # 假设 D 线列名格式

            if k_col in data.columns and d_col in data.columns:  # 检查列是否存在
                k_val = data[k_col]
                d_val = data[d_col]
                # STOCH 信号：1 (超卖区金叉), -1 (超买区死叉), 0.5 (超卖), -0.5 (超买), 0 (中间)
                analysis_df['stoch_signal'] = 0.0 # 默认中性

                # 判断金叉和死叉，并处理 NaN
                turning_up = (k_val > d_val) & (k_val.shift(1) <= d_val.shift(1))
                turning_down = (k_val < d_val) & (k_val.shift(1) >= d_val.shift(1))

                # 超卖区金叉 (看涨)
                oversold_golden_cross = (k_val < self.stoch_oversold_threshold) & (d_val < self.stoch_oversold_threshold) & turning_up
                analysis_df.loc[oversold_golden_cross.fillna(False), 'stoch_signal'] = 1.0

                # 超买区死叉 (看跌)
                overbought_death_cross = (k_val > self.stoch_overbought_threshold) & (d_val > self.stoch_overbought_threshold) & turning_down
                analysis_df.loc[overbought_death_cross.fillna(False), 'stoch_signal'] = -1.0

                # 仅超卖 (非金叉)
                only_oversold = (k_val < self.stoch_oversold_threshold) & (d_val < self.stoch_oversold_threshold) & ~turning_up
                analysis_df.loc[only_oversold.fillna(False), 'stoch_signal'] = 0.5

                # 仅超买 (非死叉)
                only_overbought = (k_val > self.stoch_overbought_threshold) & (d_val > self.stoch_overbought_threshold) & ~turning_down
                analysis_df.loc[only_overbought.fillna(False), 'stoch_signal'] = -0.5

                # 处理可能的 NaN 值，如果 K 或 D 任一为 NaN，则信号为 NaN
                analysis_df.loc[k_val.isna() | d_val.isna(), 'stoch_signal'] = np.nan

            else:
                logger.warning(f"[{self.strategy_name}] 缺少 STOCH K/D 列 ({k_col}, {d_col})，无法计算 STOCH 信号。")
                analysis_df['stoch_signal'] = np.nan  # 默认无法计算则为 NaN
        else:
            analysis_df['stoch_signal'] = np.nan  # STOCH 未启用则为 NaN


        # --- 10. VWAP 偏离判断 (使用 focus_timeframe) ---
        vwap_col = f'VWAP_{focus_tf}'
        close_col = f'close_{focus_tf}'
        if vwap_col in data.columns and close_col in data.columns:  # 检查列是否存在
            vwap = data[vwap_col]
            close_price = data[close_col]
            # 计算价格相对 VWAP 的偏离度百分比
            # 避免除以零，如果 VWAP 为零或 NaN，偏离度也为 NaN
            deviation = pd.Series(np.nan, index=data.index)
            valid_vwap = vwap.fillna(0) != 0 # 找到 VWAP 不为零且不为 NaN 的位置
            deviation.loc[valid_vwap] = ((close_price.loc[valid_vwap] - vwap.loc[valid_vwap]) / vwap.loc[valid_vwap])

            # VWAP 偏离信号：1 (显著高于 VWAP), -1 (显著低于 VWAP), 0 (接近 VWAP)
            analysis_df['vwap_deviation_signal'] = 0 # 默认接近
            # 显著高于 VWAP
            analysis_df.loc[deviation > self.vwap_deviation_threshold, 'vwap_deviation_signal'] = 1
            # 显著低于 VWAP
            analysis_df.loc[deviation < -self.vwap_deviation_threshold, 'vwap_deviation_signal'] = -1
            # 处理 NaN，如果偏离度为 NaN，则信号为 NaN
            analysis_df.loc[deviation.isna(), 'vwap_deviation_signal'] = np.nan

            # 保留原始偏离度值可能也有用
            analysis_df['vwap_deviation_percent'] = deviation * 100
        else:
            logger.warning(f"[{self.strategy_name}] 缺少 VWAP 或收盘价列 ({vwap_col}, {close_col})，无法计算 VWAP 偏离信号。")
            analysis_df['vwap_deviation_signal'] = np.nan  # 默认无法计算则为 NaN
            analysis_df['vwap_deviation_percent'] = np.nan # 默认无法计算则为 NaN


        # --- 11. BOLL 突破判断 (使用 focus_timeframe) ---
        boll_period = bs_params.get("boll_period", 20)  # 假设使用基础参数中的周期
        upper_col = f'BB_UPPER_{boll_period}_{focus_tf}'  # 需要确认列名格式
        lower_col = f'BB_LOWER_{boll_period}_{focus_tf}'
        middle_col = f'BB_MIDDLE_{boll_period}_{focus_tf}'  # 中轨也可能有用
        close_col = f'close_{focus_tf}'
        if all(c in data.columns for c in [upper_col, lower_col, close_col]):  # 检查列是否存在
            upper_band = data[upper_col]
            lower_band = data[lower_col]
            middle_band = data.get(middle_col)  # 中轨可选，使用 .get()
            close_price = data[close_col]
            # BOLL 突破信号：1 (向上突破上轨), -1 (向下突破下轨), 0 (轨道内)
            analysis_df['boll_breakout_signal'] = 0 # 默认轨道内
            # 向上突破上轨
            analysis_df.loc[close_price > upper_band, 'boll_breakout_signal'] = 1
            # 向下突破下轨
            analysis_df.loc[close_price < lower_band, 'boll_breakout_signal'] = -1
            # 处理 NaN，如果收盘价或布林带任一为 NaN，则信号为 NaN
            analysis_df.loc[close_price.isna() | upper_band.isna() | lower_band.isna(), 'boll_breakout_signal'] = np.nan

            # 可选：计算价格在布林带中的相对位置 (百分比)
            if middle_band is not None and middle_col in data.columns:
                band_width = upper_band - lower_band
                # 处理带宽为0或 NaN 的情况
                analysis_df['boll_percent_b'] = np.nan # 默认 NaN
                valid_band_width = band_width.fillna(0) != 0 # 找到带宽不为零且不为 NaN 的位置
                analysis_df.loc[valid_band_width, 'boll_percent_b'] = ((close_price.loc[valid_band_width] - lower_band.loc[valid_band_width]) / band_width.loc[valid_band_width]) * 100
                # 如果计算结果为 Inf 或 -Inf (例如价格等于上下轨)，也设为 NaN
                analysis_df['boll_percent_b'] = analysis_df['boll_percent_b'].replace([np.inf, -np.inf], np.nan)
            else:
                 logger.warning(f"缺少 BOLL 中轨列 ({middle_col})，无法计算 BOLL %b。")
                 analysis_df['boll_percent_b'] = np.nan

        else:
            logger.warning(f"[{self.strategy_name}] 缺少 BOLL 上轨/下轨或收盘价列 ({upper_col}, {lower_col}, {close_col})，无法计算 BOLL 突破信号或 %b。")
            analysis_df['boll_breakout_signal'] = np.nan  # 默认无法计算则为 NaN
            analysis_df['boll_percent_b'] = np.nan # 默认无法计算则为 NaN

        # --- 12. 趋势持续时间判断 (基于 ADX 和 DI 交叉) ---
        # 需要 ADX, +DI, -DI 列 (已在 ADX 部分检查)
        if 'adx_strength_signal' in analysis_df.columns and not analysis_df['adx_strength_signal'].isnull().all():
             # 识别上升趋势 (+DI > -DI) 和下降趋势 (+DI < -DI) 的持续时间
             is_uptrend = (data.get(pdi_col, pd.Series(np.nan, index=data.index)) > data.get(mdi_col, pd.Series(np.nan, index=data.index))).fillna(False)
             is_downtrend = (data.get(pdi_col, pd.Series(np.nan, index=data.index)) < data.get(mdi_col, pd.Series(np.nan, index=data.index))).fillna(False)

             # 计算连续的上升/下降趋势周期数
             # 使用 strategy_utils.count_consecutive_true 函数 (假设存在)
             # 或者手动计算
             uptrend_duration = is_uptrend.astype(int).groupby((is_uptrend != is_uptrend.shift()).cumsum()).cumsum()
             downtrend_duration = is_downtrend.astype(int).groupby((is_downtrend != is_downtrend.shift()).cumsum()).cumsum()

             # 趋势持续时间信号：1 (强持续), 0.5 (中等持续), 0 (不稳定/反转)
             analysis_df['trend_duration_signal'] = 0.0 # 默认不稳定
             # 强上升持续
             analysis_df.loc[is_uptrend & (uptrend_duration >= self.trend_duration_threshold_strong), 'trend_duration_signal'] = 1.0
             # 中等上升持续
             analysis_df.loc[is_uptrend & (uptrend_duration >= self.trend_duration_threshold_moderate) & (uptrend_duration < self.trend_duration_threshold_strong), 'trend_duration_signal'] = 0.5
             # 强下降持续
             analysis_df.loc[is_downtrend & (downtrend_duration >= self.trend_duration_threshold_strong), 'trend_duration_signal'] = -1.0
             # 中等下降持续
             analysis_df.loc[is_downtrend & (downtrend_duration >= self.trend_duration_threshold_moderate) & (downtrend_duration < self.trend_duration_threshold_strong), 'trend_duration_signal'] = -0.5

             # 如果 ADX, PDI, MDI 任一为 NaN，则持续时间信号为 NaN
             analysis_df.loc[data.get(adx_col).isna() | data.get(pdi_col).isna() | data.get(mdi_col).isna(), 'trend_duration_signal'] = np.nan

        else:
             logger.warning("缺少 ADX/PDI/MDI 列，无法计算趋势持续时间信号。")
             analysis_df['trend_duration_signal'] = np.nan # 默认无法计算则为 NaN


        # --- 13. 量能确认信号 (基于量能均线和 CMF) ---
        vc_params = self.params.get('volume_confirmation', {})
        if vc_params.get('enabled', False):
            vol_tf = vc_params.get('tf', self.focus_timeframe)
            amount_ma_col = f'AMT_MA_{vc_params.get("amount_ma_period", 15)}_{vol_tf}'
            cmf_col = f'CMF_{vc_params.get("cmf_period", 20)}_{vol_tf}'
            volume_col = f'volume_{vol_tf}'
            amount_col = f'amount_{vol_tf}'

            if all(c in data.columns for c in [amount_ma_col, cmf_col, volume_col, amount_col]):
                 amount = data[amount_col]
                 amount_ma = data[amount_ma_col]
                 cmf = data[cmf_col]

                 # 量能信号：1 (量能放大且 CMF > 0), -1 (量能缩小且 CMF < 0), 0 (中性)
                 analysis_df['volume_confirm_signal'] = 0 # 默认中性

                 # 量能放大确认 (当前成交额 > 均线) 且 CMF > 0
                 volume_boost_condition = (amount > amount_ma) & (cmf > 0)
                 analysis_df.loc[volume_boost_condition.fillna(False), 'volume_confirm_signal'] = 1

                 # 量能缩小确认 (当前成交额 < 均线) 且 CMF < 0
                 volume_penalty_condition = (amount < amount_ma) & (cmf < 0)
                 analysis_df.loc[volume_penalty_condition.fillna(False), 'volume_confirm_signal'] = -1

                 # 处理 NaN
                 analysis_df.loc[amount.isna() | amount_ma.isna() | cmf.isna(), 'volume_confirm_signal'] = np.nan
            else:
                 logger.warning(f"缺少量能确认所需列 ({amount_ma_col}, {cmf_col}, {volume_col}, {amount_col})，无法计算量能确认信号。")
                 analysis_df['volume_confirm_signal'] = np.nan
        else:
            analysis_df['volume_confirm_signal'] = np.nan # 量能确认未启用则为 NaN


        # --- 14. 量价背离信号 (从数据中获取，假设已计算) ---
        # 假设背离检测函数已将背离信号添加到数据中，列名如 'divergence_signal_bullish', 'divergence_signal_bearish'
        # 这里只检查是否存在并添加到分析结果中
        # 假设背离信号列名格式为 divergence_signal_type_indicator_tf
        # 例如: divergence_signal_regular_bullish_macd_hist_15
        # 我们需要一个汇总的背离信号
        # 假设存在一个汇总列 'total_divergence_signal' 或类似
        # 如果没有汇总列，可以根据参数中的背离指标和类型手动汇总
        dd_params = self.params.get('divergence_detection', {})
        if dd_params.get('enabled', True):
             div_tf = dd_params.get('tf', self.focus_timeframe)
             div_indicators_check = dd_params.get('indicators', {})
             divergence_signal_sum = pd.Series(0.0, index=data.index)
             divergence_count = 0

             # 检查并累加各种背离信号 (假设信号值为 1 或 -1)
             if div_indicators_check.get('macd_hist', True):
                 macd_fast = bs_params.get("macd_fast", 12)
                 macd_slow = bs_params.get("macd_slow", 26)
                 macd_signal = bs_params.get("macd_signal", 9)
                 col_bull = f'divergence_signal_regular_bullish_macd_hist_{div_tf}'
                 col_bear = f'divergence_signal_regular_bearish_macd_hist_{div_tf}'
                 col_h_bull = f'divergence_signal_hidden_bullish_macd_hist_{div_tf}'
                 col_h_bear = f'divergence_signal_hidden_bearish_macd_hist_{div_tf}'
                 for col in [col_bull, col_bear, col_h_bull, col_h_bear]:
                      if col in data.columns:
                           # 将背离信号转换为数值 (例如，看涨为正，看跌为负)
                           # 假设原始信号是布尔值或 1/-1
                           signal_val = data[col]
                           if col.startswith('divergence_signal_regular_bullish') or col.startswith('divergence_signal_hidden_bullish'):
                                divergence_signal_sum += signal_val.replace({True: 1, False: 0}).fillna(0) # 看涨信号加 1
                           elif col.startswith('divergence_signal_regular_bearish') or col.startswith('divergence_signal_hidden_bearish'):
                                divergence_signal_sum += signal_val.replace({True: -1, False: 0}).fillna(0) # 看跌信号减 1
                           # 如果原始信号已经是数值 1/-1，则直接累加
                           # divergence_signal_sum += signal_val.fillna(0)
                           divergence_count += 1 # 简单计数，或者更复杂的权重

             # 对其他指标 (RSI, MFI, OBV) 进行类似处理...
             # RSI
             if div_indicators_check.get('rsi', True):
                 rsi_period = bs_params.get("rsi_period", 14)
                 col_bull = f'divergence_signal_regular_bullish_rsi_{rsi_period}_{div_tf}'
                 col_bear = f'divergence_signal_regular_bearish_rsi_{rsi_period}_{div_tf}'
                 col_h_bull = f'divergence_signal_hidden_bullish_rsi_{rsi_period}_{div_tf}'
                 col_h_bear = f'divergence_signal_hidden_bearish_rsi_{rsi_period}_{div_tf}'
                 for col in [col_bull, col_bear, col_h_bull, col_h_bear]:
                      if col in data.columns:
                           signal_val = data[col]
                           if col.startswith('divergence_signal_regular_bullish') or col.startswith('divergence_signal_hidden_bullish'):
                                divergence_signal_sum += signal_val.replace({True: 1, False: 0}).fillna(0)
                           elif col.startswith('divergence_signal_regular_bearish') or col.startswith('divergence_signal_hidden_bearish'):
                                divergence_signal_sum += signal_val.replace({True: -1, False: 0}).fillna(0)
                           divergence_count += 1

             # MFI
             if div_indicators_check.get('mfi', True):
                 mfi_period = bs_params.get("mfi_period", 14)
                 col_bull = f'divergence_signal_regular_bullish_mfi_{mfi_period}_{div_tf}'
                 col_bear = f'divergence_signal_regular_bearish_mfi_{mfi_period}_{div_tf}'
                 col_h_bull = f'divergence_signal_hidden_bullish_mfi_{mfi_period}_{div_tf}'
                 col_h_bear = f'divergence_signal_hidden_bearish_mfi_{mfi_period}_{div_tf}'
                 for col in [col_bull, col_bear, col_h_bull, col_h_bear]:
                      if col in data.columns:
                           signal_val = data[col]
                           if col.startswith('divergence_signal_regular_bullish') or col.startswith('divergence_signal_hidden_bullish'):
                                divergence_signal_sum += signal_val.replace({True: 1, False: 0}).fillna(0)
                           elif col.startswith('divergence_signal_regular_bearish') or col.startswith('divergence_signal_hidden_bearish'):
                                divergence_signal_sum += signal_val.replace({True: -1, False: 0}).fillna(0)
                           divergence_count += 1

             # OBV
             if div_indicators_check.get('obv', False):
                 col_bull = f'divergence_signal_regular_bullish_obv_{div_tf}'
                 col_bear = f'divergence_signal_regular_bearish_obv_{div_tf}'
                 col_h_bull = f'divergence_signal_hidden_bullish_obv_{div_tf}'
                 col_h_bear = f'divergence_signal_hidden_bearish_obv_{div_tf}'
                 for col in [col_bull, col_bear, col_h_bull, col_h_bear]:
                      if col in data.columns:
                           signal_val = data[col]
                           if col.startswith('divergence_signal_regular_bullish') or col.startswith('divergence_signal_hidden_bullish'):
                                divergence_signal_sum += signal_val.replace({True: 1, False: 0}).fillna(0)
                           elif col.startswith('divergence_signal_regular_bearish') or col.startswith('divergence_signal_hidden_bearish'):
                                divergence_signal_sum += signal_val.replace({True: -1, False: 0}).fillna(0)
                           divergence_count += 1

             # 将累加的信号作为背离信号，可以归一化或直接使用
             # 这里简单使用累加值，正值表示看涨背离多，负值表示看跌背离多
             analysis_df['divergence_signal'] = divergence_signal_sum
             if divergence_count == 0:
                  logger.warning("背离检测启用，但没有找到任何背离信号列，背离信号设为 NaN。")
                  analysis_df['divergence_signal'] = np.nan
        else:
             analysis_df['divergence_signal'] = np.nan # 背离检测未启用则为 NaN


        # --- 15. K线形态信号 (从数据中获取，假设已计算) ---
        # 假设 K线形态检测函数已将形态信号添加到数据中，列名如 'CDLENGULFING_15', 'CDLHAMMER_15' 等
        # 这里只检查是否存在并添加到分析结果中
        kp_params = self.params.get('kline_pattern_detection', {})
        if kp_params.get('enabled', True):
             kp_tf = kp_params.get('tf', self.focus_timeframe)
             # 假设形态列以 'CDL' 开头，且值为 100 (看涨) 或 -100 (看跌) 或 0 (无形态)
             kline_signal_sum = pd.Series(0.0, index=data.index)
             kline_pattern_cols = [col for col in data.columns if col.startswith('CDL') and col.endswith(f'_{kp_tf}')]

             if kline_pattern_cols:
                  # 将所有形态信号列相加 (假设 100/-100 信号)
                  for col in kline_pattern_cols:
                       # 确保列是数值类型，将非数值转换为 0
                       kline_signal_sum += pd.to_numeric(data[col], errors='coerce').fillna(0)

                  # 可以根据总和判断强弱，例如 > 0 看涨，< 0 看跌
                  analysis_df['kline_pattern_signal'] = 0 # 默认中性
                  analysis_df.loc[kline_signal_sum > 0, 'kline_pattern_signal'] = 1 # 看涨形态多
                  analysis_df.loc[kline_signal_sum < 0, 'kline_pattern_signal'] = -1 # 看跌形态多
                  # 如果所有形态列都为 NaN 或 0，则总和为 0，信号为 0 (中性)
                  # 如果总和为 NaN (所有形态列都为 NaN)，则信号为 NaN
                  analysis_df.loc[kline_signal_sum.isna(), 'kline_pattern_signal'] = np.nan

             else:
                  logger.warning(f"K线形态检测启用，但没有找到任何形态信号列 (以 'CDL' 开头，时间框架 {kp_tf})，形态信号设为 NaN。")
                  analysis_df['kline_pattern_signal'] = np.nan
        else:
             analysis_df['kline_pattern_signal'] = np.nan # K线形态检测未启用则为 NaN


        # --- 16. 汇总辅助信号 ---
        # 将所有辅助信号列合并到一个 DataFrame 中，方便后续使用
        auxiliary_signals = analysis_df[[col for col in analysis_df.columns if col.endswith('_signal') or col.endswith('_context') or col.endswith('_strength') or col.endswith('_percent')]]
        # 确保所有辅助信号列都存在，如果某个信号计算失败 (NaN)，则该列为 NaN
        # analysis_df 现在包含了所有计算出的分析结果列

        # 返回包含所有分析结果的 DataFrame
        return analysis_df

    def _adjust_volatility_parameters(self, data: pd.DataFrame):
        """
        根据股票波动率动态调整参数，如波动率阈值。
        :param data: 包含波动率相关数据的 DataFrame
        """
        focus_tf = self.focus_timeframe
        close_col = f'close_{focus_tf}'
        if close_col in data:
            # 计算价格的波动率（基于收盘价的标准差）
            volatility_window = self.params['trend_analysis'].get('volatility_window', 10)
            price_volatility = data[close_col].rolling(window=volatility_window).std()
            if not price_volatility.isnull().all():
                latest_volatility = price_volatility.iloc[-1] if not price_volatility.empty else 0
                # 根据波动率调整 volatility_threshold_high 和 volatility_threshold_low
                self.volatility_adjust_factor = max(1.0, latest_volatility / 5.0)  # 假设基准波动率为5.0
                self.volatility_threshold_high = self.tf_params.get('volatility_threshold_high', 10) * self.volatility_adjust_factor
                self.volatility_threshold_low = self.tf_params.get('volatility_threshold_low', 5) * self.volatility_adjust_factor
                # logger.info(f"动态调整波动率阈值: high={self.volatility_threshold_high}, low={self.volatility_threshold_low}, factor={self.volatility_adjust_factor}")
            else:
                logger.warning("波动率数据不可用，无法动态调整参数。")
        else:
            logger.warning(f"缺少收盘价列 {close_col}，无法动态调整参数。")

    def _apply_adx_boost(self, final_signal: pd.Series, adx_strength_norm: pd.Series, normalized_contribution: pd.Series) -> pd.Series:
        """
        模块化调整逻辑：使用 ADX 强度增强信号。
        :param final_signal: 当前信号
        :param adx_strength_norm: ADX 强度归一化值
        :param normalized_contribution: 归一化贡献值，用于确定趋势方向
        :return: 调整后的信号
        """
        adx_boost = np.abs(adx_strength_norm)
        adjustment = np.sign(normalized_contribution) * adx_boost * 5  # ADX 强时最多调整 +/- 5 分
        logger.debug(f"ADX 增强调整: {adjustment.iloc[-1] if not adjustment.empty else 'N/A'}")
        return final_signal + adjustment

    def _apply_divergence_penalty(self, final_signal: pd.Series, divergence_signals: pd.DataFrame, dd_params: Dict) -> pd.Series:
        """
        模块化调整逻辑：应用背离惩罚。
        :param final_signal: 当前信号
        :param divergence_signals: 背离信号 DataFrame
        :param dd_params: 背离检测参数
        :return: 调整后的信号
        """
        if not divergence_signals.empty:
            penalty_factor = dd_params.get('divergence_penalty_factor', 0.45)
            is_bullish_trend = final_signal > 55
            is_bearish_trend = final_signal < 45
            has_bearish_div = divergence_signals.get('has_bearish_divergence', pd.Series(False, index=final_signal.index))
            has_bullish_div = divergence_signals.get('has_bullish_divergence', pd.Series(False, index=final_signal.index))

            # 使用 Pandas Series 操作，避免直接使用 np.where 返回 NumPy 数组
            adjusted_signal = final_signal.copy()
            mask_bullish_div = is_bullish_trend & has_bearish_div
            mask_bearish_div = is_bearish_trend & has_bullish_div

            adjusted_signal = adjusted_signal.where(~mask_bullish_div, 50 + (final_signal - 50) * (1 - penalty_factor))
            adjusted_signal = adjusted_signal.where(~mask_bearish_div, 50 + (final_signal - 50) * (1 - penalty_factor))
            
            logger.debug(f"背离惩罚调整后信号: {adjusted_signal.iloc[-1] if not adjusted_signal.empty else 'N/A'}")
        return adjusted_signal

    def _apply_trend_confirmation(self, final_signal: pd.Series) -> pd.Series:
        """
        增强假信号过滤：要求信号突破阈值后持续若干周期才视为有效。
        :param final_signal: 当前信号
        :return: 过滤后的信号
        """
        confirmed_signal = final_signal.copy()
        trend_threshold_upper = 55
        trend_threshold_lower = 45
        for i in range(len(final_signal)):
            if i < self.trend_confirmation_periods:
                continue
            window = final_signal.iloc[i - self.trend_confirmation_periods + 1:i + 1]
            if all(window > trend_threshold_upper):
                confirmed_signal.iloc[i] = final_signal.iloc[i]
            elif all(window < trend_threshold_lower):
                confirmed_signal.iloc[i] = final_signal.iloc[i]
            else:
                confirmed_signal.iloc[i] = 50.0  # 未确认趋势，返回中性分
        logger.debug(f"趋势确认过滤后信号: {confirmed_signal.iloc[-1] if not confirmed_signal.empty else 'N/A'}")
        return confirmed_signal

    def generate_signals(self, data: pd.DataFrame, stock_code: str) -> pd.Series:
        """
        生成趋势跟踪信号，整合基础分、趋势分析、量能、背离等信息，并结合LSTM模型预测。
        """
        logger.info(f"开始执行策略: {self.strategy_name} (Focus: {self.focus_timeframe})，股票代码: {stock_code}")

        # 1. 计算规则基础信号 (final_signal)
        # 假设 _calculate_rule_based_signal 方法存在并返回规则信号和中间结果
        final_signal, intermediate_results = self._calculate_rule_based_signal(data=data, stock_code=stock_code)

        # 2. LSTM模型预测
        # 初始化 LSTM 信号为中性分
        lstm_signal = pd.Series(50.0, index=data.index)
        # 获取 LSTM 需要的特征列 (与 prepare_data_for_lstm 中的 required_columns 一致)
        required_cols = self.get_required_columns()

        # 设置股票特定的模型路径
        self.set_model_paths(stock_code)

        # 检查模型和 scaler 是否存在，如果不存在则训练
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            logger.info(f"股票 {stock_code} 的 LSTM 模型或 Scaler 不存在，开始训练...")
            # 训练模型，训练完成后 self.lstm_model, self.feature_scaler, self.target_scaler 会被设置
            # 注意：这里需要将规则生成的 final_signal 添加到 data 中，作为 LSTM 的目标变量
            data_with_target = data.copy()
            data_with_target['final_signal'] = final_signal # 将规则信号作为 LSTM 的训练目标
            self.train_lstm_model_for_stock(data_with_target, stock_code, required_cols)
        else:
            # 如果模型和 scaler 存在，则加载
            self.load_lstm_model(stock_code)

        # 如果模型和 scaler 成功加载或训练，则进行预测
        if self.lstm_model is not None and self.feature_scaler is not None and self.target_scaler is not None:
            try:
                # 准备用于预测的数据
                # 使用 get_required_columns 获取特征列，并确保这些列在 data 中存在
                # 填充 NaN 值，与 prepare_data_for_lstm 中的处理一致
                # 只选择实际存在的列进行预测
                features_for_prediction = data.loc[:, [col for col in required_cols if col in data.columns]].copy()
                features_for_prediction = features_for_prediction.ffill().bfill()

                if features_for_prediction.isnull().any().any():
                    logger.warning(f"股票 {stock_code} 用于 LSTM 预测的数据中仍有 NaN 值，可能影响预测结果。将尝试填充为0。")
                    features_for_prediction.fillna(0, inplace=True)


                if not features_for_prediction.empty and len(features_for_prediction) >= self.window_size:
                    # 应用特征缩放 (使用加载或训练好的 feature_scaler)
                    features_scaled = self.feature_scaler.transform(features_for_prediction.values)

                    # 构建预测所需的窗口数据
                    X_predict = []
                    # 从倒数 window_size 个数据点开始构建最后一个窗口
                    if len(features_scaled) >= self.window_size:
                         X_predict.append(features_scaled[-self.window_size:])
                         X_predict = np.array(X_predict)

                         if X_predict.shape[0] > 0:
                             # 进行预测
                             # predict 返回的是缩放后的结果
                             lstm_pred_scaled = self.lstm_model.predict(X_predict, verbose=0)

                             # 逆缩放预测结果 (使用加载或训练好的 target_scaler)
                             # lstm_pred_scaled 是二维数组 (n_samples, 1)
                             lstm_pred_original_scale = self.target_scaler.inverse_transform(lstm_pred_scaled)

                             # 将预测结果映射到信号 Series 的最后一个位置
                             # 注意：LSTM 预测的是窗口 *之后* 的一个点
                             # 如果我们预测的是当前时间点的信号，那么需要将预测结果对应到 data 的最后一个索引
                             # 如果预测的是未来一个时间点的信号，则需要不同的处理
                             # 假设这里预测的是当前时间点的信号
                             if not lstm_pred_original_scale.flatten().tolist():
                                 logger.warning(f"股票 {stock_code} LSTM 预测结果为空。")
                             else:
                                 # 将预测结果（原始尺度）转换为 0-100 的信号分
                                 # 假设原始尺度的目标变量 (final_signal) 范围大致就是 0-100 分
                                 # 那么逆缩放后的值就是预测的分数
                                 lstm_signal_score = lstm_pred_original_scale.flatten()[-1] # 取最后一个预测值
                                 # 将预测分数限制在 0-100 范围内
                                 lstm_signal.iloc[-1] = np.clip(lstm_signal_score, 0, 100)
                                 logger.info(f"股票 {stock_code} LSTM 模型预测完成，最新预测信号: {lstm_signal.iloc[-1]:.2f}")
                         else:
                             logger.warning(f"股票 {stock_code} 数据不足以构建 LSTM 预测窗口。")
                    else:
                         logger.warning(f"股票 {stock_code} 数据长度 {len(features_for_prediction)} 小于窗口大小 {self.window_size}，无法进行 LSTM 预测。")
                else:
                    logger.warning(f"股票 {stock_code} 用于 LSTM 预测的数据为空或长度不足。")

            except Exception as e:
                logger.error(f"股票 {stock_code} LSTM 模型预测出错: {e}", exc_info=True)
                # 预测出错时，lstm_signal 保持默认的 50.0

        else:
            logger.warning(f"股票 {stock_code} 的 LSTM 模型或 Scaler 未成功加载/训练，跳过 LSTM 预测。")

        # 3. 结合规则信号和LSTM信号
        # 确保 combined_signal 的长度与原始数据一致
        combined_signal = pd.Series(50.0, index=data.index) # 初始化为中性分
        # 只有当 final_signal 和 lstm_signal 都有值时才进行组合
        # 注意：lstm_signal 目前只在最后一个点有预测值，其他点是 50.0
        # 如果需要对历史数据也进行 LSTM 预测，需要修改上面的预测逻辑
        # 假设我们只关心最新的组合信号
        if not final_signal.empty and not lstm_signal.empty:
             # 组合最新的信号点
             latest_final_signal = final_signal.iloc[-1] if not final_signal.isnull().all() else 50.0
             latest_lstm_signal = lstm_signal.iloc[-1] if not lstm_signal.isnull().all() else 50.0
             latest_combined_signal = 0.7 * latest_final_signal + 0.3 * latest_lstm_signal
             combined_signal.iloc[-1] = np.clip(latest_combined_signal, 0, 100).round(2)
             # 对于历史数据，combined_signal 可以等于 final_signal，或者也进行历史 LSTM 预测
             # 为了简化，这里只组合最新的点，历史点使用规则信号
             combined_signal.iloc[:-1] = final_signal.iloc[:-1] # 历史数据使用规则信号
        else:
             # 如果规则信号或 LSTM 信号为空，则组合信号也为空或中性
             logger.warning(f"股票 {stock_code} 规则信号或 LSTM 信号为空，组合信号将不完整或为中性。")
             combined_signal = final_signal.copy() # 如果规则信号存在，使用规则信号


        # --- 存储中间数据 ---
        try:
            # 从 intermediate_results 获取规则计算的中间数据
            base_scores_df = intermediate_results.get('base_scores_df', pd.DataFrame(index=data.index))
            trend_analysis_df = intermediate_results.get('trend_analysis_df', pd.DataFrame(index=data.index))
            divergence_signals = intermediate_results.get('divergence_signals', pd.DataFrame(index=data.index))

            # 将规则信号、LSTM信号和组合信号添加到中间数据中
            self.intermediate_data = pd.concat([
                base_scores_df,
                trend_analysis_df,
                divergence_signals.add_prefix('div_'), # 给背离信号列加前缀，避免冲突
                pd.DataFrame({
                    'final_signal': final_signal,
                    'lstm_signal': lstm_signal, # 这里的 lstm_signal 只有最后一个点有预测值
                    'combined_signal': combined_signal
                }, index=data.index)
            ], axis=1)
        except Exception as e:
            logger.warning(f"存储中间数据时出错: {e}，将仅存储最终信号数据。")
            self.intermediate_data = pd.DataFrame({
                'final_signal': final_signal,
                'lstm_signal': lstm_signal,
                'combined_signal': combined_signal
            }, index=data.index)


        logger.info(f"{self.strategy_name}: 信号生成完毕，股票代码: {stock_code}，最新组合信号: {combined_signal.iloc[-1] if not combined_signal.empty else 'N/A'}")

        # 调用 analyze_signals 方法进行分析，传入股票代码
        self.analyze_signals(stock_code)

        return combined_signal
    
    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回中间计算结果"""
        return self.intermediate_data

    def _calculate_trend_duration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算趋势的持续时间和强度，并根据 focus_timeframe 转换为具体时间单位。
        使用参数文件中定义的持续时间阈值。
        :param data: 包含趋势指标的 DataFrame
        :return: 包含趋势持续时间和强度的字典
        """
        trend_duration_info = {
            'bullish_duration': 0,
            'bearish_duration': 0,
            'bullish_duration_text': '0分钟',
            'bearish_duration_text': '0分钟',
            'current_trend': 'neutral',
            'trend_strength': 'weak',
            'duration_status': 'short' # 新增：持续时间状态 (short, moderate, long)
        }

        if 'final_signal' not in data or data['final_signal'].isnull().all():
            return trend_duration_info

        final_signal = data['final_signal'].dropna()
        if final_signal.empty:
            return trend_duration_info

        # 计算持续时间（数据点）
        current_bullish_streak = 0
        current_bearish_streak = 0
        trend_threshold_upper = 55
        trend_threshold_lower = 45
        for signal in final_signal.iloc[::-1]:
            if signal >= trend_threshold_upper:
                current_bullish_streak += 1
                current_bearish_streak = 0
            elif signal <= trend_threshold_lower:
                current_bearish_streak += 1
                current_bullish_streak = 0
            else: # 中性区域，中断计数
                 # 如果需要，可以设置一个更宽的中性区来维持计数，但当前逻辑是中断
                 break

        trend_duration_info['bullish_duration'] = current_bullish_streak
        trend_duration_info['bearish_duration'] = current_bearish_streak

        # 根据 focus_timeframe 转换为具体时间单位
        try:
            timeframe_value = int(self.focus_timeframe)  # 假设 focus_timeframe 是类似 '30' 的字符串，表示分钟
            # 计算持续时间（分钟）
            bullish_duration_minutes = current_bullish_streak * timeframe_value
            bearish_duration_minutes = current_bearish_streak * timeframe_value

            # 转换为合适的单位（分钟、小时、天）
            def format_duration(minutes):
                if minutes < 60:
                    return f"{minutes}分钟"
                elif minutes < 1440:  # 24小时 = 1440分钟
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
            logger.warning(f"无法将 focus_timeframe '{self.focus_timeframe}' 转换为时间单位，持续时间将以周期数显示")
            trend_duration_info['bullish_duration_text'] = f"{current_bullish_streak}个周期" if current_bullish_streak > 0 else '0个周期'
            trend_duration_info['bearish_duration_text'] = f"{current_bearish_streak}个周期" if current_bearish_streak > 0 else '0个周期'

        # 确定当前趋势和强度
        latest_signal = final_signal.iloc[-1]
        if latest_signal >= 75: # 更高的强趋势阈值
            trend_duration_info['current_trend'] = 'bullish'
            trend_duration_info['trend_strength'] = 'very strong'
        elif latest_signal >= 65: # 强趋势
            trend_duration_info['current_trend'] = 'bullish'
            trend_duration_info['trend_strength'] = 'strong'
        elif latest_signal >= trend_threshold_upper: # 温和看涨
            trend_duration_info['current_trend'] = 'bullish'
            trend_duration_info['trend_strength'] = 'moderate'
        elif latest_signal <= 25: # 很强看跌
            trend_duration_info['current_trend'] = 'bearish'
            trend_duration_info['trend_strength'] = 'very strong'
        elif latest_signal <= 35: # 强看跌
            trend_duration_info['current_trend'] = 'bearish'
            trend_duration_info['trend_strength'] = 'strong'
        elif latest_signal <= trend_threshold_lower: # 温和看跌
            trend_duration_info['current_trend'] = 'bearish'
            trend_duration_info['trend_strength'] = 'moderate'
        else: # 中性
            trend_duration_info['current_trend'] = 'neutral'
            trend_duration_info['trend_strength'] = 'weak'

        # 判断持续时间状态
        current_duration = max(current_bullish_streak, current_bearish_streak)
        if current_duration >= self.trend_duration_threshold_strong:
             trend_duration_info['duration_status'] = 'long'
        elif current_duration >= self.trend_duration_threshold_moderate:
             trend_duration_info['duration_status'] = 'moderate'
        else:
             trend_duration_info['duration_status'] = 'short'
        return trend_duration_info

    # 修改 analyze_signals 方法，主要分析 combined_signal
    def analyze_signals(self, stock_code: str) -> Optional[Dict[str, Any]]: # Changed return type to Dict
        """
        分析趋势策略信号，增加对新指标、趋势增强/枯竭、背离的解读，优化操作建议。
        适应A股 T+1 交易制度，增加止损止盈建议。
        """
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
        if 'combined_signal' in data:
            combined_signal = data['combined_signal'].dropna()
            if not combined_signal.empty:
                analysis_results['combined_signal_mean'] = combined_signal.mean()
                analysis_results['combined_signal_median'] = combined_signal.median()
                analysis_results['combined_signal_std'] = combined_signal.std()
                analysis_results['combined_signal_bullish_ratio'] = (combined_signal > 55).mean()
                analysis_results['combined_signal_bearish_ratio'] = (combined_signal < 45).mean()
                analysis_results['combined_signal_strong_bullish_ratio'] = (combined_signal >= 70).mean()
                analysis_results['combined_signal_strong_bearish_ratio'] = (combined_signal <= 30).mean()
        # Keep analysis for other intermediate signals if needed
        if 'alignment_signal' in data:
            alignment = data['alignment_signal'].dropna()
            if not alignment.empty:
                analysis_results['alignment_fully_bullish_ratio'] = (alignment == 3).mean()
                analysis_results['alignment_fully_bearish_ratio'] = (alignment == -3).mean()
                analysis_results['alignment_bullish_ratio'] = (alignment > 0).mean()
                analysis_results['alignment_bearish_ratio'] = (alignment < 0).mean()
        if 'long_term_context' in data:
            context = data['long_term_context'].dropna()
            if not context.empty:
                analysis_results['long_term_bullish_ratio'] = (context == 1).mean()
                analysis_results['long_term_bearish_ratio'] = (context == -1).mean()
        if 'trend_strength_score' in data:
            trend_strength_score_series = data['trend_strength_score'].dropna()
            if not trend_strength_score_series.empty:
                analysis_results['trend_strength_mean'] = trend_strength_score_series.mean()
                analysis_results['trend_strength_strong_bull_ratio'] = (trend_strength_score_series >= 1.5).mean()
                analysis_results['trend_strength_strong_bear_ratio'] = (trend_strength_score_series <= -1.5).mean()

        # --- 计算趋势持续时间 (基于 final_signal，因为 combined_signal 历史数据可能不包含 LSTM 预测) ---
        # 或者，如果 combined_signal 历史数据也包含规则信号，可以使用 combined_signal
        # 假设 trend duration 仍然基于规则信号更稳定
        trend_duration_info = self._calculate_trend_duration(data) # This method uses 'final_signal'
        analysis_results.update(trend_duration_info)

        # --- 最新信号判断和细化操作建议 (基于 combined_signal) ---
        signal_judgment = {}
        operation_advice = "中性观望"
        risk_warning = ""
        t_plus_1_note = "（受 T+1 限制，建议次日操作）"
        stop_loss_profit_advice = ""
        # 使用最新的 combined_signal 进行判断
        final_score = latest_data.get('combined_signal', 50.0) # Use combined_signal here
        current_trend = trend_duration_info['current_trend'] # Trend status from rule-based signal
        trend_strength = trend_duration_info['trend_strength']
        duration_status = trend_duration_info['duration_status']

        # 基础判断 (基于 combined_signal 的值)
        if final_score >= 60: # 调整阈值以适应 0-100 分数
             signal_judgment['overall_signal'] = "看涨信号"
             if final_score >= 75:
                  signal_judgment['overall_signal'] += " (强)"
                  if duration_status == 'long':
                      operation_advice = f"持有或逢低加仓 (信号强劲且趋势持续) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止盈（接近近期高点）"
                  elif duration_status == 'moderate':
                      operation_advice = f"持有或试探加仓 (信号强劲但需确认趋势持续) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止损（近期低点下方）"
                  else:
                      operation_advice = f"关注买入信号 (信号强劲但趋势刚启动) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止损（入场价下方3-5%）"
             else: # 60 <= final_score < 75
                  signal_judgment['overall_signal'] += " (温和)"
                  if duration_status == 'long':
                      operation_advice = f"谨慎持有 (信号温和但趋势持续较长) {t_plus_1_note}"
                  elif duration_status == 'moderate':
                      operation_advice = f"持有观察 (信号温和且趋势持续中) {t_plus_1_note}"
                  else:
                      operation_advice = f"观望或轻仓试多 (信号温和启动) {t_plus_1_note}"

        elif final_score <= 40: # 调整阈值
             signal_judgment['overall_signal'] = "看跌信号"
             if final_score <= 25:
                  signal_judgment['overall_signal'] += " (强)"
                  if duration_status == 'long':
                      operation_advice = f"卖出或逢高减仓 (信号强劲且趋势持续) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止盈（接近近期低点）"
                  elif duration_status == 'moderate':
                      operation_advice = f"减仓或准备卖出 (信号强劲但需确认趋势持续) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止损（近期高点上方）"
                  else:
                      operation_advice = f"关注卖出信号 (信号强劲但趋势刚启动) {t_plus_1_note}"
                      stop_loss_profit_advice = "建议设置止损（入场价上方3-5%）"
             else: # 25 < final_score <= 40
                  signal_judgment['overall_signal'] += " (温和)"
                  if duration_status == 'long':
                      operation_advice = f"谨慎持有空头或卖出 (信号温和但趋势持续较长) {t_plus_1_note}"
                  elif duration_status == 'moderate':
                      operation_advice = f"持有空头或观望 (信号温和且趋势持续中) {t_plus_1_note}"
                  else:
                      operation_advice = f"观望或轻仓试空 (信号温和启动) {t_plus_1_note}"
        else:
            signal_judgment['overall_signal'] = "中性信号"
            operation_advice = "中性观望，等待信号明朗"

        # --- 结合其他指标细化判断和建议 ---
        # EMA 排列 (基于规则信号的中间结果)
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

        # 长期背景 (基于规则信号的中间结果)
        long_context = latest_data.get('long_term_context', 0)
        if long_context == 1:
            signal_judgment['long_term_view'] = "长期看涨"
        elif long_context == -1:
            signal_judgment['long_term_view'] = "长期看跌"
        else:
            signal_judgment['long_term_view'] = "长期不明"

        # ADX 强度 (基于规则信号的中间结果)
        adx_signal = latest_data.get('adx_strength_signal', 0)
        if adx_signal >= 0.5:
            signal_judgment['adx_status'] = f"趋势明确 (上升)"
        elif adx_signal == 0:
            signal_judgment['adx_status'] = "无明显趋势"
        else:
            signal_judgment['adx_status'] = f"趋势明确 (下降)"
        # 如果 combined_signal 显示强趋势，但 ADX 弱，则提示风险
        if abs(final_score - 50) > 20 and abs(adx_signal) < 0.5:
            risk_warning += "ADX显示趋势强度不足，注意假突破风险。 "

        # STOCH 状态与风险提示 (基于规则信号的中间结果)
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
        # 如果 combined_signal 看涨，但 STOCH 超买，则提示风险
        if final_score > 50 and stoch_signal <= -0.5:
            risk_warning += "STOCH进入超买区，警惕回调。 "
            if "止盈" not in stop_loss_profit_advice: # 避免重复建议
                 stop_loss_profit_advice = "建议设置止盈（当前价上方5-8%）"
        # 如果 combined_signal 看跌，但 STOCH 超卖，则提示风险
        if final_score < 50 and stoch_signal >= 0.5:
            risk_warning += "STOCH进入超卖区，警惕反弹。 "
            if "止盈" not in stop_loss_profit_advice: # 避免重复建议
                 stop_loss_profit_advice = "建议设置止盈（当前价下方5-8%）"

        # BOLL 突破 (基于规则信号的中间结果)
        boll_signal = latest_data.get('boll_breakout_signal', 0)
        if boll_signal == 1:
            signal_judgment['boll_status'] = "向上突破布林带"
            if final_score > 50: operation_advice += " - BOLL突破确认"
        elif boll_signal == -1:
            signal_judgment['boll_status'] = "向下突破布林带"
            if final_score < 5_0: operation_advice += " - BOLL突破确认"
        else:
            signal_judgment['boll_status'] = "布林带轨道内运行"

        # 量能确认 (基于规则信号的中间结果)
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
            if abs(final_score - 50) > 10: # 如果信号偏离中性且放量
                operation_advice += " (放量)"
            else: # 如果信号中性但放量
                operation_advice += " (放量关注突破)"

        # 背离信号解读与风险提示 (基于规则信号的中间结果)
        has_bearish_div = latest_data.get('div_has_bearish_divergence', False)
        has_bullish_div = latest_data.get('div_has_bullish_divergence', False)
        if has_bearish_div and final_score > 50: # 看涨信号但有顶背离
            signal_judgment['divergence_status'] = "检测到顶背离"
            risk_warning += "检测到顶背离，趋势可能衰竭或反转！ "
            operation_advice = operation_advice.replace("加仓", "观望").replace("买入", "谨慎买入").replace("持有", "谨慎持有")
            if "止损" not in stop_loss_profit_advice: # 避免重复建议
                 stop_loss_profit_advice = "建议设置止损（当前价下方3-5%）"
        elif has_bullish_div and final_score < 50: # 看跌信号但有底背离
            signal_judgment['divergence_status'] = "检测到底背离"
            risk_warning += "检测到底背离，趋势可能衰竭或反转！ "
            operation_advice = operation_advice.replace("减仓", "观望").replace("卖出", "谨慎卖出").replace("持有空头", "谨慎持有空头")
            if "止损" not in stop_loss_profit_advice: # 避免重复建议
                 stop_loss_profit_advice = "建议设置止损（当前价上方3-5%）"
        else:
            signal_judgment['divergence_status'] = "无明显背离"

        # --- 生成中文解读 ---
        bullish_duration_text = trend_duration_info['bullish_duration_text']
        bearish_duration_text = trend_duration_info['bearish_duration_text']
        duration_text = f"看涨持续 {bullish_duration_text}" if trend_duration_info['current_trend'] == 'bullish' and trend_duration_info['bullish_duration'] > 0 else \
                        f"看跌持续 {bearish_duration_text}" if trend_duration_info['current_trend'] == 'bearish' and trend_duration_info['bearish_duration'] > 0 else \
                        "趋势持续时间不足"

        now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        chinese_interpretation = (
            f"【趋势跟踪策略分析 - {stock_code} - {now_str}】\n"
            f"最新组合信号分: {final_score:.2f} (规则信号: {latest_data.get('final_signal', 50.0):.2f}, LSTM预测: {latest_data.get('lstm_signal', 50.0):.2f})\n" # 显示规则和LSTM原始分
            f"当前趋势状态: {signal_judgment.get('overall_signal', '中性')}\n" # 使用组合信号判断整体状态
            f"规则趋势判断: {trend_duration_info['current_trend'].capitalize()} ({trend_duration_info['trend_strength'].capitalize()})\n" # 显示规则判断的趋势状态和强度
            f"趋势持续: {duration_text} ({trend_duration_info['duration_status'].capitalize()})\n"
            f"EMA排列: {signal_judgment.get('alignment_status', '未知')}\n"
            f"长期背景: {signal_judgment.get('long_term_view', '未知')}\n"
            f"ADX强度: {signal_judgment.get('adx_status', '未知')}\n"
            f"STOCH状态: {signal_judgment.get('stoch_status', '未知')}\n"
            f"BOLL状态: {signal_judgment.get('boll_status', '未知')}\n"
            f"量能状态: {signal_judgment.get('volume_status', '未知')}{f' ({signal_judgment["volume_spike"]})' if 'volume_spike' in signal_judgment else ''}\n"
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
        logger.info(chinese_interpretation) # 打印分析结果

        return analysis_results

    def get_analysis_results(self) -> Optional[Dict[str, Any]]:
        """返回信号分析结果"""
        return self.analysis_results

    def save_analysis_results(self, stock_code: str, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        保存趋势跟踪策略的分析结果到数据库
        """
        from stock_models.stock_analytics import StockScoreAnalysis
        from stock_models.stock_basic import StockInfo
        import logging

        logger = logging.getLogger("strategy_trend_following")

        try:
            stock = StockInfo.objects.get(stock_code=stock_code)
            analysis_data = self.analysis_results.iloc[0].to_dict() if self.analysis_results is not None and not self.analysis_results.empty else {}
            intermediate_data = self.intermediate_data.iloc[-1].to_dict() if self.intermediate_data is not None and not self.intermediate_data.empty else {}
            latest_data = data.iloc[-1].to_dict() if data is not None and not data.empty else {}

            def convert_nan_to_none(value):
                # 处理 inf 和 -inf
                if isinstance(value, (int, float)) and (np.isinf(value) or np.isnan(value)):
                     return None
                return None if pd.isna(value) else value

            # --- 准备要保存的数据字典 ---
            defaults_dict = {
                'score': convert_nan_to_none(intermediate_data.get('final_signal', None)),
                'base_score_raw': convert_nan_to_none(intermediate_data.get('base_score_raw', None)),
                'base_score_volume_adjusted': convert_nan_to_none(intermediate_data.get('base_score_volume_adjusted', None)),
                'alignment_signal': convert_nan_to_none(intermediate_data.get('alignment_signal', None)),
                'long_term_context': convert_nan_to_none(intermediate_data.get('long_term_context', None)),
                'trend_strength_score': convert_nan_to_none(intermediate_data.get('trend_strength_score', None)), # 新增趋势强度评分
                'score_momentum': convert_nan_to_none(intermediate_data.get('score_momentum', None)),
                'score_volatility': convert_nan_to_none(intermediate_data.get('score_volatility', None)), # 新增波动率
                'adx_strength_signal': convert_nan_to_none(intermediate_data.get('adx_strength_signal', None)), # 新增 ADX 信号
                'stoch_signal': convert_nan_to_none(intermediate_data.get('stoch_signal', None)), # 新增 STOCH 信号
                'vwap_deviation_signal': convert_nan_to_none(intermediate_data.get('vwap_deviation_signal', None)), # 新增 VWAP 偏离
                'boll_breakout_signal': convert_nan_to_none(intermediate_data.get('boll_breakout_signal', None)), # 新增 BOLL 突破
                'ema_cross_signal': convert_nan_to_none(intermediate_data.get('ema_cross_signal', None)), # 新增 EMA 交叉
                'volume_confirmation_signal': convert_nan_to_none(intermediate_data.get('volume_confirmation_signal', None)), # 新增量能确认信号
                'volume_spike_signal': convert_nan_to_none(intermediate_data.get('volume_spike_signal', None)), # 新增量能突增信号
                'div_has_bearish_divergence': convert_nan_to_none(intermediate_data.get('div_has_bearish_divergence', None)), # 新增顶背离标志
                'div_has_bullish_divergence': convert_nan_to_none(intermediate_data.get('div_has_bullish_divergence', None)), # 新增底背离标志
                'close_price': convert_nan_to_none(latest_data.get(f'close_{self.focus_timeframe}', None)),
                # 添加分析结果中的关键信息
                'current_trend': analysis_data.get('current_trend', None),
                'trend_strength': analysis_data.get('trend_strength', None),
                'bullish_duration': convert_nan_to_none(analysis_data.get('bullish_duration', None)),
                'bearish_duration': convert_nan_to_none(analysis_data.get('bearish_duration', None)),
                'operation_advice': analysis_data.get('operation_advice', None), # 保存操作建议
                'risk_warning': analysis_data.get('risk_warning', None), # 保存风险提示
                # 保存参数快照
                'params_snapshot': self.params, # 保存整个参数文件或仅相关部分
            }

            # --- 清理 NaN 值 ---
            defaults_cleaned = {k: convert_nan_to_none(v) for k, v in defaults_dict.items()}

            # --- 数据库操作 ---
            StockScoreAnalysis.objects.update_or_create(
                stock=stock,
                strategy_name=self.strategy_name,
                timestamp=timestamp,
                time_level=self.focus_timeframe,
                defaults=defaults_cleaned
            )
            logger.info(f"成功保存 {stock_code} 的趋势跟踪策略分析结果，时间戳: {timestamp}")
        except StockInfo.DoesNotExist:
            logger.error(f"股票 {stock_code} 未找到，无法保存分析结果")
        except Exception as e:
            # 打印更详细的错误信息，特别是 defaults_cleaned 的内容
            logger.error(f"保存 {stock_code} 的趋势跟踪策略分析结果时出错: {e}", exc_info=True)
            # logger.error(f"尝试保存的数据: {defaults_cleaned}") # 调试时可以取消注释此行

    def load_lstm_model(self, stock_code: str):
        """
        为特定股票加载 LSTM 模型和 scaler。
        """
        self.set_model_paths(stock_code) # 设置股票特定的路径

        # 检查模型和 scaler 文件是否存在
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                # 加载模型
                # 如果模型使用了自定义层、激活函数、损失函数或指标，需要提供 custom_objects
                # 目前 build_lstm_model 使用标准 Keras 层和激活函数，所以可能不需要
                self.lstm_model = tf.keras.models.load_model(self.model_path)
                logger.info(f"股票 {stock_code} LSTM 模型从 {self.model_path} 加载成功。")

                # 加载 Scaler
                scalers = joblib.load(self.scaler_path)
                self.feature_scaler = scalers.get('feature_scaler')
                self.target_scaler = scalers.get('target_scaler')
                if self.feature_scaler and self.target_scaler:
                    logger.info(f"股票 {stock_code} Scaler 从 {self.scaler_path} 加载成功。")
                else:
                    logger.warning(f"股票 {stock_code} Scaler 文件 {self.scaler_path} 内容不完整，加载失败。")
                    self.feature_scaler = None
                    self.target_scaler = None
                    self.lstm_model = None # Scaler 加载失败，模型也视为无效

            except Exception as e:
                logger.error(f"股票 {stock_code} 加载 LSTM 模型或 Scaler 出错: {e}", exc_info=True)
                self.lstm_model = None
                self.feature_scaler = None
                self.target_scaler = None
        else:
            logger.warning(f"股票 {stock_code} 的 LSTM 模型或 Scaler 文件不存在 ({self.model_path}, {self.scaler_path})，将跳过加载。")
            self.lstm_model = None
            self.feature_scaler = None
            self.target_scaler = None

    def train_and_save_lstm_model(self, data: pd.DataFrame, stock_code: str):
        """
        训练LSTM模型并保存（模型和Scaler分开保存，支持多股票）。
        """
        self.set_model_paths(stock_code) # 设置该股票模型的保存路径
        required_cols = self.get_required_columns() # 获取策略所需的所有列名

        # --- 确保目标列存在且是纯规则信号 ---
        target_col_name = 'final_signal' # 定义目标列名
        if target_col_name not in data.columns:
            logger.info(f"[{stock_code}] 目标列 '{target_col_name}' 不存在，将计算纯规则信号作为目标。")
            data = data.copy()  # 避免修改原始传入的 DataFrame
            # 调用内部方法计算基于规则的信号
            rule_signal, _ = self._calculate_rule_based_signal(data, stock_code)
            data[target_col_name] = rule_signal # 将规则信号作为目标列
        # 可选：如果目标列已存在，但想确保它是最新的纯规则信号，可以取消下面的注释
        # elif self.lstm_model is not None: # 如果目标列已存在，但可能被污染了，重新计算
        #     logger.warning(f"[{stock_code}] 目标列 '{target_col_name}' 已存在，为确保是纯规则信号，将重新计算。")
        #     data = data.copy()
        #     rule_signal, _ = self._calculate_rule_based_signal(data, stock_code)
        #     data[target_col_name] = rule_signal

        # --- 检查数据是否足够 ---
        if data.shape[0] < self.window_size + 10: # 至少需要 window_size + 少量样本来训练和测试
            logger.warning(f"[{stock_code}] 数据量 ({data.shape[0]}) 过少，不足以进行窗口大小为 {self.window_size} 的LSTM训练。跳过训练。")
            return None

        try:
            # --- 调用 prepare_data_for_lstm 并接收所有 8 个返回值 ---
            # --- 修改：明确设置 use_feature_selection=False 以避免与 use_pca=True 冲突 ---
            X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = prepare_data_for_lstm(
                data=data,                      # 包含特征和目标列的完整数据
                required_columns=required_cols, # 策略所需的特征列（函数内部会排除目标列）
                target_column=target_col_name,  # 指定目标列名
                window_size=self.window_size,   # LSTM 时间窗口大小
                scaler_type='minmax',           # 特征缩放器类型 ('minmax' 或 'standard')
                target_scaler_type='minmax',    # 目标变量缩放器类型 ('minmax' 或 'standard')
                train_split=0.7,                # 训练集比例
                val_split=0.15,                 # 验证集比例
                use_pca=False,                   # 保持启用PCA降维 (因为原代码是这样写的)
                use_feature_selection=True,    # <--- 显式禁用基于模型的特征选择
                feature_selector_model='rf',  # <--- 使用随机森林作为选择器
                max_features_fs=20,             # <--- 选择前20个重要特征
                feature_selection_threshold='median',  # <--- 阈值设置
                n_components=0.99,              # PCA保留的方差比例 (如果 use_pca=True)
                # apply_variance_threshold=False # 可选：是否应用方差阈值过滤
            )

            # --- 将 feature_scaler 赋值给 self.scaler ---
            # 注意：prepare_data_for_lstm 返回的是 feature_scaler，我们将其赋给 self.scaler
            self.scaler = feature_scaler

            # --- 验证数据准备结果 ---
            logger.info(f"[{stock_code}] LSTM数据集 shape: X_train={X_train.shape}, y_train={y_train.shape}, "
                        f"X_val={X_val.shape}, y_val={y_val.shape}, "
                        f"X_test={X_test.shape}, y_test={y_test.shape}")
            logger.info(f"[{stock_code}] 特征 Scaler对象: {self.scaler}")
            logger.info(f"[{stock_code}] 目标 Scaler对象: {target_scaler}") # 记录目标 scaler

            # --- 检查训练数据是否有效 ---
            if X_train.shape[0] == 0:
                logger.warning(f"[{stock_code}] 准备数据后 X_train 为空，无法训练LSTM模型。")
                return None # 无法训练，直接返回

            # --- 如果训练数据有效，则继续 ---
            # 动态获取处理后的特征维度 (PCA可能改变维度)
            num_features = X_train.shape[2]
            logger.info(f"[{stock_code}] 动态获取的特征维度: {num_features}") # 修改日志说明

            # --- 构建新模型，确保输入形状正确 ---
            self.lstm_model = build_lstm_model(
                window_size=self.window_size, # 时间窗口大小
                num_features=num_features,    # 使用处理后的实际特征维度
                model_config=self.model_config,# 模型配置字典
                model_type='lstm',            # 模型类型 (lstm, bilstm, gru)
                summary=True                  # 是否打印模型结构
            )

            # --- 训练模型，并将 target_scaler 传递给训练函数 ---
            history = train_lstm_model(
                X_train, y_train, # 训练数据 (特征和目标都已缩放)
                X_val, y_val,     # 验证数据 (特征和目标都已缩放)
                X_test, y_test,   # 测试数据 (特征和目标都已缩放)
                model=self.lstm_model, # 已构建和编译的模型
                target_scaler=target_scaler, # <--- 传入目标变量的缩放器
                training_config=self.training_config, # 训练配置 (epochs, batch_size 等)
                checkpoint_path=self.checkpoint_path, # 最佳模型保存路径 <--- 使用 self.checkpoint_path
                plot_training_history=True # 是否绘制训练历史图
            )
            if 'val_loss' in history:
                logger.info(f"[{stock_code}] 训练完成，最新 val_loss: {history['val_loss'][-1]:.4f}，是否改善: {'是' if history['val_loss'][-1] < 0.02362 else '否'}")

            # --- 保存最终模型和特征缩放器 ---
            # train_lstm_model 中的 ModelCheckpoint 会保存最佳模型到 checkpoint_path
            # 这里我们假设 train_lstm_model 恢复了最佳权重，然后保存到 self.model_path
            self.lstm_model.save(self.model_path)
            logger.info(f"[{stock_code}] 最终LSTM模型已保存至: {self.model_path}")

            # 保存特征缩放器 (self.scaler)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"[{stock_code}] 特征Scaler已保存至: {self.scaler_path}")

            # --- (可选) 保存目标缩放器 ---
            target_scaler_path = self.model_path.replace('.keras', '_target_scaler.save')
            joblib.dump(target_scaler, target_scaler_path)
            logger.info(f"[{stock_code}] 目标Scaler已保存至: {target_scaler_path}")

            # --- (可选) 在测试集上评估最终模型 ---
            # train_lstm_model 内部已经进行了评估并打印日志
            # 如果需要在这里再次评估或获取评估结果，可以取消注释
            # if X_test.shape[0] > 0:
            #     test_loss, test_mae_scaled = self.lstm_model.evaluate(X_test, y_test, verbose=0)
            #     # 使用 target_scaler 估算原始范围的 MAE
            #     try:
            #         dummy_values = np.array([[0.0], [test_mae_scaled]])
            #         original_values = target_scaler.inverse_transform(dummy_values)
            #         mae_original_approx = abs(original_values[1][0] - original_values[0][0])
            #     except Exception: mae_original_approx = np.nan
            #     logger.info(f"[{stock_code}] 最终模型在测试集上的损失: {test_loss:.4f}, MAE (缩放后): {test_mae_scaled:.4f}, MAE (原始范围估算): {mae_original_approx:.4f}")
            # else:
            #     logger.warning(f"[{stock_code}] 测试集为空，无法评估最终LSTM模型。")

            return history # 返回训练历史

        except ValueError as ve: # 捕获特定的 ValueError
             if "unpack" in str(ve):
                 logger.error(f"[{stock_code}] 训练时发生解包错误: {ve}。请检查 prepare_data_for_lstm 的返回值数量和接收变量数量是否匹配。", exc_info=True)
             else:
                 logger.error(f"[{stock_code}] 训练时发生值错误: {ve}", exc_info=True)
             # 清理可能未完全初始化的模型和 scaler
             self.lstm_model = None
             self.scaler = None
             return None
        except Exception as e:
            logger.error(f"[{stock_code}] 训练和保存LSTM模型时发生未知错误: {e}", exc_info=True)
            # 清理可能未完全初始化的模型和 scaler
            self.lstm_model = None
            self.scaler = None
            return None
        
    def _calculate_rule_based_signal(self, data: pd.DataFrame, stock_code: str) -> Tuple[pd.Series, Dict]:
        """
        计算基于规则的信号，并返回中间结果。
        Args:
            data (pd.DataFrame): 输入数据。
            stock_code (str): 股票代码。
        Returns:
            Tuple[pd.Series, Dict]: 最终信号和中间结果字典。
        """
        if data is None or data.empty:
            logger.warning("输入数据为空，无法生成信号。")
            return pd.Series(dtype=float), {}

        # --- 检查必需列 ---
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"[{self.strategy_name}] 输入数据缺少必需列: {missing_cols}。策略无法运行。")
            return pd.Series(50.0, index=data.index), {}

        # 检查数据完整性（是否有过多 NaN）
        nan_check_cols = [f'close_{self.focus_timeframe}', f'ADX_{self.params["base_scoring"]["dmi_period"]}_{self.focus_timeframe}']
        if data[nan_check_cols].isnull().all().any():
            logger.error(f"[{self.strategy_name}] 关键输入数据 ({nan_check_cols}) 全为 NaN。策略无法运行。")
            return pd.Series(50.0, index=data.index), {}

        # --- 动态调整参数 ---
        self._adjust_volatility_parameters(data)

        # --- 计算趋势导向的基础评分 ---
        base_scores_df = self._calculate_trend_focused_score(data)

        # --- 应用量能调整 ---
        vc_params = self.params.get('volume_confirmation', {})
        vc_params_adjusted = vc_params.copy()
        vc_params_adjusted['boost_factor'] = self.volume_boost_factor
        vc_params_adjusted['penalty_factor'] = self.volume_penalty_factor
        vc_params_adjusted['volume_spike_threshold'] = self.volume_spike_threshold
        dd_params = self.params.get('divergence_detection', {})
        bs_params = self.params.get('base_scoring', {})
        base_score_adjusted, volume_analysis = strategy_utils.adjust_score_with_volume(
            base_scores_df['base_score_raw'], data, vc_params_adjusted, dd_params, bs_params, return_analysis=True
        )
        base_scores_df['base_score_volume_adjusted'] = base_score_adjusted
        base_scores_df = pd.concat([base_scores_df, volume_analysis], axis=1)

        # --- 执行趋势分析 ---
        trend_analysis_df = self._perform_trend_analysis(data, base_scores_df['base_score_volume_adjusted'])

        # --- 检测背离信号 ---
        divergence_signals = pd.DataFrame(index=data.index)
        if dd_params.get('enabled', True):
            try:
                divergence_signals = strategy_utils.detect_divergence(data, dd_params, bs_params)
                # logger.info(f"背离检测完成，发现信号: {divergence_signals.iloc[-1].to_dict() if not divergence_signals.empty else '无'}")
            except Exception as e:
                logger.error(f"执行背离检测时出错: {e}")

        # --- 组合最终信号 ---
        final_signal = pd.Series(50.0, index=data.index)
        base_score_norm = (base_scores_df['base_score_volume_adjusted'].fillna(50.0) - 50) / 50
        alignment_norm = trend_analysis_df.get('alignment_signal', pd.Series(0, index=data.index)).fillna(0) / 3
        long_context_norm = trend_analysis_df.get('long_term_context', pd.Series(0, index=data.index)).fillna(0)
        momentum_norm = np.sign(trend_analysis_df.get('score_momentum', pd.Series(0, index=data.index)).fillna(0))
        ema_cross_norm = trend_analysis_df.get('ema_cross_signal', pd.Series(0, index=data.index)).fillna(0)
        boll_breakout_norm = trend_analysis_df.get('boll_breakout_signal', pd.Series(0, index=data.index)).fillna(0)
        adx_strength_norm = trend_analysis_df.get('adx_strength_signal', pd.Series(0, index=data.index)).fillna(0)
        vwap_dev_norm = trend_analysis_df.get('vwap_deviation_signal', pd.Series(0, index=data.index)).fillna(0)
        volume_spike_signal = base_scores_df.get('volume_spike_signal', pd.Series(0, index=data.index)).fillna(0)
        total_weighted_contribution = pd.Series(0.0, index=data.index)
        total_weight = 0.0
        w_base = self.signal_weights.get('base_score', 0.5)
        total_weighted_contribution += base_score_norm * w_base
        total_weight += w_base
        w_align = self.signal_weights.get('alignment', 0.25)
        total_weighted_contribution += alignment_norm * w_align
        total_weight += w_align
        w_context = self.signal_weights.get('long_context', 0.1)
        total_weighted_contribution += long_context_norm * w_context
        total_weight += w_context
        w_momentum = self.signal_weights.get('momentum', 0.15)
        total_weighted_contribution += momentum_norm * w_momentum
        total_weight += w_momentum
        w_ema_cross = self.ema_cross_weight
        total_weighted_contribution += ema_cross_norm * w_ema_cross
        total_weight += w_ema_cross
        w_boll = self.boll_breakout_weight
        total_weighted_contribution += boll_breakout_norm * w_boll
        total_weight += w_boll
        if total_weight > 0:
            normalized_contribution = (total_weighted_contribution / total_weight).clip(-1, 1)
            final_signal = 50.0 + normalized_contribution * 50.0
        else:
            logger.warning("信号权重总和为0，使用中性分50.0作为最终信号。")
            final_signal = pd.Series(50.0, index=data.index)
        final_signal = self._apply_adx_boost(final_signal, adx_strength_norm, normalized_contribution)
        final_signal = self._apply_divergence_penalty(final_signal, divergence_signals, dd_params)
        vwap_adjustment = -vwap_dev_norm * np.sign(normalized_contribution) * 2
        final_signal += vwap_adjustment
        logger.debug(f"VWAP 偏离调整: {vwap_adjustment.iloc[-1] if not vwap_adjustment.empty else 'N/A'}")
        volume_spike_adjustment = volume_spike_signal * np.sign(normalized_contribution) * 3
        final_signal += volume_spike_adjustment
        logger.debug(f"成交量突增调整: {volume_spike_adjustment.iloc[-1] if not volume_spike_adjustment.empty else 'N/A'}")
        final_signal = self._apply_trend_confirmation(final_signal)
        rule_signal = final_signal.clip(0, 100).round(2)
        
        # 返回最终信号和中间结果
        intermediate_results = {
            'base_scores_df': base_scores_df,
            'trend_analysis_df': trend_analysis_df,
            'divergence_signals': divergence_signals
        }
        return rule_signal, intermediate_results









