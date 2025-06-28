# 此策略非常简单，主要基于价格与 VWAP 的偏离度，并以 5 分钟级别为主要参考。
# t_plus_0_strategy.py
import asyncio
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, Any, List, Optional

from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao

# 假设 BaseStrategy 和常量在 .base 或 core.constants
from .base import BaseStrategy
# 注意：T+0 策略可能不需要 strategy_utils

logger = logging.getLogger("strategy_t_plus_0")

class TPlus0Strategy(BaseStrategy):
    """
    T+0 日内交易策略：
    - 主要基于价格与 VWAP 的偏离度。
    - 以指定的时间框架（默认为 '5'）为主要参考。
    - 可选：结合长期趋势过滤。
    - 输出离散信号：1 (买入), -1 (卖出), 0 (无信号)。
    """
    strategy_name = "TPlus0Strategy"
    focus_timeframe = '5' # 主要关注的时间框架

    def __init__(self, params_file: str = "strategies/indicator_parameters.json"):
        """初始化策略，加载参数"""
        self.params_file = params_file
        self.params = self._load_params()
        self.strategy_name = self.params.get('t_plus_0_strategy_name', self.strategy_name)
        self.focus_timeframe = self.params.get('t_plus_0_signals', {}).get('focus_timeframe', self.focus_timeframe)
        # T+0 策略通常不计算复杂分数，直接生成信号
        self.intermediate_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[pd.DataFrame] = None

        super().__init__(self.params)

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
        if 't_plus_0_signals' not in self.params:
             raise ValueError("参数中必须包含 't_plus_0_signals' 部分。")
        t0_params = self.params['t_plus_0_signals']
        bs_params = self.params.get('base_scoring', {}) # 可能需要 timeframes

        if 'timeframes' not in bs_params or not isinstance(bs_params['timeframes'], list):
             logger.warning("'base_scoring.timeframes' 未定义，无法检查 focus_timeframe")
        elif self.focus_timeframe not in bs_params['timeframes']:
             raise ValueError(f"'focus_timeframe' ({self.focus_timeframe}) 必须在 'base_scoring.timeframes' 中")

        if 'buy_dev_threshold' not in t0_params or 'sell_dev_threshold' not in t0_params:
             raise ValueError("'t_plus_0_signals' 必须包含 'buy_dev_threshold' 和 'sell_dev_threshold'")
        if t0_params['buy_dev_threshold'] >= 0 or t0_params['sell_dev_threshold'] <= 0:
             raise ValueError("'buy_dev_threshold' 应为负数, 'sell_dev_threshold' 应为正数")

        logger.info(f"[{self.strategy_name}] 参数验证通过，主要关注时间框架: {self.focus_timeframe}")

    def get_required_columns(self) -> List[str]:
        """返回 T+0 策略所需的列"""
        required = set()
        t0_params = self.params['t_plus_0_signals']
        analysis_tf = self.focus_timeframe # T+0 分析使用焦点时间框架

        # VWAP (假设 IndicatorService 提供，可能是 vwap 或 vwap_{tf})
        # 尝试两种命名方式
        required.add(f'vwap_{analysis_tf}')
        # required.add('vwap') # 如果不区分时间框架

        required.add(f'close_{analysis_tf}') # 需要当前价格

        # 如果启用了长期趋势过滤
        if t0_params.get('use_long_term_filter', False):
             # 需要计算长期趋势的列，假设来自 TrendFollowStrategy 或单独计算
             # 这里需要确定 long_term_context 的来源
             # 假设它是由另一个策略或服务预先计算并合并到 data 中的
             required.add('long_term_context') # 需要这个列

        return list(required)

    def _generate_t0_signals_internal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        内部函数，生成 T+0 交易建议信号。
        :param data: 包含所需列的 DataFrame
        :return: 包含 't0_signal' 列的 DataFrame
        """
        signals = pd.DataFrame(index=data.index)
        signals['t0_signal'] = 0 # 0: No signal, 1: Buy, -1: Sell
        t0_params = self.params['t_plus_0_signals']
        analysis_tf = self.focus_timeframe

        # 确定 VWAP 列名
        vwap_col_tf = f'VWAP_{analysis_tf}'
        vwap_col_no_tf = 'VWAP'
        vwap_col = None
        if vwap_col_tf in data and data[vwap_col_tf].notna().any():
            vwap_col = vwap_col_tf
        elif vwap_col_no_tf in data and data[vwap_col_no_tf].notna().any():
            vwap_col = vwap_col_no_tf

        close_col = f'close_{analysis_tf}'
        long_term_context_col = 'long_term_context'

        if vwap_col is None:
             logger.warning(f"缺少有效的 VWAP 列 ({vwap_col_tf} or {vwap_col_no_tf})，无法生成 T+0 信号。")
             return signals
        if close_col not in data or data[close_col].isnull().all():
             logger.warning(f"缺少 Close 列 ({close_col}) 或全为 NaN，无法生成 T+0 信号。")
             return signals

        vwap = data[vwap_col]
        close = data[close_col]

        if vwap.isnull().all() or close.isnull().all():
            logger.warning(f"VWAP ({vwap_col}) 或 Close ({close_col}) 列数据无效，无法生成 T+0 信号。")
            return signals

        # 计算价格与 VWAP 的偏离度 (%)
        # deviation = (close - vwap) / vwap * 100 # 百分比形式
        deviation = (close - vwap) / vwap # 小数形式，阈值也用小数
        deviation = deviation.fillna(0) # 处理可能的 NaN

        # 买入条件：价格低于 VWAP 达到阈值
        buy_threshold = t0_params['buy_dev_threshold'] # 应为负数, e.g., -0.005 (-0.5%)
        buy_condition = deviation <= buy_threshold

        # 卖出条件：价格高于 VWAP 达到阈值
        sell_threshold = t0_params['sell_dev_threshold'] # 应为正数, e.g., 0.005 (+0.5%)
        sell_condition = deviation >= sell_threshold

        # 应用长期趋势过滤
        if t0_params.get('use_long_term_filter', False):
             if long_term_context_col not in data:
                  logger.warning(f"启用了长期趋势过滤，但缺少 {long_term_context_col} 列。将不应用过滤。")
             else:
                  long_term_context = data[long_term_context_col].fillna(0) # 填充 NaN 为中性
                  # 只有在长期趋势向上 (>=0, 即 1 或 0) 时才考虑买入信号
                  buy_condition &= (long_term_context >= 0)
                  # 只有在长期趋势向下 (<=0, 即 -1 或 0) 时才考虑卖出信号
                  sell_condition &= (long_term_context <= 0)
        signals.loc[buy_condition, 't0_signal'] = 1
        signals.loc[sell_condition, 't0_signal'] = -1
        return signals

    def generate_signals(self, data: pd.DataFrame, stock_code: str) -> pd.Series:
        """
        生成 T+0 信号 (1 买, -1 卖, 0 无)。
        """
        logger.info(f"开始执行策略: {self.strategy_name} (Focus: {self.focus_timeframe})")
        if data is None or data.empty:
            logger.warning("输入数据为空，无法生成信号。")
            return pd.Series(dtype=int) # 返回整数信号
        logger.info(f"data数据: {data.columns}")
        # --- 检查必需列 ---
        required_cols = self.get_required_columns()
        # 处理 VWAP 列名的不确定性
        has_vwap = False
        vwap_cols_to_check = [f'VWAP_{self.focus_timeframe}', 'vwap']
        actual_vwap_cols = [col for col in vwap_cols_to_check if col in data and data[col].notna().any()]
        if actual_vwap_cols:
            has_vwap = True
        else:
            # 如果 VWAP 列不存在，从 required_cols 中移除它们，避免报错
            required_cols = [col for col in required_cols if col not in vwap_cols_to_check]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if not has_vwap:
             missing_cols.append("vwap (找不到有效的列)")
        if missing_cols:
            logger.error(f"[{self.strategy_name}] 输入数据缺少必需列: {missing_cols}。策略无法运行。")
            return pd.Series(0, index=data.index) # 返回无信号
        # --- 步骤 1: 生成 T+0 信号 ---
        t0_signals_df = self._generate_t0_signals_internal(data)
        # --- 步骤 2: 存储中间数据 (可选，T+0 比较简单) ---
        self.intermediate_data = t0_signals_df # 只包含 t0_signal 列
        final_signal = t0_signals_df['t0_signal']

        logger.info(f"{self.strategy_name}: 信号生成完毕。")
        self.analyze_signals(stock_code) # 执行分析
        return final_signal # 直接返回信号 Series

    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回中间计算结果"""
        return self.intermediate_data

    def analyze_signals(self, stock_code: str) -> Optional[pd.DataFrame]:
        """分析 T+0 策略信号"""
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning("中间数据为空，无法进行信号分析。")
            return None

        analysis_results = {}
        data = self.intermediate_data
        latest_data = data.iloc[-1] if not data.empty else None
        # stock_basic_dao = StockBasicDAO()
        # stock = asyncio.run(stock_basic_dao.get_stock_by_code(stock_code))

        # --- 统计分析 ---
        if 't0_signal' in data:
            t0_signal = data['t0_signal'].dropna()
            if not t0_signal.empty:
                 analysis_results['t0_buy_ratio'] = (t0_signal == 1).mean()
                 analysis_results['t0_sell_ratio'] = (t0_signal == -1).mean()
                 analysis_results['t0_no_signal_ratio'] = (t0_signal == 0).mean()
                 analysis_results['t0_buy_count'] = (t0_signal == 1).sum()
                 analysis_results['t0_sell_count'] = (t0_signal == -1).sum()


        # --- 最新信号判断 ---
        signal_judgment = {}
        if latest_data is not None:
            if 't0_signal' in latest_data:
                 signal = latest_data['t0_signal']
                 if signal == 1:
                      signal_judgment['t0_status'] = "T+0 买入信号"
                      signal_judgment['t0_suggestion'] = "价格相对 VWAP 偏低，存在日内做多机会"
                 elif signal == -1:
                      signal_judgment['t0_status'] = "T+0 卖出信号"
                      signal_judgment['t0_suggestion'] = "价格相对 VWAP 偏高，存在日内做空/卖出机会"
                 else:
                      signal_judgment['t0_status'] = "无 T+0 信号"
                      signal_judgment['t0_suggestion'] = "价格接近 VWAP 或不满足过滤条件，建议观望"

            # 中文解读
            chinese_interpretation = (
                f"【T+0 策略分析 - {stock_code} - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}】\n"
                f"核心观点:\n"
                f" - 当前 T+0 信号: {signal_judgment.get('t0_status', '未知')}\n"
                f"操作建议:\n"
                f" - {signal_judgment.get('t0_suggestion', '暂无建议')}\n"
                f"统计数据:\n"
                f" - T+0 买入信号比例: {analysis_results.get('t0_buy_ratio', np.nan)*100:.2f}%\n"
                f" - T+0 卖出信号比例: {analysis_results.get('t0_sell_ratio', np.nan)*100:.2f}%\n"
                f" - 无 T+0 信号比例: {analysis_results.get('t0_no_signal_ratio', np.nan)*100:.2f}%"
            )
            logger.info(chinese_interpretation)
            print(chinese_interpretation)

        analysis_results.update(signal_judgment)
        self.analysis_results = pd.DataFrame([analysis_results])
        return self.analysis_results

    def save_analysis_results(self, stock_code: str, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        保存 T+0 策略的分析结果到数据库
        """
        from stock_models.stock_analytics import StockScoreAnalysis
        from stock_models.stock_basic import StockInfo
        import logging

        logger = logging.getLogger("strategy_t_plus_0")

        try:
            stock = StockInfo.objects.get(stock_code=stock_code)
            analysis_data = self.analysis_results.iloc[0] if self.analysis_results is not None and not self.analysis_results.empty else {}
            intermediate_data = self.intermediate_data.iloc[-1] if self.intermediate_data is not None and not self.intermediate_data.empty else {}
            latest_data = data.iloc[-1] if data is not None and not data.empty else {}
            # 将 NaN 转换为 None 以兼容 MySQL
            def convert_nan_to_none(value):
                return None if pd.isna(value) else value

            StockScoreAnalysis.objects.update_or_create(
                stock=stock,
                strategy_name=self.strategy_name,
                timestamp=timestamp,
                time_level=self.focus_timeframe,
                defaults={
                    'score': convert_nan_to_none(intermediate_data.get('t0_signal', None)),
                    't0_signal': convert_nan_to_none(intermediate_data.get('t0_signal', None)),
                    'price_vwap_deviation': convert_nan_to_none(None),  # 需要计算时可从 data 中提取
                    'close_price': convert_nan_to_none(latest_data.get(f'close_{self.focus_timeframe}', None)),
                    'vwap': convert_nan_to_none(latest_data.get(f'vwap_{self.focus_timeframe}', latest_data.get('vwap', None))),
                    't0_buy_ratio': convert_nan_to_none(analysis_data.get('t0_buy_ratio', None)),
                    't0_sell_ratio': convert_nan_to_none(analysis_data.get('t0_sell_ratio', None)),
                    't0_no_signal_ratio': convert_nan_to_none(analysis_data.get('t0_no_signal_ratio', None)),
                    't0_buy_count': convert_nan_to_none(analysis_data.get('t0_buy_count', None)),
                    't0_sell_count': convert_nan_to_none(analysis_data.get('t0_sell_count', None)),
                    'params_snapshot': self.params,
                }
            )
            logger.info(f"成功保存 {stock_code} 的 T+0 策略分析结果，时间戳: {timestamp}")
        except StockInfo.DoesNotExist:
            logger.error(f"股票 {stock_code} 未找到，无法保存分析结果")
        except Exception as e:
            logger.error(f"保存 {stock_code} 的 T+0 策略分析结果时出错: {e}")
