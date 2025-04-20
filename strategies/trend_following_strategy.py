# 此策略侧重于识别和跟随趋势，主要使用 EMA 排列、DMI、SAR 等指标，并以 30 分钟级别为主要权重。
# trend_following_strategy.py
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, Any, List, Optional

# 假设 BaseStrategy 和常量在 .base 或 core.constants
from .base import BaseStrategy
from . import strategy_utils # 导入公共工具

logger = logging.getLogger("strategy_trend_following")

# --- 动态导入 pandas_ta ---
try:
    import pandas_ta as ta
    if not hasattr(ta, 'ema'):
        logger.warning("pandas_ta 的 EMA 功能不可用，趋势分析将受限。")
        ta = None # 如果核心功能缺失，则禁用
except ImportError:
    ta = None
    logger.warning("pandas_ta 库未安装或导入失败，趋势分析将受限。")

class TrendFollowingStrategy(BaseStrategy):
    """
    趋势跟踪策略：
    - 主要关注中长期趋势信号（如 EMA 排列, DMI, SAR）。
    - 以指定的时间框架（默认为 '30'）为主要权重。
    - 结合量能确认。
    """
    strategy_name = "TrendFollowingStrategy"
    focus_timeframe = '30' # 主要关注的时间框架

    def __init__(self, params_file: str = "strategies/indicator_parameters.json"):
        """初始化策略，加载参数"""
        self.params_file = params_file
        self.params = self._load_params()
        self.strategy_name = self.params.get('trend_following_strategy_name', self.strategy_name)
        self.focus_timeframe = self.params.get('trend_following_params', {}).get('focus_timeframe', self.focus_timeframe)
        self.intermediate_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[pd.DataFrame] = None

        if ta is None:
             logger.error(f"[{self.strategy_name}] pandas_ta 未加载或 EMA 不可用，无法计算趋势指标。")
             raise ImportError("pandas_ta with EMA is required for TrendFollowingStrategy.")

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
        if 'trend_following_params' not in self.params:
            logger.warning("参数中缺少 'trend_following_params' 部分，将使用默认值。")
        tf_params = self.params.get('trend_following_params', {})
        bs_params = self.params.get('base_scoring', {})
        ta_params = self.params.get('trend_analysis', {})

        if 'timeframes' not in bs_params or not isinstance(bs_params['timeframes'], list):
             raise ValueError("'base_scoring.timeframes' 必须是一个列表")
        if self.focus_timeframe not in bs_params['timeframes']:
             raise ValueError(f"'focus_timeframe' ({self.focus_timeframe}) 必须在 'base_scoring.timeframes' 中")
        if 'trend_indicators' not in tf_params or not isinstance(tf_params['trend_indicators'], list):
            logger.warning("'trend_following_params.trend_indicators' 未定义，将使用默认趋势指标 ['dmi', 'sar', 'macd']")
            tf_params['trend_indicators'] = ['dmi', 'sar', 'macd'] # 设置默认值
        if not ta_params.get('ema_periods') or not isinstance(ta_params['ema_periods'], list):
             raise ValueError("'trend_analysis.ema_periods' 必须是一个列表")
        if ta_params.get('long_term_ema_period') not in ta_params['ema_periods']:
             raise ValueError("'trend_analysis.long_term_ema_period' 必须在 'ema_periods' 列表中")

        logger.info(f"[{self.strategy_name}] 参数验证通过，主要关注时间框架: {self.focus_timeframe}")

    def get_required_columns(self) -> List[str]:
        """返回趋势跟踪策略所需的列"""
        required = set()
        bs_params = self.params['base_scoring']
        vc_params = self.params['volume_confirmation']
        ta_params = self.params['trend_analysis']
        tf_params = self.params.get('trend_following_params', {'trend_indicators': ['dmi', 'sar', 'macd']})
        timeframes = bs_params['timeframes']
        trend_indicators = tf_params['trend_indicators']

        # 趋势分析所需的 EMA 基础 (通常基于收盘价或基础分)
        # 基础分计算需要指标列
        for tf in timeframes:
             required.add(f'close_{tf}') # SAR/BOLL 等需要
             if 'macd' in trend_indicators:
                  required.update([f'MACD_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}',
                                   f'MACDh_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}',
                                   f'MACDs_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'])
             if 'dmi' in trend_indicators:
                  required.update([f'+DI_{bs_params["dmi_period"]}_{tf}', f'-DI_{bs_params["dmi_period"]}_{tf}', f'ADX_{bs_params["dmi_period"]}_{tf}'])
             if 'sar' in trend_indicators:
                  required.add(f'SAR_{tf}')
             # 其他可能用于基础分的指标 (即使非主要趋势指标)
             if 'rsi' in bs_params.get('score_indicators', []): required.add(f'RSI_{bs_params["rsi_period"]}_{tf}')
             if 'kdj' in bs_params.get('score_indicators', []): required.update([f'K_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{tf}', f'D_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{tf}', f'J_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{tf}'])
             # ... 添加 base_scoring 中定义的其他指标列 ...

        # 量能确认指标 (如果启用)
        if vc_params.get('enabled', False):
            vol_tf = vc_params.get('tf', self.focus_timeframe) # 默认使用焦点时间框架
            required.add(f'close_{vol_tf}')
            required.add(f'high_{vol_tf}')
            required.add(f'amount_{vol_tf}')
            required.add(f'AMT_MA_{vc_params["amount_ma_period"]}_{vol_tf}')
            required.add(f'CMF_{vc_params["cmf_period"]}_{vol_tf}')
            required.add(f'OBV_{vol_tf}')
            required.add(f'OBV_MA_{vc_params["obv_ma_period"]}_{vol_tf}')

        # 长期趋势分析所需的列 (如果 EMA 基于基础分计算，则不需要价格 EMA)
        # 在 generate_signals 中会计算分数 EMA

        return list(required)

    def _calculate_trend_focused_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算多时间框架加权的基础评分，重点关注趋势指标和焦点时间框架。
        """
        scores = pd.DataFrame(index=data.index)
        bs_params = self.params['base_scoring']
        tf_params = self.params.get('trend_following_params', {})
        timeframes = bs_params['timeframes']
        trend_indicators = tf_params.get('trend_indicators', ['dmi', 'sar', 'macd'])
        all_score_indicators = bs_params.get('score_indicators', trend_indicators) # 包含所有用于计算基础分的指标

        # 定义权重，焦点时间框架权重更高
        base_weight = (1.0 - tf_params.get('focus_weight', 0.5)) / (len(timeframes) - 1) if len(timeframes) > 1 else 0
        weights = {tf: base_weight for tf in timeframes if tf != self.focus_timeframe}
        weights[self.focus_timeframe] = tf_params.get('focus_weight', 0.5) # 默认焦点权重 0.5

        scores['total_weighted_score'] = 0.0
        for tf in timeframes:
            tf_score_sum = pd.Series(0.0, index=data.index)
            indicator_count_in_tf = 0
            close_price_col = f'close_{tf}'
            close_price = data.get(close_price_col, pd.Series(np.nan, index=data.index))

            # --- 使用 strategy_utils 计算各指标分数 ---
            if 'macd' in all_score_indicators:
                macd_col = f'MACD_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'
                macdh_col = f'MACDh_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'
                macds_col = f'MACDs_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'
                if all(c in data for c in [macd_col, macdh_col, macds_col]):
                    score = strategy_utils.calculate_macd_score(data[macd_col], data[macds_col], data[macdh_col])
                    tf_score_sum += score.fillna(50.0)
                    scores[f'macd_score_{tf}'] = score
                    indicator_count_in_tf += 1

            if 'rsi' in all_score_indicators:
                rsi_col = f'RSI_{bs_params["rsi_period"]}_{tf}'
                if rsi_col in data:
                     score = strategy_utils.calculate_rsi_score(data[rsi_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'rsi_score_{tf}'] = score
                     indicator_count_in_tf += 1

            if 'kdj' in all_score_indicators:
                 k_col = f'K_{bs_params["kdj_period_k"]}_{tf}'
                 d_col = f'D_{bs_params["kdj_period_k"]}_{tf}'
                 j_col = f'J_{bs_params["kdj_period_k"]}_{tf}'
                 if all(c in data for c in [k_col, d_col, j_col]):
                     score = strategy_utils.calculate_kdj_score(data[k_col], data[d_col], data[j_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'kdj_score_{tf}'] = score
                     indicator_count_in_tf += 1

            if 'boll' in all_score_indicators:
                 upper_col, mid_col, lower_col = f'BB_UPPER_{tf}', f'BB_MIDDLE_{tf}', f'BB_LOWER_{tf}'
                 if all(c in data for c in [upper_col, mid_col, lower_col]) and not close_price.isnull().all():
                     score = strategy_utils.calculate_boll_score(close_price, data[upper_col], data[mid_col], data[lower_col])
                     tf_score_sum += score.fillna(50.0)
                     scores[f'boll_score_{tf}'] = score
                     indicator_count_in_tf += 1

            if 'cci' in all_score_indicators:
                 cci_col = f'CCI_{bs_params["cci_period"]}_{tf}'
                 if cci_col in data:
                     score = strategy_utils.calculate_cci_score(data[cci_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'cci_score_{tf}'] = score
                     indicator_count_in_tf += 1

            if 'mfi' in all_score_indicators:
                 mfi_col = f'MFI_{bs_params["mfi_period"]}_{tf}'
                 if mfi_col in data:
                     score = strategy_utils.calculate_mfi_score(data[mfi_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'mfi_score_{tf}'] = score
                     indicator_count_in_tf += 1

            if 'roc' in all_score_indicators:
                 roc_col = f'ROC_{bs_params["roc_period"]}_{tf}'
                 if roc_col in data:
                     score = strategy_utils.calculate_roc_score(data[roc_col])
                     tf_score_sum += score.fillna(50.0)
                     scores[f'roc_score_{tf}'] = score
                     indicator_count_in_tf += 1

            if 'dmi' in all_score_indicators:
                 pdi_col, mdi_col, adx_col = f'+DI_{bs_params["dmi_period"]}_{tf}', f'-DI_{bs_params["dmi_period"]}_{tf}', f'ADX_{bs_params["dmi_period"]}_{tf}'
                 if all(c in data for c in [pdi_col, mdi_col, adx_col]):
                     score = strategy_utils.calculate_dmi_score(data[pdi_col], data[mdi_col], data[adx_col], bs_params)
                     tf_score_sum += score.fillna(50.0)
                     scores[f'dmi_score_{tf}'] = score
                     indicator_count_in_tf += 1

            if 'sar' in all_score_indicators:
                 sar_col = f'SAR_{tf}'
                 if sar_col in data and not close_price.isnull().all():
                     score = strategy_utils.calculate_sar_score(close_price, data[sar_col])
                     tf_score_sum += score.fillna(50.0)
                     scores[f'sar_score_{tf}'] = score
                     indicator_count_in_tf += 1

            # --- 计算加权平均分 ---
            if indicator_count_in_tf > 0:
                avg_tf_score = tf_score_sum / indicator_count_in_tf
                # 优先使用焦点时间框架的趋势指标进行加权
                if tf == self.focus_timeframe:
                     scores['total_weighted_score'] += avg_tf_score * weights[tf]
                else: # 非焦点时间框架权重较低
                     scores['total_weighted_score'] += avg_tf_score * weights[tf]
            else:
                scores['total_weighted_score'] += 50.0 * weights[tf] # 无指标贡献中性分

        scores['base_score_raw'] = scores['total_weighted_score'].clip(0, 100)
        return scores

    def _perform_trend_analysis(self, base_score_series: pd.Series) -> pd.DataFrame:
        """
        执行趋势分析 (EMA排列, 长期背景, 动量, 波动率)。
        :param base_score_series: 用于计算趋势的基础分数 Series (通常是量能调整后的)
        :return: 包含趋势分析结果的 DataFrame
        """
        analysis_df = pd.DataFrame(index=base_score_series.index)
        ta_params = self.params['trend_analysis']

        if base_score_series.isnull().all():
             logger.warning("基础分数全为 NaN，无法执行趋势分析。")
             return analysis_df # 返回空的分析结果

        score_series = base_score_series

        # 1. 计算 EMA
        all_ema_periods = ta_params['ema_periods']
        for period in all_ema_periods:
            try:
                analysis_df[f'ema_score_{period}'] = ta.ema(score_series, length=period)
            except Exception as e:
                logger.error(f"计算 EMA Score {period} 时出错: {e}")
                analysis_df[f'ema_score_{period}'] = np.nan

        # 2. 计算 EMA 排列信号 (例如 5, 13, 21, 55)
        ema_cols_align = [f'ema_score_{p}' for p in [5, 13, 21, 55] if p in all_ema_periods]
        if len(ema_cols_align) >= 2 and all(col in analysis_df for col in ema_cols_align):
             # 使用参数中定义的短期/中期/长期 EMA 进行比较
             short_ema_col = f'ema_score_{all_ema_periods[0]}'
             mid1_ema_col = f'ema_score_{all_ema_periods[1]}'
             mid2_ema_col = f'ema_score_{all_ema_periods[2]}'
             long_ema_col = f'ema_score_{all_ema_periods[3]}' # 假设至少有4个周期

             signal_s_m1 = np.where(analysis_df[short_ema_col] > analysis_df[mid1_ema_col], 1, np.where(analysis_df[short_ema_col] < analysis_df[mid1_ema_col], -1, 0))
             signal_m1_m2 = np.where(analysis_df[mid1_ema_col] > analysis_df[mid2_ema_col], 1, np.where(analysis_df[mid1_ema_col] < analysis_df[mid2_ema_col], -1, 0))
             signal_m2_l = np.where(analysis_df[mid2_ema_col] > analysis_df[long_ema_col], 1, np.where(analysis_df[mid2_ema_col] < analysis_df[long_ema_col], -1, 0))

             analysis_df['alignment_signal'] = signal_s_m1 + signal_m1_m2 + signal_m2_l
             # 处理 NaN
             analysis_df.loc[analysis_df[ema_cols_align].isna().any(axis=1), 'alignment_signal'] = np.nan
        else:
             analysis_df['alignment_signal'] = np.nan

        # 3. 计算排列反转信号 (可选，趋势策略可能不关注反转)
        # analysis_df['alignment_reversal'] = 0
        # ... (省略或弱化反转计算)

        # 4. 计算 EMA 强度 (例如 短期 vs 长期)
        short_ema_col = f'ema_score_{all_ema_periods[0]}'
        long_term_ema_period_for_strength = all_ema_periods[-1] # 使用最长 EMA 作为比较
        long_ema_col = f'ema_score_{long_term_ema_period_for_strength}'
        if short_ema_col in analysis_df and long_ema_col in analysis_df:
             analysis_df['ema_strength'] = analysis_df[short_ema_col] - analysis_df[long_ema_col]
        else:
             analysis_df['ema_strength'] = np.nan

        # 5. 计算得分动量
        analysis_df['score_momentum'] = score_series.diff()

        # 6. 计算得分波动率 (可选)
        # volatility_window = ta_params.get('volatility_window', 10)
        # analysis_df['score_volatility'] = score_series.rolling(window=volatility_window).std()

        # 7. 长期趋势背景 (基于分数与长期 EMA)
        long_term_ema_period_context = ta_params['long_term_ema_period']
        long_term_ema_col_context = f'ema_score_{long_term_ema_period_context}'
        if long_term_ema_col_context in analysis_df:
             analysis_df['long_term_context'] = np.where(
                 score_series > analysis_df[long_term_ema_col_context], 1, # 价格在长均线上方 - 看涨背景
                 np.where(score_series < analysis_df[long_term_ema_col_context], -1, 0) # 看跌背景
             )
             analysis_df.loc[score_series.isna() | analysis_df[long_term_ema_col_context].isna(), 'long_term_context'] = np.nan
        else:
             analysis_df['long_term_context'] = np.nan

        return analysis_df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成趋势跟踪信号"""
        logger.info(f"开始执行策略: {self.strategy_name} (Focus: {self.focus_timeframe})")
        if data is None or data.empty:
            logger.warning("输入数据为空，无法生成信号。")
            return pd.Series(dtype=float)
        # --- 检查必需列 ---
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns or data[col].isnull().all()]
        if missing_cols:
            logger.error(f"[{self.strategy_name}] 输入数据缺少必需列: {missing_cols}。策略无法运行。")
            return pd.Series(50.0, index=data.index) # 返回中性分
        # --- 步骤 1: 计算趋势导向的基础评分 ---
        base_scores_df = self._calculate_trend_focused_score(data)
        # --- 步骤 2: 应用量能调整 ---
        vc_params = self.params.get('volume_confirmation', {})
        dd_params = self.params.get('divergence_detection', {}) # 可能需要用于量价背离检查
        bs_params = self.params.get('base_scoring', {})
        base_score_adjusted = strategy_utils.adjust_score_with_volume(
            base_scores_df['base_score_raw'], data, vc_params, dd_params, bs_params
        )
        base_scores_df['base_score_volume_adjusted'] = base_score_adjusted
        # --- 步骤 3: 执行趋势分析 ---
        # 使用量能调整后的分数进行趋势分析，更能反映带量趋势
        trend_analysis_df = self._perform_trend_analysis(base_scores_df['base_score_volume_adjusted'])

        # --- 步骤 4: 组合最终信号 ---
        # 趋势策略的最终信号主要基于趋势分析结果和基础分
        final_signal = pd.Series(50.0, index=data.index) # 从中性分开始
        tf_params = self.params.get('trend_following_params', {})
        signal_weights = tf_params.get('signal_weights', {
            'base_score': 0.4,
            'alignment': 0.3,
            'long_context': 0.2,
            'momentum': 0.1
        })
        # 基础分贡献 (量能调整后)
        score_contribution = (base_scores_df['base_score_volume_adjusted'] - 50) * signal_weights.get('base_score', 0.4)

        # EMA 排列贡献 (-3 到 +3 范围 -> 映射到 -50 到 +50 范围?)
        alignment_signal = trend_analysis_df.get('alignment_signal', pd.Series(0, index=data.index)).fillna(0)
        # 简单线性映射: 3 -> +15, -3 -> -15 (假设权重 0.3, 范围 50*0.3=15)
        alignment_contribution = alignment_signal * (50 * signal_weights.get('alignment', 0.3) / 3)

        # 长期趋势背景贡献 (-1, 0, 1 -> -10, 0, 10?)
        long_context = trend_analysis_df.get('long_term_context', pd.Series(0, index=data.index)).fillna(0)
        context_contribution = long_context * (50 * signal_weights.get('long_context', 0.2))

        # 动量贡献 (正/负 -> 轻微加/减分)
        momentum = trend_analysis_df.get('score_momentum', pd.Series(0, index=data.index)).fillna(0)
        momentum_contribution = np.sign(momentum) * (50 * signal_weights.get('momentum', 0.1)) # 只取方向

        final_signal = 50.0 + score_contribution + alignment_contribution + context_contribution + momentum_contribution
        final_signal = final_signal.clip(0, 100).round(2)

        # --- 存储中间数据 ---
        self.intermediate_data = pd.concat([
            base_scores_df,
            trend_analysis_df,
            pd.DataFrame({'final_signal': final_signal}, index=data.index)
        ], axis=1)

        logger.info(f"{self.strategy_name}: 信号生成完毕。")
        self.analyze_signals() # 执行分析
        return final_signal

    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回中间计算结果"""
        return self.intermediate_data

    def analyze_signals(self) -> Optional[pd.DataFrame]:
        """分析趋势策略信号"""
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning("中间数据为空，无法进行信号分析。")
            return None

        analysis_results = {}
        data = self.intermediate_data
        latest_data = data.iloc[-1] if not data.empty else None

        # --- 统计分析 ---
        if 'final_signal' in data:
            final_signal = data['final_signal'].dropna()
            if not final_signal.empty:
                 analysis_results['final_signal_mean'] = final_signal.mean()
                 analysis_results['final_signal_bullish_ratio'] = (final_signal > 55).mean() # 趋势偏多
                 analysis_results['final_signal_bearish_ratio'] = (final_signal < 45).mean() # 趋势偏空
                 analysis_results['final_signal_strong_bullish_ratio'] = (final_signal >= 70).mean() # 强趋势
                 analysis_results['final_signal_strong_bearish_ratio'] = (final_signal <= 30).mean() # 强趋势

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

        # --- 最新信号判断 ---
        signal_judgment = {}
        if latest_data is not None:
             if 'final_signal' in latest_data:
                  score = latest_data['final_signal']
                  if score >= 70: signal_judgment['trend_strength'] = "强劲上升趋势"
                  elif score >= 55: signal_judgment['trend_strength'] = "温和上升趋势"
                  elif score <= 30: signal_judgment['trend_strength'] = "强劲下降趋势"
                  elif score <= 45: signal_judgment['trend_strength'] = "温和下降趋势"
                  else: signal_judgment['trend_strength'] = "趋势不明/震荡"

             if 'alignment_signal' in latest_data:
                 alignment = latest_data['alignment_signal']
                 if alignment == 3: signal_judgment['alignment_status'] = "完全多头排列"
                 elif alignment > 0: signal_judgment['alignment_status'] = "部分多头排列"
                 elif alignment == -3: signal_judgment['alignment_status'] = "完全空头排列"
                 elif alignment < 0: signal_judgment['alignment_status'] = "部分空头排列"
                 else: signal_judgment['alignment_status'] = "排列混乱"

             if 'long_term_context' in latest_data:
                 context = latest_data['long_term_context']
                 if context == 1: signal_judgment['long_term_view'] = "长期看涨背景"
                 elif context == -1: signal_judgment['long_term_view'] = "长期看跌背景"
                 else: signal_judgment['long_term_view'] = "长期趋势不明"

             # 中文解读
             chinese_interpretation = (
                 f"【趋势跟踪策略分析 - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}】\n"
                 f"核心观点:\n"
                 f" - 当前趋势强度: {signal_judgment.get('trend_strength', '未知')}\n"
                 f" - EMA 排列状态: {signal_judgment.get('alignment_status', '未知')}\n"
                 f" - 长期趋势背景: {signal_judgment.get('long_term_view', '未知')}\n"
                 f"操作建议:\n"
                 f" - {signal_judgment.get('trend_strength', '未知')} 时，通常建议顺应趋势方向操作。\n"
                 f" - 结合 {signal_judgment.get('alignment_status', '未知')} 和 {signal_judgment.get('long_term_view', '未知')} 评估趋势的稳固性。\n"
                 f"统计数据:\n"
                 f" - 最终信号平均值: {analysis_results.get('final_signal_mean', np.nan):.2f}\n"
                 f" - 强看涨信号比例: {analysis_results.get('final_signal_strong_bullish_ratio', np.nan)*100:.2f}%\n"
                 f" - 强看跌信号比例: {analysis_results.get('final_signal_strong_bearish_ratio', np.nan)*100:.2f}%\n"
                 f" - 完全多头排列比例: {analysis_results.get('alignment_fully_bullish_ratio', np.nan)*100:.2f}%\n"
                 f" - 完全空头排列比例: {analysis_results.get('alignment_fully_bearish_ratio', np.nan)*100:.2f}%\n"
                 f" - 长期看涨背景比例: {analysis_results.get('long_term_bullish_ratio', np.nan)*100:.2f}%\n"
                 f" - 长期看跌背景比例: {analysis_results.get('long_term_bearish_ratio', np.nan)*100:.2f}%"
             )
            #  logger.info(chinese_interpretation)
            #  print(chinese_interpretation)

        analysis_results.update(signal_judgment)
        self.analysis_results = pd.DataFrame([analysis_results])
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
                    'score': convert_nan_to_none(intermediate_data.get('final_signal', None)),
                    'base_score_raw': convert_nan_to_none(intermediate_data.get('base_score_raw', None)),
                    'base_score_volume_adjusted': convert_nan_to_none(intermediate_data.get('base_score_volume_adjusted', None)),
                    'alignment_signal': convert_nan_to_none(intermediate_data.get('alignment_signal', None)),
                    'long_term_context': convert_nan_to_none(intermediate_data.get('long_term_context', None)),
                    'ema_score_5': convert_nan_to_none(intermediate_data.get('ema_score_5', None)),
                    'ema_score_13': convert_nan_to_none(intermediate_data.get('ema_score_13', None)),
                    'ema_score_21': convert_nan_to_none(intermediate_data.get('ema_score_21', None)),
                    'ema_score_55': convert_nan_to_none(intermediate_data.get('ema_score_55', None)),
                    'ema_score_233': convert_nan_to_none(intermediate_data.get('ema_score_233', None)),
                    'ema_strength': convert_nan_to_none(intermediate_data.get('ema_strength', None)),
                    'score_momentum': convert_nan_to_none(intermediate_data.get('score_momentum', None)),
                    'close_price': convert_nan_to_none(latest_data.get(f'close_{self.focus_timeframe}', None)),
                    'final_signal_mean': convert_nan_to_none(analysis_data.get('final_signal_mean', None)),
                    'final_signal_bullish_ratio': convert_nan_to_none(analysis_data.get('final_signal_bullish_ratio', None)),
                    'final_signal_bearish_ratio': convert_nan_to_none(analysis_data.get('final_signal_bearish_ratio', None)),
                    'final_signal_strong_bullish_ratio': convert_nan_to_none(analysis_data.get('final_signal_strong_bullish_ratio', None)),
                    'final_signal_strong_bearish_ratio': convert_nan_to_none(analysis_data.get('final_signal_strong_bearish_ratio', None)),
                    'alignment_fully_bullish_ratio': convert_nan_to_none(analysis_data.get('alignment_fully_bullish_ratio', None)),
                    'alignment_fully_bearish_ratio': convert_nan_to_none(analysis_data.get('alignment_fully_bearish_ratio', None)),
                    'alignment_bullish_ratio': convert_nan_to_none(analysis_data.get('alignment_bullish_ratio', None)),
                    'alignment_bearish_ratio': convert_nan_to_none(analysis_data.get('alignment_bearish_ratio', None)),
                    'long_term_bullish_ratio': convert_nan_to_none(analysis_data.get('long_term_bullish_ratio', None)),
                    'long_term_bearish_ratio': convert_nan_to_none(analysis_data.get('long_term_bearish_ratio', None)),
                    'params_snapshot': self.params,
                }
            )
            logger.info(f"成功保存 {stock_code} 的趋势跟踪策略分析结果，时间戳: {timestamp}")
        except StockInfo.DoesNotExist:
            logger.error(f"股票 {stock_code} 未找到，无法保存分析结果")
        except Exception as e:
            logger.error(f"保存 {stock_code} 的趋势跟踪策略分析结果时出错: {e}")

