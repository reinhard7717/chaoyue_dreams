# 此策略侧重于识别潜在的趋势转折点，主要使用振荡器（RSI, KDJ, CCI, MFI）、布林带、背离信号和 K 线形态，并以 15 分钟级别为主要权重。
# trend_reversal_strategy.py
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
from . import strategy_utils # 导入公共工具

logger = logging.getLogger("strategy_trend_reversal")

class TrendReversalStrategy(BaseStrategy):
    """
    趋势反转策略：
    - 主要关注超买超卖信号 (RSI, KDJ, CCI, MFI)、布林带边界、背离、K线反转形态。
    - 以指定的时间框架（默认为 '30'）为主要权重。
    - 结合量能、波动率、动量等多维度确认或背离。
    """
    strategy_name = "TrendReversalStrategy"
    focus_timeframe = '30'  # 主要关注的时间框架，参数中调整为30分钟

    def __init__(self, params_file: str = "strategies/indicator_parameters.json"):
        """初始化策略，加载参数"""
        self.params_file = params_file
        self.params = self._load_params()
        self.strategy_name = self.params.get('trend_reversal_strategy_name', self.strategy_name)
        self.focus_timeframe = self.params.get('trend_reversal_params', {}).get('focus_timeframe', self.focus_timeframe)
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
        if 'trend_reversal_params' not in self.params:
            logger.warning("参数中缺少 'trend_reversal_params' 部分，将使用默认值。")
        tr_params = self.params.get('trend_reversal_params', {})
        bs_params = self.params.get('base_scoring', {})
        dd_params = self.params.get('divergence_detection', {})
        kpd_params = self.params.get('kline_pattern_detection', {})
        ia_params = self.params.get('indicator_analysis_params', {}) # 用于 OBOS 反转参数

        if 'timeframes' not in bs_params or not isinstance(bs_params['timeframes'], list):
             raise ValueError("'base_scoring.timeframes' 必须是一个列表")
        if self.focus_timeframe not in bs_params['timeframes']:
             raise ValueError(f"'focus_timeframe' ({self.focus_timeframe}) 必须在 'base_scoring.timeframes' 中")
        if 'reversal_indicators' not in tr_params or not isinstance(tr_params['reversal_indicators'], list):
            logger.warning("'trend_reversal_params.reversal_indicators' 未定义，将使用默认反转指标 ['rsi', 'kdj', 'cci', 'boll']")
            tr_params['reversal_indicators'] = ['rsi', 'kdj', 'cci', 'boll'] # 设置默认值

        # 检查背离和 K 线参数（如果启用）
        # if dd_params.get('enabled', False) and (dd_params.get('tf') != self.focus_timeframe):
        #      logger.warning(f"背离检测时间框 ({dd_params.get('tf')}) 与反转策略焦点时间框 ({self.focus_timeframe}) 不同")
        # if kpd_params.get('enabled', False) and (kpd_params.get('tf') != self.focus_timeframe):
        #      logger.warning(f"K线形态检测时间框 ({kpd_params.get('tf')}) 与反转策略焦点时间框 ({self.focus_timeframe}) 不同")

        # logger.info(f"[{self.strategy_name}] 参数验证通过，主要关注时间框架: {self.focus_timeframe}")

    def get_required_columns(self) -> List[str]:
        """
        返回趋势反转策略所需的列名列表。
        通过分析策略参数，动态生成所有必要的技术指标和数据列，确保不重复添加列名。
        
        Returns:
            List[str]: 策略所需的唯一列名列表
        """
        logger.debug("开始生成趋势反转策略所需的列名")
        required = set()  # 使用集合避免重复列名
        bs_params = self.params['base_scoring']
        vc_params = self.params['volume_confirmation']
        dd_params = self.params['divergence_detection']
        kpd_params = self.params['kline_pattern_detection']
        ia_params = self.params['indicator_analysis_params']
        tr_params = self.params.get('trend_reversal_params', {'reversal_indicators': ['rsi', 'kdj', 'cci', 'boll']})
        
        timeframes = bs_params['timeframes']
        reversal_indicators = tr_params['reversal_indicators']
        all_score_indicators = bs_params.get('score_indicators', reversal_indicators)
        
        logger.debug(f"时间框架: {timeframes}, 评分指标: {all_score_indicators}")

        # 辅助函数：为特定时间框架添加指标列
        def add_indicator_columns(tf: str, indicators: List[str], params: Dict[str, Any], target_set: set):
            """为指定时间框架添加技术指标列到目标集合中"""
            target_set.add(f'close_{tf}')
            if 'rsi' in indicators:
                target_set.add(f'RSI_{params["rsi_period"]}_{tf}')
            if 'kdj' in indicators:
                target_set.update([
                    f'K_{params["kdj_period_k"]}_{params["kdj_period_d"]}_{tf}',
                    f'D_{params["kdj_period_k"]}_{params["kdj_period_d"]}_{tf}',
                    f'J_{params["kdj_period_k"]}_{params["kdj_period_d"]}_{tf}'
                ])
            if 'boll' in indicators:
                target_set.update([
                    f'BB_UPPER_{params["boll_period"]}_{tf}',
                    f'BB_MIDDLE_{params["boll_period"]}_{tf}',
                    f'BB_LOWER_{params["boll_period"]}_{tf}'
                ])
            if 'cci' in indicators:
                target_set.add(f'CCI_{params["cci_period"]}_{tf}')
            if 'mfi' in indicators:
                target_set.add(f'MFI_{params["mfi_period"]}_{tf}')
            if 'macd' in indicators:
                target_set.update([
                    f'MACD_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}_{tf}',
                    f'MACDh_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}_{tf}',
                    f'MACDs_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}_{tf}'
                ])
            if 'roc' in indicators:
                target_set.add(f'ROC_{params["roc_period"]}_{tf}')
            if 'dmi' in indicators:
                target_set.update([
                    f'+DI_{params["dmi_period"]}_{tf}',
                    f'-DI_{params["dmi_period"]}_{tf}',
                    f'ADX_{params["dmi_period"]}_{tf}'
                ])
            if 'sar' in indicators:
                target_set.add(f'SAR_{tf}')

        # 1. 基础评分指标列（所有时间框架）
        for tf in timeframes:
            add_indicator_columns(tf, all_score_indicators, bs_params, required)

        # 2. 量能确认指标（如果启用）
        if vc_params.get('enabled', False):
            vol_tf = vc_params.get('tf', self.focus_timeframe)
            required.update([f'close_{vol_tf}', f'high_{vol_tf}', f'amount_{vol_tf}'])
            required.add(f'AMT_MA_{vc_params["amount_ma_period"]}_{vol_tf}')
            required.add(f'CMF_{vc_params["cmf_period"]}_{vol_tf}')
            required.add(f'OBV_{vol_tf}')
            required.add(f'OBV_MA_{vc_params["obv_ma_period"]}_{vol_tf}')
            if dd_params.get('enabled', False) and dd_params.get('indicators', {}).get('mfi'):
                required.add(f'MFI_{bs_params["mfi_period"]}_{vol_tf}')

        # 3. 背离检测所需指标（焦点时间框架）
        if dd_params.get('enabled', False):
            div_tf = dd_params.get('tf', self.focus_timeframe)
            required.add(f'close_{div_tf}')
            indicators = dd_params.get('indicators', {})
            if indicators.get('macd_hist'):
                required.add(f'MACDh_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{div_tf}')
            if indicators.get('rsi'):
                required.add(f'RSI_{bs_params["rsi_period"]}_{div_tf}')
            if indicators.get('mfi'):
                required.add(f'MFI_{bs_params["mfi_period"]}_{div_tf}')
            if indicators.get('obv'):
                required.add(f'OBV_{div_tf}')

        # 4. K线形态检测所需列（指定时间框架）
        if kpd_params.get('enabled', False):
            kline_tf = kpd_params.get('tf', self.focus_timeframe)
            required.update([f'open_{kline_tf}', f'high_{kline_tf}', f'low_{kline_tf}', f'close_{kline_tf}'])

        # 5. 指标超买超卖反转信号所需列（焦点时间框架）
        analysis_tf = self.focus_timeframe
        required.update([
            f'close_{analysis_tf}',
            f'RSI_{bs_params["rsi_period"]}_{analysis_tf}',
            f'STOCH_K_{ia_params["stoch_k"]}_{analysis_tf}',
            f'CCI_{bs_params["cci_period"]}_{analysis_tf}',
            f'BB_UPPER_{bs_params["boll_period"]}_{analysis_tf}',
            f'BB_LOWER_{bs_params["boll_period"]}_{analysis_tf}',
            f'volume_{analysis_tf}',
            f'VOL_MA_{ia_params["volume_ma_period"]}_{analysis_tf}'
        ])

        # 6. 长期趋势过滤所需列
        if tr_params.get('long_term_trend_filter', {}).get('enabled', False):
            tf_long = tr_params['long_term_trend_filter'].get('timeframe', 'D')
            required.update([f'close_{tf_long}', f'ADX_{bs_params["dmi_period"]}_{tf_long}'])

        # 7. 波动率过滤 (ATR) 所需列
        if tr_params.get('volatility_filter', {}).get('enabled', False):
            vol_tf = tr_params['volatility_filter'].get('timeframe', self.focus_timeframe)
            required.update([f'high_{vol_tf}', f'low_{vol_tf}', f'close_{vol_tf}'])

        # 8. 动量反转 (Williams %R) 所需列
        if tr_params.get('momentum_reversal', {}).get('enabled', False):
            mom_tf = tr_params['momentum_reversal'].get('timeframe', self.focus_timeframe)
            required.update([f'high_{mom_tf}', f'low_{mom_tf}', f'close_{mom_tf}'])

        # 9. 历史波动率 (HV) 所需列
        if tr_params.get('historical_volatility', {}).get('enabled', False):
            hv_tf = tr_params['historical_volatility'].get('timeframe', self.focus_timeframe)
            required.add(f'close_{hv_tf}')

        # 10. 换手率过滤所需列
        if tr_params.get('turnover_filter', {}).get('enabled', False):
            turnover_tf = tr_params['turnover_filter'].get('timeframe', self.focus_timeframe)
            required.add(f'turnover_rate_{turnover_tf}')

        required_list = list(required)
        logger.debug(f"生成必需列名完成，共 {len(required_list)} 个列")
        return required_list
    
    def _calculate_reversal_focused_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算多时间框架加权的基础评分，重点关注反转指标和焦点时间框架。
        """
        scores = pd.DataFrame(index=data.index)
        bs_params = self.params['base_scoring']
        tr_params = self.params.get('trend_reversal_params', {})
        timeframes = bs_params['timeframes']
        reversal_indicators = tr_params.get('reversal_indicators', ['rsi', 'kdj', 'cci', 'boll'])
        all_score_indicators = bs_params.get('score_indicators', reversal_indicators)

        # 定义权重，焦点时间框架权重更高
        base_weight = (1.0 - tr_params.get('focus_weight', 0.5)) / (len(timeframes) - 1) if len(timeframes) > 1 else 0
        weights = {tf: base_weight for tf in timeframes if tf != self.focus_timeframe}
        weights[self.focus_timeframe] = tr_params.get('focus_weight', 0.5)

        scores['total_weighted_score'] = 0.0
        for tf in timeframes:
            tf_score_sum = pd.Series(0.0, index=data.index)
            indicator_count_in_tf = 0
            close_price_col = f'close_{tf}'
            close_price = data.get(close_price_col, pd.Series(np.nan, index=data.index))

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

            # 计算加权平均分
            if indicator_count_in_tf > 0:
                avg_tf_score = tf_score_sum / indicator_count_in_tf
                scores['total_weighted_score'] += avg_tf_score * weights[tf]
            else:
                scores['total_weighted_score'] += 50.0 * weights[tf]

        scores['base_score_raw'] = scores['total_weighted_score'].clip(0, 100)
        return scores
    
    def _perform_signal_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行反转相关的信号分析 (背离, K线, 指标反转, 波动率, 动量)。
        :param data: 包含基础数据和指标的 DataFrame
        :return: 包含信号分析结果的 DataFrame
        """
        analysis_df = pd.DataFrame(index=data.index)
        dd_params = self.params['divergence_detection']
        kpd_params = self.params['kline_pattern_detection']
        bs_params = self.params['base_scoring']
        ia_params = self.params['indicator_analysis_params']
        tr_params = self.params.get('trend_reversal_params', {})
        # 更新信号权重
        signal_weights = tr_params.get('signal_weights', {
            'regular_div': 0.3,
            'hidden_div': 0.05,
            'obos_reversal': 0.18,
            'bb_reversal': 0.12,
            'kline_reversal_strong': 0.22,
            'kline_reversal_weak': 0.08,
            'kline_continuation': -0.05,
            'kline_marubozu': -0.1,
            'kline_doji': 0.0,
            'volume_spike': 0.08,
            'willr_reversal': 0.1,
            'atr_volatility': 0.07,
            'hv_environment': 0.05,
            'max_abs_signal': 1.0
        })
        analysis_tf = self.focus_timeframe  # 主要分析焦点时间框架的信号
        # 1. 背离检测
        if dd_params.get('enabled', False):
            div_tf = dd_params.get('tf', analysis_tf)
            price_col = f'close_{div_tf}'
            if price_col not in data.columns or data[price_col].isnull().all():
                logger.warning(f"缺少价格列 {price_col}，无法进行背离检测。")
                for indicator_key in dd_params.get('indicators', {}).keys():
                    analysis_df[f'{indicator_key}_divergence'] = 0
            else:
                # 调用 detect_divergence 函数，传递正确的参数
                divergence_signals = strategy_utils.detect_divergence(
                    data=data,
                    dd_params=dd_params,
                    bs_params=bs_params,
                    vc_params=self.params.get('volume_confirmation', {})
                )
                # 将背离信号合并到 analysis_df 中
                for indicator_key in dd_params.get('indicators', {}).keys():
                    if dd_params['indicators'].get(indicator_key):
                        analysis_df[f'{indicator_key}_divergence'] = 0
                        for div_type in ['regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish']:
                            col_name = f'div_{indicator_key}_{div_type}'
                            if col_name in divergence_signals.columns:
                                # 合并背离信号，优先使用常规背离信号
                                if div_type in ['regular_bullish', 'regular_bearish']:
                                    analysis_df.loc[divergence_signals[col_name] != 0, f'{indicator_key}_divergence'] = divergence_signals[col_name]
                                elif analysis_df[f'{indicator_key}_divergence'].eq(0).all() and div_type in ['hidden_bullish', 'hidden_bearish']:
                                    analysis_df.loc[divergence_signals[col_name] != 0, f'{indicator_key}_divergence'] = divergence_signals[col_name]
                    else:
                        analysis_df[f'{indicator_key}_divergence'] = 0
        else:
            for indicator_key in dd_params.get('indicators', {}).keys():
                analysis_df[f'{indicator_key}_divergence'] = 0
        # 2. K 线形态检测
        if kpd_params.get('enabled', False):
            kline_tf = kpd_params.get('tf', analysis_tf)
            ohlc_cols = [f'open_{kline_tf}', f'high_{kline_tf}', f'low_{kline_tf}', f'close_{kline_tf}']
            if all(col in data for col in ohlc_cols):
                temp_df = data[ohlc_cols].rename(columns={
                    f'open_{kline_tf}': 'open', f'high_{kline_tf}': 'high',
                    f'low_{kline_tf}': 'low', f'close_{kline_tf}': 'close'
                })
                analysis_df['kline_pattern'] = strategy_utils.detect_kline_patterns(temp_df)
            else:
                logger.warning(f"缺少 OHLC 列 for tf={kline_tf}，无法进行 K 线形态检测。")
                analysis_df['kline_pattern'] = 0
        else:
            analysis_df['kline_pattern'] = 0
        # 3. 技术指标反转信号 (OB/OS, BB, 新增 Williams %R)
        analysis_df['rsi_obos_reversal'] = 0
        analysis_df['stoch_obos_reversal'] = 0
        analysis_df['cci_obos_reversal'] = 0
        analysis_df['bb_reversal'] = 0
        analysis_df['volume_spike'] = 0
        analysis_df['willr_reversal'] = 0  # 新增 Williams %R 反转信号
        price_col = f'close_{analysis_tf}'
        if price_col not in data or data[price_col].isnull().all():
            logger.warning(f"缺少价格列 {price_col}，无法计算指标反转信号。")
        else:
            close_series = data[price_col]
            # RSI Reversal
            rsi_col = f'RSI_{bs_params["rsi_period"]}_{analysis_tf}'
            if rsi_col in data and data[rsi_col].notna().any():
                rsi = data[rsi_col]; ob, os = bs_params.get('rsi_overbought', 70), bs_params.get('rsi_oversold', 30)
                ext_ob, ext_os = bs_params.get('rsi_extreme_overbought', 80), bs_params.get('rsi_extreme_oversold', 20)
                sell_cond = ((rsi.shift(1) > ob) & (rsi <= ob)) | ((rsi.shift(1) > ext_ob) & (rsi <= ext_ob))
                buy_cond = ((rsi.shift(1) < os) & (rsi >= os)) | ((rsi.shift(1) < ext_os) & (rsi >= ext_os))
                analysis_df.loc[sell_cond, 'rsi_obos_reversal'] = -1
                analysis_df.loc[buy_cond, 'rsi_obos_reversal'] = 1
            # Stoch Reversal
            stoch_k_col = f'STOCH_K_{ia_params["stoch_k"]}_{analysis_tf}'
            if stoch_k_col in data and data[stoch_k_col].notna().any():
                stoch_k = data[stoch_k_col]; ob, os = ia_params.get('stoch_ob', 80), ia_params.get('stoch_os', 20)
                sell_cond = (stoch_k.shift(1) > ob) & (stoch_k <= ob)
                buy_cond = (stoch_k.shift(1) < os) & (stoch_k >= os)
                analysis_df.loc[sell_cond, 'stoch_obos_reversal'] = -1
                analysis_df.loc[buy_cond, 'stoch_obos_reversal'] = 1
            # CCI Reversal
            cci_col = f'CCI_{bs_params["cci_period"]}_{analysis_tf}'
            if cci_col in data and data[cci_col].notna().any():
                cci = data[cci_col]; ob, os = bs_params.get('cci_threshold', 100), -bs_params.get('cci_threshold', 100)
                ext_ob, ext_os = bs_params.get('cci_extreme_threshold', 200), -bs_params.get('cci_extreme_threshold', 200)
                sell_cond = ((cci.shift(1) > ob) & (cci <= ob)) | ((cci.shift(1) > ext_ob) & (cci <= ext_ob))
                buy_cond = ((cci.shift(1) < os) & (cci >= os)) | ((cci.shift(1) < ext_os) & (cci >= ext_os))
                analysis_df.loc[sell_cond, 'cci_obos_reversal'] = -1
                analysis_df.loc[buy_cond, 'cci_obos_reversal'] = 1
            # Bollinger Band Reversal - 新增前瞻性条件
            bb_upper_col, bb_lower_col = f'BB_UPPER_{analysis_tf}', f'BB_LOWER_{analysis_tf}'
            anticipation_params = tr_params.get('signal_anticipation', {})
            if bb_upper_col in data and bb_lower_col in data and data[bb_upper_col].notna().any() and data[bb_lower_col].notna().any():
                upper, lower = data[bb_upper_col], data[bb_lower_col]
                sell_cond = (close_series.shift(1) > upper.shift(1)) & (close_series <= upper)
                buy_cond = (close_series.shift(1) < lower.shift(1)) & (close_series >= lower)
                if anticipation_params.get('enabled', False):
                    proximity_factor = anticipation_params.get('bb_proximity_factor', 0.95)
                    upper_proximity = upper * proximity_factor
                    lower_proximity = lower * (2 - proximity_factor)
                    sell_pre_cond = (close_series > upper_proximity) & (close_series < upper) & (close_series < close_series.shift(1))
                    buy_pre_cond = (close_series < lower_proximity) & (close_series > lower) & (close_series > close_series.shift(1))
                    analysis_df.loc[sell_cond, 'bb_reversal'] = -1
                    analysis_df.loc[buy_cond, 'bb_reversal'] = 1
                    analysis_df.loc[sell_pre_cond & ~sell_cond, 'bb_reversal'] = -0.5
                    analysis_df.loc[buy_pre_cond & ~buy_cond, 'bb_reversal'] = 0.5
                    logger.info(f"应用布林带前瞻性信号检测，接近因子: {proximity_factor}")
                else:
                    analysis_df.loc[sell_cond, 'bb_reversal'] = -1
                    analysis_df.loc[buy_cond, 'bb_reversal'] = 1
            # Volume Spike (放量可能确认反转)
            volume_col = f'volume_{analysis_tf}'
            vol_ma_col = f'VOL_MA_{ia_params["volume_ma_period"]}_{analysis_tf}'
            if volume_col in data and vol_ma_col in data and data[volume_col].notna().any() and data[vol_ma_col].notna().any():
                vol, vol_ma = data[volume_col], data[vol_ma_col]
                vol_spike_factor = ia_params.get('volume_spike_factor', 2.0)
                analysis_df.loc[vol > vol_ma * vol_spike_factor, 'volume_spike'] = 1
            # Williams %R Reversal
            momentum_reversal = tr_params.get('momentum_reversal', {})
            if momentum_reversal.get('enabled', False):
                mom_tf = momentum_reversal.get('timeframe', analysis_tf)
                high_col, low_col, close_col = f'high_{mom_tf}', f'low_{mom_tf}', f'close_{mom_tf}'
                if all(col in data for col in [high_col, low_col, close_col]) and data[high_col].notna().any():
                    period = momentum_reversal.get('period', 14)
                    # 计算 Williams %R
                    highest_high = data[high_col].rolling(window=period).max()
                    lowest_low = data[low_col].rolling(window=period).min()
                    willr = -100 * (highest_high - data[close_col]) / (highest_high - lowest_low + 1e-10)
                    ob, os = momentum_reversal.get('overbought', -20), momentum_reversal.get('oversold', -80)
                    ext_ob, ext_os = momentum_reversal.get('extreme_overbought', -10), momentum_reversal.get('extreme_oversold', -90)
                    sell_cond = ((willr.shift(1) > ob) & (willr <= ob)) | ((willr.shift(1) > ext_ob) & (willr <= ext_ob))
                    buy_cond = ((willr.shift(1) < os) & (willr >= os)) | ((willr.shift(1) < ext_os) & (willr >= ext_os))
                    analysis_df.loc[sell_cond, 'willr_reversal'] = -1
                    analysis_df.loc[buy_cond, 'willr_reversal'] = 1
                    logger.info(f"应用 Williams %R 反转信号检测，周期: {period}")
        # 波动率过滤 (ATR)
        analysis_df['atr_volatility_signal'] = 0
        volatility_filter = tr_params.get('volatility_filter', {})
        if volatility_filter.get('enabled', False):
            vol_tf = volatility_filter.get('timeframe', analysis_tf)
            high_col, low_col, close_col = f'high_{vol_tf}', f'low_{vol_tf}', f'close_{vol_tf}'
            if all(col in data for col in [high_col, low_col, close_col]) and data[high_col].notna().any():
                period = volatility_filter.get('period', 14)
                # 计算 ATR
                tr = np.maximum(np.maximum(data[high_col] - data[low_col], 
                                           data[high_col] - data[close_col].shift(1).fillna(data[close_col])),
                               data[close_col].shift(1).fillna(data[close_col]) - data[low_col])
                atr = tr.rolling(window=period).mean()
                atr_ma = atr.rolling(window=period).mean()
                high_vol_threshold = volatility_filter.get('high_vol_threshold', 1.5)
                low_vol_threshold = volatility_filter.get('low_vol_threshold', 0.5)
                high_vol_cond = atr > atr_ma * high_vol_threshold
                low_vol_cond = atr < atr_ma * low_vol_threshold
                analysis_df.loc[high_vol_cond, 'atr_volatility_signal'] = 1  # 高波动率增强信号
                analysis_df.loc[low_vol_cond, 'atr_volatility_signal'] = -1  # 低波动率削弱信号
                logger.info(f"应用 ATR 波动率过滤，周期: {period}")
        # 历史波动率 (HV) 环境过滤
        analysis_df['hv_environment_signal'] = 0
        hv_filter = tr_params.get('historical_volatility', {})
        if hv_filter.get('enabled', False):
            hv_tf = hv_filter.get('timeframe', analysis_tf)
            close_col = f'close_{hv_tf}'
            if close_col in data and data[close_col].notna().any():
                period = hv_filter.get('period', 20)
                # 计算历史波动率 (基于收盘价的对数收益率标准差)
                log_returns = np.log(data[close_col] / data[close_col].shift(1))
                hv = log_returns.rolling(window=period).std() * np.sqrt(252)  # 年化波动率
                hv_ma = hv.rolling(window=period).mean()
                high_hv_threshold = hv_filter.get('high_hv_threshold', 1.5)
                low_hv_threshold = hv_filter.get('low_hv_threshold', 0.5)
                high_hv_cond = hv > hv_ma * high_hv_threshold
                low_hv_cond = hv < hv_ma * low_hv_threshold
                analysis_df.loc[high_hv_cond, 'hv_environment_signal'] = 1  # 高波动率环境增强信号
                analysis_df.loc[low_hv_cond, 'hv_environment_signal'] = -1  # 低波动率环境削弱信号
                logger.info(f"应用历史波动率环境过滤，周期: {period}")
        # 4. 计算加权反转确认信号
        analysis_df['reversal_confirmation_signal'] = 0.0
        def get_signal_value(col_name):
            if col_name in analysis_df and analysis_df[col_name].notna().any():
                return analysis_df[col_name].fillna(0)
            else:
                return pd.Series(0.0, index=analysis_df.index)
        # 背离信号贡献
        if dd_params.get('enabled', False):
            for indicator_key in dd_params.get('indicators', {}).keys():
                if dd_params['indicators'].get(indicator_key):
                    div_col = f'{indicator_key}_divergence'
                    div_series = get_signal_value(div_col)
                    analysis_df['reversal_confirmation_signal'] += (div_series == 1) * signal_weights.get('regular_div', 0)
                    analysis_df['reversal_confirmation_signal'] -= (div_series == -1) * signal_weights.get('regular_div', 0)
        # 指标 OB/OS 反转贡献
        obos_reversal_total = (get_signal_value('rsi_obos_reversal') +
                               get_signal_value('stoch_obos_reversal') +
                               get_signal_value('cci_obos_reversal'))
        analysis_df['reversal_confirmation_signal'] += np.sign(obos_reversal_total) * signal_weights.get('obos_reversal', 0)
        # BB 反转贡献 - 考虑前瞻性信号权重
        bb_reversal = get_signal_value('bb_reversal')
        if anticipation_params.get('enabled', False):
            anticipation_weight = anticipation_params.get('anticipation_weight', 0.1)
            full_weight = signal_weights.get('bb_reversal', 0.12)
            analysis_df['reversal_confirmation_signal'] += (bb_reversal == 1) * full_weight
            analysis_df['reversal_confirmation_signal'] -= (bb_reversal == -1) * full_weight
            analysis_df['reversal_confirmation_signal'] += (bb_reversal == 0.5) * anticipation_weight
            analysis_df['reversal_confirmation_signal'] -= (bb_reversal == -0.5) * anticipation_weight
        else:
            analysis_df['reversal_confirmation_signal'] += bb_reversal * signal_weights.get('bb_reversal', 0)
        # K 线形态贡献
        kline_series = get_signal_value('kline_pattern')
        w_rev_strong = signal_weights.get('kline_reversal_strong', 0)
        w_rev_weak = signal_weights.get('kline_reversal_weak', 0)
        w_cont = signal_weights.get('kline_continuation', 0)
        w_maru = signal_weights.get('kline_marubozu', 0)
        analysis_df['reversal_confirmation_signal'] += (kline_series.isin([1, 3, 15])) * w_rev_strong
        analysis_df['reversal_confirmation_signal'] -= (kline_series.isin([-1, -3, -12, -15])) * w_rev_strong
        analysis_df['reversal_confirmation_signal'] += (kline_series == 2) * w_rev_strong
        analysis_df['reversal_confirmation_signal'] -= (kline_series == -2) * w_rev_strong
        analysis_df['reversal_confirmation_signal'] += (kline_series.isin([4, 7, 8, 9])) * w_rev_weak
        analysis_df['reversal_confirmation_signal'] -= (kline_series.isin([-4, -7, -8, -9])) * w_rev_weak
        analysis_df['reversal_confirmation_signal'] += (kline_series.isin([6, 11, 13, 14])) * w_cont
        analysis_df['reversal_confirmation_signal'] -= (kline_series.isin([-6, -11, -13, -14])) * w_cont
        analysis_df['reversal_confirmation_signal'] += (kline_series == 10) * w_maru
        analysis_df['reversal_confirmation_signal'] -= (kline_series == -10) * w_maru
        # 放量确认反转
        vol_spike = get_signal_value('volume_spike')
        current_signal_direction = np.sign(analysis_df['reversal_confirmation_signal'])
        analysis_df.loc[(vol_spike == 1) & (current_signal_direction != 0), 'reversal_confirmation_signal'] += current_signal_direction * signal_weights.get('volume_spike', 0)
        # Williams %R 反转贡献
        willr_reversal = get_signal_value('willr_reversal')
        analysis_df['reversal_confirmation_signal'] += willr_reversal * signal_weights.get('willr_reversal', 0.1)
        # ATR 波动率信号贡献
        atr_vol_signal = get_signal_value('atr_volatility_signal')
        vol_filter_params = tr_params.get('volatility_filter', {})
        if vol_filter_params.get('enabled', False):
            high_vol_boost = vol_filter_params.get('high_vol_boost', 0.2)
            low_vol_penalty = vol_filter_params.get('low_vol_penalty', 0.3)
            atr_weight = signal_weights.get('atr_volatility', 0.07)
            analysis_df.loc[atr_vol_signal == 1, 'reversal_confirmation_signal'] += current_signal_direction * atr_weight * high_vol_boost
            analysis_df.loc[atr_vol_signal == -1, 'reversal_confirmation_signal'] -= current_signal_direction * atr_weight * low_vol_penalty
        # 历史波动率环境信号贡献
        hv_env_signal = get_signal_value('hv_environment_signal')
        hv_params = tr_params.get('historical_volatility', {})
        if hv_params.get('enabled', False):
            high_hv_weight = hv_params.get('high_hv_weight', 0.1)
            low_hv_penalty = hv_params.get('low_hv_penalty', 0.1)
            hv_weight = signal_weights.get('hv_environment', 0.05)
            analysis_df.loc[hv_env_signal == 1, 'reversal_confirmation_signal'] += current_signal_direction * hv_weight * high_hv_weight
            analysis_df.loc[hv_env_signal == -1, 'reversal_confirmation_signal'] -= current_signal_direction * hv_weight * low_hv_penalty
        # 归一化或限制信号范围
        max_abs_signal = signal_weights.get('max_abs_signal', 1.0)
        analysis_df['reversal_confirmation_signal'] = analysis_df['reversal_confirmation_signal'].clip(-max_abs_signal, max_abs_signal)
        # 触发强确认信号
        threshold = tr_params.get('confirmation_threshold', 0.7)
        analysis_df['strong_reversal_confirmation'] = 0
        analysis_df.loc[analysis_df['reversal_confirmation_signal'] >= threshold, 'strong_reversal_confirmation'] = 1
        analysis_df.loc[analysis_df['reversal_confirmation_signal'] <= -threshold, 'strong_reversal_confirmation'] = -1
        return analysis_df
    
    def generate_signals(self, data: pd.DataFrame, stock_code: str) -> pd.Series:
        """生成趋势反转信号"""
        logger.info(f"开始执行策略: {self.strategy_name} (Focus: {self.focus_timeframe})")
        # print("传入generate_signals的DataFrame列名：", data.columns.tolist())
        if data is None or data.empty:
            logger.warning("输入数据为空，无法生成信号。")
            return pd.Series(dtype=float)
        # 检查必需列
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"[{self.strategy_name}] 输入数据缺少必需列: {missing_cols}。策略无法运行。")
            return pd.Series(50.0, index=data.index)
        # 步骤 1: 计算反转导向的基础评分
        base_scores_df = self._calculate_reversal_focused_score(data)
        # 步骤 2: 应用量能调整
        vc_params = self.params.get('volume_confirmation', {})
        dd_params = self.params.get('divergence_detection', {})
        bs_params = self.params.get('base_scoring', {})
        # 正确解包 adjust_score_with_volume 的返回值
        base_score_adjusted, _ = strategy_utils.adjust_score_with_volume(
            base_scores_df['base_score_raw'], data, vc_params, dd_params, bs_params, return_analysis=False
        )
        # 确保 base_score_adjusted 的长度与 base_scores_df 的索引长度一致
        if len(base_score_adjusted) != len(base_scores_df.index):
            logger.warning(f"base_score_adjusted 长度 ({len(base_score_adjusted)}) 与 base_scores_df 索引长度 ({len(base_scores_df.index)}) 不匹配，进行对齐处理。")
            base_score_adjusted = base_score_adjusted.reindex(base_scores_df.index, fill_value=base_scores_df['base_score_raw'])
        base_scores_df['base_score_volume_adjusted'] = base_score_adjusted
        # 步骤 3: 执行信号分析 (背离, K线, 指标反转, 波动率, 动量)
        combined_data_for_signals = pd.concat([data, base_scores_df], axis=1)
        signal_analysis_df = self._perform_signal_analysis(combined_data_for_signals)
        # 步骤 4: 组合最终信号
        final_signal = pd.Series(50.0, index=data.index)
        tr_params = self.params.get('trend_reversal_params', {})
        final_weights = tr_params.get('final_signal_weights', {
            'base_score': 0.6,
            'reversal_confirm': 0.4
        })
        # 基础分贡献 (量能调整后)
        score_contribution = (base_scores_df['base_score_volume_adjusted'] - 50)
        # 反转确认信号贡献 (范围 -1 到 1) -> 映射到 -50 到 50
        reversal_confirm = signal_analysis_df.get('reversal_confirmation_signal', pd.Series(0.0, index=data.index)).fillna(0)
        reversal_contribution = reversal_confirm * 50
        # 加权求和
        final_signal = 50.0 + (score_contribution * final_weights.get('base_score', 0.6) +
                           reversal_contribution * final_weights.get('reversal_confirm', 0.4))
        # 长期趋势过滤（日线级别）
        long_term_filter = tr_params.get('long_term_trend_filter', {})
        if long_term_filter.get('enabled', False):
            tf_long = long_term_filter.get('timeframe', 'D')
            adx_col = f'ADX_14_{tf_long}'
            close_col = f'close_{tf_long}'
            if adx_col in data and close_col in data:
                adx_threshold = long_term_filter.get('adx_threshold', 30)
                filter_strength = long_term_filter.get('filter_strength', 0.3)
                adx = data[adx_col]
                trend_adjustment = pd.Series(0.0, index=data.index)
                strong_trend_cond = adx > adx_threshold
                trend_adjustment.loc[strong_trend_cond & (final_signal > 50)] = -filter_strength * 50
                trend_adjustment.loc[strong_trend_cond & (final_signal < 50)] = filter_strength * 50
                final_signal += trend_adjustment
                logger.info(f"应用长期趋势过滤，ADX 阈值: {adx_threshold}，调整强度: {filter_strength}")
        # 市场情绪过滤（预留，当前数据不支持时不执行）
        sentiment_filter = tr_params.get('market_sentiment_filter', {})
        if sentiment_filter.get('enabled', False):
            logger.warning("市场情绪过滤已启用，但当前数据不支持，跳过此步骤。")

        # --- 换手率过滤逻辑 ---
        turnover_filter = tr_params.get('turnover_filter', {})
        if turnover_filter.get('enabled', False):
            tf_turnover = turnover_filter.get('timeframe', self.focus_timeframe)
            turnover_col = f'turnover_rate_{tf_turnover}'
            if turnover_col in data and data[turnover_col].notna().any():
                turnover_rate = data[turnover_col]
                high_turnover_threshold = turnover_filter.get('high_turnover_threshold', 2.0)
                low_turnover_threshold = turnover_filter.get('low_turnover_threshold', 0.5)
                high_turnover_boost = turnover_filter.get('high_turnover_boost', 0.15)
                low_turnover_penalty = turnover_filter.get('low_turnover_penalty', 0.2)
                
                # 计算换手率均线作为基准
                turnover_ma_period = turnover_filter.get('period', 5)
                turnover_ma = turnover_rate.rolling(window=turnover_ma_period).mean()
                
                # 换手率调整信号
                turnover_adjustment = pd.Series(0.0, index=data.index)
                high_turnover_cond = turnover_rate > turnover_ma * high_turnover_threshold
                low_turnover_cond = turnover_rate < turnover_ma * low_turnover_threshold
                
                # 高换手率增强信号，低换手率削弱信号
                signal_direction = np.sign(final_signal - 50)  # 信号方向：>50 为买入，<50 为卖出
                turnover_adjustment.loc[high_turnover_cond] = signal_direction * high_turnover_boost * 50
                turnover_adjustment.loc[low_turnover_cond] = -signal_direction * low_turnover_penalty * 50
                
                final_signal += turnover_adjustment
                # logger.info(f"[{self.strategy_name}] 应用换手率过滤，时间框架: {tf_turnover}，高换手阈值: {high_turnover_threshold}，低换手阈值: {low_turnover_threshold}")
            else:
                logger.warning(f"[{self.strategy_name}] 换手率列 {turnover_col} 不存在或数据为空，跳过换手率过滤。")


        # 存储中间数据
        self.intermediate_data = pd.concat([
            base_scores_df,
            signal_analysis_df,
            pd.DataFrame({'final_signal': final_signal}, index=data.index)
        ], axis=1)

        logger.info(f"{self.strategy_name}: 信号生成完毕。")
        self.analyze_signals(stock_code)
        return final_signal
    
    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回中间计算结果"""
        return self.intermediate_data

    def analyze_signals(self, stock_code: str) -> Optional[pd.DataFrame]:
        """分析反转策略信号"""
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning("中间数据为空，无法进行信号分析。")
            return None
        
        analysis_results = {}
        data = self.intermediate_data
        latest_data = data.iloc[-1] if not data.empty else None
        dd_params = self.params.get('divergence_detection', {})
        kpd_params = self.params.get('kline_pattern_detection', {})

        # 统计分析
        if 'final_signal' in data:
            final_signal = data['final_signal'].dropna()
            if not final_signal.empty:
                analysis_results['final_signal_mean'] = final_signal.mean()
                analysis_results['final_signal_potential_buy_ratio'] = (final_signal >= 65).mean()
                analysis_results['final_signal_potential_sell_ratio'] = (final_signal <= 35).mean()
                analysis_results['final_signal_strong_buy_ratio'] = (final_signal >= 75).mean()
                analysis_results['final_signal_strong_sell_ratio'] = (final_signal <= 25).mean()
        if 'strong_reversal_confirmation' in data:
            strong_conf = data['strong_reversal_confirmation'].dropna()
            if not strong_conf.empty:
                analysis_results['strong_buy_reversal_ratio'] = (strong_conf == 1).mean()
                analysis_results['strong_sell_reversal_ratio'] = (strong_conf == -1).mean()
        # 背离统计
        if dd_params.get('enabled', False):
            for indicator_key in dd_params.get('indicators', {}).keys():
                if dd_params['indicators'].get(indicator_key):
                    div_col = f'{indicator_key}_divergence'
                    if div_col in data:
                        div_series = data[div_col].dropna()
                        if not div_series.empty:
                            analysis_results[f'{indicator_key}_regular_bullish_count'] = (div_series == 1).sum()
                            analysis_results[f'{indicator_key}_regular_bearish_count'] = (div_series == -1).sum()
                            analysis_results[f'{indicator_key}_hidden_bullish_count'] = (div_series == 2).sum()
                            analysis_results[f'{indicator_key}_hidden_bearish_count'] = (div_series == -2).sum()
        # K线形态统计
        if kpd_params.get('enabled', False) and 'kline_pattern' in data:
            kline_series = data['kline_pattern'].dropna()
            if not kline_series.empty:
                analysis_results['kline_bull_engulf_count'] = (kline_series == 1).sum()
                analysis_results['kline_bear_engulf_count'] = (kline_series == -1).sum()
                analysis_results['kline_morning_star_count'] = (kline_series == 3).sum()
                analysis_results['kline_evening_star_count'] = (kline_series == -3).sum()
                analysis_results['kline_hammer_count'] = (kline_series == 2).sum()
                analysis_results['kline_hanging_man_count'] = (kline_series == -2).sum()
                analysis_results['kline_bull_counter_count'] = (kline_series == 15).sum()
                analysis_results['kline_bear_counter_count'] = (kline_series == -15).sum()
                analysis_results['kline_upside_gap_two_crows_count'] = (kline_series == -12).sum()
                analysis_results['kline_piercing_count'] = (kline_series == 4).sum()
                analysis_results['kline_dark_cloud_count'] = (kline_series == -4).sum()
                analysis_results['kline_bull_harami_count'] = (kline_series == 7).sum()
                analysis_results['kline_bear_harami_count'] = (kline_series == -7).sum()
                analysis_results['kline_bull_harami_cross_count'] = (kline_series == 8).sum()
                analysis_results['kline_bear_harami_cross_count'] = (kline_series == -8).sum()
                analysis_results['kline_tweezer_bottom_count'] = (kline_series == 9).sum()
                analysis_results['kline_tweezer_top_count'] = (kline_series == -9).sum()
                analysis_results['kline_three_soldiers_count'] = (kline_series == 6).sum()
                analysis_results['kline_three_crows_count'] = (kline_series == -6).sum()
                analysis_results['kline_rising_three_count'] = (kline_series == 11).sum()
                analysis_results['kline_falling_three_count'] = (kline_series == -11).sum()
                analysis_results['kline_upside_tasuki_count'] = (kline_series == 13).sum()
                analysis_results['kline_downside_tasuki_count'] = (kline_series == -13).sum()
                analysis_results['kline_bull_sep_lines_count'] = (kline_series == 14).sum()
                analysis_results['kline_bear_sep_lines_count'] = (kline_series == -14).sum()
                analysis_results['kline_doji_count'] = (kline_series == 5).sum()
                analysis_results['kline_bull_marubozu_count'] = (kline_series == 10).sum()
                analysis_results['kline_bear_marubozu_count'] = (kline_series == -10).sum()
        # Williams %R 反转统计
        if 'willr_reversal' in data:
            willr_series = data['willr_reversal'].dropna()
            if not willr_series.empty:
                analysis_results['willr_buy_count'] = (willr_series == 1).sum()
                analysis_results['willr_sell_count'] = (willr_series == -1).sum()
        # ATR 波动率信号统计
        if 'atr_volatility_signal' in data:
            atr_series = data['atr_volatility_signal'].dropna()
            if not atr_series.empty:
                analysis_results['atr_high_vol_count'] = (atr_series == 1).sum()
                analysis_results['atr_low_vol_count'] = (atr_series == -1).sum()
        # 历史波动率环境统计
        if 'hv_environment_signal' in data:
            hv_series = data['hv_environment_signal'].dropna()
            if not hv_series.empty:
                analysis_results['hv_high_vol_count'] = (hv_series == 1).sum()
                analysis_results['hv_low_vol_count'] = (hv_series == -1).sum()

        # 最新信号判断
        signal_judgment = {}
        is_strong_trend = False  # 标识是否强趋势
        if latest_data is not None:
            if 'final_signal' in latest_data:
                score = latest_data['final_signal']
                if score >= 75:
                    signal_judgment['reversal_potential'] = "极强买入反转潜力"
                elif score >= 65:
                    signal_judgment['reversal_potential'] = "较强买入反转潜力"
                elif score <= 25:
                    signal_judgment['reversal_potential'] = "极强卖出反转潜力"
                elif score <= 35:
                    signal_judgment['reversal_potential'] = "较强卖出反转潜力"
                else:
                    signal_judgment['reversal_potential'] = "反转信号不明确"
            if 'strong_reversal_confirmation' in latest_data:
                strong_conf = latest_data['strong_reversal_confirmation']
                if strong_conf == 1:
                    signal_judgment['confirmation_status'] = "强确认看涨反转"
                elif strong_conf == -1:
                    signal_judgment['confirmation_status'] = "强确认看跌反转"
                else:
                    signal_judgment['confirmation_status'] = "无强反转确认"
                # 判断强趋势条件
                if ((score >= 75 or score <= 25) and strong_conf in [1, -1]):
                    is_strong_trend = True

            # 检查是否有活跃的反转信号
            active_signals = []
            if dd_params.get('enabled', False):
                for indicator_key in dd_params.get('indicators', {}).keys():
                    if dd_params['indicators'].get(indicator_key):
                        div_col = f'{indicator_key}_divergence'
                        if div_col in latest_data and latest_data[div_col] != 0:
                            active_signals.append(f"{indicator_key} 背离 (值: {latest_data[div_col]})")
            if 'kline_pattern' in latest_data and latest_data['kline_pattern'] != 0 and latest_data['kline_pattern'] != 5:
                active_signals.append(f"K线形态 (值: {latest_data['kline_pattern']})")
            if 'rsi_obos_reversal' in latest_data and latest_data['rsi_obos_reversal'] != 0:
                active_signals.append(f"RSI 超买/超卖 (值: {latest_data['rsi_obos_reversal']})")
            if 'willr_reversal' in latest_data and latest_data['willr_reversal'] != 0:
                active_signals.append(f"Williams %R 反转 (值: {latest_data['willr_reversal']})")
            if 'atr_volatility_signal' in latest_data and latest_data['atr_volatility_signal'] != 0:
                active_signals.append(f"ATR 波动率 (值: {latest_data['atr_volatility_signal']})")
            if 'hv_environment_signal' in latest_data and latest_data['hv_environment_signal'] != 0:
                active_signals.append(f"历史波动率环境 (值: {latest_data['hv_environment_signal']})")
            signal_judgment['active_reversal_triggers'] = ", ".join(active_signals) if active_signals else "暂无活跃反转信号"
        
        def get_count(key):
            return analysis_results.get(key, 0)

        kline_bull_rev_strong_count = sum([get_count(f'kline_{k}_count') for k in ['bull_engulf', 'morning_star', 'hammer', 'bull_counter']])
        kline_bear_rev_strong_count = sum([get_count(f'kline_{k}_count') for k in ['bear_engulf', 'evening_star', 'hanging_man', 'bear_counter', 'upside_gap_two_crows']])
        kline_bull_rev_weak_count = sum([get_count(f'kline_{k}_count') for k in ['piercing', 'bull_harami', 'bull_harami_cross', 'tweezer_bottom']])
        kline_bear_rev_weak_count = sum([get_count(f'kline_{k}_count') for k in ['dark_cloud', 'bear_harami', 'bear_harami_cross', 'tweezer_top']])
        kline_bull_cont_count = sum([get_count(f'kline_{k}_count') for k in ['three_soldiers', 'rising_three', 'upside_tasuki', 'bull_sep_lines', 'bull_marubozu']])
        kline_bear_cont_count = sum([get_count(f'kline_{k}_count') for k in ['three_crows', 'falling_three', 'downside_tasuki', 'bear_sep_lines', 'bear_marubozu']])

        chinese_interpretation = (
            f"【趋势反转策略分析报告 - 股票代码: {stock_code} - 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}】\n"
            f"核心观点:\n"
            f" - 当前反转潜力评估: {signal_judgment.get('reversal_potential', '未知')}\n"
            f" - 确认信号状态: {signal_judgment.get('confirmation_status', '未知')}\n"
            f" - 当前触发信号: {signal_judgment.get('active_reversal_triggers', '暂无')}\n"
            f"操作建议:\n"
            f" - {signal_judgment.get('reversal_potential', '未知')} 指示了潜在的交易机会方向。\n"
            f" - 结合 {signal_judgment.get('confirmation_status', '未知')} 评估信号的可靠性。\n"
            f" - 关注 {signal_judgment.get('active_reversal_triggers', '暂无')} 提供的具体反转依据。\n"
            f"统计数据 (分析周期内):\n"
            f" - 最终信号平均值: {analysis_results.get('final_signal_mean', np.nan):.2f}\n"
            f" - 强买入反转确认比例: {analysis_results.get('strong_buy_reversal_ratio', np.nan)*100:.2f}%\n"
            f" - 强卖出反转确认比例: {analysis_results.get('strong_sell_reversal_ratio', np.nan)*100:.2f}%\n"
            f" - K线强反转信号: 看涨 {kline_bull_rev_strong_count} 次, 看跌 {kline_bear_rev_strong_count} 次\n"
            f" - K线弱反转信号: 看涨 {kline_bull_rev_weak_count} 次, 看跌 {kline_bear_rev_weak_count} 次\n"
            f" - K线持续信号: 看涨 {kline_bull_cont_count} 次, 看跌 {kline_bear_cont_count} 次\n"
            f" - Williams %R 反转信号: 买入 {get_count('willr_buy_count')} 次, 卖出 {get_count('willr_sell_count')} 次\n"
            f" - ATR 波动率信号: 高波动 {get_count('atr_high_vol_count')} 次, 低波动 {get_count('atr_low_vol_count')} 次\n"
            f" - 历史波动率环境: 高波动 {get_count('hv_high_vol_count')} 次, 低波动 {get_count('hv_low_vol_count')} 次\n"
            f"温馨提示: 本分析仅供参考，不构成投资建议，请结合市场实际情况和个人风险承受能力谨慎决策。"
        )

        # 生成路径并保存文件
        base_path = os.path.join("analysis_results", "TrendReversalStrategy", stock_code)
        os.makedirs(base_path, exist_ok=True)
        filename = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S.txt')
        file_path = os.path.join(base_path, filename)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(chinese_interpretation)
            logger.info(f"分析结果已保存至文件: {file_path}")
        except Exception as e:
            logger.error(f"保存分析结果文件失败: {e}")

        # 强趋势时输出到屏幕
        if is_strong_trend:
            print(chinese_interpretation)

        logger.info(chinese_interpretation)

        analysis_results.update(signal_judgment)
        self.analysis_results = pd.DataFrame([analysis_results])
        return self.analysis_results

    
    def save_analysis_results(self, stock_code: str, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        保存趋势反转策略的分析结果到数据库
        """
        from stock_models.stock_analytics import StockScoreAnalysis
        from stock_models.stock_basic import StockInfo
        import logging

        logger = logging.getLogger("strategy_trend_reversal")

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
                    'reversal_confirmation_signal': convert_nan_to_none(intermediate_data.get('reversal_confirmation_signal', None)),
                    'strong_reversal_confirmation': convert_nan_to_none(intermediate_data.get('strong_reversal_confirmation', None)),
                    'macd_hist_divergence': convert_nan_to_none(intermediate_data.get('macd_hist_divergence', None)),
                    'rsi_divergence': convert_nan_to_none(intermediate_data.get('rsi_divergence', None)),
                    'mfi_divergence': convert_nan_to_none(intermediate_data.get('mfi_divergence', None)),
                    'obv_divergence': convert_nan_to_none(intermediate_data.get('obv_divergence', None)),
                    'kline_pattern': convert_nan_to_none(intermediate_data.get('kline_pattern', None)),
                    'rsi_obos_reversal': convert_nan_to_none(intermediate_data.get('rsi_obos_reversal', None)),
                    'stoch_obos_reversal': convert_nan_to_none(intermediate_data.get('stoch_obos_reversal', None)),
                    'cci_obos_reversal': convert_nan_to_none(intermediate_data.get('cci_obos_reversal', None)),
                    'bb_reversal': convert_nan_to_none(intermediate_data.get('bb_reversal', None)),
                    'volume_spike': convert_nan_to_none(intermediate_data.get('volume_spike', None)),
                    'close_price': convert_nan_to_none(latest_data.get(f'close_{self.focus_timeframe}', None)),
                    'final_signal_mean': convert_nan_to_none(analysis_data.get('final_signal_mean', None)),
                    'final_signal_potential_buy_ratio': convert_nan_to_none(analysis_data.get('final_signal_potential_buy_ratio', None)),
                    'final_signal_potential_sell_ratio': convert_nan_to_none(analysis_data.get('final_signal_potential_sell_ratio', None)),
                    'final_signal_strong_buy_ratio': convert_nan_to_none(analysis_data.get('final_signal_strong_buy_ratio', None)),
                    'final_signal_strong_sell_ratio': convert_nan_to_none(analysis_data.get('final_signal_strong_sell_ratio', None)),
                    'strong_buy_reversal_ratio': convert_nan_to_none(analysis_data.get('strong_buy_reversal_ratio', None)),
                    'strong_sell_reversal_ratio': convert_nan_to_none(analysis_data.get('strong_sell_reversal_ratio', None)),
                    'params_snapshot': self.params,
                }
            )
            logger.info(f"成功保存 {stock_code} 的趋势反转策略分析结果，时间戳: {timestamp}")
        except StockInfo.DoesNotExist:
            logger.error(f"股票 {stock_code} 未找到，无法保存分析结果")
        except Exception as e:
            logger.error(f"保存 {stock_code} 的趋势反转策略分析结果时出错: {e}")







