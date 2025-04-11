# strategies/macd_rsi_kdj_boll_enhanced_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# 从 base 导入基类和信号常量
from .base import BaseStrategy, SIGNAL_STRONG_BUY, SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_STRONG_SELL, SIGNAL_NONE

# 假设 core.constants 定义了 TimeLevel 枚举 (虽然这里不直接用，但保持一致性)
# from core.constants import TimeLevel

class MacdRsiKdjBollEnhancedStrategy(BaseStrategy):
    """
    多时间周期 MACD+RSI+KDJ+BOLL+量能+趋势 组合增强策略。

    策略逻辑:
    1. 以 15 分钟 K 线为主要操作周期。
    2. 结合 5, 15, 30, 60 四个时间周期的 MACD, RSI, KDJ, BOLL, CCI, MFI, ROC, DMI, SAR 指标。
    3. 为每个指标在每个时间周期生成初步信号 (-1, 0, 1)。
    4. 根据时间周期为信号分配权重 (短周期权重低，长周期权重高)。
    5. 计算加权总分。
    6. **增加量能确认**: 使用 15 分钟周期的 Amount MA, CMF, OBV 趋势对初步信号进行确认或降级。
    7. 根据最终调整后的分数或信号状态确定最终的买卖信号强度。
    """
    strategy_name = "MACD_RSI_KDJ_BOLL_Enhanced_MultiTimeframe"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略参数。
        """
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'kdj_period_k': 9,
            'kdj_period_d': 3,
            'kdj_period_j': 3,
            'kdj_oversold': 20,
            'kdj_overbought': 80,
            'boll_period': 20,
            'boll_std_dev': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'cci_period': 14,       # CCI 周期
            'cci_threshold': 100,   # CCI 超买/超卖阈值 (+/-)
            'mfi_period': 14,       # MFI 周期
            'mfi_oversold': 20,     # MFI 超卖
            'mfi_overbought': 80,   # MFI 超买
            'roc_period': 12,       # ROC 周期
            'dmi_period': 14,       # DMI (ADX, PDI, MDI) 周期
            'adx_threshold': 20,    # ADX 趋势强度阈值
            'sar_step': 0.02,       # SAR 加速因子
            'sar_max': 0.2,         # SAR 最大加速因子
            'amount_ma_period': 20, # 成交量移动平均周期 (用于确认)
            'obv_ma_period': 10,    # OBV 移动平均周期 (用于确认 OBV 趋势)
            'cmf_period': 20,       # CMF 周期 (用于确认)
            # --- 权重和阈值 ---
            'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2},
            'score_thresholds': { # 可能需要根据指标数量调整
                'strong_buy': 2.5, # 示例值，需要调整
                'buy': 0.8,        # 示例值，需要调整
                'sell': -0.8,       # 示例值，需要调整
                'strong_sell': -2.5 # 示例值，需要调整
            },
            'volume_confirmation': True, # 是否启用量能确认
            'volume_tf': '15',         # 使用哪个时间周期的量能指标进行确认
            # --- 量能背离检查 ---
            'check_bearish_divergence': True, # 是否检查顶背离作为卖出信号
            'divergence_price_period': 5,     # 检测价格新高的周期 (例如最近5根K线)
            'divergence_threshold_cmf': -0.05, # CMF 低于此值视为量能弱
            'divergence_threshold_mfi': 40, # MFI 低于此值视为量能弱 (可选)
        }
        # 合并默认参数和传入参数
        merged_params = {**default_params, **(params or {})}
        # 深层合并字典参数
        if params and 'weights' in params:
            merged_params['weights'] = {**default_params['weights'], **params['weights']}
        if params and 'score_thresholds' in params:
            merged_params['score_thresholds'] = {**default_params['score_thresholds'], **params['score_thresholds']}

        self.timeframes = ['5', '15', '30', '60'] # 策略使用的时间周期
        # 定义哪些指标参与评分
        self.score_indicators = ['macd', 'rsi', 'kdj', 'boll', 'cci', 'mfi', 'roc', 'dmi', 'sar']
        # 定义用于量能确认的指标 (及其参数)
        self.volume_confirm_indicators = ['amount_ma', 'cmf', 'obv']
        # 用于存储中间计算结果的属性
        self.intermediate_data: Optional[pd.DataFrame] = None

        super().__init__(merged_params)


    def _validate_params(self):
        """验证参数"""
        super()._validate_params() # 调用基类或其他已有的验证
        # 验证权重
        if not isinstance(self.params['weights'], dict) or \
           not all(tf in self.params['weights'] for tf in self.timeframes) or \
           abs(sum(self.params['weights'].values()) - 1.0) > 1e-6:
            raise ValueError("参数 'weights' 必须是包含所有时间周期 ('5', '15', '30', '60') 且总和为 1.0 的字典")
        # 验证阈值
        if not isinstance(self.params['score_thresholds'], dict) or \
           not all(k in self.params['score_thresholds'] for k in ['strong_buy', 'buy', 'sell', 'strong_sell']):
            raise ValueError("参数 'score_thresholds' 必须是包含 'strong_buy', 'buy', 'sell', 'strong_sell' 键的字典")
        # 验证量能确认时间周期
        if self.params['volume_confirmation'] and self.params['volume_tf'] not in self.timeframes:
             raise ValueError(f"参数 'volume_tf' ({self.params['volume_tf']}) 必须是 {self.timeframes} 中的一个")
        # TODO: 添加新指标参数的验证


    def get_required_columns(self) -> List[str]:
        """返回策略运行所需的 DataFrame 列名。"""
        required = []
        price_tf = self.params.get('volume_tf', '15')
        required.append(f'close_{price_tf}')
        required.append(f'high_{price_tf}') # 用于计算价格新高
        required.append(f'low_{price_tf}')
        # 主操作周期的收盘价 (用于 BOLL, SAR, KC 等比较)
        # 使用量能确认周期的价格数据，或者固定的 15m 数据
        compare_price_tf = self.params.get('volume_tf', '15')
        required.append(f'close_{compare_price_tf}') # 主要用于比较的价格序列
        required.append(f'high_{compare_price_tf}') # SAR 可能需要
        required.append(f'low_{compare_price_tf}')  # SAR 可能需要

        # 参与评分的指标
        for tf in self.timeframes:
            # MACD
            required.extend([f'diff_{tf}', f'dea_{tf}', f'macd_{tf}'])
            # RSI (简化假设列名)
            required.append(f'rsi_{tf}')
            # KDJ (简化假设列名)
            required.extend([f'k_{tf}', f'd_{tf}', f'j_{tf}'])
            # BOLL (简化假设列名)
            required.extend([f'upper_{tf}', f'mid_{tf}', f'lower_{tf}'])
            # CCI (简化假设列名)
            required.append(f'cci_{tf}')
            # MFI (简化假设列名)
            required.append(f'mfi_{tf}')
            # ROC (简化假设列名)
            required.append(f'roc_{tf}')
            # DMI (简化假设列名)
            required.extend([f'pdi_{tf}', f'mdi_{tf}', f'adx_{tf}'])
            # SAR (简化假设列名)
            required.append(f'sar_{tf}')
            # # KC (如果加入)
            # required.extend([f'kc_upper_{tf}', f'kc_mid_{tf}', f'kc_lower_{tf}'])

        # 量能确认指标 (仅需特定时间周期)
        vol_tf = self.params['volume_tf']
        # Amount MA (需要成交量数据)
        required.append(f'amount_{vol_tf}') # 需要原始成交额
        required.append(f'amount_ma_{vol_tf}') # 需要计算好的成交额均线
        # CMF
        required.append(f'cmf_{vol_tf}')
        # OBV (需要原始 OBV 和其均线)
        required.append(f'obv_{vol_tf}')
        required.append(f'obv_ma_{vol_tf}') # 假设 OBV 的均线列名
        # --- 如果使用 MFI 判断背离，确保包含 ---
        # if self.params.get('check_bearish_divergence'):
        #     required.append(f'mfi_{vol_tf}') # 假设 MFI 列名是 mfi_TF

        # 去重后返回
        return list(set(required))

    # --- 单个指标信号生成函数 (保持向量化) ---
    # 这些函数已经使用了向量化操作，无需修改
    def _get_macd_signal(self, diff: pd.Series, dea: pd.Series, macd: pd.Series) -> pd.Series:
        signal = pd.Series(0, index=diff.index, dtype=np.int8) # 使用更小的整数类型
        buy_signal = (macd.shift(1) < 0) & (macd > 0)
        sell_signal = (macd.shift(1) > 0) & (macd < 0)
        signal.loc[buy_signal] = 1
        signal.loc[sell_signal] = -1
        return signal

    def _get_rsi_signal(self, rsi: pd.Series) -> pd.Series:
        signal = pd.Series(0, index=rsi.index, dtype=np.int8)
        oversold = self.params['rsi_oversold']
        overbought = self.params['rsi_overbought']
        buy_signal = (rsi.shift(1) < oversold) & (rsi > oversold)
        sell_signal = (rsi.shift(1) > overbought) & (rsi < overbought)
        signal.loc[buy_signal] = 1
        signal.loc[sell_signal] = -1
        return signal

    def _get_kdj_signal(self, k: pd.Series, d: pd.Series, j: pd.Series) -> pd.Series:
        signal = pd.Series(0, index=k.index, dtype=np.int8)
        # oversold = self.params['kdj_oversold'] # J值超卖/超买条件可以加入，但交叉更常用
        # overbought = self.params['kdj_overbought']
        buy_cross = (k.shift(1) < d.shift(1)) & (k > d) & (j < 80) # 金叉发生在非超买区
        sell_cross = (k.shift(1) > d.shift(1)) & (k < d) & (j > 20) # 死叉发生在非超卖区
        signal.loc[buy_cross] = 1
        signal.loc[sell_cross] = -1
        return signal

    def _get_boll_signal(self, close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
        signal = pd.Series(0, index=close.index, dtype=np.int8)
        buy_support = (close.shift(1) < lower.shift(1)) & (close > lower) # 下轨支撑
        sell_pressure = (close.shift(1) > upper.shift(1)) & (close < upper) # 上轨压制
        # 也可以考虑突破中轨
        # buy_mid_cross = (close.shift(1) < mid.shift(1)) & (close > mid)
        # sell_mid_cross = (close.shift(1) > mid.shift(1)) & (close < mid)
        signal.loc[buy_support] = 1
        signal.loc[sell_pressure] = -1
        return signal

    def _get_cci_signal(self, cci: pd.Series) -> pd.Series:
        """根据 CCI 指标生成信号 (1: 从下向上穿过阈值, -1: 从上向下穿过阈值)"""
        signal = pd.Series(0, index=cci.index, dtype=np.int8)
        threshold = self.params['cci_threshold']
        # 上穿 -threshold (买入信号)
        buy_signal = (cci.shift(1) < -threshold) & (cci > -threshold)
        # 下穿 +threshold (卖出信号)
        sell_signal = (cci.shift(1) > threshold) & (cci < threshold)
        signal.loc[buy_signal] = 1
        signal.loc[sell_signal] = -1
        return signal

    def _get_mfi_signal(self, mfi: pd.Series) -> pd.Series:
        """根据 MFI 指标生成信号 (1: 超卖区反弹, -1: 超买区回调)"""
        signal = pd.Series(0, index=mfi.index, dtype=np.int8)
        oversold = self.params['mfi_oversold']
        overbought = self.params['mfi_overbought']
        # 上穿超卖线
        buy_signal = (mfi.shift(1) < oversold) & (mfi > oversold)
        # 下穿超买线
        sell_signal = (mfi.shift(1) > overbought) & (mfi < overbought)
        signal.loc[buy_signal] = 1
        signal.loc[sell_signal] = -1
        return signal

    def _get_roc_signal(self, roc: pd.Series) -> pd.Series:
        """根据 ROC 指标生成信号 (1: 上穿 0 轴, -1: 下穿 0 轴)"""
        signal = pd.Series(0, index=roc.index, dtype=np.int8)
        # ROC 由负转正
        buy_signal = (roc.shift(1) < 0) & (roc > 0)
        # ROC 由正转负
        sell_signal = (roc.shift(1) > 0) & (roc < 0)
        signal.loc[buy_signal] = 1
        signal.loc[sell_signal] = -1
        return signal

    def _get_dmi_signal(self, pdi: pd.Series, mdi: pd.Series, adx: pd.Series) -> pd.Series:
        """根据 DMI 指标生成信号 (1: PDI 上穿 MDI 且 ADX 确认趋势, -1: MDI 上穿 PDI 且 ADX 确认趋势)"""
        signal = pd.Series(0, index=pdi.index, dtype=np.int8)
        adx_threshold = self.params['adx_threshold']
        # PDI 上穿 MDI，且 ADX 大于阈值 (或 ADX 正在上升)
        buy_signal = (pdi.shift(1) < mdi.shift(1)) & (pdi > mdi) & (adx > adx_threshold)
        # MDI 上穿 PDI，且 ADX 大于阈值 (或 ADX 正在上升)
        sell_signal = (mdi.shift(1) < pdi.shift(1)) & (mdi > pdi) & (adx > adx_threshold)
        signal.loc[buy_signal] = 1
        signal.loc[sell_signal] = -1
        return signal

    def _get_sar_signal(self, close: pd.Series, sar: pd.Series) -> pd.Series:
        """根据 SAR 指标生成信号 (1: SAR 向上翻转, -1: SAR 向下翻转)"""
        signal = pd.Series(0, index=close.index, dtype=np.int8)
        # SAR 向上翻转 (前一天 SAR > close, 今天 SAR < close) - 买入信号
        # 使用 .loc 避免 SettingWithCopyWarning
        buy_signal = (sar.shift(1) > close.shift(1)) & (sar < close) # 使用当前 close 和 sar 判断
        # SAR 向下翻转 (前一天 SAR < close, 今天 SAR > close) - 卖出信号
        sell_signal = (sar.shift(1) < close.shift(1)) & (sar > close) # 使用当前 close 和 sar 判断
        signal.loc[buy_signal] = 1
        signal.loc[sell_signal] = -1
        return signal

    # --- 量能确认逻辑 (保持向量化) ---
    def _confirm_with_volume(self, preliminary_signal: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        使用量能指标确认或调整初步信号。
        :param preliminary_signal: 初步生成的信号 Series (包含 BUY, SELL 等)
        :param data: 包含量能指标数据的 DataFrame
        :return: 调整后的信号 Series
        """
        if not self.params['volume_confirmation']:
            return preliminary_signal

        vol_tf = self.params['volume_tf']
        # 复制一份以修改，确保原始信号不变
        confirmed_signal = preliminary_signal.copy()

        # 获取价格和量能数据 (确保列存在)
        close = data.get(f'close_{vol_tf}')
        high = data.get(f'high_{vol_tf}') # 用于检测价格新高
        amount = data.get(f'amount_{vol_tf}')
        amount_ma = data.get(f'amount_ma_{vol_tf}')
        cmf = data.get(f'cmf_{vol_tf}')
        obv = data.get(f'obv_{vol_tf}')
        obv_ma = data.get(f'obv_ma_{vol_tf}')
        mfi = data.get(f'mfi_{vol_tf}') # 如果使用 MFI

        # 检查所需列是否存在，避免 KeyError
        required_cols_exist = all(col is not None for col in [close, high, amount, amount_ma, cmf, obv, obv_ma])
        if not required_cols_exist:
             # 如果缺少关键列，无法进行量能确认，返回原始信号
             # 可以考虑添加日志警告
             print(f"Warning: Missing required columns for volume confirmation on timeframe {vol_tf}. Skipping volume confirmation.") # 或者使用 logging
             return preliminary_signal

        # 1. 量能确认买入信号
        # 买入确认: 成交额高于均线 OR CMF > 0 OR OBV 高于其均线 (三者满足其一或多个)
        buy_volume_confirm = pd.Series(False, index=data.index)
        if amount is not None and amount_ma is not None:
            buy_volume_confirm |= (amount > amount_ma)
        if cmf is not None:
            buy_volume_confirm |= (cmf > 0) # CMF 大于 0 表示资金流入
        if obv is not None and obv_ma is not None:
            buy_volume_confirm |= (obv > obv_ma) # OBV 趋势向上

        # 应用确认逻辑: 对初步的买入/强力买入信号进行检查
        is_buy_signal = (preliminary_signal == SIGNAL_BUY) | (preliminary_signal == SIGNAL_STRONG_BUY)
        # 如果是买入信号但量能不确认，则降级为持有
        # 使用 .loc 避免 SettingWithCopyWarning
        confirmed_signal.loc[is_buy_signal & (~buy_volume_confirm)] = SIGNAL_HOLD

        # 2. 检查量能顶背离
        if self.params.get('check_bearish_divergence'):
            price_period = self.params['divergence_price_period']
            cmf_threshold = self.params['divergence_threshold_cmf']
            mfi_threshold = self.params['divergence_threshold_mfi'] # 如果使用 MFI

            # 条件1: 价格创近期新高 (例如，当前 high 是最近 price_period 内的最高点)
            # 确保 rolling 有足够的数据
            if len(high) >= price_period:
                is_price_high = high == high.rolling(window=price_period, min_periods=price_period).max()
            else:
                is_price_high = pd.Series(False, index=high.index) # 不足周期无法判断

            # 并且价格是上涨的 (可选，避免横盘时误判)
            is_price_rising = close > close.shift(1)
            # 条件2: 量能指标表现疲软
            is_cmf_weak = cmf < cmf_threshold # CMF 为负或低于某个阈值
            is_obv_weak = obv < obv_ma       # OBV 低于其移动平均线
            is_mfi_weak = mfi < mfi_threshold # MFI 处于较低水平 (可选)

            # 组合判断顶背离: 价格创新高 + 量能疲软 (至少满足一个量能弱条件)
            # 注意：更严格的背离需要比较指标的高点，这里简化为当前状态判断
            bearish_divergence = is_price_high & is_price_rising & (is_cmf_weak | is_obv_weak) # 可以加入 is_mfi_weak

            # 如果检测到顶背离，且当前信号不是卖出信号，则强制设置为卖出
            # 可以根据需要设置为 SIGNAL_SELL 或 SIGNAL_STRONG_SELL
            # 这里我们覆盖掉非卖出信号
            is_not_sell = (confirmed_signal != SIGNAL_SELL) & (confirmed_signal != SIGNAL_STRONG_SELL)
            # 使用 .loc 避免 SettingWithCopyWarning
            confirmed_signal.loc[bearish_divergence & is_not_sell] = SIGNAL_SELL # 或 SIGNAL_STRONG_SELL

        return confirmed_signal


    # --- 主信号生成逻辑 ---
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        根据输入的包含多时间周期指标的 DataFrame 生成最终交易信号。
        """
        # 初始化用于存储中间信号和总分的 DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['total_score'] = 0.0 # 初始化总分列

        weights = self.params['weights']
        thresholds = self.params['score_thresholds']
        vol_tf = self.params['volume_tf'] # 量能确认的时间周期

        # 获取用于比较的价格序列 (例如 SAR, BOLL 需要)
        # 使用量能确认周期的价格数据，或者固定的 15m 数据
        price_tf = vol_tf # 或者 '15'
        # 确保价格列存在
        if f'close_{price_tf}' not in data.columns:
            raise ValueError(f"在输入数据中找不到所需的列: 'close_{price_tf}'.")
        close_price = data[f'close_{price_tf}']
        high_price = data.get(f'high_{price_tf}') # 如果 SAR 需要
        low_price = data.get(f'low_{price_tf}')  # 如果 SAR 需要

        # 1. 为每个时间周期计算各指标的初步信号并加权计分
        for tf in self.timeframes:
            tf_signal_sum = pd.Series(0.0, index=data.index) # 当前时间周期的信号总和 (float)

            # --- MACD ---
            diff_col, dea_col, macd_col = f'diff_{tf}', f'dea_{tf}', f'macd_{tf}'
            if all(c in data for c in [diff_col, dea_col, macd_col]):
                macd_sig = self._get_macd_signal(data[diff_col], data[dea_col], data[macd_col])
                if 'macd' in self.score_indicators: tf_signal_sum += macd_sig.astype(float).fillna(0) # 计算时用 float
                signals[f'macd_signal_{tf}'] = macd_sig.astype('category') # 存储时用 category

            # --- RSI ---
            rsi_col = f'rsi_{tf}'
            if rsi_col in data:
                rsi_sig = self._get_rsi_signal(data[rsi_col])
                if 'rsi' in self.score_indicators: tf_signal_sum += rsi_sig.astype(float).fillna(0)
                signals[f'rsi_signal_{tf}'] = rsi_sig.astype('category')

            # --- KDJ ---
            k_col, d_col, j_col = f'k_{tf}', f'd_{tf}', f'j_{tf}'
            if all(c in data for c in [k_col, d_col, j_col]):
                kdj_sig = self._get_kdj_signal(data[k_col], data[d_col], data[j_col])
                if 'kdj' in self.score_indicators: tf_signal_sum += kdj_sig.astype(float).fillna(0)
                signals[f'kdj_signal_{tf}'] = kdj_sig.astype('category')

            # --- BOLL ---
            upper_col, mid_col, lower_col = f'upper_{tf}', f'mid_{tf}', f'lower_{tf}'
            if all(c in data for c in [upper_col, mid_col, lower_col]):
                # BOLL 信号统一使用 price_tf 的收盘价比较
                boll_sig = self._get_boll_signal(close_price, data[upper_col], data[mid_col], data[lower_col])
                if 'boll' in self.score_indicators: tf_signal_sum += boll_sig.astype(float).fillna(0)
                signals[f'boll_signal_{tf}'] = boll_sig.astype('category')

            # --- CCI ---
            cci_col = f'cci_{tf}'
            if cci_col in data:
                cci_sig = self._get_cci_signal(data[cci_col])
                if 'cci' in self.score_indicators: tf_signal_sum += cci_sig.astype(float).fillna(0)
                signals[f'cci_signal_{tf}'] = cci_sig.astype('category')

            # --- MFI ---
            mfi_col = f'mfi_{tf}'
            if mfi_col in data:
                mfi_sig = self._get_mfi_signal(data[mfi_col])
                if 'mfi' in self.score_indicators: tf_signal_sum += mfi_sig.astype(float).fillna(0)
                signals[f'mfi_signal_{tf}'] = mfi_sig.astype('category')

            # --- ROC ---
            roc_col = f'roc_{tf}'
            if roc_col in data:
                roc_sig = self._get_roc_signal(data[roc_col])
                if 'roc' in self.score_indicators: tf_signal_sum += roc_sig.astype(float).fillna(0)
                signals[f'roc_signal_{tf}'] = roc_sig.astype('category')

            # --- DMI ---
            pdi_col, mdi_col, adx_col = f'pdi_{tf}', f'mdi_{tf}', f'adx_{tf}'
            if all(c in data for c in [pdi_col, mdi_col, adx_col]):
                dmi_sig = self._get_dmi_signal(data[pdi_col], data[mdi_col], data[adx_col])
                if 'dmi' in self.score_indicators: tf_signal_sum += dmi_sig.astype(float).fillna(0)
                signals[f'dmi_signal_{tf}'] = dmi_sig.astype('category')

            # --- SAR ---
            sar_col = f'sar_{tf}'
            if sar_col in data:
                 # SAR 信号统一使用 price_tf 的收盘价比较
                sar_sig = self._get_sar_signal(close_price, data[sar_col])
                if 'sar' in self.score_indicators: tf_signal_sum += sar_sig.astype(float).fillna(0)
                signals[f'sar_signal_{tf}'] = sar_sig.astype('category')

            # 将当前时间周期的信号总和加权计入总分
            signals['total_score'] += tf_signal_sum * weights[tf]


        # 2. 根据总分映射到初步信号级别
        # 使用 np.select 实现更高效的条件赋值
        conditions = [
            signals['total_score'] >= thresholds['strong_buy'],
            (signals['total_score'] >= thresholds['buy']) & (signals['total_score'] < thresholds['strong_buy']),
            (signals['total_score'] < thresholds['sell']) & (signals['total_score'] >= thresholds['strong_sell']),
            signals['total_score'] < thresholds['strong_sell'],
            signals['total_score'].isna() # 处理 NaN
        ]
        choices = [
            SIGNAL_STRONG_BUY,
            SIGNAL_BUY,
            SIGNAL_SELL,
            SIGNAL_STRONG_SELL,
            SIGNAL_NONE # 对应 NaN
        ]
        preliminary_signal = pd.Series(np.select(conditions, choices, default=SIGNAL_HOLD), index=data.index)

        # 3. 应用量能确认逻辑 (如果启用)
        final_signal = self._confirm_with_volume(preliminary_signal, data)

        # 存储中间结果，方便调试分析
        self.intermediate_data = signals # 包含所有初步信号(category)和总分(float)

        # 4. 返回最终信号，使用 category 类型优化内存
        # category 类型可以处理 SIGNAL_NONE (np.nan)
        return final_signal.astype('category')

    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回包含中间计算结果（如各指标信号、总分）的 DataFrame，用于分析。"""
        return self.intermediate_data

