# strategies/macd_rsi_kdj_boll_enhanced_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
# 从 base 导入基类 (假设它不强制要求返回离散信号)
# 如果 BaseStrategy 或后续处理强制要求离散信号，需要在最后一步将分数映射回去，
# 或者修改调用端的逻辑以处理分数。这里我们让策略本身返回分数。
from .base import BaseStrategy #, SIGNAL_STRONG_BUY, SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_STRONG_SELL, SIGNAL_NONE

# 假设 core.constants 定义了 TimeLevel 枚举 (虽然这里不直接用，但保持一致性)
# from core.constants import TimeLevel

logger = logging.getLogger("strategy")

class MacdRsiKdjBollEnhancedStrategy(BaseStrategy):
    """
    多时间周期 MACD+RSI+KDJ+BOLL+量能+趋势 组合增强策略 (百分制评分)。
    策略逻辑:
    1. 结合 5, 15, 30, 60 四个时间周期的 MACD, RSI, KDJ, BOLL, CCI, MFI, ROC, DMI, SAR 指标。
    2. 为每个指标在每个时间周期生成初步的 0-100 分数 (100=最强买, 0=最强卖, 50=中性)。
    3. 根据时间周期为分数分配权重 (短周期权重低，长周期权重高)。
    4. 计算加权总分，并归一化到 0-100 范围。
    5. **增加量能确认**: 使用 15 分钟周期的 Amount MA, CMF, OBV 趋势对初步分数进行调整。
    6. **增加顶背离检查**: 检测到顶背离时显著降低分数。
    7. 返回最终的 0-100 分数。
    """
    strategy_name = "MACD_RSI_KDJ_BOLL_Enhanced_MultiTimeframe_Score"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略参数。
        """
        default_params = {
            'rsi_period': 12,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_extreme_oversold': 20,
            'rsi_extreme_overbought': 80,
            'kdj_period_k': 9,
            'kdj_period_d': 3,
            'kdj_period_j': 3,
            'kdj_oversold': 20,
            'kdj_overbought': 80,
            'boll_period': 20,
            'boll_std_dev': 2,
            'macd_fast': 10,
            'macd_slow': 26,
            'macd_signal': 9,
            'cci_period': 14,
            'cci_threshold': 100,
            'cci_extreme_threshold': 200,
            'mfi_period': 14,
            'mfi_oversold': 20,
            'mfi_overbought': 80,
            'mfi_extreme_oversold': 10,
            'mfi_extreme_overbought': 90,
            'roc_period': 12,
            'dmi_period': 14,
            'adx_threshold': 20,
            'adx_strong_threshold': 30,
            'sar_step': 0.02,
            'sar_max': 0.2,
            'amount_ma_period': 20,
            'obv_ma_period': 10,
            'cmf_period': 20,
            'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2},
            'volume_confirmation': True,
            'volume_tf': '15',
            'volume_confirm_boost': 1.1,
            'volume_fail_penalty': 0.8,
            'divergence_penalty': 0.3,
            'check_bearish_divergence': True,
            'divergence_price_period': 5,
            'divergence_threshold_cmf': -0.05,
            'divergence_threshold_mfi': 40,
        }
        # 合并默认参数和传入参数
        merged_params = {**default_params, **(params or {})}
        # 深层合并字典参数
        if params and 'weights' in params:
            merged_params['weights'] = {**default_params['weights'], **params['weights']}

        # 先定义实例属性，这些属性可能在 _validate_params 中用到
        self.timeframes = ['5', '15', '30', '60']
        self.score_indicators = ['macd', 'rsi', 'kdj', 'boll', 'cci', 'mfi', 'roc', 'dmi', 'sar']
        self.volume_confirm_indicators = ['amount_ma', 'cmf', 'obv']
        self.intermediate_data: Optional[pd.DataFrame] = None

        self._num_score_indicators = len(self.score_indicators)

        # 现在调用父类的 __init__，它会调用 _validate_params
        super().__init__(merged_params)

    def _validate_params(self):
        """验证参数"""
        super()._validate_params()
        if not isinstance(self.params['weights'], dict) or \
           not all(tf in self.params['weights'] for tf in self.timeframes) or \
           abs(sum(self.params['weights'].values()) - 1.0) > 1e-6:
            raise ValueError("参数 'weights' 必须是包含所有时间周期 ('5', '15', '30', '60') 且总和为 1.0 的字典")
        if self.params['volume_confirmation'] and self.params['volume_tf'] not in self.timeframes:
             raise ValueError(f"参数 'volume_tf' ({self.params['volume_tf']}) 必须是 {self.timeframes} 中的一个")
        if self._num_score_indicators == 0:
             raise ValueError("参数 'score_indicators' 不能为空列表")
        # TODO: 添加新指标参数的验证 (如 extreme thresholds)

    def get_required_columns(self) -> List[str]:
        """返回策略运行所需的 DataFrame 列名。"""
        # (与原代码基本一致，因为需要的原始指标数据不变)
        required = []
        price_tf = self.params.get('volume_tf', '15')
        required.append(f'close_{price_tf}')
        required.append(f'high_{price_tf}') # 用于计算价格新高和 CMF/MFI
        required.append(f'low_{price_tf}') # 用于计算 CMF/MFI
        required.append(f'volume_{price_tf}') # CMF/MFI 需要原始 volume

        # 主操作周期的收盘价 (用于 BOLL, SAR, KC 等比较)
        compare_price_tf = self.params.get('volume_tf', '15')
        required.append(f'close_{compare_price_tf}') # 主要用于比较的价格序列
        required.append(f'high_{compare_price_tf}') # SAR 可能需要
        required.append(f'low_{compare_price_tf}')  # SAR 可能需要

        # 参与评分的指标
        for tf in self.timeframes:
            required.extend([f'diff_{tf}', f'dea_{tf}', f'macd_{tf}'])
            required.append(f'rsi_{tf}')
            required.extend([f'k_{tf}', f'd_{tf}', f'j_{tf}'])
            required.extend([f'upper_{tf}', f'mid_{tf}', f'lower_{tf}'])
            required.append(f'cci_{tf}')
            required.append(f'mfi_{tf}')
            required.append(f'roc_{tf}')
            required.extend([f'pdi_{tf}', f'mdi_{tf}', f'adx_{tf}'])
            required.append(f'sar_{tf}')

        # 量能确认指标 (仅需特定时间周期)
        vol_tf = self.params['volume_tf']
        required.append(f'amount_{vol_tf}') # 需要原始成交额
        required.append(f'amount_ma_{vol_tf}') # 需要计算好的成交额均线
        required.append(f'cmf_{vol_tf}')
        required.append(f'obv_{vol_tf}')
        required.append(f'obv_ma_{vol_tf}') # 假设 OBV 的均线列名
        # MFI 用于背离检查
        if self.params.get('check_bearish_divergence'):
            required.append(f'mfi_{vol_tf}')

        # 去重后返回
        return list(set(required))

    # --- 单个指标评分函数 (0-100分) ---

    def _get_macd_score(self, diff: pd.Series, dea: pd.Series, macd: pd.Series) -> pd.Series:
        """MACD 评分 (0-100)"""
        score = pd.Series(50.0, index=diff.index) # 默认中性分
        # 金叉信号 (柱子由负变正)
        buy_cross = (macd.shift(1) < 0) & (macd > 0)
        score.loc[buy_cross] = 75.0
        # 死叉信号 (柱子由正变负)
        sell_cross = (macd.shift(1) > 0) & (macd < 0)
        score.loc[sell_cross] = 25.0
        # 零轴上方运行，柱子扩大 (趋势加强) - 可选，增加复杂性
        # score.loc[(macd > 0) & (dea > 0) & (macd > macd.shift(1))] = 65.0 # 示例
        # 零轴下方运行，柱子缩小 (趋势减弱) - 可选
        # score.loc[(macd < 0) & (dea < 0) & (macd > macd.shift(1))] = 40.0 # 示例
        return score

    def _get_rsi_score(self, rsi: pd.Series) -> pd.Series:
        """RSI 评分 (0-100)"""
        score = pd.Series(50.0, index=rsi.index)
        os = self.params['rsi_oversold']
        ob = self.params['rsi_overbought']
        ext_os = self.params['rsi_extreme_oversold']
        ext_ob = self.params['rsi_extreme_overbought']

        # 极度超卖区
        score.loc[rsi < ext_os] = 95.0
        # 超卖区
        score.loc[(rsi >= ext_os) & (rsi < os)] = 85.0
        # 上穿超卖线 (买入信号)
        buy_signal = (rsi.shift(1) < os) & (rsi >= os)
        score.loc[buy_signal] = 75.0

        # 极度超买区
        score.loc[rsi > ext_ob] = 5.0
        # 超买区
        score.loc[(rsi <= ext_ob) & (rsi > ob)] = 15.0
        # 下穿超买线 (卖出信号)
        sell_signal = (rsi.shift(1) > ob) & (rsi <= ob)
        score.loc[sell_signal] = 25.0

        # 中性区域 (os 到 ob 之间，非信号点) - 可以细化，例如靠近50给50分
        score.loc[(rsi >= os) & (rsi <= ob) & (~buy_signal) & (~sell_signal)] = 50.0 # 默认就是50，这里确保覆盖

        return score

    def _get_kdj_score(self, k: pd.Series, d: pd.Series, j: pd.Series) -> pd.Series:
        """KDJ 评分 (0-100)"""
        score = pd.Series(50.0, index=k.index)
        os = self.params['kdj_oversold']
        ob = self.params['kdj_overbought']

        # J 值超卖
        score.loc[j < os] = 85.0
        # J 值极度超卖 (例如 < 10)
        score.loc[j < 10] = 95.0
        # 金叉发生在非超买区
        buy_cross = (k.shift(1) < d.shift(1)) & (k > d) & (j < ob)
        score.loc[buy_cross] = 75.0

        # J 值超买
        score.loc[j > ob] = 15.0
        # J 值极度超买 (例如 > 90)
        score.loc[j > 90] = 5.0
        # 死叉发生在非超卖区
        sell_cross = (k.shift(1) > d.shift(1)) & (k < d) & (j > os)
        score.loc[sell_cross] = 25.0

        # 覆盖交叉信号优先级高于 J 值区域信号
        score.loc[buy_cross] = 75.0
        score.loc[sell_cross] = 25.0

        return score

    def _get_boll_score(self, close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
        """BOLL 评分 (0-100)"""
        score = pd.Series(50.0, index=close.index)

        # 价格低于下轨 (极度超卖)
        score.loc[close < lower] = 90.0
        # 价格触及下轨反弹 (上穿下轨)
        buy_support = (close.shift(1) < lower.shift(1)) & (close >= lower)
        score.loc[buy_support] = 80.0

        # 价格高于上轨 (极度超买)
        score.loc[close > upper] = 10.0
        # 价格触及上轨回落 (下穿上轨)
        sell_pressure = (close.shift(1) > upper.shift(1)) & (close <= upper)
        score.loc[sell_pressure] = 20.0

        # 考虑中轨突破
        buy_mid_cross = (close.shift(1) < mid.shift(1)) & (close > mid)
        score.loc[buy_mid_cross] = 65.0 # 上穿中轨给买入倾向分
        sell_mid_cross = (close.shift(1) > mid.shift(1)) & (close < mid)
        score.loc[sell_mid_cross] = 35.0 # 下穿中轨给卖出倾向分

        # 在上下轨之间，靠近中轨时分数趋近50
        # 可以根据价格在 (mid, upper) 或 (lower, mid) 的相对位置进行线性插值，但会增加复杂性
        # 简化处理：非信号点，且在中轨上方给略高于50分，下方给略低于50分
        is_signal = buy_support | sell_pressure | buy_mid_cross | sell_mid_cross
        score.loc[(~is_signal) & (close > mid) & (close < upper)] = 55.0 # 中轨上方
        score.loc[(~is_signal) & (close < mid) & (close > lower)] = 45.0 # 中轨下方

        return score

    def _get_cci_score(self, cci: pd.Series) -> pd.Series:
        """CCI 评分 (0-100)"""
        score = pd.Series(50.0, index=cci.index)
        threshold = self.params['cci_threshold']
        ext_threshold = self.params['cci_extreme_threshold']

        # 极度超卖区
        score.loc[cci < -ext_threshold] = 95.0
        # 超卖区
        score.loc[(cci >= -ext_threshold) & (cci < -threshold)] = 85.0
        # 上穿 -threshold (买入信号)
        buy_signal = (cci.shift(1) < -threshold) & (cci >= -threshold)
        score.loc[buy_signal] = 75.0

        # 极度超买区
        score.loc[cci > ext_threshold] = 5.0
        # 超买区
        score.loc[(cci <= ext_threshold) & (cci > threshold)] = 15.0
        # 下穿 +threshold (卖出信号)
        sell_signal = (cci.shift(1) > threshold) & (cci <= threshold)
        score.loc[sell_signal] = 25.0

        # 中性区域
        score.loc[(cci >= -threshold) & (cci <= threshold) & (~buy_signal) & (~sell_signal)] = 50.0

        return score

    def _get_mfi_score(self, mfi: pd.Series) -> pd.Series:
        """MFI 评分 (0-100)"""
        score = pd.Series(50.0, index=mfi.index)
        os = self.params['mfi_oversold']
        ob = self.params['mfi_overbought']
        ext_os = self.params['mfi_extreme_oversold']
        ext_ob = self.params['mfi_extreme_overbought']

        # 极度超卖区
        score.loc[mfi < ext_os] = 95.0
        # 超卖区
        score.loc[(mfi >= ext_os) & (mfi < os)] = 85.0
        # 上穿超卖线 (买入信号)
        buy_signal = (mfi.shift(1) < os) & (mfi >= os)
        score.loc[buy_signal] = 75.0

        # 极度超买区
        score.loc[mfi > ext_ob] = 5.0
        # 超买区
        score.loc[(mfi <= ext_ob) & (mfi > ob)] = 15.0
        # 下穿超买线 (卖出信号)
        sell_signal = (mfi.shift(1) > ob) & (mfi <= ob)
        score.loc[sell_signal] = 25.0

        # 中性区域
        score.loc[(mfi >= os) & (mfi <= ob) & (~buy_signal) & (~sell_signal)] = 50.0

        return score

    def _get_roc_score(self, roc: pd.Series) -> pd.Series:
        """ROC 评分 (0-100)"""
        score = pd.Series(50.0, index=roc.index)
        # ROC 由负转正 (买入信号)
        buy_signal = (roc.shift(1) < 0) & (roc > 0)
        score.loc[buy_signal] = 70.0 # 动量转强
        # ROC 由正转负 (卖出信号)
        sell_signal = (roc.shift(1) > 0) & (roc < 0)
        score.loc[sell_signal] = 30.0 # 动量转弱

        # ROC > 0 且持续上升 (多头动能)
        score.loc[(roc > 0) & (roc > roc.shift(1)) & (~buy_signal)] = 60.0
        # ROC < 0 且持续下降 (空头动能)
        score.loc[(roc < 0) & (roc < roc.shift(1)) & (~sell_signal)] = 40.0

        return score

    def _get_dmi_score(self, pdi: pd.Series, mdi: pd.Series, adx: pd.Series) -> pd.Series:
        """DMI 评分 (0-100)"""
        score = pd.Series(50.0, index=pdi.index)
        adx_th = self.params['adx_threshold']
        adx_strong_th = self.params['adx_strong_threshold']

        # PDI 上穿 MDI (买入信号)
        buy_cross = (pdi.shift(1) < mdi.shift(1)) & (pdi > mdi)
        # MDI 上穿 PDI (卖出信号)
        sell_cross = (mdi.shift(1) < pdi.shift(1)) & (mdi > pdi)

        # 根据 ADX 强度调整信号分数
        # 买入信号 + ADX 确认趋势
        score.loc[buy_cross & (adx > adx_th)] = 75.0
        score.loc[buy_cross & (adx > adx_strong_th)] = 85.0 # 强趋势下信号更可靠
        # 卖出信号 + ADX 确认趋势
        score.loc[sell_cross & (adx > adx_th)] = 25.0
        score.loc[sell_cross & (adx > adx_strong_th)] = 15.0 # 强趋势下信号更可靠

        # PDI > MDI 但非交叉点 (多头占优)
        is_bullish = (pdi > mdi) & (~buy_cross)
        score.loc[is_bullish & (adx > adx_strong_th)] = 65.0 # 强趋势多头
        score.loc[is_bullish & (adx <= adx_strong_th) & (adx > adx_th)] = 60.0 # 普通趋势多头
        score.loc[is_bullish & (adx <= adx_th)] = 55.0 # 弱趋势/无趋势多头

        # MDI > PDI 但非交叉点 (空头占优)
        is_bearish = (mdi > pdi) & (~sell_cross)
        score.loc[is_bearish & (adx > adx_strong_th)] = 35.0 # 强趋势空头
        score.loc[is_bearish & (adx <= adx_strong_th) & (adx > adx_th)] = 40.0 # 普通趋势空头
        score.loc[is_bearish & (adx <= adx_th)] = 45.0 # 弱趋势/无趋势空头

        return score

    def _get_sar_score(self, close: pd.Series, sar: pd.Series) -> pd.Series:
        """SAR 评分 (0-100)"""
        score = pd.Series(50.0, index=close.index)
        # SAR 向上翻转 (买入信号)
        buy_signal = (sar.shift(1) > close.shift(1)) & (sar < close)
        score.loc[buy_signal] = 75.0
        # SAR 向下翻转 (卖出信号)
        sell_signal = (sar.shift(1) < close.shift(1)) & (sar > close)
        score.loc[sell_signal] = 25.0

        # 价格在 SAR 上方 (多头趋势)
        score.loc[(close > sar) & (~buy_signal)] = 60.0
        # 价格在 SAR 下方 (空头趋势)
        score.loc[(close < sar) & (~sell_signal)] = 40.0

        return score

    # --- 量能调整逻辑 ---
    def _adjust_score_with_volume(self, preliminary_score: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        使用量能指标调整初步的 0-100 分数。
        :param preliminary_score: 初步生成的 0-100 分数 Series
        :param data: 包含量能指标数据的 DataFrame
        :return: 调整后的 0-100 分数 Series
        """
        if not self.params['volume_confirmation']:
            return preliminary_score

        vol_tf = self.params['volume_tf']
        adjusted_score = preliminary_score.copy()

        # 获取价格和量能数据
        close = data.get(f'close_{vol_tf}')
        high = data.get(f'high_{vol_tf}')
        amount = data.get(f'amount_{vol_tf}')
        amount_ma = data.get(f'amount_ma_{vol_tf}')
        cmf = data.get(f'cmf_{vol_tf}')
        obv = data.get(f'obv_{vol_tf}')
        obv_ma = data.get(f'obv_ma_{vol_tf}')
        mfi = data.get(f'mfi_{vol_tf}') # 如果使用 MFI

        required_cols_exist = all(col is not None for col in [close, high, amount, amount_ma, cmf, obv, obv_ma])
        if not required_cols_exist:
             logger.warning(f"缺少用于量能确认的列 (时间周期 {vol_tf})。跳过量能调整。")
             return preliminary_score

        # 1. 量能确认/不确认调整 (只调整偏离中性区 50 的分数)
        boost = self.params['volume_confirm_boost']
        penalty = self.params['volume_fail_penalty']

        # 买入确认条件
        buy_volume_confirm = pd.Series(False, index=data.index)
        if amount is not None and amount_ma is not None:
            buy_volume_confirm |= (amount > amount_ma)
        if cmf is not None:
            buy_volume_confirm |= (cmf > 0)
        if obv is not None and obv_ma is not None:
            buy_volume_confirm |= (obv > obv_ma)

        # 对 > 50 的分数进行调整
        is_bullish_score = adjusted_score > 50
        # 量能确认，提升分数
        adjusted_score.loc[is_bullish_score & buy_volume_confirm] *= boost
        # 量能不确认，降低分数
        adjusted_score.loc[is_bullish_score & (~buy_volume_confirm)] *= penalty

        # 2. 检查量能顶背离并惩罚分数
        if self.params.get('check_bearish_divergence'):
            price_period = self.params['divergence_price_period']
            cmf_threshold = self.params['divergence_threshold_cmf']
            mfi_threshold = self.params['divergence_threshold_mfi']
            divergence_penalty_factor = self.params['divergence_penalty']

            if len(high) >= price_period:
                is_price_high = high == high.rolling(window=price_period, min_periods=price_period).max()
            else:
                is_price_high = pd.Series(False, index=high.index)

            is_price_rising = close > close.shift(1)
            is_cmf_weak = cmf < cmf_threshold
            is_obv_weak = obv < obv_ma
            is_mfi_weak = mfi < mfi_threshold if mfi is not None else pd.Series(False, index=data.index) # MFI 弱势可选

            # 顶背离条件: 价格新高 + 量能弱 (CMF 或 OBV 或 MFI)
            bearish_divergence = is_price_high & is_price_rising & (is_cmf_weak | is_obv_weak | is_mfi_weak)

            # 如果检测到顶背离，且分数 > 50，则显著降低分数
            adjusted_score.loc[bearish_divergence & is_bullish_score] *= divergence_penalty_factor

        # 3. 确保分数在 0-100 范围内
        adjusted_score = adjusted_score.clip(0, 100)

        return adjusted_score

    # --- 主评分逻辑 ---
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        根据输入的包含多时间周期指标的 DataFrame 生成最终的 0-100 评分。
        """
        if self._num_score_indicators == 0:
             logger.error("没有配置用于评分的指标 ('score_indicators' 为空)。")
             return pd.Series(50.0, index=data.index) # 返回中性分

        # 初始化用于存储中间分数和总分的 DataFrame
        scores = pd.DataFrame(index=data.index)
        scores['total_weighted_score'] = 0.0 # 初始化加权总分列

        weights = self.params['weights']
        vol_tf = self.params['volume_tf']
        price_tf = vol_tf # 使用量能确认周期的价格

        # 确保价格列存在
        close_price_col = f'close_{price_tf}'
        if close_price_col not in data.columns:
            raise ValueError(f"在输入数据中找不到所需的列: '{close_price_col}'.")
        close_price = data[close_price_col]
        # high_price = data.get(f'high_{price_tf}') # SAR 可能需要，但在 _get_sar_score 中已处理
        # low_price = data.get(f'low_{price_tf}')

        # 1. 为每个时间周期计算各指标的得分并加权
        for tf in self.timeframes:
            tf_weighted_score_sum = pd.Series(0.0, index=data.index) # 当前时间周期的指标得分总和 (未加权)
            indicator_count_in_tf = 0 # 计算该时间周期实际有多少指标参与评分

            # --- MACD ---
            diff_col, dea_col, macd_col = f'diff_{tf}', f'dea_{tf}', f'macd_{tf}'
            if 'macd' in self.score_indicators and all(c in data for c in [diff_col, dea_col, macd_col]):
                macd_score = self._get_macd_score(data[diff_col], data[dea_col], data[macd_col])
                tf_weighted_score_sum += macd_score.fillna(50.0) # 用中性分填充 NaN
                scores[f'macd_score_{tf}'] = macd_score # 存储中间结果
                indicator_count_in_tf += 1

            # --- RSI ---
            rsi_col = f'rsi_{tf}'
            if 'rsi' in self.score_indicators and rsi_col in data:
                rsi_score = self._get_rsi_score(data[rsi_col])
                tf_weighted_score_sum += rsi_score.fillna(50.0)
                scores[f'rsi_score_{tf}'] = rsi_score
                indicator_count_in_tf += 1

            # --- KDJ ---
            k_col, d_col, j_col = f'k_{tf}', f'd_{tf}', f'j_{tf}'
            if 'kdj' in self.score_indicators and all(c in data for c in [k_col, d_col, j_col]):
                kdj_score = self._get_kdj_score(data[k_col], data[d_col], data[j_col])
                tf_weighted_score_sum += kdj_score.fillna(50.0)
                scores[f'kdj_score_{tf}'] = kdj_score
                indicator_count_in_tf += 1

            # --- BOLL ---
            upper_col, mid_col, lower_col = f'upper_{tf}', f'mid_{tf}', f'lower_{tf}'
            if 'boll' in self.score_indicators and all(c in data for c in [upper_col, mid_col, lower_col]):
                boll_score = self._get_boll_score(close_price, data[upper_col], data[mid_col], data[lower_col])
                tf_weighted_score_sum += boll_score.fillna(50.0)
                scores[f'boll_score_{tf}'] = boll_score
                indicator_count_in_tf += 1

            # --- CCI ---
            cci_col = f'cci_{tf}'
            if 'cci' in self.score_indicators and cci_col in data:
                cci_score = self._get_cci_score(data[cci_col])
                tf_weighted_score_sum += cci_score.fillna(50.0)
                scores[f'cci_score_{tf}'] = cci_score
                indicator_count_in_tf += 1

            # --- MFI ---
            mfi_col = f'mfi_{tf}'
            if 'mfi' in self.score_indicators and mfi_col in data:
                mfi_score = self._get_mfi_score(data[mfi_col])
                tf_weighted_score_sum += mfi_score.fillna(50.0)
                scores[f'mfi_score_{tf}'] = mfi_score
                indicator_count_in_tf += 1

            # --- ROC ---
            roc_col = f'roc_{tf}'
            if 'roc' in self.score_indicators and roc_col in data:
                roc_score = self._get_roc_score(data[roc_col])
                tf_weighted_score_sum += roc_score.fillna(50.0)
                scores[f'roc_score_{tf}'] = roc_score
                indicator_count_in_tf += 1

            # --- DMI ---
            pdi_col, mdi_col, adx_col = f'pdi_{tf}', f'mdi_{tf}', f'adx_{tf}'
            if 'dmi' in self.score_indicators and all(c in data for c in [pdi_col, mdi_col, adx_col]):
                dmi_score = self._get_dmi_score(data[pdi_col], data[mdi_col], data[adx_col])
                tf_weighted_score_sum += dmi_score.fillna(50.0)
                scores[f'dmi_score_{tf}'] = dmi_score
                indicator_count_in_tf += 1

            # --- SAR ---
            sar_col = f'sar_{tf}'
            if 'sar' in self.score_indicators and sar_col in data:
                sar_score = self._get_sar_score(close_price, data[sar_col])
                tf_weighted_score_sum += sar_score.fillna(50.0)
                scores[f'sar_score_{tf}'] = sar_score
                indicator_count_in_tf += 1

            # 将当前时间周期的 *平均* 指标得分 (归一化到0-100) 按权重加到总分中
            if indicator_count_in_tf > 0:
                 # 计算该时间框架下所有参与评分指标的平均分
                 avg_tf_score = tf_weighted_score_sum / indicator_count_in_tf
                 scores['total_weighted_score'] += avg_tf_score * weights[tf]
            else:
                 # 如果这个时间周期没有任何指标数据，则贡献中性分 50 * 权重
                 scores['total_weighted_score'] += 50.0 * weights[tf]


        # 2. 归一化总加权分数到 0-100
        # 因为每个时间周期的 avg_tf_score 已经是 0-100，并且权重和为 1，
        # 所以 total_weighted_score 理论上已经是 0-100 范围了。
        # 但为了安全起见，还是 clip 一下。
        preliminary_final_score = scores['total_weighted_score'].clip(0, 100)
        scores['preliminary_score'] = preliminary_final_score # 存储量能调整前的分数

        # 3. 应用量能调整逻辑
        final_score = self._adjust_score_with_volume(preliminary_final_score, data)

        # 存储中间结果，方便调试分析
        self.intermediate_data = scores # 包含所有初步分数和最终分数

        # 4. 返回最终分数 (0-100)
        final_data = final_score.round(2) # 保留两位小数
        logger.info(f"{self.strategy_name}.最终分数: 计算完成") # 避免打印整个 Series
        # logger.debug(f"Sample final scores:\n{final_data.tail()}") # Debug 时可以查看尾部数据
        return final_data

    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回包含中间计算结果（如各指标分数、总分）的 DataFrame，用于分析。"""
        return self.intermediate_data

