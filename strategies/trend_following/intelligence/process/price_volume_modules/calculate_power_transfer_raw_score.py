# strategies\trend_following\intelligence\process\price_volume_modules\calculate_power_transfer_raw_score.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import jit, float64, int64
from typing import Dict, List, Optional, Any, Tuple

from strategies.trend_following.utils import get_param_value
@jit(nopython=True)
def _numba_power_activation(x, alpha=0.01, gain=1.5):
    """V32.0 · 非对称动力学激活算子：强化极端正向爆发，抑制负向噪音"""
    res = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            # 正向信号线性增益，捕捉“夺权”爆发力
            res[i] = x[i] * gain
        else:
            # 负向信号渗漏抑制，保留风险底色
            res[i] = x[i] * alpha
    return res

class CalculatePowerTransferRawScore:
    """
    PROCESS_META_POWER_TRANSFER_RAW_SCORE
    """
    def __init__(self, is_debug, probe_ts):
        self.is_debug = is_debug
        self.probe_ts = probe_ts
        pass

    def _calculate_dynamic_impulse_norm(self, activated_impulse: pd.Series, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V54.0 · 冲量单位标准化：引入增益系数以平衡量纲级差"""
        is_debug, probe_ts, _ = self.is_debug, self.probe_ts, method_name
        # 1. 获取流动性质量因子
        current_amount = raw['amount_D'].replace(0, 1e-9)
        avg_amount = current_amount.rolling(21).mean().fillna(current_amount)
        rel_intensity = (current_amount / avg_amount).clip(0.5, 5.0)
        # 2. 计算标准化系数
        mass_factor = np.log10(current_amount + 1).clip(5, 12)
        norm_factor = mass_factor * np.sqrt(rel_intensity)
        # 3. 执行标准化转换 (引入 * 5.0 增益)
        normalized_impulse = (activated_impulse * 5.0) / norm_factor
        # 4. 探针
        if is_debug and probe_ts in df_index:
            print(f"\n[冲量标准化探针 V54 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    质量基准: {mass_factor.loc[probe_ts]:.4f}, 相对强度: {rel_intensity.loc[probe_ts]:.4f}")
            print(f"    原始激活冲量: {activated_impulse.loc[probe_ts]:.4f}")
            print(f"    >>> 标准化后冲量: {normalized_impulse.loc[probe_ts]:.6f} (含 5.0x 增益)")
        return normalized_impulse.astype(np.float32)

    def _calculate_limit_price_compensation(self, norm_impulse: pd.Series, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V35.0 · 全维度价格限制补偿：包含跌停负压映射与动态权重标定"""
        is_debug, probe_ts, _ = self.is_debug, self.probe_ts, method_name
        # 1. 动态权重标定：基于流一致性 (Consistency Weight)
        # 一致性越高，越相信协同攻击信号；一致性低，则增加压力释放指数的权重
        cons_w = raw['flow_consistency_D'].clip(0.3, 0.9)
        # 2. 涨停补偿 (虚拟正压)
        is_limit_up = (raw['close_D'] >= raw['up_limit_D'] * 0.999)
        up_virtual_impulse = (raw['SMART_MONEY_HM_COORDINATED_ATTACK_D'] * cons_w + raw['pressure_release_index_D'] * (1 - cons_w))
        # 3. 跌停补偿 (虚拟负压)
        is_limit_down = (raw['close_D'] <= raw['down_limit_D'] * 1.001)
        # 跌停时使用 net_amount_rate 的负向 JERK 作为核心负压分量
        down_virtual_impulse = (raw['net_amount_rate_D'].clip(-1, 0) * 1.5 + raw['JERK_3_net_amount_rate_D'].clip(-1, 0))
        # 4. 执行多路补偿方案
        compensated = np.where(is_limit_up, np.maximum(norm_impulse, up_virtual_impulse), norm_impulse)
        compensated = np.where(is_limit_down, np.minimum(compensated, down_virtual_impulse), compensated)
        res = pd.Series(compensated, index=df_index)
        if is_debug and probe_ts in df_index:
            if is_limit_up.loc[probe_ts]:
                print(f"\n[补偿动态标针 - 涨停 @ {probe_ts.strftime('%Y-%m-%d')}]")
                print(f"    流一致性: {cons_w.loc[probe_ts]:.4f}, 补偿权重(攻击): {cons_w.loc[probe_ts]:.4f}")
                print(f"    >>> 涨停虚拟正压: {up_virtual_impulse.loc[probe_ts]:.6f}")
            elif is_limit_down.loc[probe_ts]:
                print(f"\n[补偿动态标针 - 跌停 @ {probe_ts.strftime('%Y-%m-%d')}]")
                print(f"    净流入率(负向): {raw['net_amount_rate_D'].loc[probe_ts]:.4f}, JERK_3负压: {raw['JERK_3_net_amount_rate_D'].loc[probe_ts]:.4f}")
                print(f"    >>> 跌停虚拟负压: {down_virtual_impulse.loc[probe_ts]:.6f}")
        return res.astype(np.float32)

    def _calculate_auction_prediction(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V54.0 · T+1 竞价预判：引入对数压缩防止高强度尾盘导致的分值饱和"""
        is_debug, probe_ts, _ = self.is_debug, self.probe_ts, method_name
        # 1. 冲击项
        impulse_term = (raw['JERK_3_net_amount_rate_D'] * 0.6 + raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'] * 0.4)
        activated_imp = _numba_power_activation(impulse_term.values, gain=1.2)
        # 2. 强度项 (引入 log1p 压缩)
        intensity_log = np.log1p(raw['closing_flow_intensity_D'])
        auction_strength = intensity_log * (1.0 + activated_imp)
        auction_prediction_raw = auction_strength * (raw['T1_PREMIUM_EXPECTATION_D'] + 0.5)
        # 3. 空间映射 (tanh/20.0 * 2.5)
        auc_prediction = np.tanh(auction_prediction_raw / 20.0) * 2.5
        res = pd.Series(auc_prediction, index=df_index)
        if is_debug and probe_ts in df_index:
            print(f"\n[竞价量纲修正探针 V54 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    尾盘强度(Log): {intensity_log.loc[probe_ts]:.4f}, 冲击增益: {activated_imp[df_index.get_loc(probe_ts)]:.4f}")
            print(f"    原始竞价分: {auction_prediction_raw.loc[probe_ts]:.4f}, 压缩后分值: {res.loc[probe_ts]:.4f}")
        return res.astype(np.float32)







