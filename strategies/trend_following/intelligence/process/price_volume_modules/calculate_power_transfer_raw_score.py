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
    def __init__(self):
        pass

    def _setup_debug_info(self, df: pd.DataFrame, method_name: str) -> Tuple[bool, Optional[pd.Timestamp], Dict]:
        """
        设置调试信息，包括是否启用调试、探针日期和临时调试值字典。
        """
        is_debug_enabled_for_method = get_param_value(self.helper.debug_params.get('enabled'), False) and get_param_value(self.helper.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.helper.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.helper.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        _temp_debug_values = {}
        if is_debug_enabled_for_method and probe_ts:
            _temp_debug_values[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            _temp_debug_values[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价量动态..."] = ""
        return is_debug_enabled_for_method, probe_ts, _temp_debug_values


    def _calculate_dynamic_impulse_norm(self, activated_impulse: pd.Series, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V33.0 · 冲量单位标准化：基于 T 日成交金额的流动性能效折算"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 获取流动性质量因子 (Inertia Mass)
        # 使用成交金额及其 21 日均值衡量个股的“体量惯性”
        current_amount = raw['amount_D'].replace(0, 1e-9)
        avg_amount = current_amount.rolling(21).mean().fillna(current_amount)
        # 相对成交强度因子 (Relative Liquidity Intensity)
        rel_intensity = (current_amount / avg_amount).clip(0.5, 5.0)
        # 2. 计算标准化系数 (Normalization Coefficient)
        # 逻辑：体量越大，对冲量的吸收越强；我们使用对数基准来平衡大盘股与小盘股的差异
        # log10(amount_D) 提供了市值量级的基准
        mass_factor = np.log10(current_amount + 1).clip(5, 12) # A股金额通常在 1e6 到 1e11 之间
        # 3. 执行标准化转换
        # 公式：激活冲量 / (质量基准 * 相对强度开方)
        # 使用 sqrt(rel_intensity) 是为了在放量夺权时适当调低冲量得分，防止成交量伪造动能
        norm_factor = mass_factor * np.sqrt(rel_intensity)
        normalized_impulse = activated_impulse / norm_factor
        # 4. 深度探针：暴露流动性折算细节
        if is_debug and probe_ts in df_index:
            print(f"\n[冲量标准化探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"--- 1. 流动性原料 ---")
            print(f"    T日成交额: {current_amount.loc[probe_ts]:.2f}, 21日均值: {avg_amount.loc[probe_ts]:.2f}")
            print(f"    相对强度因子: {rel_intensity.loc[probe_ts]:.4f}")
            print(f"--- 2. 标准化计算节点 ---")
            print(f"    质量基准(MassFactor): {mass_factor.loc[probe_ts]:.4f}")
            print(f"    综合折算系数: {norm_factor.loc[probe_ts]:.4f}")
            print(f"--- 3. 结果对比 ---")
            print(f"    ReLU激活后原始冲量: {activated_impulse.loc[probe_ts]:.4f}")
            print(f"    单位流动性标准化冲量: {normalized_impulse.loc[probe_ts]:.6f}")
        return normalized_impulse.astype(np.float32)

    def _calculate_limit_price_compensation(self, norm_impulse: pd.Series, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V35.0 · 全维度价格限制补偿：包含跌停负压映射与动态权重标定"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
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
        """V34.0 · T+1 开盘竞价预判：基于尾盘动量外推模型"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 核心逻辑：尾盘强度 * (1 + 爆发冲量)
        # 捕捉 $T$ 日最后 30 分钟的资金抢筹是否具有持续性
        impulse_term = (raw['JERK_3_net_amount_rate_D'] * 0.6 + raw['ACCEL_5_SMART_MONEY_HM_NET_BUY_D'] * 0.4)
        auction_strength = raw['closing_flow_intensity_D'] * (1.0 + _numba_power_activation(impulse_term.values, gain=1.2))
        # 引入溢价预期因子
        auction_prediction = auction_strength * (raw['T1_PREMIUM_EXPECTATION_D'] + 0.5)
        if is_debug and probe_ts in df_index:
            print(f"\n[T+1 竞价预判探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    尾盘强度: {raw['closing_flow_intensity_D'].loc[probe_ts]:.4f}, 溢价预期: {raw['T1_PREMIUM_EXPECTATION_D'].loc[probe_ts]:.4f}")
            print(f"    冲击项(Impulse): {impulse_term.loc[probe_ts]:.4f}, 最终竞价分: {auction_prediction.loc[probe_ts]:.4f}")
        return auction_prediction.astype(np.float32)

    def _calculate_premium_reversal_risk(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V35.0 · 溢价回吐风险预判：基于情绪透支与换手率能效比模型"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 核心逻辑：若T日尾盘占比过高且处于情绪极端，T+1容易出现开盘脉冲后的衰竭
        # 换手率归一化因子 (假设换手率 > 15% 为高能耗区)
        exhaustion_factor = (raw['turnover_rate_f_D'] / 15.0).clip(0, 1.5)
        # 溢价回吐压力 = 尾盘占比 * 情绪极端因子 * 换手能耗
        reversal_pressure = raw['closing_flow_ratio_D'] * raw['IS_EMOTIONAL_EXTREME_D'].astype(float) * exhaustion_factor
        # 转化为风险调节系数 (0.6 代表 40% 的衰减，1.0 代表无衰减)
        risk_adjustment = (1.0 - reversal_pressure * 0.4).clip(0.6, 1.0)
        if is_debug and probe_ts in df_index:
            print(f"\n[溢价回吐风险探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    情绪极端: {raw['IS_EMOTIONAL_EXTREME_D'].loc[probe_ts]}, 自由换手: {raw['turnover_rate_f_D'].loc[probe_ts]:.2f}%")
            print(f"    尾盘占比: {raw['closing_flow_ratio_D'].loc[probe_ts]:.4f}, 能耗因子: {exhaustion_factor.loc[probe_ts]:.4f}")
            print(f"    >>> 最终风险调节系数: {risk_adjustment.loc[probe_ts]:.4f}")
        return risk_adjustment.astype(np.float32)

    def _calculate_intraday_decay_model(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V38.0 · T+1 日内衰减与修复模型：集成“分歧转一致”潜力识别"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        stability = raw['TURNOVER_STABILITY_INDEX_D'].fillna(0.5).clip(0, 1) [cite: 2]
        is_limit_up = (raw['close_D'] >= raw['up_limit_D'] * 0.999) [cite: 3]
        # 1. 基础衰减逻辑：封板占比高且稳定性差
        bad_board_mask = is_limit_up & (raw['closing_flow_ratio_D'] > 0.4) & (stability < 0.4) [cite: 2, 3]
        # 2. 核心修复逻辑：低位暴力换手识别 (分歧转一致潜力)
        # 若获利盘比例处于极低位 (winner_rate_D < 0.15) 且 换手极其充分 (稳定性低)
        # 代表主力可能在跌破成本后的重新收集 
        repair_potential = np.where((raw['winner_rate_D'] < 0.15) & (stability < 0.3), 1.5, 1.0) [cite: 4]
        # 3. 结构性惩罚
        decay_resistance = (0.6 + stability * 0.4) * np.where(bad_board_mask, 0.6, 1.0) * repair_potential
        res = pd.Series(decay_resistance, index=df_index).clip(0.3, 1.5)
        if is_debug and probe_ts in df_index:
            print(f"\n[烂板修复侦测探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    获利比例: {raw['winner_rate_D'].loc[probe_ts]:.4f}, 换手稳定性: {stability.loc[probe_ts]:.4f}")
            print(f"    检测修复系数: {repair_potential[df_index.get_loc(probe_ts)]:.2f} (1.5代表具备反转修复力)")
            print(f"    >>> 最终抗衰减系数: {res.loc[probe_ts]:.4f}")
        return res.astype(np.float32)

    def _calculate_sector_resonance_modifier(self, raw: Dict[str, pd.Series], df_index: pd.Index, method_name: str) -> pd.Series:
        """V38.0 · 板块效应共振算子：集成流一致性判定以解决热度滞后性"""
        is_debug, probe_ts, _ = self._setup_debug_info(pd.DataFrame(index=df_index), method_name)
        # 1. 板块动力学：热度斜率与加速度
        sector_impulse = (raw['SLOPE_5_THEME_HOTNESS_SCORE_D'] * 0.6 + raw['ACCEL_5_THEME_HOTNESS_SCORE_D'] * 0.4).fillna(0) [cite: 1]
        # 2. 持续性校验：行业排名加速度与板块流一致性
        # 只有在排名正在快速提升且板块内资金流向一致时，才确认热度非滞后 [cite: 2, 3]
        persistence_factor = np.where((raw['industry_rank_accel_D'] > 0) & (raw['flow_consistency_D'] > 0.6), 1.2, 0.8) [cite: 2, 3]
        # 3. 综合调节
        resonance_mod = (1.0 + _numba_power_activation(sector_impulse.values, gain=0.5)) * persistence_factor
        res = pd.Series(resonance_mod, index=df_index)
        if is_debug and probe_ts in df_index:
            print(f"\n[板块持久力探针 @ {probe_ts.strftime('%Y-%m-%d')}]")
            print(f"    热度冲量: {sector_impulse.loc[probe_ts]:.4f}, 行业排名加速度: {raw['industry_rank_accel_D'].loc[probe_ts]:.4f}")
            print(f"    持续性因子: {persistence_factor[df_index.get_loc(probe_ts)]:.2f}")
            print(f"    >>> 板块共振调节分: {res.loc[probe_ts]:.4f}")
        return res.clip(0.6, 1.8).astype(np.float32)









