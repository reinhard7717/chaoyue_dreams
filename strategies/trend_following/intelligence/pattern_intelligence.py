# 文件: strategies/trend_following/intelligence/pattern_intelligence.py

import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class PatternIntelligence:
    """
    【V6.0 · 三大公理重构版】形态智能引擎
    - 核心升级: 废弃旧的复杂模型，引入基于形态演化本质的“背离、反转、突破”三大公理。
                使引擎更聚焦、逻辑更清晰、信号更纯粹。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.0 · 超级原子信号版】形态分析总指挥
        - 核心升级: 指挥旗下三大公理诊断引擎，全面转向消费新一代的、经过深度提炼的
                      “超级原子信号”，包括“日内VWAP偏离度积分指数”、“对手盘衰竭指数”
                      和“突破质量分”，极大提升了形态识别的准确性和可靠性。
        """
        print("启动【V7.0 · 超级原子信号版】形态分析...")
        all_states = {}
        p_conf = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("形态智能引擎已在配置中禁用，跳过。")
            return {}
        
        norm_window = get_param_value(p_conf.get('norm_window'), 60)

        # --- 步骤一: 诊断三大公理 ---
        print("工序一: 正在诊断三大形态公理...")
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        axiom_reversal = self._diagnose_axiom_reversal(df, norm_window)
        axiom_breakout = self._diagnose_axiom_breakout(df, norm_window)

        all_states['SCORE_PATTERN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_PATTERN_AXIOM_REVERSAL'] = axiom_reversal
        all_states['SCORE_PATTERN_AXIOM_BREAKOUT'] = axiom_breakout

        # --- 步骤二: 融合三大公理，合成终极信号 ---
        print("工序二: 正在合成终极形态共振信号...")
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'divergence': 0.4, 'reversal': 0.4, 'breakout': 0.2
        })
        
        bipolar_health = (
            axiom_divergence * axiom_weights['divergence'] +
            axiom_reversal * axiom_weights['reversal'] +
            axiom_breakout * axiom_weights['breakout']
        ).clip(-1, 1)

        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_health)
        all_states['SCORE_PATTERN_BULLISH_RESONANCE'] = bullish_resonance
        all_states['SCORE_PATTERN_BEARISH_RESONANCE'] = bearish_resonance

        print(f"【V7.0 · 超级原子信号版】形态分析完成。最终看涨共振分: {bullish_resonance.iloc[-1]:.4f}, 看跌共振分: {bearish_resonance.iloc[-1]:.4f}")
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.0 · 超级原子信号版】形态公理一：诊断“背离”
        - 核心升级: 废弃对RSI/MACD的间接依赖，直接使用更可靠的“日内VWAP偏离度积分指数”
                      (`intraday_vwap_div_index_D`)作为核心动量信号，以捕捉更真实的价势背离。
        """
        print("    -- [形态公理一: 背离] 正在使用 '日内VWAP偏离度积分指数' 进行诊断...")
        price_slope = df.get('SLOPE_13_close_D', pd.Series(0, index=df.index))
        # 获取新的超级原子信号的斜率
        momentum_slope = df.get('SLOPE_13_intraday_vwap_div_index_D', pd.Series(0, index=df.index))

        # 看涨背离：价格下跌，但日内真实力量（VWAP偏离积分）在增强
        bullish_divergence_strength = ((price_slope < 0) & (momentum_slope > 0)).astype(float)

        # 看跌背离：价格上涨，但日内真实力量在减弱
        bearish_divergence_strength = ((price_slope > 0) & (momentum_slope < 0)).astype(float)

        # 融合成双极性分数
        raw_divergence_score = bullish_divergence_strength - bearish_divergence_strength
        
        # 使用双极归一化进行最终裁决
        divergence_score = normalize_to_bipolar(raw_divergence_score, df.index, window=norm_window)
        print(f"    -- [形态公理一: 背离] 诊断完成，最新分值: {divergence_score.iloc[-1]:.4f}")
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_reversal(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.1 · 探针植入版】形态公理二：诊断“反转”
        - 核心升级: 废弃对RSI/MACD的间接推断，直接使用“对手盘衰竭指数”
                      (`counterparty_exhaustion_index_D`)作为核心判断依据。
        - 调试升级: 植入探针，检查原始输入信号的数据质量。
        """
        print("    -- [形态公理二: 反转] 正在使用 '对手盘衰竭指数' 进行诊断...")
        # [代码修改开始]
        # 直接获取衡量对手盘是否力竭的超级原子信号
        raw_reversal_score = df.get('counterparty_exhaustion_index_D', pd.Series(0, index=df.index))
        
        # --- [探针] ---
        print("        [探针-反转公理] 检查 'counterparty_exhaustion_index_D' 数据:")
        if 'counterparty_exhaustion_index_D' in df.columns and not df['counterparty_exhaustion_index_D'].isnull().all():
            print(f"        -> 数据存在且不全为NaN。最后5个值:\n{df['counterparty_exhaustion_index_D'].iloc[-5:].to_string()}")
            # 使用.describe()提供更全面的统计信息，并用.to_string()保证格式
            print(f"        -> 统计信息:\n{df['counterparty_exhaustion_index_D'].describe().to_string()}")
        else:
            print("        -> 数据不存在或全为NaN/0，使用默认值0。")
        # --- [探针结束] ---
        
        # 使用双极归一化进行最终裁决，使其与其他公理分值范围对齐
        reversal_score = normalize_to_bipolar(raw_reversal_score, df.index, window=norm_window)
        print(f"    -- [形态公理二: 反转] 诊断完成，最新分值: {reversal_score.iloc[-1]:.4f}")
        return reversal_score.astype(np.float32)
        # [代码修改结束]

    def _diagnose_axiom_breakout(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.1 · 探针植入版】形态公理三：诊断“突破”
        - 核心升级: 废弃对BBW等能量指标的间接推断，直接使用经过“质量审查”的
                      “突破质量分”(`breakout_quality_score_D`)作为核心判断依据。
        - 调试升级: 植入探针，检查突破方向、突破质量分和最终原始分。
        """
        print("    -- [形态公理三: 突破] 正在使用 '突破质量分' 进行诊断...")
        # [代码修改开始]
        # 1. 突破方向判断 (逻辑保留)
        is_breakout_up = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        is_breakout_down = (df['close_D'] < df.get('dynamic_consolidation_low_D', -np.inf)).astype(float)
        breakout_direction = is_breakout_up - is_breakout_down

        # 2. 获取突破质量分 (新的超级原子信号)
        breakout_quality = df.get('breakout_quality_score_D', pd.Series(0, index=df.index))

        # --- [探针] ---
        print("        [探针-突破公理] 检查中间变量:")
        print(f"        -> 最后5个突破方向(breakout_direction):\n{breakout_direction.iloc[-5:].to_string()}")
        if 'breakout_quality_score_D' in df.columns and not df['breakout_quality_score_D'].isnull().all():
            print(f"        -> 'breakout_quality_score_D' 数据存在。最后5个值:\n{breakout_quality.iloc[-5:].to_string()}")
            print(f"        -> 突破质量分统计信息:\n{breakout_quality.describe().to_string()}")
        else:
            print("        -> 'breakout_quality_score_D' 数据不存在或全为NaN/0。")
        # --- [探针结束] ---

        # 3. 融合：突破方向 * 突破质量
        raw_breakout_score = breakout_direction * breakout_quality
        
        # --- [探针] ---
        print(f"        [探针-突破公理] 检查融合后的原始分:")
        print(f"        -> 最后5个原始突破分(raw_breakout_score):\n{raw_breakout_score.iloc[-5:].to_string()}")
        # --- [探针结束] ---
        
        # 使用双极归一化进行最终裁决
        breakout_score = normalize_to_bipolar(raw_breakout_score, df.index, window=norm_window)
        print(f"    -- [形态公理三: 突破] 诊断完成，最新分值: {breakout_score.iloc[-1]:.4f}")
        return breakout_score.astype(np.float32)
        # [代码修改结束]
