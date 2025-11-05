# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class StructuralIntelligence:
    """
    【V3.0 · 三大公理重构版】
    - 核心升级: 废弃旧的复杂四支柱模型，引入基于结构本质的“趋势形态、多周期协同、结构稳定性”三大公理。
                使引擎更聚焦、逻辑更清晰、信号更纯粹。
    """
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化结构情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def diagnose_structural_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 · 矛与盾融合版】结构情报分析总指挥
        - 核心升级: 指挥旗下三大公理诊断引擎。其中“结构稳定性”公理已升级，
                      融合了EMA的短期收敛性(矛)与MA的长期支撑性(盾)，实现攻守兼备的结构评估。
        """
        print("启动【V4.0 · 矛与盾融合版】结构情报分析...")
        all_states = {}
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("结构情报引擎已在配置中禁用，跳过。")
            return {}
        
        norm_window = get_param_value(p_conf.get('norm_window'), 55)

        # --- 步骤一: 诊断三大公理 ---
        print("工序一: 正在诊断三大结构公理...")
        axiom_trend_form = self._diagnose_axiom_trend_form(df, norm_window)
        axiom_mtf_cohesion = self._diagnose_axiom_mtf_cohesion(df, norm_window, axiom_trend_form)
        axiom_stability = self._diagnose_axiom_stability(df, norm_window)

        all_states['SCORE_STRUCT_AXIOM_TREND_FORM'] = axiom_trend_form
        all_states['SCORE_STRUCT_AXIOM_MTF_COHESION'] = axiom_mtf_cohesion
        all_states['SCORE_STRUCT_AXIOM_STABILITY'] = axiom_stability

        # --- 步骤二: 融合三大公理，合成终极信号 ---
        print("工序二: 正在合成终极结构共振信号...")
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'trend_form': 0.5, 'mtf_cohesion': 0.3, 'stability': 0.2
        })
        
        bipolar_health = (
            axiom_trend_form * axiom_weights['trend_form'] +
            axiom_mtf_cohesion * axiom_weights['mtf_cohesion'] +
            axiom_stability * axiom_weights['stability']
        ).clip(-1, 1)

        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_health)
        all_states['SCORE_STRUCTURE_BULLISH_RESONANCE'] = bullish_resonance
        all_states['SCORE_STRUCTURE_BEARISH_RESONANCE'] = bearish_resonance

        print(f"【V4.0 · 矛与盾融合版】结构情报分析完成。")
        return all_states

    def _diagnose_axiom_trend_form(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """【V1.0 · 新增】结构公理一：诊断“趋势形态”"""
        print("    -- [结构公理一: 趋势形态] 正在诊断日线级别均线形态...")
        ma_periods = [5, 13, 21, 55]
        required_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in required_cols):
            logger.warning("诊断趋势形态失败：缺少必要的EMA列。")
            return pd.Series(0.0, index=df.index)

        # 证据1: 排列健康度 (多头/空头排列)
        bull_alignment = np.mean([(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).astype(float) for i in range(len(ma_periods) - 1)], axis=0)
        bear_alignment = np.mean([(df[f'EMA_{ma_periods[i]}_D'] < df[f'EMA_{ma_periods[i+1]}_D']).astype(float) for i in range(len(ma_periods) - 1)], axis=0)
        
        # 证据2: 斜率健康度 (均线方向)
        slope_cols = [f'SLOPE_5_EMA_{p}_D' for p in ma_periods if f'SLOPE_5_EMA_{p}_D' in df.columns]
        if not slope_cols:
            return pd.Series(0.0, index=df.index)
        bull_velocity = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in slope_cols], axis=0)
        bear_velocity = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in slope_cols], axis=0)

        # 融合看涨分和看跌分
        bull_score = bull_alignment * bull_velocity
        bear_score = bear_alignment * bear_velocity

        # 生成双极性分数
        trend_form_score = pd.Series(bull_score - bear_score, index=df.index).clip(-1, 1)
        print(f"    -- [结构公理一: 趋势形态] 诊断完成，最新分值: {trend_form_score.iloc[-1]:.4f}")
        return trend_form_score.astype(np.float32)

    def _diagnose_axiom_mtf_cohesion(self, df: pd.DataFrame, norm_window: int, daily_trend_form_score: pd.Series) -> pd.Series:
        """【V1.0 · 新增】结构公理二：诊断“多周期协同”"""
        print("    -- [结构公理二: 多周期协同] 正在诊断日线与周线结构的协同性...")
        ma_periods_w = [5, 13, 21, 55]
        required_cols_w = [f'EMA_{p}_W' for p in ma_periods_w]
        if not all(col in df.columns for col in required_cols_w):
            logger.warning("诊断多周期协同失败：缺少必要的周线EMA列，将仅使用日线结构。")
            return pd.Series(0.0, index=df.index)

        # 计算周线级别的趋势形态分 (逻辑与日线相同)
        bull_alignment_w = np.mean([(df[f'EMA_{ma_periods_w[i]}_W'] > df[f'EMA_{ma_periods_w[i+1]}_W']).astype(float) for i in range(len(ma_periods_w) - 1)], axis=0)
        bear_alignment_w = np.mean([(df[f'EMA_{ma_periods_w[i]}_W'] < df[f'EMA_{ma_periods_w[i+1]}_W']).astype(float) for i in range(len(ma_periods_w) - 1)], axis=0)
        slope_cols_w = [f'SLOPE_5_EMA_{p}_W' for p in ma_periods_w if f'SLOPE_5_EMA_{p}_W' in df.columns]
        if not slope_cols_w:
            return pd.Series(0.0, index=df.index)
        bull_velocity_w = np.mean([normalize_score(df[col], df.index, norm_window, ascending=True).values for col in slope_cols_w], axis=0)
        bear_velocity_w = np.mean([normalize_score(df[col], df.index, norm_window, ascending=False).values for col in slope_cols_w], axis=0)
        weekly_trend_form_score = pd.Series(bull_alignment_w * bull_velocity_w - bear_alignment_w * bear_velocity_w, index=df.index).clip(-1, 1)

        # 协同分 = 日线分 * 周线分
        # 如果方向一致（同正或同负），结果为正；如果方向相反，结果为负。
        cohesion_score = (daily_trend_form_score * weekly_trend_form_score).fillna(0)
        print(f"    -- [结构公理二: 多周期协同] 诊断完成，最新分值: {cohesion_score.iloc[-1]:.4f}")
        return cohesion_score.astype(np.float32)

    def _diagnose_axiom_stability(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.2 · 物理直观重构版】结构公理三：诊断“结构稳定性”
        - 核心重构: 废除对间接且脆弱的 'MA_CONV_CV_SHORT_D' 的依赖。将“能量积蓄度”的评估核心完全聚焦于更直观、更稳健的布林带宽度(BBW)指标。
        - 诊断维度:
          1. 能量积蓄度 (Energy Accumulation): 直接使用BBW的收缩程度。
          2. 基石支撑度 (Foundation Support): 价格是否站稳在关键长期MA(55, 144)之上。
          3. 长期趋势健康度 (Long-term Trend Health): 关键长期MA自身的斜率方向。
        """
        # [代码修改开始]
        print("    -- [结构公理三: 结构稳定性] 正在诊断均线收敛度、长期支撑与趋势健康度...")
        # --- 证据1: 能量积蓄度 (Energy Accumulation) ---
        bbw_col = 'BBW_21_2.0_D'
        if bbw_col not in df.columns:
            print(f"诊断结构稳定性失败：缺少核心列 '{bbw_col}'。")
            return pd.Series(0.0, index=df.index)
        
        # BBW越小，波动率越低，能量积蓄度越高。因此使用 1 - normalize_score。
        energy_accumulation_score = 1 - normalize_score(df[bbw_col], df.index, norm_window, ascending=True)
        energy_accumulation_score = energy_accumulation_score.fillna(0.5)

        # --- 证据2 & 3: 基石支撑度 & 长期趋势健康度 (MA as Shield) ---
        long_term_ma_periods = [55, 144] # 使用MA而非EMA作为成本线
        required_ma_cols = [f'MA_{p}_D' for p in long_term_ma_periods]
        required_slope_cols = [f'SLOPE_5_MA_{p}_D' for p in long_term_ma_periods]
        
        if not all(col in df.columns for col in required_ma_cols + required_slope_cols):
            print("诊断结构稳定性失败：缺少必要的长期MA或其斜率列，长期结构评估将跳过。")
            foundation_health_score = pd.Series(0.5, index=df.index)
        else:
            # 证据2: 基石支撑度 - 价格是否在关键MA之上
            support_scores = []
            for p in long_term_ma_periods:
                # 价格高于MA越多，分数越高
                support_score = normalize_score(df['close_D'] - df[f'MA_{p}_D'], df.index, norm_window).clip(0, 1)
                support_scores.append(support_score)
            foundation_support_score = np.mean(support_scores, axis=0)
            
            # 证据3: 长期趋势健康度 - 关键MA的斜率是否为正
            health_scores = []
            for p in long_term_ma_periods:
                # 斜率越大，分数越高
                health_score = normalize_score(df[f'SLOPE_5_MA_{p}_D'], df.index, norm_window).clip(0, 1)
                health_scores.append(health_score)
            long_term_trend_health_score = np.mean(health_scores, axis=0)
            
            # 融合基石支撑与长期趋势健康度
            foundation_health_score = (pd.Series(foundation_support_score, index=df.index) * pd.Series(long_term_trend_health_score, index=df.index)).pow(0.5)

        # --- 最终融合 ---
        # 融合能量积蓄度与长期结构健康度
        # 当长期结构健康时(foundation_health_score高)，能量的积蓄(energy_accumulation_score高)才是有效的看涨信号
        raw_stability_score = (energy_accumulation_score * foundation_health_score).fillna(0.5)
        
        # 转换为双极性分数：高稳定性为正，低稳定性为负
        stability_score = (raw_stability_score * 2 - 1).clip(-1, 1)
        print(f"    -- [结构公理三: 结构稳定性] 诊断完成，最新分值: {stability_score.iloc[-1]:.4f}")
        return stability_score.astype(np.float32)
        # [代码修改结束]
