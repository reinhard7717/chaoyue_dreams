# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.trend_following.utils import get_params_block, get_param_value

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _calculate_normalized_score(self, series: pd.Series, window: int, ascending: bool = True) -> pd.Series:
        """
        【V12.0 性能优化版】计算一个系列在滚动窗口内的归一化得分 (0-1)。
        """
        if series is None or series.isnull().all():
            return pd.Series(0.5, index=self.strategy.df.index if series is None else series.index)
        return series.rolling(
            window=window, 
            min_periods=int(window * 0.2)
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5)

    def _get_consensus_metric(self, df: pd.DataFrame, available_cols: set, base_name: str) -> pd.Series: # [新增] 辅助函数
        """
        【新增 V12.0】获取三方数据源的共识指标。
        - 核心: 动态查找 'ths', 'dc', 'tushare' 三个来源的指标列，并计算其均值。
                能优雅地处理部分数据源缺失的情况。
        :param df: 主DataFrame。
        :param available_cols: 可用列的集合，用于高效查找。
        :param base_name: 指标的基础名称模板，用 '{source}' 作为占位符。
        :return: 共识指标的Series，如果所有来源都缺失则返回None。
        """
        sources = ['ths', 'dc', 'tushare']
        found_cols = [base_name.format(source=source) for source in sources if base_name.format(source=source) in available_cols]
        if not found_cols:
            return None
        return df[found_cols].mean(axis=1)

    def _diagnose_fund_flow_dynamics(self, df: pd.DataFrame) -> pd.DataFrame: # [新增] 核心诊断引擎
        """
        【新增 V12.0 - 动态共振与反转引擎】
        - 核心: 基于多时间维度(5, 13, 21, 55日)的静态、斜率、加速度指标，进行深度交叉验证。
        - 产出: 生成4类核心动态信号得分：上升共振、下跌共振、底部反转、顶部反转。
        """
        print("            -> [资金流动态引擎 V12.0] 启动交叉验证...")
        available_cols = set(df.columns)
        periods = [5, 13, 21, 55]
        norm_window = 120

        # --- 步骤 1: 预计算所有周期的共识指标得分 ---
        metrics = {}
        for p in periods:
            # 静态存量 (Static Value)
            static_base = f"net_d{p}_net_amount_fund_flow_{{source}}_D"
            metrics[f'static_{p}'] = self._get_consensus_metric(df, available_cols, static_base)
            
            # 动态斜率 (Slope)
            slope_base = f"SLOPE_{p}_net_d{p}_net_amount_fund_flow_{{source}}_D"
            metrics[f'slope_{p}'] = self._get_consensus_metric(df, available_cols, slope_base)
            
            # 动态加速度 (Acceleration)
            accel_base = f"ACCEL_{p}_net_d{p}_net_amount_fund_flow_{{source}}_D"
            metrics[f'accel_{p}'] = self._get_consensus_metric(df, available_cols, accel_base)

        # --- 步骤 2: 生成上升共振信号 (Upward Resonance) ---
        # 上升共振: 资金存量、趋势、加速度均呈现健康的多头状态。
        s5 = self._calculate_normalized_score(metrics['static_5'], norm_window)
        sl5 = self._calculate_normalized_score(metrics['slope_5'], norm_window)
        a5 = self._calculate_normalized_score(metrics['accel_5'], norm_window)
        
        s21 = self._calculate_normalized_score(metrics['static_21'], norm_window)
        sl21 = self._calculate_normalized_score(metrics['slope_21'], norm_window)
        
        s55 = self._calculate_normalized_score(metrics['static_55'], norm_window)
        sl55 = self._calculate_normalized_score(metrics['slope_55'], norm_window)

        # 低置信度: 短期(5日)共振
        df['FF_SCORE_RESONANCE_UP_LOW'] = s5 * sl5 * a5
        # 中置信度: 中短期(5, 21日)共振
        df['FF_SCORE_RESONANCE_UP_MID'] = df['FF_SCORE_RESONANCE_UP_LOW'] * s21 * sl21
        # 高置信度: 全周期(5, 21, 55日)共振
        df['FF_SCORE_RESONANCE_UP_HIGH'] = df['FF_SCORE_RESONANCE_UP_MID'] * s55 * sl55
        print("               - 上升共振信号已生成 (低/中/高置信度)")

        # --- 步骤 3: 生成下跌共振信号 (Downward Resonance) ---
        # 下跌共振: 资金存量、趋势、加速度均呈现明确的空头状态。
        s5_neg = self._calculate_normalized_score(metrics['static_5'], norm_window, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics['slope_5'], norm_window, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics['accel_5'], norm_window, ascending=False)
        
        s21_neg = self._calculate_normalized_score(metrics['static_21'], norm_window, ascending=False)
        sl21_neg = self._calculate_normalized_score(metrics['slope_21'], norm_window, ascending=False)
        
        s55_neg = self._calculate_normalized_score(metrics['static_55'], norm_window, ascending=False)
        sl55_neg = self._calculate_normalized_score(metrics['slope_55'], norm_window, ascending=False)

        df['FF_SCORE_RESONANCE_DOWN_LOW'] = s5_neg * sl5_neg * a5_neg
        df['FF_SCORE_RESONANCE_DOWN_MID'] = df['FF_SCORE_RESONANCE_DOWN_LOW'] * s21_neg * sl21_neg
        df['FF_SCORE_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_RESONANCE_DOWN_MID'] * s55_neg * sl55_neg
        print("               - 下跌共振信号已生成 (低/中/高置信度)")

        # --- 步骤 4: 生成底部反转信号 (Bottom Reversal) ---
        # 底部反转: 长期趋势(55日)仍在探底，但中期趋势(21日)开始企稳，短期(5日)动能已强力反转。
        long_term_bottoming = self._calculate_normalized_score(metrics['slope_55'], norm_window, ascending=False)
        mid_term_stabilizing = self._calculate_normalized_score(metrics['slope_21'], norm_window)
        short_term_reversing = self._calculate_normalized_score(metrics['slope_5'], norm_window)
        short_term_accelerating = self._calculate_normalized_score(metrics['accel_5'], norm_window)

        # 中置信度: 中期企稳 + 短期反转
        df['FF_SCORE_REVERSAL_BOTTOM_MID'] = mid_term_stabilizing * short_term_reversing
        # 高置信度: 长期探底背景 + 中期企稳 + 短期强力加速反转
        df['FF_SCORE_REVERSAL_BOTTOM_HIGH'] = long_term_bottoming * mid_term_stabilizing * short_term_reversing * short_term_accelerating
        print("               - 底部反转信号已生成 (中/高置信度)")

        # --- 步骤 5: 生成顶部反转信号 (Top Reversal) ---
        # 顶部反转: 长期趋势(55日)仍在向上，但中期趋势(21日)开始走平，短期(5日)动能已显著恶化。
        long_term_topping = self._calculate_normalized_score(metrics['slope_55'], norm_window)
        mid_term_stalling = self._calculate_normalized_score(metrics['slope_21'], norm_window, ascending=False)
        short_term_diverging = self._calculate_normalized_score(metrics['slope_5'], norm_window, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics['accel_5'], norm_window, ascending=False)

        # 中置信度: 中期失速 + 短期背离
        df['FF_SCORE_REVERSAL_TOP_MID'] = mid_term_stalling * short_term_diverging
        # 高置信度: 长期顶部背景 + 中期失速 + 短期加速恶化
        df['FF_SCORE_REVERSAL_TOP_HIGH'] = long_term_topping * mid_term_stalling * short_term_diverging * short_term_decelerating
        print("               - 顶部反转信号已生成 (中/高置信度)")
        
        return df

    def _diagnose_capital_structure_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V15.0 - 资金结构动态共振引擎】
        - 核心: 模仿聚合流引擎，对主力资金（超大单+大单）的买入行为进行多时间维度(5, 13, 21日)的静态、斜率、加速度交叉验证。
        - 产出: 生成多置信度的主力资金上升共振与下跌共振（主力撤退）信号。
        """
        print("            -> [资金结构引擎 V15.0] 启动交叉验证...")
        available_cols = set(df.columns)
        norm_window = 120
        periods = [5, 13, 21]
        metrics = {}

        # --- 步骤 1: 预计算主力资金（超大单+大单）共识指标 ---
        main_force_metrics = {
            'static': (df.get('buy_elg_amount_rate_fund_flow_dc_D', 0) + df.get('buy_lg_amount_rate_fund_flow_ths_D', 0)) / 2,
            'slope_5': (df.get('SLOPE_5_buy_elg_amount_rate_fund_flow_dc_D', 0) + df.get('SLOPE_5_buy_lg_amount_rate_fund_flow_ths_D', 0)) / 2,
            'slope_13': (df.get('SLOPE_13_buy_elg_amount_rate_fund_flow_dc_D', 0) + df.get('SLOPE_13_buy_lg_amount_rate_fund_flow_ths_D', 0)) / 2,
            'slope_21': (df.get('SLOPE_21_buy_elg_amount_rate_fund_flow_dc_D', 0) + df.get('SLOPE_21_buy_lg_amount_rate_fund_flow_ths_D', 0)) / 2,
            'accel_5': (df.get('ACCEL_5_buy_elg_amount_rate_fund_flow_dc_D', 0) + df.get('ACCEL_5_buy_lg_amount_rate_fund_flow_ths_D', 0)) / 2,
        }

        # --- 步骤 2: 生成主力资金上升共振信号 ---
        mf_s5 = self._calculate_normalized_score(main_force_metrics['static'], norm_window)
        mf_sl5 = self._calculate_normalized_score(main_force_metrics['slope_5'], norm_window)
        mf_a5 = self._calculate_normalized_score(main_force_metrics['accel_5'], norm_window)
        
        mf_s13 = self._calculate_normalized_score(main_force_metrics['static'], norm_window) # 静态值共用
        mf_sl13 = self._calculate_normalized_score(main_force_metrics['slope_13'], norm_window)
        
        mf_s21 = self._calculate_normalized_score(main_force_metrics['static'], norm_window) # 静态值共用
        mf_sl21 = self._calculate_normalized_score(main_force_metrics['slope_21'], norm_window)

        df['FF_SCORE_STRUCTURE_RESONANCE_UP_LOW'] = mf_s5 * mf_sl5 * mf_a5
        df['FF_SCORE_STRUCTURE_RESONANCE_UP_MID'] = df['FF_SCORE_STRUCTURE_RESONANCE_UP_LOW'] * mf_s13 * mf_sl13
        df['FF_SCORE_STRUCTURE_RESONANCE_UP_HIGH'] = df['FF_SCORE_STRUCTURE_RESONANCE_UP_MID'] * mf_s21 * mf_sl21
        print("               - [结构]上升共振信号已生成 (低/中/高置信度)")

        # --- 步骤 3: 生成主力资金下跌共振信号 (主力撤退) ---
        mf_s5_neg = self._calculate_normalized_score(main_force_metrics['static'], norm_window, ascending=False)
        mf_sl5_neg = self._calculate_normalized_score(main_force_metrics['slope_5'], norm_window, ascending=False)
        mf_a5_neg = self._calculate_normalized_score(main_force_metrics['accel_5'], norm_window, ascending=False)

        mf_s13_neg = self._calculate_normalized_score(main_force_metrics['static'], norm_window, ascending=False)
        mf_sl13_neg = self._calculate_normalized_score(main_force_metrics['slope_13'], norm_window, ascending=False)

        mf_s21_neg = self._calculate_normalized_score(main_force_metrics['static'], norm_window, ascending=False)
        mf_sl21_neg = self._calculate_normalized_score(main_force_metrics['slope_21'], norm_window, ascending=False)

        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_LOW'] = mf_s5_neg * mf_sl5_neg * mf_a5_neg
        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_MID'] = df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_LOW'] * mf_s13_neg * mf_sl13_neg
        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_MID'] * mf_s21_neg * mf_sl21_neg
        print("               - [结构]下跌共振(主力撤退)信号已生成 (低/中/高置信度)")

        return df

    def _diagnose_synergy_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【新增 V14.0 - 协同诊断引擎】
        - 核心: 将聚合流分析与结构流分析的结果进行交叉验证，生成更高置信度的协同信号。
        - 产出: 强化共振、强化反转、诱多风险、聪明钱抄底等高阶信号。
        """
        print("            -> [协同诊断引擎 V14.0] 启动...")
        available_cols = set(df.columns)

        # --- 协同信号1: 强化上升共振 (高置信度聚合流共振 + 结构共振) ---
        if {'FF_SCORE_RESONANCE_UP_HIGH', 'FF_SCORE_STRUCTURE_RESONANCE_UP'}.issubset(available_cols):
            df['FF_SCORE_SYNERGY_REINFORCED_UP'] = df['FF_SCORE_RESONANCE_UP_HIGH'] * df['FF_SCORE_STRUCTURE_RESONANCE_UP']
            print("               - [协同]强化上升共振信号已生成")

        # --- 协同信号2: 强化顶部反转风险 (高置信度聚合流反转 + 结构背离) ---
        if {'FF_SCORE_REVERSAL_TOP_HIGH', 'FF_SCORE_STRUCTURE_DIVERGENCE_RISK'}.issubset(available_cols):
            df['FF_SCORE_SYNERGY_REINFORCED_TOP_RISK'] = df['FF_SCORE_REVERSAL_TOP_HIGH'] * df['FF_SCORE_STRUCTURE_DIVERGENCE_RISK']
            print("               - [协同]强化顶部反转风险信号已生成")

        # --- 协同信号3: 诱多拉升风险 (中置信度聚合流上升 + 结构背离) ---
        if {'FF_SCORE_RESONANCE_UP_MID', 'FF_SCORE_STRUCTURE_DIVERGENCE_RISK'}.issubset(available_cols):
            df['FF_SCORE_SYNERGY_DECEPTIVE_RALLY_RISK'] = df['FF_SCORE_RESONANCE_UP_MID'] * df['FF_SCORE_STRUCTURE_DIVERGENCE_RISK']
            print("               - [协同]诱多拉升风险信号已生成")

        # --- 协同信号4: 聪明钱抄底 (高置信度聚合流反转 + 结构共振) ---
        if {'FF_SCORE_REVERSAL_BOTTOM_HIGH', 'FF_SCORE_STRUCTURE_RESONANCE_UP'}.issubset(available_cols):
            df['FF_SCORE_SYNERGY_SMART_BOTTOM_FISHING'] = df['FF_SCORE_REVERSAL_BOTTOM_HIGH'] * df['FF_SCORE_STRUCTURE_RESONANCE_UP']
            print("               - [协同]聪明钱抄底信号已生成")
            
        return df

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]: # [修改] 全面重构
        """
        【V12.0 - 动态共振与反转引擎】
        - 核心升级:
          1.  【架构重构】: 废弃旧的、离散的信号计算，转向调用统一的交叉验证引擎 `_diagnose_fund_flow_dynamics`。
          2.  【数值化输出】: 不再生成布尔信号，而是直接输出由诊断引擎生成的、所有以 'FF_SCORE_' 开头的数值化评分系列。
          3.  【数据驱动】: 假设所有需要的衍生指标（静态、斜率、加速度）均由数据层提供，符合最新架构原则。
        """
        print("        -> [资金流情报模块 V12.0] 启动...")
        states = {}
        p = get_params_block(self.strategy, 'fund_flow_params')
        if not get_param_value(p.get('enabled'), False):
            return states
        
        # --- 步骤一: 调用新的核心诊断引擎 ---
        df = self._diagnose_fund_flow_dynamics(df)

        # --- 步骤二: 调用资金结构诊断引擎 ---
        df = self._diagnose_capital_structure_dynamics(self, df)

        # --- 步骤三: 调用协同诊断引擎 ---
        df = self._diagnose_synergy_dynamics(df)

        # --- 步骤四: 收集所有生成的数值化评分 ---
        for col in df.columns:
            if col.startswith('FF_SCORE_'):
                states[col] = df[col]
        
        print(f"        -> [资金流情报模块 V12.0] 诊断完毕，生成了 {len(states)} 个数值化动态信号。")
        return states
















