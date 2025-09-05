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
            return pd.Series(0.5, index=self.strategy.df.index)
        return series.rolling(
            window=window, 
            min_periods=int(window * 0.2)
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5)

    def _diagnose_fund_flow_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
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
            metrics[f'static_{p}'] = df.get(f"net_flow_consensus_sum_{p}d_D")
            metrics[f'slope_{p}'] = df.get(f"SLOPE_{p}_net_flow_consensus_sum_{p}d_D")
            if p in [5, 13, 21]: # 仅计算模型中存在的加速度周期
                 metrics[f'accel_{p}'] = df.get(f"ACCEL_{p}_net_flow_consensus_D")
        # --- 步骤 2: 生成上升共振信号 (Upward Resonance) ---
        s5 = self._calculate_normalized_score(metrics['static_5'], norm_window)
        sl5 = self._calculate_normalized_score(metrics['slope_5'], norm_window)
        a5 = self._calculate_normalized_score(metrics['accel_5'], norm_window)
        s21 = self._calculate_normalized_score(metrics['static_21'], norm_window)
        sl21 = self._calculate_normalized_score(metrics['slope_21'], norm_window)
        s55 = self._calculate_normalized_score(metrics['static_55'], norm_window)
        sl55 = self._calculate_normalized_score(metrics['slope_55'], norm_window)
        df['FF_SCORE_RESONANCE_UP_LOW'] = s5 * sl5 * a5
        df['FF_SCORE_RESONANCE_UP_MID'] = df['FF_SCORE_RESONANCE_UP_LOW'] * s21 * sl21
        df['FF_SCORE_RESONANCE_UP_HIGH'] = df['FF_SCORE_RESONANCE_UP_MID'] * s55 * sl55
        print("               - 上升共振信号已生成 (低/中/高置信度)")
        # --- 步骤 3: 生成下跌共振信号 (Downward Resonance) ---
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
        long_term_bottoming = self._calculate_normalized_score(metrics['slope_55'], norm_window, ascending=False)
        mid_term_stabilizing = self._calculate_normalized_score(metrics['slope_21'], norm_window)
        short_term_reversing = self._calculate_normalized_score(metrics['slope_5'], norm_window)
        short_term_accelerating = self._calculate_normalized_score(metrics['accel_5'], norm_window)
        df['FF_SCORE_REVERSAL_BOTTOM_MID'] = mid_term_stabilizing * short_term_reversing
        df['FF_SCORE_REVERSAL_BOTTOM_HIGH'] = long_term_bottoming * mid_term_stabilizing * short_term_reversing * short_term_accelerating
        print("               - 底部反转信号已生成 (中/高置信度)")
        # --- 步骤 5: 生成顶部反转信号 (Top Reversal) ---
        long_term_topping = self._calculate_normalized_score(metrics['slope_55'], norm_window)
        mid_term_stalling = self._calculate_normalized_score(metrics['slope_21'], norm_window, ascending=False)
        short_term_diverging = self._calculate_normalized_score(metrics['slope_5'], norm_window, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics['accel_5'], norm_window, ascending=False)
        df['FF_SCORE_REVERSAL_TOP_MID'] = mid_term_stalling * short_term_diverging
        df['FF_SCORE_REVERSAL_TOP_HIGH'] = long_term_topping * mid_term_stalling * short_term_diverging * short_term_decelerating
        print("               - 顶部反转信号已生成 (中/高置信度)")
        return df

    # 整个 _diagnose_capital_structure_dynamics 方法块
    def _diagnose_capital_structure_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V16.0 - 主力资金共振与反转引擎】
        - 核心重构: 逻辑与聚合流引擎对齐，使用更可靠的指标组合。
            - 静态背景: 使用【累计】主力净流入 (`_sum_`)，代表主力资金的存量和仓位。
            - 趋势方向: 使用【累计】主力净流入的斜率，代表主力建仓/减仓的趋势。
            - 动能变化: 使用【每日】主力净流入的加速度，代表主力动作的瞬时变化率。
        - 新增信号: 增加了主力资金行为的顶部/底部反转信号。
        """
        print("            -> [资金结构引擎 V16.0] 启动交叉验证...") # 更新版本号
        periods = [5, 13, 21, 55]
        norm_window = 120
        # --- 步骤 1: 获取预计算好的主力资金共识指标 ---
        # 修复了复制粘贴错误，将指标从超大单(xl)修正为主力(main_force)
        metrics = {}
        for p in periods:
            metrics[f'static_{p}'] = df.get(f"main_force_net_flow_consensus_sum_{p}d_D")
            metrics[f'slope_{p}'] = df.get(f"SLOPE_{p}_main_force_net_flow_consensus_sum_{p}d_D")
            if p in [5, 13, 21]: # 仅计算模型中存在的加速度周期
                 metrics[f'accel_{p}'] = df.get(f"ACCEL_{p}_main_force_net_flow_consensus_D")
        # --- 步骤 2: 生成主力资金上升共振信号 ---
        s5 = self._calculate_normalized_score(metrics.get('static_5'), norm_window)
        sl5 = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        a5 = self._calculate_normalized_score(metrics.get('accel_5'), norm_window)
        s21 = self._calculate_normalized_score(metrics.get('static_21'), norm_window)
        sl21 = self._calculate_normalized_score(metrics.get('slope_21'), norm_window)
        s55 = self._calculate_normalized_score(metrics.get('static_55'), norm_window)
        sl55 = self._calculate_normalized_score(metrics.get('slope_55'), norm_window)
        df['FF_SCORE_STRUCTURE_RESONANCE_UP_LOW'] = s5 * sl5 * a5
        df['FF_SCORE_STRUCTURE_RESONANCE_UP_MID'] = df['FF_SCORE_STRUCTURE_RESONANCE_UP_LOW'] * s21 * sl21
        df['FF_SCORE_STRUCTURE_RESONANCE_UP_HIGH'] = df['FF_SCORE_STRUCTURE_RESONANCE_UP_MID'] * s55 * sl55
        print("               - [结构]上升共振信号已生成 (低/中/高置信度)")
        # --- 步骤 3: 生成主力资金下跌共振信号 (主力撤退) ---
        s5_neg = self._calculate_normalized_score(metrics.get('static_5'), norm_window, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, ascending=False)
        s21_neg = self._calculate_normalized_score(metrics.get('static_21'), norm_window, ascending=False)
        sl21_neg = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, ascending=False)
        s55_neg = self._calculate_normalized_score(metrics.get('static_55'), norm_window, ascending=False)
        sl55_neg = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, ascending=False)
        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_LOW'] = s5_neg * sl5_neg * a5_neg
        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_MID'] = df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_LOW'] * s21_neg * sl21_neg
        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_MID'] * s55_neg * sl55_neg
        print("               - [结构]下跌共振(主力撤退)信号已生成 (低/中/高置信度)")
        # --- 步骤 4: 生成主力资金底部反转信号 ---
        long_term_selling = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, ascending=False)
        mid_term_stabilizing = self._calculate_normalized_score(metrics.get('slope_21'), norm_window)
        short_term_reversing = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        short_term_accelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window)
        df['FF_SCORE_STRUCTURE_REVERSAL_BOTTOM_MID'] = mid_term_stabilizing * short_term_reversing
        df['FF_SCORE_STRUCTURE_REVERSAL_BOTTOM_HIGH'] = long_term_selling * mid_term_stabilizing * short_term_reversing * short_term_accelerating
        print("               - [结构]底部反转信号已生成 (中/高置信度)")
        # --- 步骤 5: 生成主力资金顶部反转信号 ---
        long_term_buying = self._calculate_normalized_score(metrics.get('slope_55'), norm_window)
        mid_term_stalling = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, ascending=False)
        short_term_diverging = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, ascending=False)
        df['FF_SCORE_STRUCTURE_REVERSAL_TOP_MID'] = mid_term_stalling * short_term_diverging
        df['FF_SCORE_STRUCTURE_REVERSAL_TOP_HIGH'] = long_term_buying * mid_term_stalling * short_term_diverging * short_term_decelerating
        print("               - [结构]顶部反转信号已生成 (中/高置信度)")
        return df

    def _diagnose_capital_conflict_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V19.0 - 主力散户分歧·多维交叉验证引擎】
        - 核心升级: 全面利用数据层提供的分歧度静态、斜率、加速度指标，构建共振与反转信号。
        - 信号体系:
            - 基础分歧: 当前主力与散户的对立状态。
            - 分歧共振: 确认主力吸筹或派发趋势的持续性与强度。
            - 分歧反转: 预警主力行为发生关键性逆转的顶部或底部。
        - 数据依赖: 依赖 `flow_divergence_mf_vs_retail_D` 及其 `SLOPE` 和 `accel` 衍生列。
        """
        print("            -> [资金冲突引擎 V19.0] 启动多维交叉验证...") # 更新版本号和描述
        norm_window = 120
        periods = [5, 13, 21] # 定义用于共振的核心周期
        # --- 步骤 1: 获取分歧度的静态、斜率、加速度全套指标 ---
        metrics = {
            'static': df.get('flow_divergence_mf_vs_retail_D'),
            'slope_5': df.get('SLOPE_5_flow_divergence_mf_vs_retail_D'),
            'slope_13': df.get('SLOPE_13_flow_divergence_mf_vs_retail_D'),
            'slope_21': df.get('SLOPE_21_flow_divergence_mf_vs_retail_D'),
            'accel_5': df.get('accel_5d_flow_divergence_mf_vs_retail_D'),
            'accel_13': df.get('accel_13d_flow_divergence_mf_vs_retail_D'),
            'accel_21': df.get('accel_21d_flow_divergence_mf_vs_retail_D'),
        }
        # --- 步骤 2: [保留并优化] 计算基础分歧信号 (当前状态) ---
        df['FF_SCORE_CONFLICT_MF_BUYS_RETAIL_SELLS'] = self._calculate_normalized_score(metrics['static'], norm_window)
        df['FF_SCORE_CONFLICT_MF_SELLS_RETAIL_BUYS'] = self._calculate_normalized_score(metrics['static'], norm_window, ascending=False)
        print("               - [冲突]基础分歧信号已生成 (当前状态)")
        # --- 步骤 3: 生成分歧共振信号 (趋势确认) ---
        sl5_pos = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        sl13_pos = self._calculate_normalized_score(metrics.get('slope_13'), norm_window)
        sl21_pos = self._calculate_normalized_score(metrics.get('slope_21'), norm_window)
        df['FF_SCORE_CONFLICT_RESONANCE_UP_LOW'] = sl5_pos
        df['FF_SCORE_CONFLICT_RESONANCE_UP_MID'] = sl5_pos * sl13_pos
        df['FF_SCORE_CONFLICT_RESONANCE_UP_HIGH'] = sl5_pos * sl13_pos * sl21_pos
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        sl13_neg = self._calculate_normalized_score(metrics.get('slope_13'), norm_window, ascending=False)
        sl21_neg = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, ascending=False)
        df['FF_SCORE_CONFLICT_RESONANCE_DOWN_LOW'] = sl5_neg
        df['FF_SCORE_CONFLICT_RESONANCE_DOWN_MID'] = sl5_neg * sl13_neg
        df['FF_SCORE_CONFLICT_RESONANCE_DOWN_HIGH'] = sl5_neg * sl13_neg * sl21_neg
        print("               - [冲突]分歧共振信号已生成 (上升/下跌趋势确认)")
        # --- 步骤 4: 生成分歧反转信号 (顶部/底部预警) ---
        static_high_score = df['FF_SCORE_CONFLICT_MF_BUYS_RETAIL_SELLS'] # 复用基础分歧信号
        slope_reversing_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        accel_reversing_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, ascending=False)
        df['FF_SCORE_CONFLICT_REVERSAL_TOP_HIGH'] = static_high_score * slope_reversing_neg * accel_reversing_neg
        static_low_score = df['FF_SCORE_CONFLICT_MF_SELLS_RETAIL_BUYS'] # 复用基础分歧信号
        slope_reversing_pos = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        accel_reversing_pos = self._calculate_normalized_score(metrics.get('accel_5'), norm_window)
        df['FF_SCORE_CONFLICT_REVERSAL_BOTTOM_HIGH'] = static_low_score * slope_reversing_pos * accel_reversing_pos
        print("               - [冲突]分歧反转信号已生成 (高置信度顶部/底部预警)")
        return df

    def _diagnose_cmf_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【新增 V15.0 - CMF动态诊断引擎】
        - 核心: 基于Chaikin Money Flow (CMF)指标的静态、斜率、加速度进行交叉验证。
        - 视角: 从成交量加权的角度衡量买卖压力及其动态变化。
        - 产出: 生成买卖压力的共振与反转信号。
        """
        print("            -> [CMF动态引擎 V1.0] 启动交叉验证...")
        norm_window = 120
        # --- 步骤 1: 获取CMF全套指标 ---
        metrics = {
            'static': df.get('CMF_21_D'),
            'slope_5': df.get('SLOPE_5_CMF_21_D'),
            'slope_21': df.get('SLOPE_21_CMF_21_D'),
            'accel_5': df.get('ACCEL_5_CMF_21_D'),
            'accel_21': df.get('ACCEL_21_CMF_21_D'),
        }
        # --- 步骤 2: 生成买压共振信号 (Upward Resonance) ---
        s_pos = self._calculate_normalized_score(metrics['static'], norm_window)
        sl5_pos = self._calculate_normalized_score(metrics['slope_5'], norm_window)
        sl21_pos = self._calculate_normalized_score(metrics['slope_21'], norm_window)
        a5_pos = self._calculate_normalized_score(metrics['accel_5'], norm_window)
        df['FF_SCORE_CMF_RESONANCE_UP_LOW'] = s_pos * sl5_pos * a5_pos
        df['FF_SCORE_CMF_RESONANCE_UP_HIGH'] = df['FF_SCORE_CMF_RESONANCE_UP_LOW'] * sl21_pos
        print("               - [CMF]买压共振信号已生成 (低/高置信度)")
        # --- 步骤 3: 生成卖压共振信号 (Downward Resonance) ---
        s_neg = self._calculate_normalized_score(metrics['static'], norm_window, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics['slope_5'], norm_window, ascending=False)
        sl21_neg = self._calculate_normalized_score(metrics['slope_21'], norm_window, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics['accel_5'], norm_window, ascending=False)
        df['FF_SCORE_CMF_RESONANCE_DOWN_LOW'] = s_neg * sl5_neg * a5_neg
        df['FF_SCORE_CMF_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_CMF_RESONANCE_DOWN_LOW'] * sl21_neg
        print("               - [CMF]卖压共振信号已生成 (低/高置信度)")
        # --- 步骤 4: 生成底部反转信号 (Bottom Reversal) ---
        df['FF_SCORE_CMF_REVERSAL_BOTTOM_HIGH'] = s_neg * sl5_pos * a5_pos
        print("               - [CMF]底部反转信号已生成")
        # --- 步骤 5: 生成顶部反转信号 (Top Reversal) ---
        df['FF_SCORE_CMF_REVERSAL_TOP_HIGH'] = s_pos * sl5_neg * a5_neg
        print("               - [CMF]顶部反转信号已生成")
        return df

    # 整个 _diagnose_xl_order_dynamics 方法块
    def _diagnose_xl_order_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【新增 V16.0 - 超大单动态诊断引擎】
        - 核心: 聚焦于net_xl_amount (超大单净额)，分析市场中最核心力量的动态。
        - 逻辑: 与主力资金引擎对齐，采用“累计值(静态)+累计值斜率(趋势)+每日值加速度(动能)”的黄金组合。
        - 产出: 生成关于“聪明钱”的共振与反转信号。
        - 数据假设: 依赖数据层提供 net_xl_amount 的 sum, SLOPE, ACCEL 衍生列。
        """
        print("            -> [超大单动态引擎 V1.0] 启动交叉验证...")
        periods = [5, 21, 55] # 使用简化的周期组合
        norm_window = 120
        # --- 步骤 1: 获取超大单资金指标 ---
        metrics = {}
        for p in periods:
            # 在所有列名中添加 "_consensus" 后缀以匹配军械库清单
            metrics[f'static_{p}'] = df.get(f"net_xl_amount_consensus_sum_{p}d_D")
            metrics[f'slope_{p}'] = df.get(f"SLOPE_{p}_net_xl_amount_consensus_sum_{p}d_D")
            if p in [5, 21]:
                 # 在所有列名中添加 "_consensus" 后缀以匹配军械库清单
                 metrics[f'accel_{p}'] = df.get(f"ACCEL_{p}_net_xl_amount_consensus_D")
        # --- 步骤 2: 生成超大单吸筹共振信号 ---
        s5 = self._calculate_normalized_score(metrics.get('static_5'), norm_window)
        sl5 = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        a5 = self._calculate_normalized_score(metrics.get('accel_5'), norm_window)
        s55 = self._calculate_normalized_score(metrics.get('static_55'), norm_window)
        sl55 = self._calculate_normalized_score(metrics.get('slope_55'), norm_window)
        df['FF_SCORE_XL_RESONANCE_UP_LOW'] = s5 * sl5 * a5
        df['FF_SCORE_XL_RESONANCE_UP_HIGH'] = df['FF_SCORE_XL_RESONANCE_UP_LOW'] * s55 * sl55
        print("               - [超大单]吸筹共振信号已生成 (低/高置信度)")
        # --- 步骤 3: 生成超大单派发共振信号 ---
        s5_neg = self._calculate_normalized_score(metrics.get('static_5'), norm_window, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, ascending=False)
        s55_neg = self._calculate_normalized_score(metrics.get('static_55'), norm_window, ascending=False)
        sl55_neg = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, ascending=False)
        df['FF_SCORE_XL_RESONANCE_DOWN_LOW'] = s5_neg * sl5_neg * a5_neg
        df['FF_SCORE_XL_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_XL_RESONANCE_DOWN_LOW'] * s55_neg * sl55_neg
        print("               - [超大单]派发共振信号已生成 (低/高置信度)")
        # --- 步骤 4: 生成超大单底部反转信号 ---
        long_term_selling = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, ascending=False)
        short_term_reversing = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        short_term_accelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window)
        df['FF_SCORE_XL_REVERSAL_BOTTOM_HIGH'] = long_term_selling * short_term_reversing * short_term_accelerating
        print("               - [超大单]底部反转信号已生成")
        # --- 步骤 5: 生成超大单顶部反转信号 ---
        long_term_buying = self._calculate_normalized_score(metrics.get('slope_55'), norm_window)
        short_term_diverging = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, ascending=False)
        df['FF_SCORE_XL_REVERSAL_TOP_HIGH'] = long_term_buying * short_term_diverging * short_term_decelerating
        print("               - [超大单]顶部反转信号已生成")
        return df

    def _diagnose_retail_flow_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【新增 V17.0 - 散户动态诊断引擎】
        - 核心: 独立分析散户资金流 (retail_net_flow_consensus) 的动态，捕捉市场情绪。
        - 视角: 将散户行为作为市场情绪的“温度计”和潜在的“逆向指标”。
        - 产出: 生成描述散户“买入狂热”和“恐慌杀跌”的共振与反转信号。
        - 数据假设: 依赖数据层提供 retail_net_flow_consensus 的全套衍生列。
        """
        print("            -> [散户动态引擎 V1.0] 启动交叉验证...")
        periods = [5, 21, 55]
        norm_window = 120
        # --- 步骤 1: 获取散户资金指标 ---
        metrics = {}
        for p in periods:
            metrics[f'static_{p}'] = df.get(f"retail_net_flow_consensus_sum_{p}d_D")
            metrics[f'slope_{p}'] = df.get(f"SLOPE_{p}_retail_net_flow_consensus_sum_{p}d_D")
            if p in [5, 21]:
                 metrics[f'accel_{p}'] = df.get(f"ACCEL_{p}_retail_net_flow_consensus_D")
        # --- 步骤 2: 生成散户买入狂热共振信号 (Buying Frenzy) ---
        s5 = self._calculate_normalized_score(metrics.get('static_5'), norm_window)
        sl5 = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        a5 = self._calculate_normalized_score(metrics.get('accel_5'), norm_window)
        s55 = self._calculate_normalized_score(metrics.get('static_55'), norm_window)
        sl55 = self._calculate_normalized_score(metrics.get('slope_55'), norm_window)
        df['FF_SCORE_RETAIL_RESONANCE_FRENZY_LOW'] = s5 * sl5 * a5
        df['FF_SCORE_RETAIL_RESONANCE_FRENZY_HIGH'] = df['FF_SCORE_RETAIL_RESONANCE_FRENZY_LOW'] * s55 * sl55
        print("               - [散户]买入狂热共振信号已生成")
        # --- 步骤 3: 生成散户恐慌杀跌共振信号 (Capitulation) ---
        s5_neg = self._calculate_normalized_score(metrics.get('static_5'), norm_window, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, ascending=False)
        s55_neg = self._calculate_normalized_score(metrics.get('static_55'), norm_window, ascending=False)
        sl55_neg = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, ascending=False)
        df['FF_SCORE_RETAIL_RESONANCE_CAPITULATION_LOW'] = s5_neg * sl5_neg * a5_neg
        df['FF_SCORE_RETAIL_RESONANCE_CAPITULATION_HIGH'] = df['FF_SCORE_RETAIL_RESONANCE_CAPITULATION_LOW'] * s55_neg * sl55_neg
        print("               - [散户]恐慌杀跌共振信号已生成")
        # --- 步骤 4: 生成散户抄底反转信号 ---
        long_term_selling = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, ascending=False)
        short_term_reversing = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        short_term_accelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window)
        df['FF_SCORE_RETAIL_REVERSAL_BOTTOM_FISHING'] = long_term_selling * short_term_reversing * short_term_accelerating
        print("               - [散户]抄底反转信号已生成")
        # --- 步骤 5: 生成散户顶部派发反转信号 ---
        long_term_buying = self._calculate_normalized_score(metrics.get('slope_55'), norm_window)
        short_term_diverging = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, ascending=False)
        df['FF_SCORE_RETAIL_REVERSAL_TOP_SELLING'] = long_term_buying * short_term_diverging * short_term_decelerating
        print("               - [散户]顶部派发反转信号已生成")
        return df

    def _diagnose_flow_intensity_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【新增 V18.0 - 资金流强度动态诊断引擎】
        - 核心: 分析主力资金流强度比率(主动买盘/总主动盘)，量化交易意愿和信念。
        - 视角: 从资金流的“质量”而非“数量”出发，判断主力控盘的决心。
        - 产出: 生成“信念买入”和“信念卖出”的共振与反转信号。
        - 数据假设: 依赖数据层提供 main_force_flow_intensity_ratio_D 的全套衍生列。
        """
        print("            -> [资金流强度引擎 V1.0] 启动交叉验证...")
        periods = [5, 13, 21]
        norm_window = 120
        # --- 步骤 1: 获取资金流强度指标 ---
        metrics = {
            'static': df.get('main_force_flow_intensity_ratio_D'),
            'slope_5': df.get('SLOPE_5_main_force_flow_intensity_ratio_D'),
            'slope_13': df.get('SLOPE_13_main_force_flow_intensity_ratio_D'),
            'accel_5': df.get('ACCEL_5_main_force_flow_intensity_ratio_D'),
        }
        # --- 步骤 2: 生成信念买入共振信号 (Conviction Buying) ---
        s_pos = self._calculate_normalized_score(metrics.get('static'), norm_window)
        sl5_pos = self._calculate_normalized_score(metrics.get('slope_5'), norm_window)
        sl13_pos = self._calculate_normalized_score(metrics.get('slope_13'), norm_window)
        a5_pos = self._calculate_normalized_score(metrics.get('accel_5'), norm_window)
        df['FF_SCORE_INTENSITY_RESONANCE_UP_LOW'] = s_pos * sl5_pos * a5_pos
        df['FF_SCORE_INTENSITY_RESONANCE_UP_HIGH'] = df['FF_SCORE_INTENSITY_RESONANCE_UP_LOW'] * sl13_pos
        print("               - [强度]信念买入共振信号已生成")
        # --- 步骤 3: 生成信念卖出共振信号 (Conviction Selling) ---
        s_neg = self._calculate_normalized_score(metrics.get('static'), norm_window, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, ascending=False)
        sl13_neg = self._calculate_normalized_score(metrics.get('slope_13'), norm_window, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, ascending=False)
        df['FF_SCORE_INTENSITY_RESONANCE_DOWN_LOW'] = s_neg * sl5_neg * a5_neg
        df['FF_SCORE_INTENSITY_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_INTENSITY_RESONANCE_DOWN_LOW'] * sl13_neg
        print("               - [强度]信念卖出共振信号已生成")
        # --- 步骤 4: 生成买入意愿拐点信号 (Bottom Reversal) ---
        df['FF_SCORE_INTENSITY_REVERSAL_BOTTOM_HIGH'] = s_neg * sl5_pos * a5_pos
        print("               - [强度]买入意愿拐点信号已生成")
        # --- 步骤 5: 生成卖出意愿拐点信号 (Top Reversal) ---
        df['FF_SCORE_INTENSITY_REVERSAL_TOP_HIGH'] = s_pos * sl5_neg * a5_neg
        print("               - [强度]卖出意愿拐点信号已生成")
        return df

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V18.0 - 七位一体·智能融合引擎】
        - 核心升级:
          1.  【引擎联动】: 新增资金流强度引擎，形成七大引擎矩阵。
          2.  【智能融合】: 合成“七位一体”元信号，融合六大“聪明钱”引擎并与散户逆向指标交叉验证。
             - 终极看涨: 六大聪明钱引擎看涨 + 散户恐慌杀跌
             - 终极看跌: 六大聪明钱引擎看跌 + 散户买入狂热
        """
        print("        -> [资金流情报模块 V18.0] 启动...") # [修改] 更新版本号
        states = {}
        p = get_params_block(self.strategy, 'fund_flow_params')
        if not get_param_value(p.get('enabled'), False):
            return states
        # --- 依次调用七大诊断引擎 ---
        df = self._diagnose_fund_flow_dynamics(df)
        df = self._diagnose_capital_structure_dynamics(df)
        df = self._diagnose_capital_conflict_dynamics(df)
        df = self._diagnose_cmf_dynamics(df)
        df = self._diagnose_xl_order_dynamics(df)
        df = self._diagnose_retail_flow_dynamics(df)
        df = self._diagnose_flow_intensity_dynamics(df) # 调用资金流强度诊断引擎
        # --- [终极升级] 生成资金流七位一体智能融合信号 (Septafecta Smart Resonance) ---
        print("            -> [七位一体引擎 V1.0] 启动智能信号融合...") # [修改] 升级为七位一体
        # 组合六大“聪明钱”引擎的看涨信号
        smart_money_up_low = (
            df.get('FF_SCORE_RESONANCE_UP_LOW', 0.5) *
            df.get('FF_SCORE_STRUCTURE_RESONANCE_UP_LOW', 0.5) *
            df.get('FF_SCORE_CONFLICT_RESONANCE_UP_LOW', 0.5) *
            df.get('FF_SCORE_CMF_RESONANCE_UP_LOW', 0.5) *
            df.get('FF_SCORE_XL_RESONANCE_UP_LOW', 0.5) *
            df.get('FF_SCORE_INTENSITY_RESONANCE_UP_LOW', 0.5) # 融合强度信号
        )
        smart_money_up_high = (
            df.get('FF_SCORE_RESONANCE_UP_HIGH', 0.5) *
            df.get('FF_SCORE_STRUCTURE_RESONANCE_UP_HIGH', 0.5) *
            df.get('FF_SCORE_CONFLICT_RESONANCE_UP_HIGH', 0.5) *
            df.get('FF_SCORE_CMF_RESONANCE_UP_HIGH', 0.5) *
            df.get('FF_SCORE_XL_RESONANCE_UP_HIGH', 0.5) *
            df.get('FF_SCORE_INTENSITY_RESONANCE_UP_HIGH', 0.5) # 融合强度信号
        )
        # 组合六大“聪明钱”引擎的看跌信号
        smart_money_down_low = (
            df.get('FF_SCORE_RESONANCE_DOWN_LOW', 0.5) *
            df.get('FF_SCORE_STRUCTURE_RESONANCE_DOWN_LOW', 0.5) *
            df.get('FF_SCORE_CONFLICT_RESONANCE_DOWN_LOW', 0.5) *
            df.get('FF_SCORE_CMF_RESONANCE_DOWN_LOW', 0.5) *
            df.get('FF_SCORE_XL_RESONANCE_DOWN_LOW', 0.5) *
            df.get('FF_SCORE_INTENSITY_RESONANCE_DOWN_LOW', 0.5) # 融合强度信号
        )
        smart_money_down_high = (
            df.get('FF_SCORE_RESONANCE_DOWN_HIGH', 0.5) *
            df.get('FF_SCORE_STRUCTURE_RESONANCE_DOWN_HIGH', 0.5) *
            df.get('FF_SCORE_CONFLICT_RESONANCE_DOWN_HIGH', 0.5) *
            df.get('FF_SCORE_CMF_RESONANCE_DOWN_HIGH', 0.5) *
            df.get('FF_SCORE_XL_RESONANCE_DOWN_HIGH', 0.5) *
            df.get('FF_SCORE_INTENSITY_RESONANCE_DOWN_HIGH', 0.5) # 融合强度信号
        )
        # 智能融合：看涨 = 聪明钱买 + 散户卖
        df['FF_SCORE_SEPTAFECTA_RESONANCE_UP_LOW'] = smart_money_up_low * df.get('FF_SCORE_RETAIL_RESONANCE_CAPITULATION_LOW', 0.5)
        df['FF_SCORE_SEPTAFECTA_RESONANCE_UP_HIGH'] = smart_money_up_high * df.get('FF_SCORE_RETAIL_RESONANCE_CAPITULATION_HIGH', 0.5)
        print("               - [七位一体]看涨共振信号已生成 (聪明钱买 vs 散户卖)")
        # 智能融合：看跌 = 聪明钱卖 + 散户买
        df['FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_LOW'] = smart_money_down_low * df.get('FF_SCORE_RETAIL_RESONANCE_FRENZY_LOW', 0.5)
        df['FF_SCORE_SEPTAFECTA_RESONANCE_DOWN_HIGH'] = smart_money_down_high * df.get('FF_SCORE_RETAIL_RESONANCE_FRENZY_HIGH', 0.5)
        print("               - [七位一体]看跌共振信号已生成 (聪明钱卖 vs 散户买)")
        # --- 收集所有生成的数值化评分 ---
        for col in df.columns:
            if col.startswith('FF_SCORE_'):
                states[col] = df[col]
        print(f"        -> [资金流情报模块 V18.0] 诊断完毕，生成了 {len(states)} 个数值化动态信号。") # [修改] 更新版本号
        return states
