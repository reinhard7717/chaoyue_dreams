# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class ProcessIntelligence:
    """
    【V1.0 · 过程分析中枢】
    - 核心哲学: 遵从指挥官“从头开始”的最高指示，回归本源。本引擎作为系统的“参谋部”，
                  直接在数据层提供的【原始数据】上进行回溯分析，以保证最高精度。
    - 核心职责: 根据配置文件的指令，诊断具备经典市场理论支撑的“背离”与“共振”关系，
                  并产出高维度的“过程元状态”。
    - 版本: 1.0
    """
    def __init__(self, strategy_instance):
        """
        初始化过程情报引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        # 从配置文件加载本引擎专属的参数
        self.params = get_params_block(self.strategy, 'process_intelligence_params', {})
        self.lookback_window = get_param_value(self.params.get('lookback_window'), 13)
        self.norm_window = get_param_value(self.params.get('norm_window'), 55)
        # 加载需要诊断的“背离”与“共振”的蓝图配置
        self.diagnostics_config = get_param_value(self.params.get('diagnostics'), [])

    def run_process_diagnostics(self) -> Dict[str, pd.Series]:
        """
        过程诊断总指挥。
        它将遍历配置文件中定义的所有诊断任务，并执行它们。
        """
        print("      -> [过程情报引擎 V1.0 · 过程分析中枢] 启动...")
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            print("      -> [过程情报引擎 V1.0] 警告: 数据DataFrame为空，跳过诊断。")
            return {}
        # 遍历配置中定义的每一个诊断任务
        for config in self.diagnostics_config:
            signal_name = config.get('name')
            signal_type = config.get('type')
            if not signal_name or not signal_type:
                continue
            # 根据诊断类型，调用相应的诊断方法
            if signal_type == 'divergence':
                state = self._diagnose_divergence(df, config)
                if state:
                    all_process_states.update(state)
            elif signal_type == 'resonance':
                state = self._diagnose_resonance(df, config)
                if state:
                    all_process_states.update(state)
        print(f"      -> [过程情报引擎 V1.0] 分析完毕，共生成 {len(all_process_states)} 个高维度过程元状态。")
        return all_process_states

    def _calculate_raw_process_trend(self, series: pd.Series) -> pd.Series:
        """
        【核心工具】在原始数据序列上计算趋势（斜率），保留原始量纲以确保最高精度。
        :param series: 原始数据 pd.Series。
        :return: 趋势（斜率）的 pd.Series。
        """
        # 使用线性回归计算滚动窗口内的斜率
        return series.rolling(window=self.lookback_window).apply(
            lambda x: np.polyfit(range(len(x)), x.dropna(), 1)[0] if len(x.dropna()) > 1 else 0, raw=False
        ).fillna(0)

    def _diagnose_divergence(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        诊断“背离”关系。
        :param df: 完整的指标DataFrame。
        :param config: 单个背离诊断任务的配置字典。
        :return: 包含单个背离信号的字典。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        # 从df中获取最原始的数据
        series_a = df.get(signal_a_name)
        series_b = df.get(signal_b_name)
        if series_a is None or series_b is None:
            print(f"        -> [背离诊断] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'，跳过 '{config['name']}' 的计算。")
            return {}
        # 在原始数据上计算趋势
        trend_a_raw = self._calculate_raw_process_trend(series_a)
        trend_b_raw = self._calculate_raw_process_trend(series_b)
        # 根据配置中的条件判断背离是否发生
        # 例如: condition = "A_trend < 0 and B_trend > 0"
        is_divergence = (trend_a_raw < 0) & (trend_b_raw > 0)
        # 背离的强度由“好的”那个趋势的强度决定，并进行归一化
        # 在看涨背离中，通常是B信号（如筹码集中度）的上升趋势
        strength_score = normalize_score(trend_b_raw, df.index, self.norm_window)
        # 最终分数 = 是否发生背离 * 背离强度
        final_score = (is_divergence.astype(float) * strength_score).astype(np.float32)
        return {config['name']: final_score}

    def _diagnose_resonance(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        诊断“共振”关系。
        :param df: 完整的指标DataFrame。
        :param config: 单个共振诊断任务的配置字典。
        :return: 包含单个共振信号的字典。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        series_a = df.get(signal_a_name)
        series_b = df.get(signal_b_name)
        if series_a is None or series_b is None:
            print(f"        -> [共振诊断] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'，跳过 '{config['name']}' 的计算。")
            return {}
        trend_a_raw = self._calculate_raw_process_trend(series_a)
        trend_b_raw = self._calculate_raw_process_trend(series_b)
        # 根据配置中的条件判断共振是否发生
        # 例如: condition = "A_trend > 0 and B_trend > 0"
        is_resonance = (trend_a_raw > 0) & (trend_b_raw > 0)
        # 共振的强度由两个趋势的强度共同决定，使用几何平均进行融合
        strength_a_norm = normalize_score(trend_a_raw, df.index, self.norm_window)
        strength_b_norm = normalize_score(trend_b_raw, df.index, self.norm_window)
        strength_score = (strength_a_norm * strength_b_norm)**0.5
        # 最终分数 = 是否发生共振 * 共振强度
        final_score = (is_resonance.astype(float) * strength_score).astype(np.float32)
        return {config['name']: final_score}

