# 文件: strategies/trend_following/intelligence/dynamic_mechanics_engine.py
# 动态力学引擎
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict

class DynamicMechanicsEngine:
    def __init__(self, strategy_instance):
        """
        初始化动态力学引擎。
        :param strategy_instance: 策略主实例的引用，用于访问 df_indicators。
        """
        self.strategy = strategy_instance

    def run_force_vector_analysis(self) -> Dict[str, pd.Series]:
        """
        【V317.0 新增】动态力学分析引擎
        - 核心职责: 计算进攻分和风险分的“加速度”，捕捉双向动能的剧烈变化。
                    这是判断趋势强化或转折的关键“势能”情报。
        """
        # print("        -> [动态力学分析引擎 V317.0] 启动，正在计算势能加速度...")
        states = {}
        df = self.strategy.df_indicators
        
        # 确保 entry_score 和 risk_score 已经计算完毕
        if 'entry_score' not in df.columns or 'risk_score' not in df.columns:
            print("          -> [警告] 缺少 entry_score 或 risk_score，力学分析跳过。")
            return states

        window = 5 # 使用5日窗口计算趋势和加速度

        # 1. 计算“进攻”和“风险”的趋势（斜率）
        entry_score_slope = df['entry_score'].rolling(window).apply(
            lambda y: linregress(np.arange(window), y).slope if len(y.dropna()) == window else np.nan, raw=False
        )
        risk_score_slope = df['risk_score'].rolling(window).apply(
            lambda y: linregress(np.arange(window), y).slope if len(y.dropna()) == window else np.nan, raw=False
        )

        # 2. 计算“进攻”和“风险”的加速度（斜率的差分）
        entry_score_accel = entry_score_slope.diff()
        risk_score_accel = risk_score_slope.diff()

        # 3. 定义加速度阈值，过滤掉无意义的波动
        accel_threshold = 1.0 # 当加速度变化大于1时，我们认为是有意义的

        # 4. 生成四种核心的“力学”原子状态
        states['FORCE_VECTOR_OFFENSE_ACCELERATING'] = entry_score_accel > accel_threshold
        states['FORCE_VECTOR_OFFENSE_DECELERATING'] = entry_score_accel < -accel_threshold
        states['FORCE_VECTOR_RISK_ACCELERATING'] = risk_score_accel > accel_threshold
        states['FORCE_VECTOR_RISK_DECELERATING'] = risk_score_accel < -accel_threshold
        
        self.strategy.atomic_states.update(states)
        
        # print("          -> [力学分析引擎] 进攻/风险的加速度情报已生成。")
        return states

