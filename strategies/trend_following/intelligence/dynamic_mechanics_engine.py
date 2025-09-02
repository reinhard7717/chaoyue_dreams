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
        【V317.3 宏观专属版】宏观动态力学分析引擎
        - 核心职责: (本次修改) 专门负责计算进攻分和风险分的“加速度”，捕捉双向动能的剧烈变化。
                    此方法被设计为在 entry_score 和 risk_score 计算完成之后调用。
        - 收益:     这是判断趋势强化或转折的关键“势能”情报。
        """
        # print("        -> [宏观力学分析引擎 V317.3] 启动，正在计算分数势能...")
        states = {}
        df = self.strategy.df_indicators
        if 'entry_score' not in df.columns or 'risk_score' not in df.columns:
            print("          -> [警告] 缺少 entry_score 或 risk_score，宏观力学分析跳过。")
            return states
        window = 5
        # 定义一个高效的斜率计算函数
        def calculate_slope(y):
            if np.isnan(y).any():
                return np.nan
            return linregress(np.arange(window), y).slope
        # 1. 计算“进攻”和“风险”的趋势（斜率）
        entry_score_slope = df['entry_score'].rolling(window).apply(calculate_slope, raw=True)
        risk_score_slope = df['risk_score'].rolling(window).apply(calculate_slope, raw=True)
        # 2. 计算“进攻”和“风险”的加速度（斜率的差分）
        entry_score_accel = entry_score_slope.diff()
        risk_score_accel = risk_score_slope.diff()
        # 3. 定义加速度阈值，过滤掉无意义的波动
        accel_threshold = 1.0
        # 4. 生成四种核心的“力学”原子状态
        states['FORCE_VECTOR_OFFENSE_ACCELERATING'] = entry_score_accel > accel_threshold
        states['FORCE_VECTOR_OFFENSE_DECELERATING'] = entry_score_accel < -accel_threshold
        states['FORCE_VECTOR_RISK_ACCELERATING'] = risk_score_accel > accel_threshold
        states['FORCE_VECTOR_RISK_DECELERATING'] = risk_score_accel < -accel_threshold
        # 5. 生成两种复合的“力学”中性状态
        states['FORCE_VECTOR_PURE_OFFENSIVE_MOMENTUM'] = states['FORCE_VECTOR_OFFENSE_ACCELERATING'] & states['FORCE_VECTOR_RISK_DECELERATING']
        states['FORCE_VECTOR_CHAOTIC_EXPANSION'] = states['FORCE_VECTOR_OFFENSE_ACCELERATING'] & states['FORCE_VECTOR_RISK_ACCELERATING']
        self.strategy.atomic_states.update(states)
        # print("          -> [宏观力学分析引擎] 进攻/风险的加速度情报已生成。")
        return states

    def run_dynamic_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V317.4 微观协同版】微观及协同力学分析总指挥
        - 核心职责: 统一调度微观、多时间维度及终极的“静态-动态”交叉验证分析。
        - 核心修正 (本次修改): 移除了依赖后期分数的“宏观力学分析”部分，解决了在情报层
                          前期调用时因缺少 `entry_score` 而产生警告的问题。
                          现在此方法专注于处理基础指标衍生的动态力学。
        """
        # print("        -> [微观及协同力学分析总指挥 V317.4] 启动...")
        states = {}
        df = self.strategy.df_indicators
        # --- 移除了原有的宏观力学分析部分 ---
        # --- 步骤1: 执行微观力学分析 (基于基础指标) ---
        micro_dynamic_states = self.diagnose_micro_dynamics(df)
        states.update(micro_dynamic_states)
        # --- 步骤2: 执行多时间维度力学分析 (交叉验证) ---
        multi_timeframe_states = self.diagnose_multi_timeframe_dynamics(df)
        states.update(multi_timeframe_states)
        # --- 步骤3: 执行行为力学分析 (基于情绪与效率指标) ---
        behavioral_states = self.diagnose_behavioral_mechanics(df)
        states.update(behavioral_states)
        self.strategy.atomic_states.update(states)
        # print("          -> [微观及协同力学分析总指挥] 微观/多周期/终极交叉验证情报已全部生成。")
        return states

    def diagnose_micro_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】微观力学诊断模块
        - 核心职责: 直接利用数据层预计算的基础指标（价格、成交量、波动率、筹码）的
                      斜率和加速度，生成描述市场微观力学状态的原子信号。
        - 收益: 捕捉比宏观分数更具体、更底层的市场动态，如“量价共振”、“缩量上涨”等。
        """
        print("        -> [微观力学诊断模块 V1.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查：确保所有必需的斜率和加速度列都存在 ---
        required_cols = [
            'ACCEL_5_close_D', 'ACCEL_5_volume_D',
            'SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_BBW_21_2.0_D',
            'ACCEL_5_concentration_90pct_D', 'ACCEL_5_peak_cost_D'
        ]
        if any(c not in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 微观力学诊断模块缺少关键数据: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 生成微观力学信号 ---
        # 信号1 (S级机会): “内外共振·主升确认”
        # 解读: 内部结构（筹码加速集中 & 成本加速抬升）与外部表现（价格加速上涨）完美共振，
        #      是主升浪最强烈的确认信号。
        is_internal_accelerating = (df['ACCEL_5_concentration_90pct_D'] < 0) & (df['ACCEL_5_peak_cost_D'] > 0)
        is_external_accelerating = df['ACCEL_5_close_D'] > 0
        states['OPP_DYN_INTERNAL_EXTERNAL_RESONANCE_S'] = is_internal_accelerating & is_external_accelerating

        # 信号2 (A级机会): “波动率点火”
        # 解读: 波动率经过收缩后（斜率 < 0），开始加速扩张（加速度 > 0），且价格同步加速上涨。
        #      这是经典的“Squeeze”形态突破，能量由静转动，极具爆发力。
        is_volatility_squeezing = df['SLOPE_5_BBW_21_2.0_D'] < 0
        is_volatility_igniting = df['ACCEL_5_BBW_21_2.0_D'] > 0
        states['DYN_VOLATILITY_BREAKOUT_A'] = is_volatility_squeezing.shift(1) & is_volatility_igniting & is_external_accelerating

        # 信号3 (B级中性/机会): “量价共振”
        # 解读: 价格和成交量同步加速上涨，代表上涨趋势健康、能量充沛。
        is_volume_accelerating = df['ACCEL_5_volume_D'] > 0
        states['DYN_PRICE_VOLUME_RESONANCE_B'] = is_external_accelerating & is_volume_accelerating

        # 信号4 (B级风险): “力竭上涨”
        # 解读: 价格仍在加速上涨，但成交量的加速度却为负（能量在衰减），是典型的量价背离，
        #      预示上涨动能即将衰竭，是潜在的顶部风险信号。
        is_volume_decelerating = df['ACCEL_5_volume_D'] < 0
        states['RISK_DYN_EXHAUSTION_RALLY_B'] = is_external_accelerating & is_volume_decelerating

        return states

    def diagnose_multi_timeframe_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】多时间维度力学诊断模块
        - 核心职责: 交叉验证短期（5日）与长期（21日）的动态指标（斜率/加速度），
                      以识别趋势的“共振”与“背离”，生成更高置信度的信号。
        - 收益: 区分健康的趋势与虚假的突破/反弹，精准捕捉趋势的确认、衰竭与反转点。
        """
        print("        -> [多时间维度力学诊断模块 V1.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查：确保所有必需的多周期斜率/加速度列都存在 ---
        required_cols = [
            'SLOPE_5_close_D', 'SLOPE_21_close_D', 'ACCEL_21_close_D',
            'SLOPE_5_concentration_90pct_D', 'SLOPE_21_concentration_90pct_D'
        ]
        if any(c not in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 多时间维度力学诊断模块缺少关键数据: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 定义各周期的基本动态 ---
        # 短期动态
        is_price_rising_short = df['SLOPE_5_close_D'] > 0
        is_chip_concentrating_short = df['SLOPE_5_concentration_90pct_D'] < 0

        # 长期动态
        is_price_rising_long = df['SLOPE_21_close_D'] > 0
        is_price_falling_long = df['SLOPE_21_close_D'] < 0
        is_price_fall_decelerating_long = df['ACCEL_21_close_D'] > 0 # 长期下跌趋势在减速
        is_chip_diverging_long = df['SLOPE_21_concentration_90pct_D'] > 0

        # --- 3. 生成多周期交叉验证信号 ---
        # 信号1 (S级机会): “全周期趋势共振”
        # 解读: 短期和长期的价格趋势均为上涨，且短期筹码仍在持续集中。
        #      这是最健康、最强劲的上涨形态，表明多周期力量形成合力。
        states['OPP_DYN_TREND_RESONANCE_S'] = (
            is_price_rising_short &
            is_price_rising_long &
            is_chip_concentrating_short
        )

        # 信号2 (S级风险): “结构性衰竭反弹”
        # 解读: 价格短期看似在上涨，但其赖以生存的长期筹码结构却在持续瓦解（发散）。
        #      这是典型的“拉高出货”或“无根反弹”，是极度危险的顶部背离信号。
        states['RISK_DYN_STRUCTURAL_WEAKNESS_RALLY_S'] = (
            is_price_rising_short &
            is_chip_diverging_long
        )

        # 信号3 (A级机会): “长周期底部拐点”
        # 解读: 长期下跌趋势已出现明显的减速信号，同时短期价格趋势已率先反转向上。
        #      这表明下跌动能衰竭，新的上涨周期可能正在酝酿，是左侧交易的理想信号。
        states['OPP_DYN_LONG_CYCLE_INFLECTION_A'] = (
            is_price_falling_long.shift(1) & # 确保前一天还是长期下跌趋势
            is_price_fall_decelerating_long &
            is_price_rising_short
        )

        return states

    def diagnose_behavioral_mechanics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】行为力学诊断模块
        - 核心职责: 分析资金效率、获利盘/亏损盘行为的动态变化（加速度），
                      以洞察市场情绪的真实状态和趋势的内在健康度。
        - 收益: 提供了超越传统量价分析的视角，能更早地识别趋势的“质变”。
        """
        print("        -> [行为力学诊断模块 V1.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 军备检查 ---
        required_cols = [
            'SLOPE_5_close_D', 'ACCEL_5_close_D',
            'ACCEL_5_VPA_EFFICIENCY_D',
            'ACCEL_5_turnover_from_winners_ratio_D',
            'ACCEL_5_turnover_from_losers_ratio_D'
        ]
        if any(c not in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 行为力学诊断模块缺少关键数据: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 资金效率力学 ---
        # 机会信号 (A级): “市场引擎点火”
        # 解读: 价格加速上涨的同时，资金效率也在加速提升，表明上涨健康且高效。
        is_price_accelerating = df['ACCEL_5_close_D'] > 0
        is_efficiency_accelerating = df['ACCEL_5_VPA_EFFICIENCY_D'] > 0
        states['OPP_DYN_MARKET_ENGINE_IGNITION_A'] = is_price_accelerating & is_efficiency_accelerating

        # 风险信号 (S级): “市场引擎失速”
        # 解读: 价格仍在上涨，但资金效率却在加速下滑，是严重的顶背离信号。
        is_price_rising = df['SLOPE_5_close_D'] > 0
        is_efficiency_collapsing = df['ACCEL_5_VPA_EFFICIENCY_D'] < 0
        states['RISK_DYN_MARKET_ENGINE_STALLING_S'] = is_price_rising & is_efficiency_collapsing

        # --- 3. 交易情绪力学 ---
        # 风险信号 (S级): “获利盘恐慌加速”
        # 解读: 股价下跌的同时，获利盘卖出行为在“加速”，是踩踏式下跌的预警。
        is_price_falling = df['SLOPE_5_close_D'] < 0
        is_winner_selling_accelerating = df['ACCEL_5_turnover_from_winners_ratio_D'] > 0
        states['RISK_DYN_PANIC_SELLING_ACCELERATING_S'] = is_price_falling & is_winner_selling_accelerating

        # 机会信号 (A级): “恐慌盘投降衰竭”
        # 解读: 股价下跌的同时，亏损盘卖出行为在“减速”，表明卖压衰竭，是潜在的底部信号。
        is_loser_selling_decelerating = df['ACCEL_5_turnover_from_losers_ratio_D'] < 0
        states['OPP_DYN_CAPITULATION_EXHAUSTION_A'] = is_price_falling & is_loser_selling_decelerating

        return states













