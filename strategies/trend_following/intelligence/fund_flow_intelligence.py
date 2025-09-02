# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 战术共振版】资金流原子信号诊断模块
        - 核心升级:
          1.  【协同确认】: 交叉验证多数据源信号，生成高置信度信号。
          2.  【动态动量】: 引入资金流指标的斜率，捕捉资金流入/流出的加速与减速。
          3.  【多时间维度】: 结合周线CMF指标，对日线资金流信号进行战略层面的确认或否决。
          4.  【多维动态协同】: 交叉验证周线资金趋势(斜率)与日线资金动能(加速度)。
          5.  【静态-多动态协同 (本次新增)】: 将高置信度的静态信号作为“地基”，
                                         与多个动态信号进行“共振”验证，生成最高级别的S+级战术信号。
        """
        print("        -> [资金流情报模块 V4.0 战术共振版] 启动...") # [修改代码行]
        states = {}
        default_series = pd.Series(False, index=df.index)

        p = get_params_block(self.strategy, 'fund_flow_params')
        if not get_param_value(p.get('enabled'), False):
            return states
        
        # --- 0. 军备检查 (统一检查所有需要的列) ---
        required_cols = [
            'net_mf_amount_fund_flow_tushare_D', 'net_amount_fund_flow_ths_D', 'buy_lg_amount_rate_fund_flow_ths_D',
            'net_amount_fund_flow_dc_D', 'buy_elg_amount_rate_fund_flow_dc_D', 'SLOPE_5_net_amount_fund_flow_ths_D',
            'SLOPE_5_buy_lg_amount_rate_fund_flow_ths_D', 'CMF_21_W', 'SLOPE_5_CMF_21_W', 'ACCEL_5_net_amount_fund_flow_ths_D',
            'net_d5_amount_fund_flow_ths_D', 'net_amount_rate_fund_flow_dc_D', 'buy_sm_amount_rate_fund_flow_ths_D',
            'buy_md_amount_rate_fund_flow_ths_D', 'pct_change_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [警告] 缺少资金流诊断所需列，部分信号可能无法生成。缺失: {missing_cols}")

        # --- 1. 基础静态信号 (沿用并作为后续分析的基础) ---
        # Tushare
        net_mf_amount_col = 'net_mf_amount_fund_flow_tushare_D'
        if net_mf_amount_col in df.columns:
            states['FUND_FLOW_TS_NET_INFLOW'] = df[net_mf_amount_col] > 0
        # 同花顺
        net_amount_ths_col = 'net_amount_fund_flow_ths_D'
        buy_lg_amount_rate_ths_col = 'buy_lg_amount_rate_fund_flow_ths_D'
        if net_amount_ths_col in df.columns:
            states['FUND_FLOW_THS_NET_INFLOW'] = df[net_amount_ths_col] > 0
        if buy_lg_amount_rate_ths_col in df.columns:
            large_buy_threshold = get_param_value(p.get('ths_large_buy_rate_threshold'), 0.5)
            states['FUND_FLOW_THS_LARGE_BUY_DOMINANT'] = df[buy_lg_amount_rate_ths_col] > large_buy_threshold
        # 东方财富
        net_amount_dc_col = 'net_amount_fund_flow_dc_D'
        buy_elg_amount_rate_dc_col = 'buy_elg_amount_rate_fund_flow_dc_D'
        if net_amount_dc_col in df.columns:
            states['FUND_FLOW_DC_NET_INFLOW'] = df[net_amount_dc_col] > 0
        if buy_elg_amount_rate_dc_col in df.columns:
            super_large_buy_threshold = get_param_value(p.get('dc_super_large_buy_rate_threshold'), 0.3)
            states['FUND_FLOW_DC_SUPER_LARGE_BUY_DOMINANT'] = df[buy_elg_amount_rate_dc_col] > super_large_buy_threshold

        # --- 2. 协同确认信号 (提升信号置信度) ---
        states['FUND_FLOW_MULTI_SOURCE_INFLOW_CONFIRMED'] = (
            states.get('FUND_FLOW_TS_NET_INFLOW', default_series) &
            states.get('FUND_FLOW_THS_NET_INFLOW', default_series) &
            states.get('FUND_FLOW_DC_NET_INFLOW', default_series)
        )
        states['FUND_FLOW_MAIN_FORCE_BUY_DOMINANT'] = (
            states.get('FUND_FLOW_THS_LARGE_BUY_DOMINANT', default_series) &
            states.get('FUND_FLOW_DC_SUPER_LARGE_BUY_DOMINANT', default_series)
        )

        # --- 3. 动态动量信号 (捕捉趋势变化) ---
        required_slope_cols = [
            'SLOPE_5_net_amount_fund_flow_ths_D',
            'SLOPE_5_buy_lg_amount_rate_fund_flow_ths_D'
        ]
        if all(c in df.columns for c in required_slope_cols):
            states['FUND_FLOW_DYN_INFLOW_ACCELERATING'] = df['SLOPE_5_net_amount_fund_flow_ths_D'] > 0
            states['FUND_FLOW_DYN_MAIN_FORCE_INTENSIFYING'] = df['SLOPE_5_buy_lg_amount_rate_fund_flow_ths_D'] > 0
            states['FUND_FLOW_DYN_INFLOW_DECELERATING'] = df['SLOPE_5_net_amount_fund_flow_ths_D'] < 0
        else:
            print(f"          -> [警告] 缺少资金流斜率列，动态动量信号模块已跳过。")

        # --- 4. 多时间维度协同信号 (战略级信号) ---
        cmf_w_col = 'CMF_21_W'
        if cmf_w_col in df.columns:
            is_weekly_capital_bullish = df[cmf_w_col] > 0.05
            is_daily_main_force_buying = states.get('FUND_FLOW_MAIN_FORCE_BUY_DOMINANT', default_series)
            states['OPP_FUND_FLOW_MTF_BULLISH_ALIGNMENT_A'] = is_weekly_capital_bullish & is_daily_main_force_buying
            is_weekly_capital_bearish = df[cmf_w_col] < -0.05
            is_daily_inflow = states.get('FUND_FLOW_MULTI_SOURCE_INFLOW_CONFIRMED', default_series)
            states['RISK_FUND_FLOW_MTF_BEARISH_DIVERGENCE_A'] = is_weekly_capital_bearish & is_daily_inflow
        else:
            print(f"          -> [警告] 缺少周线CMF列 '{cmf_w_col}'，多时间维度协同信号模块已跳过。")

        # --- 5. 多时间维度动态协同 (S级战略信号) ---
        required_mtf_dyn_cols = [
            'SLOPE_5_CMF_21_W',
            'ACCEL_5_net_amount_fund_flow_ths_D',
            'SLOPE_5_net_amount_fund_flow_ths_D',
            'net_amount_fund_flow_ths_D'
        ]
        is_weekly_momentum_improving = default_series # [新增代码行]
        if all(c in df.columns for c in required_mtf_dyn_cols):
            is_weekly_momentum_improving = df['SLOPE_5_CMF_21_W'] > 0
            is_daily_inflow_igniting = df['ACCEL_5_net_amount_fund_flow_ths_D'] > 0
            states['OPP_FUND_FLOW_MTF_DYN_ALIGNMENT_S'] = is_weekly_momentum_improving & is_daily_inflow_igniting
            is_weekly_momentum_deteriorating = df['SLOPE_5_CMF_21_W'] < 0
            is_daily_inflow_decelerating = (df['net_amount_fund_flow_ths_D'] > 0) & (df['SLOPE_5_net_amount_fund_flow_ths_D'] < 0)
            states['RISK_FUND_FLOW_MTF_DYN_DIVERGENCE_S'] = is_weekly_momentum_deteriorating & is_daily_inflow_decelerating
        else:
            missing_cols = [c for c in required_mtf_dyn_cols if c not in df.columns]
            print(f"          -> [警告] 缺少多维动态协同所需列: {missing_cols}，S级信号模块已跳过。")

        # --- 6. 静态-多动态协同 (S+级战术信号) ---
        # 依赖前面步骤计算出的原子信号
        # S+级机会: "主力控盘下的三维共振"
        # 静态地基: 主力资金已高度控盘 (大单+超大单联合主导)。
        # 动态确认1: 主力仍在持续加码 (主力买入占比斜率>0)。
        # 动态确认2: 战略层面资金趋势向好 (周线CMF斜率>0)。
        # 解读: 这是最强的买入信号之一。它表明控盘主力不仅没有出货，还在战略顺风的掩护下继续猛烈进攻。
        is_main_force_dominant = states.get('FUND_FLOW_MAIN_FORCE_BUY_DOMINANT', default_series)
        is_main_force_intensifying = states.get('FUND_FLOW_DYN_MAIN_FORCE_INTENSIFYING', default_series)
        states['OPP_FUND_FLOW_STATIC_DYN_CONFLUENCE_S_PLUS'] = (
            is_main_force_dominant &
            is_main_force_intensifying &
            is_weekly_momentum_improving
        )

        # S+级风险: "主力缺席下的诱多式拉升"
        # 静态背景: 主力资金并未主导买盘 (大单+超大单主导为False)。
        # 动态表象: 但日线资金却在加速流入 (资金净流入斜率>0)。
        # 战略背景: 且周线资金趋势已经恶化 (周线CMF斜率<0)。
        # 解读: 这是一个极其危险的陷阱。没有主力参与，仅靠散户资金推动的加速上涨，同时长线资金在撤离，
        # 表明这很可能是为后续派发制造空间，极易形成天地板或A杀。
        is_inflow_accelerating = states.get('FUND_FLOW_DYN_INFLOW_ACCELERATING', default_series)
        is_weekly_momentum_deteriorating = df['SLOPE_5_CMF_21_W'] < 0 if 'SLOPE_5_CMF_21_W' in df.columns else default_series
        states['RISK_FUND_FLOW_DECEPTIVE_RALLY_S_PLUS'] = (
            ~is_main_force_dominant &
            is_inflow_accelerating &
            is_weekly_momentum_deteriorating
        )
        
        # --- 7. 新增战术与风险信号 (持续性、强度、结构) ---
        # 7.1 A级机会: 持续净流入 (Sustained Net Inflow)
        # 逻辑: 过去5个交易日的累计资金净流入为正，表明买盘具有连续性。
        sustained_inflow_col = 'net_d5_amount_fund_flow_ths_D'
        if sustained_inflow_col in df.columns:
            states['FUND_FLOW_SUSTAINED_INFLOW_A'] = df[sustained_inflow_col] > 0

        # 7.2 A级机会: 高强度净流入 (High-Intensity Net Inflow)
        # 逻辑: 资金净流入额占当天总成交额的比例显著偏高，代表买盘意愿坚决。
        intensity_col = 'net_amount_rate_fund_flow_dc_D'
        if intensity_col in df.columns:
            intensity_threshold = get_param_value(p.get('dc_net_inflow_intensity_threshold'), 5.0) # 阈值: 净流入占成交额5%
            states['FUND_FLOW_HIGH_INTENSITY_INFLOW_A'] = df[intensity_col] > intensity_threshold

        # 7.3 B级风险: 散户狂热风险 (Retail FOMO Risk)
        # 逻辑: 股价大涨，但主要是由小单和中单买盘驱动，是情绪过热的危险信号。
        sm_rate_col = 'buy_sm_amount_rate_fund_flow_ths_D'
        md_rate_col = 'buy_md_amount_rate_fund_flow_ths_D'
        pct_change_col = 'pct_change_D'
        if all(c in df.columns for c in [sm_rate_col, md_rate_col, pct_change_col]):
            rally_threshold = get_param_value(p.get('fomo_rally_threshold'), 0.03) # 涨幅超过3%
            fomo_threshold = get_param_value(p.get('fomo_retail_rate_threshold'), 0.60) # 散户(小+中单)买入占比超过60%
            is_strong_rally = df[pct_change_col] > rally_threshold
            is_retail_driven = (df[sm_rate_col] + df[md_rate_col]) > fomo_threshold
            states['RISK_FUND_FLOW_RETAIL_FOMO_B'] = is_strong_rally & is_retail_driven

        print("        -> [资金流情报模块 V4.0 战术共振版] 诊断完毕。") # [修改代码行]
        return states


















