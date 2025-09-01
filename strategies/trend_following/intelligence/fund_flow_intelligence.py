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
        【V1.1 数据层适配版】资金流原子信号诊断模块
        - 核心修改: 更新了同花顺(ths)和东方财富(dc)的资金流列名，以适配数据层
                    提供的带数据源后缀的新列名。
        - 收益: 确保策略能正确读取和解析新的资金流数据。
        """
        print("        -> [资金流情报模块 V1.1 数据层适配版] 启动，正在诊断资金流原子信号...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        p = get_params_block(self.strategy, 'fund_flow_params')
        if not get_param_value(p.get('enabled'), False):
            return states

        # --- 1. Tushare 资金流 (moneyflow) ---
        # 字段: net_mf_amount_fund_flow_tushare_D (净流入额)
        net_mf_amount_col = 'net_mf_amount_fund_flow_tushare_D'
        if net_mf_amount_col in df.columns:
            states['FUND_FLOW_TS_NET_INFLOW'] = df[net_mf_amount_col] > 0
            states['FUND_FLOW_TS_NET_OUTFLOW'] = df[net_mf_amount_col] < 0
            # 连续净流入
            consecutive_days = get_param_value(p.get('ts_consecutive_inflow_days'), 3)
            states['FUND_FLOW_TS_CONSECUTIVE_INFLOW'] = (df[net_mf_amount_col] > 0).rolling(window=consecutive_days, min_periods=consecutive_days).sum() == consecutive_days
            # print(f"          -> Tushare资金流诊断完成。净流入: {states['FUND_FLOW_TS_NET_INFLOW'].sum()} 天。")
        else:
            print(f"          -> [警告] 缺少Tushare资金流列 '{net_mf_amount_col}'，跳过。")

        # --- 2. 同花顺资金流 (moneyflow_ths) ---
        # 字段: net_amount_fund_flow_ths_D (净流入额), buy_lg_amount_rate_fund_flow_ths_D (大单买入占比)
        net_amount_ths_col = 'net_amount_fund_flow_ths_D'
        buy_lg_amount_rate_ths_col = 'buy_lg_amount_rate_fund_flow_ths_D'
        
        # 检查是否存在同花顺资金流的净额列
        if net_amount_ths_col in df.columns:
            states['FUND_FLOW_THS_NET_INFLOW'] = df[net_amount_ths_col] > 0
            states['FUND_FLOW_THS_NET_OUTFLOW'] = df[net_amount_ths_col] < 0
            # print(f"          -> 同花顺资金流诊断完成。净流入: {states['FUND_FLOW_THS_NET_INFLOW'].sum()} 天。")
        else:
            print(f"          -> [警告] 缺少同花顺资金流列 '{net_amount_ths_col}'，跳过。")

        # 检查是否存在同花顺资金流的大单买入占比列
        if buy_lg_amount_rate_ths_col in df.columns:
            large_buy_threshold = get_param_value(p.get('ths_large_buy_rate_threshold'), 0.5) # 例如，大单买入占比超过50%
            states['FUND_FLOW_THS_LARGE_BUY_DOMINANT'] = df[buy_lg_amount_rate_ths_col] > large_buy_threshold
            # print(f"          -> 同花顺大单买入占比诊断完成。大单主导: {states['FUND_FLOW_THS_LARGE_BUY_DOMINANT'].sum()} 天。")
        else:
            print(f"          -> [警告] 缺少同花顺大单买入占比列 '{buy_lg_amount_rate_ths_col}'，跳过。")

        # --- 3. 东方财富资金流 (moneyflow_dc) ---
        # 字段: net_amount_fund_flow_dc_D (净流入额), buy_elg_amount_rate_fund_flow_dc_D (超大单买入占比)
        net_amount_dc_col = 'net_amount_fund_flow_dc_D'
        buy_elg_amount_rate_dc_col = 'buy_elg_amount_rate_fund_flow_dc_D'

        if net_amount_dc_col in df.columns:
            states['FUND_FLOW_DC_NET_INFLOW'] = df[net_amount_dc_col] > 0
            states['FUND_FLOW_DC_NET_OUTFLOW'] = df[net_amount_dc_col] < 0
            # print(f"          -> 东方财富资金流诊断完成。净流入: {states['FUND_FLOW_DC_NET_INFLOW'].sum()} 天。")
        else:
            print(f"          -> [警告] 缺少东方财富资金流列 '{net_amount_dc_col}'，跳过。")

        if buy_elg_amount_rate_dc_col in df.columns:
            super_large_buy_threshold = get_param_value(p.get('dc_super_large_buy_rate_threshold'), 0.3) # 例如，超大单买入占比超过30%
            states['FUND_FLOW_DC_SUPER_LARGE_BUY_DOMINANT'] = df[buy_elg_amount_rate_dc_col] > super_large_buy_threshold
            # print(f"          -> 东方财富超大单买入占比诊断完成。超大单主导: {states['FUND_FLOW_DC_SUPER_LARGE_BUY_DOMINANT'].sum()} 天。")
        else:
            print(f"          -> [警告] 缺少东方财富超大单买入占比列 '{buy_elg_amount_rate_dc_col}'，跳过。")

        print("        -> [资金流情报模块 V1.1 数据层适配版] 诊断完毕。")
        return states
