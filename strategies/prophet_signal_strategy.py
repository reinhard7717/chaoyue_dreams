# 文件: strategies/prophet_signal_strategy.py
# 版本: V1.0 · 主权独立版
import pandas as pd
from typing import Dict, List, Tuple
from stock_models.stock_analytics import TradingSignal, StrategyDailyScore
from .trend_following.utils import get_params_block, get_param_value

class ProphetSignalStrategy:
    """
    【V1.0 · 主权独立版】先知信号策略引擎
    - 核心职责: 作为一个独立的、轻量级的策略引擎，消费由主策略计算出的原子状态，
                  并独立生成“先知入场”信号的数据库记录。
    - 架构定位: 与 TrendFollowStrategy 平级的“主权策略”。
    """
    def __init__(self, orchestrator_instance, strategy_config: dict):
        """
        接收由总指挥分发的、纯净的专属配置。
        """
        self.orchestrator = orchestrator_instance
        # 不再从总指挥处继承，而是使用注入的专属配置
        self.unified_config = strategy_config
    async def apply_strategy(self, stock_code: str, df_daily: pd.DataFrame, atomic_states: dict) -> Tuple[List, List, List, List, List]:
        """
        应用先知策略，生成独立的数据库记录。
        :param stock_code: 股票代码。
        :param df_daily: 日线数据DataFrame。
        :param atomic_states: 由主策略引擎计算好的、包含所有原子状态的字典。
        :return: 一个包含五种数据库记录列表的元组。
        """
        p_prophet = get_params_block(self, 'prophet_oracle', {})
        p_judge_prophet = p_prophet.get('judgment_params', {})
        prophet_info = p_prophet.get('strategy_info', {})
        prophet_name = get_param_value(prophet_info.get('name'), 'ProphetSignal')
        prophet_entry_threshold = get_param_value(p_judge_prophet.get('prophet_entry_threshold'), 0.6)
        prophet_score_multiplier = get_param_value(p_judge_prophet.get('prophet_score_multiplier'), 1000)
        # 直接从传入的 atomic_states 中消费情报
        predictive_opp_score = atomic_states.get('PREDICTIVE_OPP_CAPITULATION_REVERSAL', pd.Series(0.0, index=df_daily.index))
        is_prophet_entry = (predictive_opp_score > prophet_entry_threshold)
        prophet_days_df = df_daily[is_prophet_entry].copy()
        if prophet_days_df.empty:
            return ([], [], [], [], []) # 返回空的五元组
        signals_to_create, daily_scores_to_create = [], []
        for trade_time, row in prophet_days_df.iterrows():
            raw_score = predictive_opp_score.get(trade_time, 0.0)
            final_score = raw_score * prophet_score_multiplier
            # 创建 TradingSignal 记录
            signal_obj = TradingSignal(
                stock_id=stock_code,
                trade_time=trade_time,
                timeframe='D',
                strategy_name=prophet_name,
                signal_type=TradingSignal.SignalType.BUY,
                entry_score=final_score,
                risk_score=0.0,
                final_score=final_score,
                close_price=row.get('close_D', 0.0)
            )
            signals_to_create.append(signal_obj)
            # 创建 StrategyDailyScore 记录
            daily_score_obj = StrategyDailyScore(
                stock_id=stock_code,
                trade_date=trade_time.date(),
                strategy_name=prophet_name,
                offensive_score=int(final_score),
                risk_score=0,
                final_score=int(final_score),
                signal_type='先知入场',
                trade_action=StrategyDailyScore.TradeActionType.PROPHET_ENTRY.value,
                score_details_json={'offense': [{'name': '【先知】恐慌投降反转', 'score': int(final_score)}], 'risk': []}
            )
            daily_scores_to_create.append(daily_score_obj)
        # 返回包含两条记录列表的五元组
        return (signals_to_create, [], daily_scores_to_create, [], [])
