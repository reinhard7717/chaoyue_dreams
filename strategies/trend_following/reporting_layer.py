# 文件: strategies/trend_following/reporting_layer.py
# 报告层
import pandas as pd
import numpy as np
from asgiref.sync import sync_to_async
from typing import Dict, List, Any, Tuple
from stock_models.stock_analytics import TradingSignal, Playbook, SignalPlaybookDetail, StrategyDailyScore, StrategyScoreComponent, StrategyDailyState
from .utils import get_params_block, get_param_value

def _convert_numpy_types_for_json(obj: Any) -> Any:
    if isinstance(obj, dict): return {key: _convert_numpy_types_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list): return [_convert_numpy_types_for_json(element) for element in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

class ReportingLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.playbooks_cache = None
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})

    async def _ensure_playbooks_cached(self):
        if self.playbooks_cache is not None: return
        try:
            self.playbooks_cache = await sync_to_async(lambda: {p.name: p for p in Playbook.objects.all()}, thread_sensitive=True)()
        except Exception as e:
            self.playbooks_cache = {}
            print(f"    -> [报告层] 警告：异步加载战法定义缓存失败。错误: {e}")

    async def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> Tuple[List, List, List, List, List]:
        """
        【V522.0 · 户籍分离版】
        - 核心升级: 为“先知”和“趋势跟踪”建立独立的户籍系统。
        - 核心逻辑: 在创建 TradingSignal 对象时，根据信号类型（'先知入场' 或其他）
                      从配置中读取并分配正确的 strategy_name。
        """
        await self._ensure_playbooks_cached()
        signals_to_create, signal_details_to_create, daily_scores_to_create, score_components_to_create, daily_states_to_create = [], [], [], [], []
        
        # 分别加载两种策略的配置信息
        trend_follow_strategy_info = get_params_block(self.strategy, 'trend_follow', {}).get('strategy_info', {})
        prophet_strategy_info = get_params_block(self.strategy, 'prophet_oracle', {}).get('strategy_info', {})
        
        save_all_days = get_param_value(trend_follow_strategy_info.get('save_all_days'), False)
        save_daily_states = get_param_value(trend_follow_strategy_info.get('save_daily_states'), False)
        
        # 获取两种策略的名称
        trend_follow_name = get_param_value(trend_follow_strategy_info.get('name'), 'TrendFollow')
        prophet_name = get_param_value(prophet_strategy_info.get('name'), 'ProphetSignal')

        signal_type_map_enum = {
            '买入信号': TradingSignal.SignalType.BUY,
            '卖出信号': TradingSignal.SignalType.SELL,
            '风险预警': TradingSignal.SignalType.WARN,
            '趋势破位离场': TradingSignal.SignalType.SELL,
            '战略失效离场': TradingSignal.SignalType.SELL,
            '风险否决': TradingSignal.SignalType.WARN,
            '先知离场': TradingSignal.SignalType.SELL,
            '先知入场': TradingSignal.SignalType.BUY,
        }
        known_signal_types = list(signal_type_map_enum.keys())
        signal_days_df = result_df[result_df['signal_type'].isin(known_signal_types)].copy()
        for trade_time, row in signal_days_df.iterrows():
            signal_enum = signal_type_map_enum.get(row['signal_type'], TradingSignal.SignalType.WARN)
            risk_score_val = row.get('risk_score', 0.0)
            db_risk_score = risk_score_val * 1000 if pd.notna(risk_score_val) else 0.0
            
            # 核心户籍分离逻辑：根据信号类型分配不同的策略名称
            if row['signal_type'] == '先知入场':
                strategy_name = prophet_name
            else:
                strategy_name = trend_follow_name

            signal_obj = TradingSignal(
                stock_id=stock_code, trade_time=trade_time, timeframe=result_timeframe, strategy_name=strategy_name,
                signal_type=signal_enum,
                entry_score=row.get('entry_score', 0.0), 
                risk_score=db_risk_score,
                final_score=row.get('final_score', 0.0),
                close_price=row.get('close_D', 0.0)
            )
            signals_to_create.append(signal_obj)

            if signal_obj.signal_type == TradingSignal.SignalType.BUY and trade_time in score_details_df.index:
                offensive_details = score_details_df.loc[trade_time][score_details_df.loc[trade_time] > 0]
                for name, score in offensive_details.items():
                    playbook_obj = self.playbooks_cache.get(name)
                    if playbook_obj:
                        signal_details_to_create.append(SignalPlaybookDetail(signal=signal_obj, playbook=playbook_obj, contributed_score=score))
        
        daily_score_map = {}
        summary_score_names = {'SCORE_REVERSAL_OFFENSE', 'SCORE_RESONANCE_OFFENSE', 'SCORE_PLAYBOOK_SYNERGY', 'SCORE_TRIGGER'}
        for trade_time, row in result_df.iterrows():
            # DailyScore 也需要记录正确的策略名称
            if row['signal_type'] == '先知入场':
                daily_score_strategy_name = prophet_name
            else:
                daily_score_strategy_name = trend_follow_name

            offensive_score_val = row.get('entry_score', 0)
            risk_score_val = row.get('risk_score', 0.0)
            db_offensive_score = int(offensive_score_val) if pd.notna(offensive_score_val) else 0
            db_risk_score = int(risk_score_val * 1000) if pd.notna(risk_score_val) else 0
            daily_score_obj = StrategyDailyScore(
                stock_id=stock_code, trade_date=trade_time.date(), strategy_name=daily_score_strategy_name,
                offensive_score=db_offensive_score,
                risk_score=db_risk_score,
                final_score=row.get('final_score', 0.0), signal_type=row.get('signal_type', '无信号'),
                trade_action=row.get('trade_action', StrategyDailyScore.TradeActionType.NO_SIGNAL.value),
                score_details_json=_convert_numpy_types_for_json(row.get('signal_details_cn', {}))
            )
            if save_all_days or (row['signal_type'] != '无信号'):
                daily_scores_to_create.append(daily_score_obj)
            daily_score_map[trade_time] = daily_score_obj
            def create_component(signal_name, score_value):
                playbook_obj = self.playbooks_cache.get(signal_name)
                if not playbook_obj: return
                signal_info = self.score_type_map.get(signal_name, {})
                score_type = signal_info.get('type', 'unknown')
                db_score_value = int(score_value) if pd.notna(score_value) else 0
                if db_score_value < 0: score_type = 'penalty'
                score_components_to_create.append(StrategyScoreComponent(
                    daily_score=daily_score_obj, 
                    playbook=playbook_obj, 
                    score_type=score_type, 
                    score_value=db_score_value
                ))
            if not score_details_df.empty and trade_time in score_details_df.index:
                for name, score in score_details_df.loc[trade_time][score_details_df.loc[trade_time] != 0].items():
                    if name not in summary_score_names: create_component(name, score)

        if save_daily_states:
            for state_dict in [self.strategy.atomic_states, self.strategy.trigger_events, self.strategy.playbook_states]:
                for state_name, state_series in state_dict.items():
                    playbook_obj = self.playbooks_cache.get(state_name)
                    if not playbook_obj: continue
                    for trade_time, is_active in state_series[state_series == True].items():
                        if trade_time in daily_score_map:
                            daily_states_to_create.append(StrategyDailyState(daily_score=daily_score_map[trade_time], playbook=playbook_obj))
        return (signals_to_create, signal_details_to_create, daily_scores_to_create, score_components_to_create, daily_states_to_create)














