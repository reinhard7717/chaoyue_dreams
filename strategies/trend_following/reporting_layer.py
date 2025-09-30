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
        【V515.0 · 坚韧版】
        - 核心加固: 在将分数转换为整数存入数据库前，增加 pd.notna() 检查，为数据链路提供“双重保险”，彻底杜绝因NaN值导致的崩溃。
        """
        await self._ensure_playbooks_cached()
        signals_to_create, signal_details_to_create, daily_scores_to_create, score_components_to_create, daily_states_to_create = [], [], [], [], []
        
        strategy_info = get_params_block(self.strategy, 'strategy_info', {})
        save_all_days = get_param_value(strategy_info.get('save_all_days'), False)
        save_daily_states = get_param_value(strategy_info.get('save_daily_states'), False)
        strategy_name = get_param_value(strategy_info.get('name'), 'TrendFollow')
        
        signal_type_map_enum = {
            '买入信号': TradingSignal.SignalType.BUY,
            '卖出信号': TradingSignal.SignalType.SELL,
            '风险预警': TradingSignal.SignalType.WARN,
            '趋势破位离场': TradingSignal.SignalType.SELL,
            '战略失效离场': TradingSignal.SignalType.SELL,
            '风险否决': TradingSignal.SignalType.WARN,
        }
        
        known_signal_types = list(signal_type_map_enum.keys())
        signal_days_df = result_df[result_df['signal_type'].isin(known_signal_types)].copy()

        for trade_time, row in signal_days_df.iterrows():
            signal_enum = signal_type_map_enum.get(row['signal_type'], TradingSignal.SignalType.WARN)
            
            # [代码新增] 为 risk_score 增加 NaN 防护
            risk_score_val = row.get('risk_score', 0.0)
            db_risk_score = risk_score_val * 1000 if pd.notna(risk_score_val) else 0.0

            signal_obj = TradingSignal(
                stock_id=stock_code, trade_time=trade_time, timeframe=result_timeframe, strategy_name=strategy_name,
                signal_type=signal_enum,
                entry_score=row.get('entry_score', 0.0), 
                risk_score=db_risk_score, # [代码修改] 使用经过防护的 risk_score
                final_score=row.get('final_score', 0.0),
                close_price=row.get('close_D', 0.0)
            )
            signals_to_create.append(signal_obj)
            if trade_time in score_details_df.index:
                offensive_details = score_details_df.loc[trade_time][score_details_df.loc[trade_time] > 0]
                for name, score in offensive_details.items():
                    playbook_obj = self.playbooks_cache.get(name)
                    if playbook_obj:
                        signal_details_to_create.append(SignalPlaybookDetail(signal=signal_obj, playbook=playbook_obj, contributed_score=score))

        daily_score_map = {}
        summary_score_names = {'SCORE_REVERSAL_OFFENSE', 'SCORE_RESONANCE_OFFENSE', 'SCORE_PLAYBOOK_SYNERGY', 'SCORE_TRIGGER'}
        for trade_time, row in result_df.iterrows():
            
            # [代码新增] 为 offensive_score 和 risk_score 增加 NaN 防护
            offensive_score_val = row.get('entry_score', 0)
            risk_score_val = row.get('risk_score', 0.0)
            db_offensive_score = int(offensive_score_val) if pd.notna(offensive_score_val) else 0
            db_risk_score = int(risk_score_val * 1000) if pd.notna(risk_score_val) else 0

            daily_score_obj = StrategyDailyScore(
                stock_id=stock_code, trade_date=trade_time.date(), strategy_name=strategy_name,
                offensive_score=db_offensive_score, # [代码修改] 使用经过防护的 offensive_score
                risk_score=db_risk_score, # [代码修改] 使用经过防护的 risk_score
                final_score=row.get('final_score', 0.0), signal_type=row.get('signal_type', '无信号'),
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
                
                # [代码新增] 为 score_value 增加 NaN 防护
                db_score_value = int(score_value) if pd.notna(score_value) else 0
                if db_score_value < 0: score_type = 'penalty'
                
                score_components_to_create.append(StrategyScoreComponent(
                    daily_score=daily_score_obj, 
                    playbook=playbook_obj, 
                    score_type=score_type, 
                    score_value=db_score_value # [代码修改] 使用经过防护的 score_value
                ))

            if not score_details_df.empty and trade_time in score_details_df.index:
                for name, score in score_details_df.loc[trade_time][score_details_df.loc[trade_time] != 0].items():
                    if name not in summary_score_names: create_component(name, score)
            if not risk_details_df.empty and trade_time in risk_details_df.index:
                for name, score in risk_details_df.loc[trade_time][risk_details_df.loc[trade_time] > 0].items():
                    create_component(name, score)

        if save_daily_states:
            for state_dict in [self.strategy.atomic_states, self.strategy.trigger_events, self.strategy.playbook_states]:
                for state_name, state_series in state_dict.items():
                    playbook_obj = self.playbooks_cache.get(state_name)
                    if not playbook_obj: continue
                    for trade_time, is_active in state_series[state_series == True].items():
                        if trade_time in daily_score_map:
                            daily_states_to_create.append(StrategyDailyState(daily_score=daily_score_map[trade_time], playbook=playbook_obj))
                            
        return (signals_to_create, signal_details_to_create, daily_scores_to_create, score_components_to_create, daily_states_to_create)














