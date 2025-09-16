# 文件: strategies/trend_following/reporting_layer.py
# 报告层 (V400.0 - ORM重构版)
import pandas as pd
import numpy as np
from asgiref.sync import sync_to_async
from typing import Dict, List, Any, Tuple
from stock_models.stock_analytics import TradingSignal, Playbook, SignalPlaybookDetail, StrategyDailyScore, StrategyScoreComponent, StrategyDailyState
from .utils import get_params_block, get_param_value

# 辅助函数，用于递归地将数据结构（字典、列表）中的 NumPy 数值类型，转换为 Python 原生类型，以解决 JSON 序列化错误。
def _convert_numpy_types_for_json(obj: Any) -> Any:
    """
    递归遍历数据结构，将numpy的整数和浮点数转换为Python原生类型。
    """
    if isinstance(obj, dict):
        return {key: _convert_numpy_types_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_types_for_json(element) for element in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class ReportingLayer:
    def __init__(self, strategy_instance):
        """
        【V401.0 异步安全版】
        - 核心重构: 采用延迟加载模式。__init__ 不再执行任何数据库操作。
                    缓存将在第一次需要时，通过一个专门的异步方法加载。
        """
        self.strategy = strategy_instance
        self.playbooks_cache = None
        # print(f"    -> [报告层] 初始化完成 (V401.0 异步安全版)，战法定义将延迟加载。")

    async def _ensure_playbooks_cached(self):
        """
        【V401.0 新增】确保战法定义已被缓存的异步方法。
        - 核心逻辑: 如果缓存尚未加载，则以异步安全的方式从数据库加载。
        """
        # 如果缓存已加载，则直接返回，避免重复查询
        if self.playbooks_cache is not None:
            return
        # print("    -> [报告层] 首次需要，正在异步加载战法定义缓存...")
        try:
            sync_db_call = lambda: {p.name: p for p in Playbook.objects.all()}
            # 异步执行它
            self.playbooks_cache = await sync_to_async(sync_db_call, thread_sensitive=True)()
            # print(f"    -> [报告层] 已成功缓存 {len(self.playbooks_cache)} 个战法定义。")
        except Exception as e:
            # 如果加载失败，初始化为空字典以保证后续代码不会出错
            self.playbooks_cache = {}
            print(f"    -> [报告层] 警告：异步加载战法定义缓存失败。错误: {e}")

    async def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> Tuple[List, List, List, List, List]:
        """
        【V511.4 · 终极修复版】
        - 核心修复 (本次修改):
          - [逻辑解耦] 彻底将 StrategyScoreComponent (分数组件) 的创建逻辑从 save_daily_states 的判断中剥离。
          - 新逻辑确保了无论 save_daily_states 开关状态如何，只要有分数，其对应的分数组件就一定会被创建并返回。
          - 这保证了即使在生产模式下（save_daily_states=false），调试报告依然能获取到构成总分的详细条目。
        - 收益: 完美解决了“生产模式下调试报告无明细”的问题，同时保留了节省空间的优势。
        """
        await self._ensure_playbooks_cached()
        signals_to_create = []
        signal_details_to_create = []
        daily_scores_to_create = []
        score_components_to_create = []
        daily_states_to_create = []
        strategy_info = params.get('strategy_params', {}).get('trend_follow', {}).get('strategy_info', {})
        save_all_days = get_param_value(strategy_info.get('save_all_days'), False)
        save_daily_states = get_param_value(strategy_info.get('save_daily_states'), False)
        strategy_name = get_param_value(strategy_info.get('name'), 'TrendFollow')
        # 现在路径修复后，这里能拿到完整的信号字典
        scoring_params = params.get('strategy_params', {}).get('trend_follow', {}).get('four_layer_scoring_params', {})
        score_type_map = scoring_params.get('score_type_map', {})
       
        # --- Part 1: 生成 TradingSignal (逻辑不变) ---
        signal_days_df = result_df[result_df['signal_type'].isin(['买入信号', '卖出信号', '风险预警'])].copy()
        for trade_time, row in signal_days_df.iterrows():
            signal_type_map_enum = {
                '买入信号': TradingSignal.SignalType.BUY,
                '卖出信号': TradingSignal.SignalType.SELL,
                '风险预警': TradingSignal.SignalType.WARN,
            }
            clean_health_summary = _convert_numpy_types_for_json(row.get('health_change_summary', {}))
            signal_obj = TradingSignal(
                stock_id=stock_code,
                trade_time=trade_time,
                timeframe=result_timeframe,
                strategy_name=strategy_name,
                signal_type=signal_type_map_enum.get(row['signal_type']),
                entry_score=row.get('entry_score', 0.0),
                risk_score=row.get('risk_score', 0.0),
                final_score=row.get('final_score', 0.0),
                veto_votes=int(row.get('veto_votes', 0)),
                close_price=row.get('close_D', 0.0),
                health_change_summary=clean_health_summary
            )
            signals_to_create.append(signal_obj)
            if trade_time in score_details_df.index:
                offensive_details = score_details_df.loc[trade_time][score_details_df.loc[trade_time] > 0]
                for name, score in offensive_details.items():
                    original_name = name.replace('PLAYBOOK_', '').replace('TRIGGER_', '').replace('SETUP_', '')
                    playbook_obj = self.playbooks_cache.get(original_name)
                    if playbook_obj:
                        signal_details_to_create.append(SignalPlaybookDetail(
                            signal=signal_obj,
                            playbook=playbook_obj,
                            contributed_score=score
                        ))
        
        # --- Part 2 & 3: 生成 StrategyDailyScore, StrategyScoreComponent, StrategyDailyState ---
        daily_score_map = {}
        for trade_time, row in result_df.iterrows():
            # --- Part 2: 生成 StrategyDailyScore ---
            trade_action_value = row.get('trade_action', StrategyDailyScore.TradeActionType.NO_SIGNAL.value)
            if trade_action_value not in StrategyDailyScore.TradeActionType.values:
                print(f"    -> [报告层-警告] 日期 {trade_time.date()} 的 trade_action '{trade_action_value}' 不在有效选项中，将使用默认值 'NO_SIGNAL'。")
                trade_action_value = StrategyDailyScore.TradeActionType.NO_SIGNAL.value
            
            setup_score = score_details_df.get('SCORE_SETUP', pd.Series(0, index=result_df.index)).get(trade_time, 0)
            trigger_score = score_details_df.get('SCORE_TRIGGER', pd.Series(0, index=result_df.index)).get(trade_time, 0)
            playbook_synergy_score = score_details_df.get('SCORE_PLAYBOOK_SYNERGY', pd.Series(0, index=result_df.index)).get(trade_time, 0)

            daily_score_obj = StrategyDailyScore(
                stock_id=stock_code,
                trade_date=trade_time.date(),
                strategy_name=strategy_name,
                offensive_score=int(row.get('entry_score', 0)),
                risk_score=int(row.get('risk_score', 0)),
                final_score=row.get('final_score', 0.0),
                positional_score=int(setup_score),
                dynamic_score=int(trigger_score),
                composite_score=int(playbook_synergy_score),
                signal_type=row.get('signal_type', '无信号'),
                score_details_json={},
                trade_action=trade_action_value
            )
            if save_all_days or (row['signal_type'] != '无信号'):
                daily_scores_to_create.append(daily_score_obj)
            daily_score_map[trade_time] = daily_score_obj

            # --- Part 2.1: 生成 StrategyScoreComponent (逻辑解耦，永远执行) ---
            all_details_for_json = {}
            
            def create_component(signal_name, score_value):
                nonlocal all_details_for_json
                prefixes_to_strip = ['SETUP_', 'TRIGGER_', 'PLAYBOOK_', 'DYN_', 'STRATEGIC_']
                base_signal_name = signal_name
                for prefix in prefixes_to_strip:
                    if signal_name.startswith(prefix):
                        base_signal_name = signal_name[len(prefix):]
                        break
                # 从战法缓存中获取 Playbook 对象实例
                playbook_obj = self.playbooks_cache.get(base_signal_name)
                # 如果在缓存中找不到对应的战法定义，则跳过此条，避免数据库出错
                if not playbook_obj:
                    print(f"    -> [报告层-警告] 在Playbook缓存中未找到 '{base_signal_name}' 的定义，跳过创建其分数组件。")
                    return
                signal_info = score_type_map.get(base_signal_name, {})
                score_type = signal_info.get('type', 'unknown')
                cn_name = signal_info.get('cn_name', base_signal_name)
                if score_value < 0:
                    score_type = 'penalty'
                score_components_to_create.append(StrategyScoreComponent(
                    daily_score=daily_score_obj,
                    playbook=playbook_obj,  # 使用 playbook 对象替代旧的 signal_name 和 signal_cn_name
                    score_type=score_type,
                    score_value=int(score_value)
                ))
                if score_type not in all_details_for_json:
                    all_details_for_json[score_type] = []
                all_details_for_json[score_type].append({'name': cn_name, 'score': int(score_value)})
            if not score_details_df.empty and trade_time in score_details_df.index:
                offensive_components = score_details_df.loc[trade_time]
                active_offensive = offensive_components[offensive_components != 0]
                for signal_name, score_value in active_offensive.items():
                    create_component(signal_name, score_value)
            if not risk_details_df.empty and trade_time in risk_details_df.index:
                risk_components = risk_details_df.loc[trade_time]
                active_risk = risk_components[risk_components > 0]
                for signal_name, score_value in active_risk.items():
                    create_component(signal_name, score_value)
            daily_score_obj.score_details_json = all_details_for_json
        # --- Part 3: 生成 StrategyDailyState (由 save_daily_states 控制) ---
        if save_daily_states:
            print(f"  [探针-报告层] 每日状态保存功能已开启，将为调试生成详细状态记录。")
            for trade_time, daily_score_obj in daily_score_map.items():
                for state_name, state_series in self.strategy.atomic_states.items():
                    if state_series.get(trade_time, False):
                        playbook_obj = self.playbooks_cache.get(signal_name)
                        if not playbook_obj:
                            print(f"    -> [报告层-警告] 在Playbook缓存中未找到 '{signal_name}' 的定义，跳过创建其每日状态。")
                            continue
                        daily_states_to_create.append(StrategyDailyState(
                            daily_score=daily_score_obj,
                            playbook=playbook_obj,  # 使用 playbook 对象
                            signal_type=signal_type
                        ))
                for trigger_name, trigger_series in self.strategy.trigger_events.items():
                    if trigger_series.get(trade_time, False):
                        daily_states_to_create.append(StrategyDailyState(
                            daily_score=daily_score_obj,
                            signal_name=trigger_name,
                            signal_cn_name=score_type_map.get(trigger_name, {}).get('cn_name', trigger_name),
                            signal_type=StrategyDailyState.SignalType.TRIGGER
                        ))
        else:
            # print(f"  [探针-报告层] 每日状态保存功能已关闭，跳过生成详细状态记录以节省空间。")
            pass
        
        print(f"  [探针-报告层] 股票 {stock_code}: 准备返回 {len(signals_to_create)} 条交易信号, "
            f"{len(daily_scores_to_create)} 条每日分数, "
            f"{len(score_components_to_create)} 条分数组件, "
            f"{len(daily_states_to_create)} 条每日状态。")
        return (signals_to_create, signal_details_to_create, daily_scores_to_create, score_components_to_create, daily_states_to_create)




